import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import WavLMModel, AutoFeatureExtractor
import soundfile as sf
import warnings

# -------------------- CONFIG --------------------
MODEL_NAME = "microsoft/wavlm-base-plus"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SPLIT = "dev"  # change to "train" / "test" as needed

# Paths
CSV_PATH = f"/home/bubai-maji/bubai/English/Processed_csv/dev_segment_metadata.csv"
#CSV_PATH = f"/home/bubai-maji/bubai/English/Processed_csv/train_balanced_metadata.csv"
BASE_DIR = "/home/bubai-maji/bubai"
SAVE_DIR = os.path.join(
    "/home/bubai-maji/bubai/English/features_npy",
    MODEL_NAME.replace("/", "-"),
    SPLIT
)
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- WARNINGS --------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------- LOAD MODEL --------------------
print(f"Loading model: {MODEL_NAME} on {DEVICE}")
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False

hidden_size = getattr(model.config, "hidden_size", 768)
print(f"Model loaded successfully | Hidden size = {hidden_size}\n")

# -------------------- SAFE AUDIO LOADER --------------------
def safe_load(filepath):
    """Safely load audio using torchaudio, fallback to soundfile."""
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception:
        waveform_np, sr = sf.read(filepath, dtype="float32")
        if len(waveform_np.shape) == 1:
            waveform = torch.tensor(waveform_np).unsqueeze(0)
        else:
            waveform = torch.tensor(waveform_np.mean(axis=1)).unsqueeze(0)
    return waveform, sr

# -------------------- FEATURE EXTRACTION --------------------
def extract_segment_feature(filepath):
    """Extract pooled (mean + std) WavLM embeddings (multi-layer averaged)."""
    waveform, sr = safe_load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)  # (num_layers, 1, T, D)

        # Average middle-to-late layers (WavLM-large has 24 layers)
        if hidden_states.shape[0] >= 18:
            selected = hidden_states[12:18]  # good range for emotion/depression cues
        else:
            selected = hidden_states[-4:]    # fallback for smaller models

        hidden = selected.mean(0).squeeze(0)  # (T, D)

        # Statistical pooling
        mean_vec = hidden.mean(dim=0)
        std_vec = hidden.std(dim=0)
        pooled = torch.cat([mean_vec, std_vec])
        return pooled.cpu().numpy()

# -------------------- LOAD METADATA --------------------
df = pd.read_csv(CSV_PATH)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

print(f"Loaded {len(df)} segments for {SPLIT} set.\n")

# -------------------- SPEAKER-LEVEL AGGREGATION --------------------
speaker_features, speaker_labels, speaker_scores = {}, {}, {}

for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {SPLIT} features"):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_segment_feature(path)
        if pid not in speaker_features:
            speaker_features[pid] = []
            speaker_labels[pid] = label
            speaker_scores[pid] = score
        speaker_features[pid].append(feat)
    except Exception as e:
        print(f"[Skip] {path}: {e}")

# -------------------- POOL TO SPEAKER LEVEL --------------------
speaker_ids, X, y, scores, speaker_index = [], [], [], [], []
for idx, pid in enumerate(speaker_features.keys()):
    feats = np.stack(speaker_features[pid])  # (N_segments, D)
    pooled_feat = feats.mean(axis=0)
    X.append(pooled_feat)
    y.append(speaker_labels[pid])
    scores.append(speaker_scores[pid])
    speaker_ids.append(pid)
    speaker_index.append(idx)

X = np.stack(X)
y = np.array(y)
scores = np.array(scores)
speaker_ids = np.array(speaker_ids)
speaker_index = np.array(speaker_index)

# -------------------- SAVE --------------------
np.save(os.path.join(SAVE_DIR, "speaker_X.npy"), X)
np.save(os.path.join(SAVE_DIR, "speaker_y.npy"), y)
np.save(os.path.join(SAVE_DIR, "speaker_score.npy"), scores)
np.save(os.path.join(SAVE_DIR, "speaker_id.npy"), speaker_ids)
np.save(os.path.join(SAVE_DIR, "speaker_index.npy"), speaker_index)

print(f"\n Speaker-level WavLM feature extraction complete for {SPLIT} set.")
print(f"Speakers processed: {len(speaker_ids)}")
print(f"Saved in: {SAVE_DIR}")
print(f"Feature dimension: {X.shape[1]} (2×{hidden_size})\n")
