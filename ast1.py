import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
import soundfile as sf
import warnings

# -------------------- Config --------------------
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SPLIT = "dev"
# Paths
CSV_PATH = "/home/bubai-maji/bubai/English/Processed_csv/dev_segment_metadata.csv"
BASE_DIR = "/home/bubai-maji/bubai"
SAVE_DIR = os.path.join("/home/bubai-maji/bubai/English/features_npy",
                        MODEL_NAME.replace("/", "-"), SPLIT) 

os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- Warnings --------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------------------- Load Model --------------------
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False

hidden_size = getattr(model.config, "hidden_size", 768)
layer_norm = torch.nn.LayerNorm(hidden_size).to(DEVICE)

print(f"Loaded model {MODEL_NAME} on {DEVICE}, hidden size = {hidden_size}")

# -------------------- Safe Audio Loader --------------------
def safe_load(filepath):
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception:
        waveform_np, sr = sf.read(filepath)
        if len(waveform_np.shape) == 1:
            waveform = torch.tensor(waveform_np).unsqueeze(0)
        else:
            waveform = torch.tensor(waveform_np.mean(axis=1)).unsqueeze(0)
        waveform = waveform.to(torch.float32)
    return waveform, sr

# -------------------- Feature Extraction --------------------
def extract_segment_feature(filepath):
    waveform, sr = safe_load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)
        normed = layer_norm(last_hidden)
        pooled = normed.mean(dim=0)
    return pooled.cpu().numpy()

# -------------------- Load Metadata --------------------
df = pd.read_csv(CSV_PATH)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")
print(f"Loaded {len(df)} segments.")

# -------------------- Speaker-Level Aggregation --------------------
speaker_features = {}
speaker_labels = {}
speaker_scores = {}

for _, row in tqdm(df.iterrows(), total=len(df)):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    path = row["audio_path"]

    try:
        feat = extract_segment_feature(path)
        if pid not in speaker_features:
            speaker_features[pid] = []
            speaker_labels[pid] = label
            speaker_scores[pid] = score
        speaker_features[pid].append(feat)
    except Exception as e:
        print(f"[Skip] {path}: {e}")

# Pool features per speaker
speaker_ids, X, y, scores, speaker_index = [], [], [], [], []
for idx, pid in enumerate(speaker_features.keys()):
    feats = np.stack(speaker_features[pid])
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

# -------------------- Save --------------------
np.save(os.path.join(SAVE_DIR, "speaker_X.npy"), X)
np.save(os.path.join(SAVE_DIR, "speaker_y.npy"), y)
np.save(os.path.join(SAVE_DIR, "speaker_score.npy"), scores)
np.save(os.path.join(SAVE_DIR, "speaker_id.npy"), speaker_ids)
np.save(os.path.join(SAVE_DIR, "speaker_index.npy"), speaker_index)

print("\n Speaker-level feature extraction complete.")
print(f"Speakers processed: {len(speaker_ids)}")
print(f"Saved in: {SAVE_DIR}")
