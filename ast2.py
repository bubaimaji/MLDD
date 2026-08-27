import os
import warnings
import torch
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

# -------------------- Config --------------------
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SPLIT = "test"

BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv_edic/test_segment_metadata.csv"
SAVE_DIR = os.path.join(
    BASE_DIR, "English/edic_features_npy",
    MODEL_NAME.replace("/", "-"), SPLIT
)
os.makedirs(SAVE_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")

# -------------------- Load Model --------------------
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False

hidden_size = getattr(model.config, "hidden_size", 768)
print(f"Loaded model. Hidden size = {hidden_size}\n")

# -------------------- Safe Audio Loader --------------------
def safe_load(filepath):
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception:
        x, sr = sf.read(filepath, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        waveform = torch.tensor(x).unsqueeze(0)
    return waveform, sr

# -------------------- Feature Extraction --------------------
def extract_segment_feature(filepath):
    wav, sr = safe_load(filepath)

    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample to 16k (AST expects 16k via HF feature extractor)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    inputs = processor(wav.squeeze(), sampling_rate=sr, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state          # (1, T, D)
        feat = last_hidden.mean(dim=1).cpu().numpy().flatten()  # (D,)

    return feat

# -------------------- Load Metadata --------------------
df = pd.read_csv(CSV_PATH)
required = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV missing columns: {missing}")
print(f"Loaded {len(df)} segments for split = {SPLIT}\n")

# -------------------- Storage --------------------
segment_feats = []
segment_labels = []
segment_scores = []
segment_speakers = []
segment_index = []

# -------------------- Loop --------------------
for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {SPLIT}"):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    # ensure absolute path
    seg_path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_segment_feature(seg_path)
        segment_feats.append(feat)
        segment_labels.append(label)
        segment_scores.append(score)
        segment_speakers.append(pid)
        segment_index.append(i)
    except Exception as e:
        print(f"[Skip] {seg_path}: {e}")

# -------------------- Save --------------------
segment_feats = np.stack(segment_feats)
np.save(os.path.join(SAVE_DIR, "segment_X.npy"), segment_feats)
np.save(os.path.join(SAVE_DIR, "segment_y.npy"), np.array(segment_labels))
np.save(os.path.join(SAVE_DIR, "segment_score.npy"), np.array(segment_scores))
np.save(os.path.join(SAVE_DIR, "segment_speaker_id.npy"), np.array(segment_speakers))
np.save(os.path.join(SAVE_DIR, "segment_index.npy"), np.array(segment_index))

print("\n Segment-level AST feature extraction DONE!")
print(f"Saved to: {SAVE_DIR}")
print(f"Embeddings shape: {segment_feats.shape}  (segments × {hidden_size})")
