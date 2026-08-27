import os
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch
from funasr import AutoModel
import warnings

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "iic/emotion2vec_plus_large"  # or "iic/emotion2vec"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
SPLIT = "dev"  

# Paths
CSV_PATH = f"/home/bubai-maji/bubai/English/Processed_csv/dev_segment_metadata"
BASE_DIR = "/home/bubai-maji/bubai"
SAVE_DIR = os.path.join(
    "/home/bubai-maji/bubai/English/features_npy",
    MODEL_NAME.replace("/", "-"),
    SPLIT
)
os.makedirs(SAVE_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

print(f"Loading Emotion2Vec model: {MODEL_NAME} on {DEVICE}")
model = AutoModel(model=MODEL_NAME, device=DEVICE)
print("Model loaded successfully.\n")

# ======================================================
# SAFE AUDIO LOADER
# ======================================================
def safe_load(filepath):
    waveform, sr = sf.read(filepath)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, sr

# ======================================================
# FEATURE EXTRACTION (Emotion2Vec)
# ======================================================
def extract_emotion2vec_feature(filepath):
    """Extract utterance-level emotion embedding."""
    result = model.generate(
        input=filepath,
        output_dir=None,
        granularity="utterance",
        extract_embedding=True
    )
    if isinstance(result, list):
        result = result[0]
    if "feats" in result:
        emb = np.array(result["feats"]).squeeze()
    else:
        raise ValueError(f"Unexpected output keys: {list(result.keys())}")
    return emb

# ======================================================
# LOAD METADATA
# ======================================================
df = pd.read_csv(CSV_PATH)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

print(f"Loaded {len(df)} audio segments for {SPLIT} set.\n")

# ======================================================
# SPEAKER-LEVEL FEATURE EXTRACTION
# ======================================================
speaker_features, speaker_labels, speaker_scores = {}, {}, {}

for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {SPLIT} features"):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    seg_path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_emotion2vec_feature(seg_path)
        if pid not in speaker_features:
            speaker_features[pid] = []
            speaker_labels[pid] = label
            speaker_scores[pid] = score
        speaker_features[pid].append(feat)
    except Exception as e:
        print(f"[Skip] {seg_path}: {e}")

# ======================================================
# POOL TO SPEAKER LEVEL
# ======================================================
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

# ======================================================
# SAVE FEATURES
# ======================================================
np.save(os.path.join(SAVE_DIR, "speaker_X.npy"), X)
np.save(os.path.join(SAVE_DIR, "speaker_y.npy"), y)
np.save(os.path.join(SAVE_DIR, "speaker_score.npy"), scores)
np.save(os.path.join(SAVE_DIR, "speaker_id.npy"), speaker_ids)
np.save(os.path.join(SAVE_DIR, "speaker_index.npy"), speaker_index)

print(f"\n Emotion2Vec+ feature extraction complete for {SPLIT} set.")
print(f"Speakers processed: {len(speaker_ids)}")
print(f"Saved in: {SAVE_DIR}")
print(f"Feature dimension: {X.shape[1]}")
