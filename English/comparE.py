import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import opensmile

# -------------------- CONFIG --------------------
SPLIT = "dev"  # change to "dev" for dev split
BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"/home/bubai-maji/bubai/English/Processed_csv/dev_segment_metadata.csv"

OUTPUT_DIR = os.path.join(BASE_DIR, "English/features_npy/ComParE_functionals", SPLIT)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- LOAD METADATA --------------------
df = pd.read_csv(CSV_PATH)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

print(f"Loaded {len(df)} segments for '{SPLIT}' split.")
print(f"Columns found: {list(df.columns)}")

# -------------------- INITIALIZE OPENSMILE --------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)
print(" OpenSMILE ComParE_2016 feature set loaded (1582-D)\n")

# -------------------- EXTRACT FEATURES --------------------
speaker_features = {}
speaker_labels = {}
speaker_scores = {}

for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {SPLIT} features"):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    seg_path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        features_df = smile.process_file(seg_path)
        feat_vec = features_df.values.flatten()  # (1582,)
        if pid not in speaker_features:
            speaker_features[pid] = []
            speaker_labels[pid] = label
            speaker_scores[pid] = score
        speaker_features[pid].append(feat_vec)
    except Exception as e:
        print(f"[Skip] {seg_path}: {e}")

# -------------------- AGGREGATE TO SPEAKER LEVEL --------------------
print("\nAggregating segment-level features to speaker-level features...")
speaker_ids, X, y, scores, speaker_index = [], [], [], [], []

for idx, pid in enumerate(speaker_features.keys()):
    feats = np.stack(speaker_features[pid])  # (num_segments, 1582)
    pooled_feat = feats.mean(axis=0)         # mean pooling
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
np.save(os.path.join(OUTPUT_DIR, "speaker_X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "speaker_y.npy"), y)
np.save(os.path.join(OUTPUT_DIR, "speaker_score.npy"), scores)
np.save(os.path.join(OUTPUT_DIR, "speaker_id.npy"), speaker_ids)
np.save(os.path.join(OUTPUT_DIR, "speaker_index.npy"), speaker_index)

# -------------------- PRINT SUMMARY --------------------
print(f"\n Speaker-level ComParE feature extraction complete for '{SPLIT}' set.")
print(f"Speakers processed: {len(speaker_ids)}")
print(f"Feature shape: {X.shape}  (Speakers × Feature_dim)")
print(f"→ Feature dimension per speaker: {X.shape[1]}")
print(f"Saved in: {OUTPUT_DIR}")
