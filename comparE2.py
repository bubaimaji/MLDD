import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import opensmile

SPLIT = "train"  
#SPLIT = "dev" 

BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv/train_balanced_metadata.csv"
#CSV_PATH = f"{BASE_DIR}/English/Processed_csv/dev_segment_metadata.csv"

OUTPUT_DIR = f"{BASE_DIR}/English/features_npy/IS10_paraling/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in metadata: {missing}")

print(f" Loaded {len(df)} segments for '{SPLIT}'")
print(f"Columns: {list(df.columns)}\n")

# -------------------- INIT OPENSMILE --------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals
)

#smile = opensmile.Smile(
    #feature_set=opensmile.FeatureSet.eGeMAPSv02,
    #feature_level=opensmile.FeatureLevel.Functionals
#)

print("Loaded openSMILE ComParE 2016 Functionals (1582-D)\n")

# -------------------- EXTRACT FEATURES (segment-level) --------------------
segment_feats = []
segment_labels = []
segment_scores = []
segment_speakers = []
segment_idx = []

print(f" Extracting features for {SPLIT} segments...\n")

for i, row in tqdm(df.iterrows(), total=len(df)):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    seg_path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        features_df = smile.process_file(seg_path)
        feat_vec = features_df.values.flatten()  # shape (1582)
        
        segment_feats.append(feat_vec)
        segment_labels.append(label)
        segment_scores.append(score)
        segment_speakers.append(pid)
        segment_idx.append(i)

    except Exception as e:
        print(f"[Skip] {seg_path}: {e}")

segment_feats = np.stack(segment_feats)
segment_labels = np.array(segment_labels)
segment_scores = np.array(segment_scores)
segment_speakers = np.array(segment_speakers)
segment_idx = np.array(segment_idx)

# -------------------- SAVE --------------------
np.save(f"{OUTPUT_DIR}/segment_X.npy", segment_feats)
np.save(f"{OUTPUT_DIR}/segment_y.npy", segment_labels)
np.save(f"{OUTPUT_DIR}/segment_score.npy", segment_scores)
np.save(f"{OUTPUT_DIR}/segment_speaker_id.npy", segment_speakers)
np.save(f"{OUTPUT_DIR}/segment_index.npy", segment_idx)

# -------------------- SUMMARY --------------------
print("\n Segment-level ComParE feature extraction DONE!")
print(f"Split: {SPLIT}")
print(f"Segments processed: {len(segment_feats)}")
print(f"Feature shape: {segment_feats.shape} ")
print(f"Saved features in: {OUTPUT_DIR}\n")

print("Saved files:")
print(f"  segment_X.npy            → features")
print(f"  segment_y.npy            → labels")
print(f"  segment_score.npy        → PHQ scores")
print(f"  segment_speaker_id.npy   → speaker IDs")
print(f"  segment_index.npy        → segment index\n")


