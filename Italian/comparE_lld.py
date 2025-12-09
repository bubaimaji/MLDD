import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import opensmile

# ---------------------------------------
# Step 1: Load metadata CSV
# ---------------------------------------
metadata_csv = "/home/bubai-maji/bubai/Itali/speaker_metadata.csv"
df = pd.read_csv(metadata_csv)

required_cols = {'seg_path', 'label', 'fold', 'speaker_id', 'utterance_id'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Metadata must contain columns: {required_cols}")

# ---------------------------------------
# Step 2: Setup output directory
# ---------------------------------------
output_dir = "Itali/features_npy/seg_ComParE_LLD_mean"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# Step 3: Initialize OpenSMILE
# ---------------------------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors
)

# ---------------------------------------
# Step 4: Extract features fold by fold
# ---------------------------------------
print("Extracting eGeMAPSv02 LLDs (mean pooled)...")

for fold in sorted(df['fold'].unique()):
    fold_df = df[df['fold'] == fold]
    X, y, speaker_ids, utterance_ids = [], [], [], []

    for row in tqdm(fold_df.itertuples(), total=len(fold_df), desc=f"Fold {fold}"):
        try:
            # Each segment → frame-level feature matrix (num_frames, num_features)
            features_df = smile.process_file(row.seg_path)
            features = features_df.values  # (T, D)

            # Mean pooling only
            mean = features.mean(axis=0)  # shape: (D,)
            X.append(mean)
            y.append(int(row.label))
            speaker_ids.append(row.speaker_id)
            utterance_ids.append(row.utterance_id)

        except Exception as e:
            print(f"[Skip] {row.seg_path}: {e}")

    # Save arrays
    np.save(os.path.join(output_dir, f"fold{fold}_X.npy"), np.stack(X))
    np.save(os.path.join(output_dir, f"fold{fold}_y.npy"), np.array(y))
    np.save(os.path.join(output_dir, f"fold{fold}_speaker.npy"), np.array(speaker_ids))
    np.save(os.path.join(output_dir, f"fold{fold}_utterance.npy"), np.array(utterance_ids))

    print(f"Saved Fold {fold} — Segments: {len(X)}, Speakers: {len(set(speaker_ids))}, Utterances: {len(set(utterance_ids))}")

print("\n eGeMAPSv02 LLD (mean-pooled) feature extraction complete.")
print("Saved in:", output_dir)
