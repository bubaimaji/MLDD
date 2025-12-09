import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import opensmile

metadata_csv = "/home/bubai-maji/bubai/Bangla/bangla_5fold_metadata.csv"
df = pd.read_csv(metadata_csv)

# Ensure necessary columns exist
required_cols = {'seg_path', 'label', 'fold', 'speaker_id', 'utterance_id'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Metadata must contain columns: {required_cols}")

# ---------------------------------------
# Step 2: Setup output directory
# ---------------------------------------
output_dir = "Bangla/bangla_features_npy/IS10_5fold"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------
# Step 3: Initialize OpenSMILE
# ---------------------------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals
)

# ---------------------------------------
# Step 4: Extract features fold by fold
# ---------------------------------------
print("Extracting eGeMAPSv02 Functionals per segment...")

for fold in sorted(df['fold'].unique()):
    fold_df = df[df['fold'] == fold]
    X, y, speaker_ids, utterance_ids = [], [], [], []

    for row in tqdm(fold_df.itertuples(), total=len(fold_df), desc=f"Fold {fold}"):
        try:
            # Extract 88-dim feature vector
            features_df = smile.process_file(row.seg_path)
            features = features_df.values.flatten()
            X.append(features)
            y.append(int(row.label))
            speaker_ids.append(row.speaker_id)
            utterance_ids.append(row.utterance_id)
        except Exception as e:
            print(f"[Skip] {row.seg_path}: {e}")

    # Save per-fold data
    np.save(os.path.join(output_dir, f"fold{fold}_X.npy"), np.stack(X))
    np.save(os.path.join(output_dir, f"fold{fold}_y.npy"), np.array(y))
    np.save(os.path.join(output_dir, f"fold{fold}_speaker.npy"), np.array(speaker_ids))
    np.save(os.path.join(output_dir, f"fold{fold}_utterance.npy"), np.array(utterance_ids))

    print(f"Saved fold{fold}: X={len(X)}, y={len(y)}, speakers={len(set(speaker_ids))}, utterances={len(set(utterance_ids))}")

print("\n Feature extraction complete.")
print("Saved in:", output_dir)
