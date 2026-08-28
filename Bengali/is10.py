import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import opensmile

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
metadata_csv = "/home/bubai-maji/bubai/Bangla/bangla_5fold_metadata.csv"
df = pd.read_csv(metadata_csv)

required_cols = {
    "seg_path",
    "label",
    "fold",
    "speaker_id",
    "utterance_id"
}

if not required_cols.issubset(df.columns):
    raise ValueError(
        f"Metadata must contain columns: {required_cols}"
    )

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
output_dir = (
    "/home/bubai-maji/bubai/bangla_feature/"
    "IS10_LLD_5fold"
)

os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# OpenSMILE IS10 LOW-LEVEL DESCRIPTORS
# -------------------------------------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

print("Extracting temporal IS10 LLD features...")

# -------------------------------------------------
# EXTRACT FOLD BY FOLD
# -------------------------------------------------
for fold in sorted(df["fold"].unique()):

    fold_df = df[df["fold"] == fold]

    X_temporal = []
    y = []
    speaker_ids = []
    utterance_ids = []
    lengths = []

    for row in tqdm(
        fold_df.itertuples(),
        total=len(fold_df),
        desc=f"Fold {fold}"
    ):

        try:

            # -----------------------------------------
            # Extract frame-level IS10 descriptors
            # -----------------------------------------
            features_df = smile.process_file(row.seg_path)

            # [T, D]
            features = features_df.values.astype(
                np.float32
            )

            X_temporal.append(features)

            y.append(int(row.label))
            speaker_ids.append(row.speaker_id)
            utterance_ids.append(row.utterance_id)

            lengths.append(features.shape[0])

        except Exception as e:

            print(
                f"[Skip] {row.seg_path}: {e}"
            )

    # -------------------------------------------------
    # VARIABLE-LENGTH SEQUENCES
    # -------------------------------------------------
    X_temporal = np.array(
        X_temporal,
        dtype=object
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_X_temporal.npy"
        ),
        X_temporal,
        allow_pickle=True
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_y.npy"
        ),
        np.array(y)
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_speaker.npy"
        ),
        np.array(speaker_ids)
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_utterance.npy"
        ),
        np.array(utterance_ids)
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_lengths.npy"
        ),
        np.array(lengths)
    )

    print(
        f"Saved fold {fold}: "
        f"samples={len(X_temporal)}, "
        f"speakers={len(set(speaker_ids))}, "
        f"utterances={len(set(utterance_ids))}, "
        f"T_min={min(lengths)}, "
        f"T_max={max(lengths)}, "
        f"T_mean={np.mean(lengths):.1f}, "
        f"D={X_temporal[0].shape[1]}"
    )

print("\nTemporal IS10 extraction complete.")
print("Saved in:", output_dir)
