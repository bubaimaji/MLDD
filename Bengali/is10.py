import os
import re
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
    "utterance_id",
}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Metadata must contain columns: {required_cols}")

SEGMENT_ORDER_COL = "segment_index"  
HAS_ORDER_COL = SEGMENT_ORDER_COL is not None and SEGMENT_ORDER_COL in df.columns

# -------------------------------------------------
# OUTPUT
# -------------------------------------------------
output_dir = (
    "/home/bubai-maji/bubai/bangla_feature/"
    "IS10_Functional_5fold"
)
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# OpenSMILE IS10 FUNCTIONALS (1582-dim per segment)
# -------------------------------------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.IS10,
    feature_level=opensmile.FeatureLevel.Functionals,
)
print("Extracting IS10 Functionals (1582-dim) per speech segment...")


def natural_sort_key(path: str):
    
    nums = [int(x) for x in re.findall(r"\d+", os.path.basename(path))]
    return nums if nums else [os.path.basename(path)]


# -------------------------------------------------
# EXTRACT FOLD BY FOLD
# -------------------------------------------------
for fold in sorted(df["fold"].unique()):
    fold_df = df[df["fold"] == fold]

    # ---- per-SEGMENT (flat) storage ----
    X_seg = []
    y_seg = []
    speaker_seg = []
    utterance_seg = []
    segpath_seg = []
    order_seg = []

    for row in tqdm(
        fold_df.itertuples(),
        total=len(fold_df),
        desc=f"Fold {fold}",
    ):
        try:
            feats_df = smile.process_file(row.seg_path)   # -> [1, 1582]
            feats = feats_df.values.astype(np.float32).squeeze(0)  # -> [1582]

            X_seg.append(feats)
            y_seg.append(int(row.label))
            speaker_seg.append(row.speaker_id)
            utterance_seg.append(row.utterance_id)
            segpath_seg.append(row.seg_path)
            order_seg.append(
                getattr(row, SEGMENT_ORDER_COL) if HAS_ORDER_COL else None
            )
        except Exception as e:
            print(f"[Skip] {row.seg_path}: {e}")

    X_seg = np.stack(X_seg).astype(np.float32)          # [N_segments, 1582]
    y_seg = np.array(y_seg)
    speaker_seg = np.array(speaker_seg)
    utterance_seg = np.array(utterance_seg)
    segpath_seg = np.array(segpath_seg)

    np.save(os.path.join(output_dir, f"fold{fold}_X_segments.npy"), X_seg)
    np.save(os.path.join(output_dir, f"fold{fold}_y_segments.npy"), y_seg)
    np.save(os.path.join(output_dir, f"fold{fold}_speaker_segments.npy"), speaker_seg)
    np.save(os.path.join(output_dir, f"fold{fold}_utterance_segments.npy"), utterance_seg)
    np.save(os.path.join(output_dir, f"fold{fold}_segpath_segments.npy"), segpath_seg)

    # ---------------------------------------------
    # GROUP BY UTTERANCE -> sequence of segment
    # vectors, in order, for cross-attention input
    # ---------------------------------------------
    X_utt_seq = []      # object array: each entry [num_segments_i, 1582]
    y_utt = []
    speaker_utt = []
    utterance_ids_utt = []
    lengths_utt = []

    seg_df = pd.DataFrame({
        "utterance_id": utterance_seg,
        "speaker_id": speaker_seg,
        "label": y_seg,
        "seg_path": segpath_seg,
        "order": order_seg,
        "row_idx": np.arange(len(X_seg)),
    })

    for utt_id, g in seg_df.groupby("utterance_id", sort=False):
        if HAS_ORDER_COL:
            g_sorted = g.sort_values("order")
        else:
            g_sorted = g.iloc[
                sorted(range(len(g)), key=lambda i: natural_sort_key(g["seg_path"].iloc[i]))
            ]

        idxs = g_sorted["row_idx"].values
        seq = X_seg[idxs]                     # [num_segments, 1582]
        labels_in_utt = g_sorted["label"].unique()

        if len(labels_in_utt) > 1:
            print(f"[Warn] utterance {utt_id} has mixed labels {labels_in_utt}; "
                  f"using the first segment's label.")

        X_utt_seq.append(seq)
        y_utt.append(int(labels_in_utt[0]))
        speaker_utt.append(g_sorted["speaker_id"].iloc[0])
        utterance_ids_utt.append(utt_id)
        lengths_utt.append(seq.shape[0])

    X_utt_seq = np.array(X_utt_seq, dtype=object)

    np.save(os.path.join(output_dir, f"fold{fold}_X_utterance_seq.npy"),
            X_utt_seq, allow_pickle=True)
    np.save(os.path.join(output_dir, f"fold{fold}_y_utterance.npy"), np.array(y_utt))
    np.save(os.path.join(output_dir, f"fold{fold}_speaker_utterance.npy"), np.array(speaker_utt))
    np.save(os.path.join(output_dir, f"fold{fold}_utterance_ids.npy"), np.array(utterance_ids_utt))
    np.save(os.path.join(output_dir, f"fold{fold}_segcount_per_utterance.npy"), np.array(lengths_utt))

    print(
        f"Saved fold {fold}: "
        f"segments={len(X_seg)}, "
        f"utterances={len(X_utt_seq)}, "
        f"speakers={len(set(speaker_seg))}, "
        f"segs/utt_min={min(lengths_utt)}, "
        f"segs/utt_max={max(lengths_utt)}, "
        f"segs/utt_mean={np.mean(lengths_utt):.1f}, "
        f"D={X_seg.shape[1]}"
    )

print("\nIS10 Functionals (1582-dim, per-segment) extraction complete.")
print("Saved in:", output_dir)
