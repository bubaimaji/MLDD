import os
import pandas as pd
import shutil

# ======= CONFIG =======
INPUT_CSV = "/home/bubai-maji/bubai/English/Processed_csv_edic/train_segment_metadata.csv"
OUTPUT_CSV = "/home/bubai-maji/bubai/English/Processed_csv_edic/train_balanced_metadata.csv"

INPUT_AUDIO_DIR = "/home/bubai-maji/bubai/English/Processed_edic_segmented/train"
OUTPUT_AUDIO_DIR = "/home/bubai-maji/bubai/English/Processed_edic_segmented/train_balanced"

os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

# ======= LOAD DATA =======
df = pd.read_csv(INPUT_CSV)

# Count subjects per class
subj_counts = df.groupby("phq8_binary")["participant_id"].nunique()
print("Original subject counts:")
print(subj_counts)

maj_class = subj_counts.idxmax()
min_class = subj_counts.idxmin()

maj_subjects = df[df.phq8_binary == maj_class]["participant_id"].unique()
min_subjects = df[df.phq8_binary == min_class]["participant_id"].unique()

target_subject_count = len(maj_subjects)

balanced_rows = []
copy_index = 1

# copy original WAVs + rows
for _, row in df.iterrows():
    src = row["audio_path"]
    dst = os.path.join(OUTPUT_AUDIO_DIR, os.path.basename(src))
    shutil.copy(src, dst)
    balanced_rows.append(row)

# oversample minority subjects — create new pseudo-subject IDs
min_sids = list(min_subjects)
while len(min_sids) < target_subject_count:

    for sid in min_subjects:
        subj_df = df[df.participant_id == sid]

        new_id = int(f"{sid}{copy_index:03d}")  # e.g. 351 → 351001

        for _, row in subj_df.iterrows():
            src = row["audio_path"]
            base = os.path.basename(src)
            new_name = base.replace(".wav", f"_dup{copy_index}.wav")
            dst = os.path.join(OUTPUT_AUDIO_DIR, new_name)

            shutil.copy(src, dst)

            new_row = row.copy()
            new_row["audio_path"] = dst
            new_row["participant_id"] = new_id   
            balanced_rows.append(new_row)

        min_sids.append(new_id)
        if len(min_sids) >= target_subject_count:
            break

    copy_index += 1

# ======= SAVE BALANCED CSV =======
balanced_df = pd.DataFrame(balanced_rows).reset_index(drop=True)
balanced_df.to_csv(OUTPUT_CSV, index=False)

print("\n Balanced dataset created.")
print("Saved WAVs to:", OUTPUT_AUDIO_DIR)
print("Saved CSV to:", OUTPUT_CSV)
print("\nBalanced subject count per class:")
print(balanced_df.groupby("phq8_binary")["participant_id"].nunique())
print("\nBalanced segment count per class:")
print(balanced_df["phq8_binary"].value_counts())
