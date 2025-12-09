import os
import pandas as pd
from pydub import AudioSegment
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
CSV_PATH = "/home/bubai-maji/bubai/Bangla/speaker_id_metadata.csv"

DEP_DIR = "/home/bubai-maji/bubai/Bangla/Data/depression"
HEA_DIR = "/home/bubai-maji/bubai/Bangla/Data/healthy"

OUTPUT_DIR = "/home/bubai-maji/bubai/Bangla/segmented_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEG_MS = 4000
MIN_MS = 500
N_FOLDS = 5

# -------------------------------------------------
# LOAD CSV SAFELY
# -------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Remove trailing empty columns caused by extra commas
df = df.dropna(how="all", axis=1)

# Clean label values
df["label"] = df["label"].astype(str).str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# Clean speaker IDs (prevent spaces)
df["speaker_id"] = df["speaker_id"].astype(str).str.strip()

print("CSV loaded. Speakers:", len(df))

# -------------------------------------------------
# BUILD SPEAKER → FILE MAP
# -------------------------------------------------
file_map = {}

# Depression speakers
for f in os.listdir(DEP_DIR):
    if f.endswith(".wav"):
        spk = f.replace(".wav", "")
        file_map[spk] = os.path.join(DEP_DIR, f)

# Healthy speakers
for f in os.listdir(HEA_DIR):
    if f.endswith(".wav"):
        spk = f.replace(".wav", "")
        file_map[spk] = os.path.join(HEA_DIR, f)

# Keep only speakers that exist in dataset
df = df[df["speaker_id"].isin(file_map.keys())].reset_index(drop=True)

print("Speakers found in audio folders:", len(df))

# -------------------------------------------------
# CREATE 3 STRATIFIED FOLDS
# -------------------------------------------------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
df["fold"] = -1

for i, (_, test_idx) in enumerate(skf.split(df["speaker_id"], df["label"]), start=1):
    df.loc[test_idx, "fold"] = i

print("\nFold distribution:")
print(df.groupby(["fold", "label"]).size())

# -------------------------------------------------
# SEGMENT AUDIO
# -------------------------------------------------
segments = []

for fold_i in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(OUTPUT_DIR, f"fold{fold_i}")
    os.makedirs(fold_dir, exist_ok=True)

    fold_spk = df[df["fold"] == fold_i]

    for _, row in tqdm(fold_spk.iterrows(), total=len(fold_spk), desc=f"Fold {fold_i}"):

        spk = row["speaker_id"]
        gender = row["gender"]
        label = row["label"]
        wav_path = file_map[spk]

        audio = AudioSegment.from_wav(wav_path)
        duration = len(audio)

        for i, start in enumerate(range(0, duration, SEG_MS)):
            end = min(start + SEG_MS, duration)
            seg = audio[start:end]

            if len(seg) < MIN_MS:
                continue

            seg_name = f"{spk}_seg{i:03}.wav"
            seg_path = os.path.join(fold_dir, seg_name)

            seg.export(seg_path, format="wav")

            utterance_id = seg_name.split("_seg")[0]

            segments.append({
                "fold": fold_i,
                "speaker_id": spk,
                "gender": gender,
                "label": label,
                "seg_path": seg_path,
                "utterance_id": utterance_id
            })

meta = pd.DataFrame(segments)
meta.to_csv("bangla_5fold_metadata.csv", index=False)

print("\nProcessing complete!")
print("Total segments saved:", len(meta))
