import os
import pandas as pd
from pydub import AudioSegment
from tqdm import tqdm

# -----------------------------------
# Config
# -----------------------------------
CSV_PATH = "/home/bubai-maji/bubai/Itali/Book1.csv"
AUDIO_ROOT = "/home/bubai-maji/bubai/Itali/Interview-Task/audio_clip"
FOLD_COLS = ["fold1", "fold2", "fold3", "fold4", "fold5"]
SEG_DIR = "Itali/segmented_audio"
os.makedirs(SEG_DIR, exist_ok=True)

segment_meta = []
df = pd.read_csv(CSV_PATH)

# -----------------------------------
# Segment Audio Files
# -----------------------------------
for fold_idx, col in enumerate(FOLD_COLS, start=1):
    fold_dir = os.path.join(SEG_DIR, f"fold{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    for fld in df[col].dropna().astype(str).str.strip("'"):
        root = os.path.join(AUDIO_ROOT, fld)
        if not os.path.isdir(root):
            continue

        label = 0 if "_C" in fld else 1

        for fname in os.listdir(root):
            if not fname.lower().endswith(".wav"):
                continue

            audio_path = os.path.join(root, fname)
            audio = AudioSegment.from_wav(audio_path)
            duration_ms = len(audio)

            for i, start in enumerate(range(0, duration_ms, 4000)):
                end = min(start + 4000, duration_ms)
                segment = audio[start:end]

                if len(segment) < 500:  # discard if <0.5s
                    continue

                seg_name = f"{fld}_{fname.replace('.wav','')}_seg{i:03}.wav"
                seg_path = os.path.join(fold_dir, seg_name)
                segment.export(seg_path, format="wav")

                # Extract speaker/session ID
                base = os.path.basename(seg_name)
                speaker_id = base.split("_seg")[0]  # everything before "_seg"

                segment_meta.append({
                    "fold": fold_idx,
                    "seg_path": seg_path,
                    "label": label,
                    "speaker_id": speaker_id
                })

# -----------------------------------
# Save metadata
# -----------------------------------
seg_df = pd.DataFrame(segment_meta)
seg_df.to_csv("segment_metadata.csv", index=False)
print(f"Saved {len(seg_df)} segments with speaker_id column.")
