import os
import pandas as pd
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm

# ================== CONFIG ==================
INPUT_CSV = "/home/bubai-maji/bubai/English/Processed_csv_edic/dev_clean_audio.csv"
OUTPUT_SEGMENT_DIR = "/home/bubai-maji/bubai/English/Processed_edic_segmented/dev"
OUTPUT_METADATA_CSV = "/home/bubai-maji/bubai/English/Processed_csv_edic/dev_segment_metadata.csv"
SEGMENT_DURATION = 6.0  

os.makedirs(OUTPUT_SEGMENT_DIR, exist_ok=True)

# ================== LOAD DATA ==================
df = pd.read_csv(INPUT_CSV)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain: {required_cols}")

# ================== SEGMENTATION FUNCTION ==================
def segment_audio(file_path, output_dir, participant_id, label, phq_score, seg_dur=6.0):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        total_duration = len(audio) / sr
        num_segments = int(total_duration // seg_dur)

        metadata_rows = []

        for i in range(num_segments):
            start = int(i * seg_dur * sr)
            end = int((i + 1) * seg_dur * sr)
            segment = audio[start:end]

            seg_filename = f"{participant_id}_seg{i:03d}.wav"
            seg_path = os.path.join(output_dir, seg_filename)
            sf.write(seg_path, segment, sr)

            metadata_rows.append({
                "audio_path": seg_path,
                "participant_id": participant_id,
                "phq8_binary": label,
                "phq8_score": phq_score,
                "segment_index": i
            })

        return metadata_rows

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

# ================== PROCESS ALL FILES ==================
all_metadata = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    file_path = row["audio_path"]
    pid = row["participant_id"]
    label = row["phq8_binary"]
    phq = row["phq8_score"]

    seg_info = segment_audio(file_path, OUTPUT_SEGMENT_DIR, pid, label, phq, SEGMENT_DURATION)
    all_metadata.extend(seg_info)

# ================== SAVE METADATA ==================
meta_df = pd.DataFrame(all_metadata)
meta_df.to_csv(OUTPUT_METADATA_CSV, index=False)

print("\n Segmentation complete!")
print(f" Segments saved to: {OUTPUT_SEGMENT_DIR}")
print(f" Metadata saved to: {OUTPUT_METADATA_CSV}")
print(f" Total segments: {len(meta_df)}")
