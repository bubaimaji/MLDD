import os
import pandas as pd
import soundfile as sf
import numpy as np
import librosa
from tqdm import tqdm
from pydub import AudioSegment, silence
import tempfile

# ================== CONFIG ==================
# Path to CSV that lists cleaned participant audio files (from previous step)
INPUT_CSV = "/home/bubai-maji/bubai/English/Processed_csv_edic/train_clean_audio.csv"   
OUTPUT_SEGMENT_DIR = "/home/bubai-maji/bubai/English/Processed_edic_segmented/train"
OUTPUT_METADATA_CSV = "/home/bubai-maji/bubai/English/Processed_csv_edic/train_segment_metadata.csv"
SEGMENT_DURATION = 6.0  # seconds

# Create output directory
os.makedirs(OUTPUT_SEGMENT_DIR, exist_ok=True)
 
# ================== LOAD DATA ==================   
df = pd.read_csv(INPUT_CSV)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain: {required_cols}")


def remove_silence(audio_path, min_silence_len=300, silence_thresh_db=-35):
    """
    Remove silence from audio and return cleaned file path.
    """
    sound = AudioSegment.from_wav(audio_path)

    # detect non-silent ranges
    nonsilence_ranges = silence.detect_nonsilent(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=sound.dBFS + silence_thresh_db,
        seek_step=5
    )

    if not nonsilence_ranges:
        return audio_path  # return original if no speech detected

    chunks = [sound[start:end] for start, end in nonsilence_ranges]

    # Combine non-silent chunks back-to-back
    combined = AudioSegment.empty()
    for c in chunks:
        combined += c

    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    combined.export(temp_file.name, format="wav")
    return temp_file.name



# ================== SEGMENTATION FUNCTION ==================
def segment_audio(file_path, output_dir, participant_id, label, phq_score, seg_dur=6.0):
    """
    Split a single audio file into non-overlapping segments of seg_dur seconds.
    Returns metadata rows for each saved segment.
    """
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

    #seg_info = segment_audio(file_path, OUTPUT_SEGMENT_DIR, pid, label, phq, SEGMENT_DURATION)
    
    # Remove silence first
    clean_path = remove_silence(file_path)
    seg_info = segment_audio(clean_path, OUTPUT_SEGMENT_DIR, pid, label, phq, SEGMENT_DURATION)

    all_metadata.extend(seg_info)

# ================== SAVE METADATA ==================
meta_df = pd.DataFrame(all_metadata)
meta_df.to_csv(OUTPUT_METADATA_CSV, index=False)

print(f"\n Segmentation complete!")
print(f"Saved {len(meta_df)} segments to: {OUTPUT_SEGMENT_DIR}")
print(f"Metadata CSV: {OUTPUT_METADATA_CSV}")
