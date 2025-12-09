import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import re

train_split_file = "/home/bubai-maji/bubai/English/label/train_split_Depression_AVEC2017.csv"
dev_split_file   = "/home/bubai-maji/bubai/English/label/dev_split_Depression_AVEC2017.csv"
test_split_file  = "/home/bubai-maji/bubai/English/label/test_split_Depression_AVEC2017.csv"

transcripts_dir  = "/home/bubai-maji/bubai/English/300-492_transcripts"
audio_dir        = "/home/bubai-maji/bubai/English/300-718_Audio"

train_audio_dir  = "/home/bubai-maji/bubai/English/Processed_audio/train"
dev_audio_dir    = "/home/bubai-maji/bubai/English/Processed_audio/dev"
test_audio_dir   = "/home/bubai-maji/bubai/English/Processed_audio/test"

train_csv_out = "/home/bubai-maji/bubai/English/Processed_csv/train_clean_audio.csv"
dev_csv_out   = "/home/bubai-maji/bubai/English/Processed_csv/dev_clean_audio.csv"
test_csv_out  = "/home/bubai-maji/bubai/English/Processed_csv/test_clean_audio.csv"

for d in [train_audio_dir, dev_audio_dir, test_audio_dir, os.path.dirname(train_csv_out)]:
    os.makedirs(d, exist_ok=True)

# -------------------- Load Splits --------------------
def load_split(file_path):
    df = pd.read_csv(file_path)
    # Ensure both PHQ columns exist
    assert 'PHQ8_Binary' in df.columns and 'PHQ8_Score' in df.columns, \
        f"Missing PHQ columns in {file_path}"
    df['PHQ8_Binary'] = df['PHQ8_Binary'].astype(int)
    df['PHQ8_Score'] = df['PHQ8_Score'].astype(int)
    df['Participant_ID'] = df['Participant_ID'].astype(int)
    return df

train_df = load_split(train_split_file)
dev_df   = load_split(dev_split_file)
test_df  = load_split(test_split_file)


def remove_ellie_and_save(file_id, transcript_file, audio_file, output_dir):
    """
    Removes Ellie's speech and saves only participant's speech as one concatenated .wav file.
    """
    # Read transcript (with header)
    transcript_df = pd.read_csv(transcript_file, sep='\t')
    transcript_df.columns = [c.strip().lower() for c in transcript_df.columns]

    # Load the corresponding audio file
    audio, sr = librosa.load(audio_file, sr=None)

    # Collect participant-only audio segments
    participant_segments = []
    for _, row in transcript_df.iterrows():
        speaker = str(row['speaker']).strip().lower()
        if speaker != 'ellie':  # Skip Ellie's lines
            start = float(row['start_time'])
            stop = float(row['stop_time'])
            segment = audio[int(start * sr):int(stop * sr)]
            participant_segments.append(segment)

    # Handle empty case
    if not participant_segments:
        print(f"No participant segments found for ID {file_id}")
        return None

    # Concatenate and save
    cleaned_audio = np.concatenate(participant_segments)
    output_path = os.path.join(output_dir, f"{file_id}_participant_only.wav")
    sf.write(output_path, cleaned_audio, sr)

    return output_path

def process_split(split_df, transcripts_dir, audio_dir, output_audio_dir):
    combined_data = []
    for _, row in split_df.iterrows():
        pid = int(row['Participant_ID'])
        label = int(row['PHQ8_Binary'])
        score = int(row['PHQ8_Score'])
        audio_file = os.path.join(audio_dir, f"{pid}_AUDIO.wav")
        transcript_file = os.path.join(transcripts_dir, f"{pid}_TRANSCRIPT.csv")

        if not os.path.exists(audio_file) or not os.path.exists(transcript_file):
            print(f"Missing files for Participant_ID: {pid}")
            continue

        cleaned_path = remove_ellie_and_save(pid, transcript_file, audio_file, output_audio_dir)
        if cleaned_path:
            combined_data.append({
                'participant_id': pid,
                'audio_path': cleaned_path,
                'phq8_binary': label,
                'phq8_score': score
            })
    return pd.DataFrame(combined_data)

# -------------------- Process Splits --------------------
train_clean_df = process_split(train_df, transcripts_dir, audio_dir, train_audio_dir)
dev_clean_df   = process_split(dev_df, transcripts_dir, audio_dir, dev_audio_dir)
test_clean_df  = process_split(test_df, transcripts_dir, audio_dir, test_audio_dir)

# -------------------- Save Outputs --------------------
train_clean_df.to_csv(train_csv_out, index=False)
dev_clean_df.to_csv(dev_csv_out, index=False)
test_clean_df.to_csv(test_csv_out, index=False)

print("Cleaned audio created successfully and saved CSVs with both PHQ8_Binary & PHQ8_Score")
