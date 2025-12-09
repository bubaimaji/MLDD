import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa

train_split_file = "/home/bubai-maji/bubai/English/E-label/train_split.csv"
dev_split_file   = "/home/bubai-maji/bubai/English/E-label/dev_split.csv"
test_split_file  = "/home/bubai-maji/bubai/English/E-label/test_split.csv"

transcripts_dir  = "/home/bubai-maji/bubai/English/300-718_transcripts"
audio_dir        = "/home/bubai-maji/bubai/English/300-718_Audio"

train_audio_dir  = "/home/bubai-maji/bubai/English/Processed_audio_edic/train"
dev_audio_dir    = "/home/bubai-maji/bubai/English/Processed_audio_edic/dev"
test_audio_dir   = "/home/bubai-maji/bubai/English/Processed_audio_edic/test"

train_csv_out = "/home/bubai-maji/bubai/English/Processed_csv_edic/train_clean_audio.csv"
dev_csv_out   = "/home/bubai-maji/bubai/English/Processed_csv_edic/dev_clean_audio.csv"
test_csv_out  = "/home/bubai-maji/bubai/English/Processed_csv_edic/test_clean_audio.csv"

for d in [train_audio_dir, dev_audio_dir, test_audio_dir, os.path.dirname(train_csv_out)]:
    os.makedirs(d, exist_ok=True)

# -------------------- Load Splits --------------------
def load_split(file_path):
    df = pd.read_csv(file_path)
    df['PHQ_Binary'] = df['PHQ_Binary'].astype(int)
    df['PHQ_Score']  = df['PHQ_Score'].astype(int)
    df['Participant_ID'] = df['Participant_ID'].astype(int)
    return df

train_df = load_split(train_split_file)
dev_df   = load_split(dev_split_file)
test_df  = load_split(test_split_file)

# -------------------- Core Function --------------------
def remove_ellie_and_save(file_id, transcript_file, audio_file, output_dir):

    # Detect delimiter (\t or ,)
    with open(transcript_file, 'r') as f:
        header_line = f.readline()
    sep = '\t' if '\t' in header_line else ','

    # Load transcript
    transcript_df = pd.read_csv(transcript_file, sep=sep)
    transcript_df.columns = [c.strip().lower().replace(" ", "_") for c in transcript_df.columns]

    # Identify if speaker column exists (Type-2)
    has_speaker = "speaker" in transcript_df.columns

    # Map column names
    if has_speaker:
        # Type-2 => "start_time", "stop_time", "speaker"
        start_col = "start_time"
        stop_col  = "stop_time"
        transcript_df["speaker"] = transcript_df["speaker"].astype(str).str.lower().str.strip()
    else:
        # Type-1 => "Start_Time, End_Time" only participant
        transcript_df.rename(columns={"start_time": "start_time", "end_time": "stop_time"}, inplace=True)
        if "start_time" not in transcript_df.columns and "start_time" in header_line.lower():
            transcript_df.rename(columns={"start_time": "start_time"}, inplace=True)
        if "stop_time" not in transcript_df.columns and "end_time" in transcript_df.columns:
            transcript_df.rename(columns={"end_time": "stop_time"}, inplace=True)

        start_col = "start_time"
        stop_col  = "stop_time"
        transcript_df["speaker"] = "participant"

    # Validate timestamps
    if start_col not in transcript_df.columns or stop_col not in transcript_df.columns:
        print(f" SKIPPED — No valid timestamps in {transcript_file}")
        print("Columns:", transcript_df.columns.tolist())
        return None

    # Load audio
    audio, sr = librosa.load(audio_file, sr=None)

    # Collect participant segments
    segments = []
    for _, row in transcript_df.iterrows():
        if row["speaker"] != "ellie":
            start = float(row[start_col])
            stop = float(row[stop_col])
            if stop > start:
                seg = audio[int(start * sr):int(stop * sr)]
                segments.append(seg)

    if not segments:
        print(f" No participant speech found for {file_id}")
        return None

    cleaned = np.concatenate(segments)
    out_path = os.path.join(output_dir, f"{file_id}_participant_only.wav")
    sf.write(out_path, cleaned, sr)

    return out_path

# -------------------- Process Splits --------------------
def process_split(split_df, transcripts_dir, audio_dir, output_audio_dir):
    results = []
    for _, row in split_df.iterrows():
        pid = int(row["Participant_ID"])
        label = int(row["PHQ_Binary"])
        score = int(row["PHQ_Score"])

        audio_file = os.path.join(audio_dir, f"{pid}_AUDIO.wav")
        transcript_file = os.path.join(transcripts_dir, f"{pid}_TRANSCRIPT.csv")

        if not os.path.exists(audio_file) or not os.path.exists(transcript_file):
            print(f" Missing files for {pid}")
            continue

        cleaned = remove_ellie_and_save(pid, transcript_file, audio_file, output_audio_dir)
        if cleaned:
            results.append({"participant_id": pid, "audio_path": cleaned,
                            "phq8_binary": label, "phq8_score": score})

    return pd.DataFrame(results)

# Run
train_clean_df = process_split(train_df, transcripts_dir, audio_dir, train_audio_dir)
dev_clean_df   = process_split(dev_df, transcripts_dir, audio_dir, dev_audio_dir)
test_clean_df  = process_split(test_df, transcripts_dir, audio_dir, test_audio_dir)

# Save CSVs
train_clean_df.to_csv(train_csv_out, index=False)
dev_clean_df.to_csv(dev_csv_out, index=False)
test_clean_df.to_csv(test_csv_out, index=False)

print("ALL DONE: Participant-only audio saved & CSVs generated.")
