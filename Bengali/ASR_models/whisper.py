import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
metadata_csv = "/home/bubai-maji/bubai/Bangla/bangla_5fold_metadata.csv"
df = pd.read_csv(metadata_csv)

required_cols = {"seg_path", "label", "fold", "speaker_id"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Metadata must contain columns: {required_cols}")

output_dir = "Bangla/bangla_features_npy/Whisper_large_5fold"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# WHISPER MODEL (ENCODER ONLY)
# -------------------------------------------------
MODEL_NAME = "openai/whisper-large"  

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

print(f"Loaded Whisper: {MODEL_NAME} on {device}")

# -------------------------------------------------
# AUDIO LOADING
# -------------------------------------------------
def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0), target_sr

# -------------------------------------------------
# FEATURE EXTRACTION PER FOLD
# -------------------------------------------------
print("\nExtracting Whisper encoder embeddings...\n")

for fold in sorted(df["fold"].unique()):
    fold_df = df[df["fold"] == fold]

    X, y, speaker_ids = [], [], []

    for row in tqdm(fold_df.itertuples(), total=len(fold_df), desc=f"Fold {fold}"):

        try:
            wav, sr = load_audio(row.seg_path)

            # Preprocess audio → log-mel spectrogram
            inputs = processor(
                wav.numpy(),
                sampling_rate=sr,
                return_tensors="pt"
            )
            feats = inputs.input_features.to(device)   # (1, 80, T)

            # -------------------------------------------------
            # ENCODER ONLY (NO DECODER INPUT REQUIRED)
            # -------------------------------------------------
            with torch.no_grad():
                enc_out = model.encoder(feats)  # Whisper encoder
                hidden = enc_out.last_hidden_state  # (1, T, H)

            # Mean pooling over time frames
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()

            X.append(emb)
            y.append(int(row.label))
            speaker_ids.append(row.speaker_id)

        except Exception as e:
            print(f"[SKIP] {row.seg_path}: {e}")

    # Save arrays
    X = np.stack(X)

    np.save(os.path.join(output_dir, f"fold{fold}_X.npy"), X)
    np.save(os.path.join(output_dir, f"fold{fold}_y.npy"), np.array(y))
    np.save(os.path.join(output_dir, f"fold{fold}_speaker.npy"), np.array(speaker_ids))

    print(f"Saved Fold {fold}: X={X.shape}, samples={len(y)}, speakers={len(set(speaker_ids))}")

print("\nWhisper feature extraction complete!")
print("Saved in:", output_dir)
