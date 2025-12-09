import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel, AutoFeatureExtractor

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
metadata_csv = "/home/bubai-maji/bubai/Bangla/bangla_5fold_metadata.csv"
df = pd.read_csv(metadata_csv)

required_cols = {"seg_path", "label", "fold", "speaker_id"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Metadata must contain columns: {required_cols}")

output_dir = "Bangla/bangla_features_npy/HuBERT_base_5fold1"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# HUBERT MODEL
# -------------------------------------------------
MODEL_NAME = "facebook/hubert-base-ls960"   # Change to your preferred model

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = HubertModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -------------------------------------------------
# AUDIO LOADING
# -------------------------------------------------
def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)

    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0), target_sr

# -------------------------------------------------
# FEATURE EXTRACTION PER FOLD
# -------------------------------------------------
print("Extracting HuBERT embeddings...\n")

for fold in sorted(df["fold"].unique()):
    fold_df = df[df["fold"] == fold]

    X, y, speaker_ids = [], [], []

    for row in tqdm(fold_df.itertuples(), total=len(fold_df), desc=f"Fold {fold}"):

        try:
            wav, sr = load_audio(row.seg_path)

            inputs = feature_extractor(
                wav.numpy(),
                sampling_rate=sr,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # -------------------------
            # Mean-pool last hidden state
            # -------------------------
            hidden = outputs.last_hidden_state        # shape: [1, T, H]
            emb = hidden.mean(dim=1).squeeze(0).cpu().numpy()

            X.append(emb)
            y.append(int(row.label))
            speaker_ids.append(row.speaker_id)

        except Exception as e:
            print(f"[SKIP] {row.seg_path} - {e}")

    # Convert to array and save
    X = np.stack(X)

    np.save(os.path.join(output_dir, f"fold{fold}_X.npy"), X)
    np.save(os.path.join(output_dir, f"fold{fold}_y.npy"), np.array(y))
    np.save(os.path.join(output_dir, f"fold{fold}_speaker.npy"), np.array(speaker_ids))

    print(f"Saved Fold {fold}: X={X.shape}, samples={len(y)}, speakers={len(set(speaker_ids))}")

print("\nHuBERT feature extraction complete!")
print("Output saved in:", output_dir)
