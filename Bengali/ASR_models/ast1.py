import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
metadata_csv = "/home/bubai-maji/bubai/Bangla/bangla_5fold_metadata.csv"
df = pd.read_csv(metadata_csv)

required_cols = {'seg_path', 'label', 'fold', 'speaker_id'}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Metadata must contain columns: {required_cols}")

output_dir = "Bangla/bangla_features_npy/AST_5fold"
os.makedirs(output_dir, exist_ok=True)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -------------------------------------------------
# AUDIO LOADING FUNCTION
# -------------------------------------------------
def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.mean(dim=0), target_sr  # mono


# -------------------------------------------------
# EXTRACT FEATURES PER FOLD
# -------------------------------------------------
print("Extracting AST features...\n")

for fold in sorted(df['fold'].unique()):
    fold_df = df[df['fold'] == fold]

    X, y, speaker_ids = [], [], []

    for row in tqdm(fold_df.itertuples(), total=len(fold_df), desc=f"Fold {fold}"):

        try:
            waveform, sr = load_audio(row.seg_path)

            inputs = feature_extractor(
                waveform.numpy(),
                sampling_rate=sr,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Pooled embedding (768-dim)
            if hasattr(outputs, "pooler_output"):
                emb = outputs.pooler_output.squeeze().cpu().numpy()
            else:
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            X.append(emb)
            y.append(int(row.label))
            speaker_ids.append(row.speaker_id)

        except Exception as e:
            print(f"[Skip] {row.seg_path} – {e}")

    X = np.stack(X)

    # Save files
    np.save(os.path.join(output_dir, f"fold{fold}_X.npy"), X)
    np.save(os.path.join(output_dir, f"fold{fold}_y.npy"), np.array(y))
    np.save(os.path.join(output_dir, f"fold{fold}_speaker.npy"), np.array(speaker_ids))

    print(f"Fold {fold} saved: X={X.shape}, samples={len(y)}, speakers={len(set(speaker_ids))}")

print("\nAST feature extraction DONE!")
print("Saved at:", output_dir)
