import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
metadata_csv = "/home/bubai-maji/bubai/Bangla/bangla_5fold_metadata.csv"
df = pd.read_csv(metadata_csv)

required_cols = {"seg_path", "label", "fold", "speaker_id"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Metadata must contain columns: {required_cols}")

output_dir = "/home/bubai-maji/bubai/revision2/bangla_feature/WavLM_large_temporal_5fold"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# WavLM LARGE
# -------------------------------------------------
MODEL_NAME = "microsoft/wavlm-base"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# -------------------------------------------------
# AUDIO LOADING
# -------------------------------------------------
def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(
            wav, sr, target_sr
        )

    return wav.squeeze(0), target_sr


# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------
print("Extracting TEMPORAL WavLM embeddings...\n")

for fold in sorted(df["fold"].unique()):

    fold_df = df[df["fold"] == fold]

    # Each element will have shape [T_i, 1024]
    X_temporal = []
    y = []
    speaker_ids = []
    lengths = []

    for row in tqdm(
        fold_df.itertuples(),
        total=len(fold_df),
        desc=f"Fold {fold}"
    ):

        try:

            wav, sr = load_audio(row.seg_path)

            inputs = feature_extractor(
                wav.numpy(),
                sampling_rate=sr,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            with torch.no_grad():

                outputs = model(**inputs)

            # -------------------------------------------------
            # IMPORTANT:
            # KEEP THE TEMPORAL REPRESENTATION
            # -------------------------------------------------
            hidden = outputs.last_hidden_state
            # [1, T, 1024]

            temporal_emb = hidden.squeeze(0).cpu().numpy()
            # [T, 1024]

            X_temporal.append(temporal_emb)

            y.append(int(row.label))
            speaker_ids.append(row.speaker_id)
            lengths.append(temporal_emb.shape[0])

        except Exception as e:

            print(f"[SKIP] {row.seg_path}: {e}")

    # -------------------------------------------------
    # SAVE AS OBJECT ARRAY BECAUSE T VARIES
    # -------------------------------------------------
    X_temporal = np.array(X_temporal, dtype=object)

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_X_temporal.npy"
        ),
        X_temporal,
        allow_pickle=True
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_y.npy"
        ),
        np.array(y)
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_speaker.npy"
        ),
        np.array(speaker_ids)
    )

    np.save(
        os.path.join(
            output_dir,
            f"fold{fold}_lengths.npy"
        ),
        np.array(lengths)
    )

    print(
        f"Saved Fold {fold}: "
        f"samples={len(y)}, "
        f"speakers={len(set(speaker_ids))}, "
        f"T_min={min(lengths)}, "
        f"T_max={max(lengths)}, "
        f"T_mean={np.mean(lengths):.1f}"
    )

print("\nTemporal WavLM feature extraction complete!")
print("Output saved in:", output_dir)