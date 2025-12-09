import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# --- Config ---
input_csv = "/home/bubai-maji/bubai/Itali/segment_metadata.csv"    # Must have: seg_path, label, fold
output_dir = "features_npy/seg_mfcc"
os.makedirs(output_dir, exist_ok=True)

SR = 16000             # Sampling rate
N_MFCC = 13            # Number of MFCCs
HOP_LENGTH = 160       # 10ms hop (160 samples at 16kHz)
N_FFT = 400            # 25ms window
MAX_FRAMES = 400       # 4 sec / 10ms = 400 frames

# --- Pad or truncate to fixed length ---
def pad_or_truncate(x, max_len=MAX_FRAMES):
    if x.shape[0] < max_len:
        pad = np.zeros((max_len - x.shape[0], x.shape[1]), dtype=np.float32)
        return np.vstack([x, pad])
    return x[:max_len]

# --- MFCC extraction ---
def extract_mfcc(path):
    y, sr = librosa.load(path, sr=SR)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    ).T  # shape: (T, N_MFCC)
    return pad_or_truncate(mfcc)  # shape: (MAX_FRAMES, N_MFCC)

# --- Load metadata ---
df = pd.read_csv(input_csv)
assert {'seg_path', 'label', 'fold'}.issubset(df.columns), "CSV must have seg_path, label, fold columns"

# --- Process each fold ---
print("Extracting MFCC features...")
for fold in sorted(df['fold'].unique()):
    fold_df = df[df['fold'] == fold]
    X, y = [], []

    print(f"\nFold {fold}: {len(fold_df)} files")
    for row in tqdm(fold_df.itertuples(), total=len(fold_df)):
        try:
            mfcc = extract_mfcc(row.seg_path)
            X.append(mfcc)
            y.append(int(row.label))
        except Exception as e:
            print(f"  Skipping {row.seg_path}: {e}")

    X = np.stack(X)  # shape: (N, MAX_FRAMES, N_MFCC)
    y = np.array(y, dtype=np.int32)

    np.save(os.path.join(output_dir, f"fold{fold}_X.npy"), X)
    np.save(os.path.join(output_dir, f"fold{fold}_y.npy"), y)
    print(f"Saved fold {fold} → X: {X.shape}, y: {y.shape}")
