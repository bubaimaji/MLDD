import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import WavLMModel, AutoFeatureExtractor
import soundfile as sf
import warnings

# =============== CONFIG ================
SPLIT = "train"   # <-- CHANGE TO: train / dev / test
BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"/home/bubai-maji/bubai/English/Processed_csv/{SPLIT}_balanced_metadata.csv"
SAVE_DIR = f"/home/bubai-maji/bubai/English/layerwise_features/wavlm-large-layer/{SPLIT}"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_NAME = "microsoft/wavlm-large"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# =============== MODEL LOAD ================
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

hidden_size = model.config.hidden_size

# =============== SAFE AUDIO ================
def safe_load(path):
    try:
        wav, sr = torchaudio.load(path)
    except:
        wav_np, sr = sf.read(path, dtype="float32")
        if wav_np.ndim == 1:
            wav = torch.tensor(wav_np).unsqueeze(0)
        else:
            wav = torch.tensor(wav_np.mean(1)).unsqueeze(0)
    return wav, sr

# =============== SEGMENT FEATURE ================
def extract_layerwise_segment(path):
    wav, sr = safe_load(path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav.mean(0).unsqueeze(0)

    inp = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        hs = out.hidden_states   # tuple of layers

        layer_feats = []
        for h in hs:
            h = h.squeeze(0)   # (T, D)
            mean = h.mean(0)
            std = h.std(0)
            pooled = torch.cat([mean, std]).cpu().numpy()
            layer_feats.append(pooled)

        return np.stack(layer_feats, axis=0)   # (n_layers, 2*D)

# =============== READ CSV ================
df = pd.read_csv(CSV_PATH)

speaker_feats = {}
speaker_labels = {}

for _, row in tqdm(df.iterrows(), total=len(df)):
    pid = int(row["participant_id"])
    path = os.path.join(BASE_DIR, row["audio_path"])
    label = int(row["phq8_binary"])

    try:
        layer_feat = extract_layerwise_segment(path)   # (L, 2D)
        if pid not in speaker_feats:
            speaker_feats[pid] = []
            speaker_labels[pid] = label
        speaker_feats[pid].append(layer_feat)
    except:
        continue

# =============== SPEAKER POOLING ================
speaker_ids = sorted(speaker_feats.keys())
first = next(iter(speaker_feats.values()))
n_layers = first[0].shape[0]
feat_dim = first[0].shape[1]

layer_X = [ [] for _ in range(n_layers) ]
y = []

for pid in speaker_ids:
    seg_feat = np.stack(speaker_feats[pid], axis=0)   # (n_seg, L, 2D)
    pooled = seg_feat.mean(0)                         # (L, 2D)

    for L in range(n_layers):
        layer_X[L].append(pooled[L])

    y.append(speaker_labels[pid])

layer_X = [ np.stack(LX) for LX in layer_X ]
y = np.array(y)
speaker_ids = np.array(speaker_ids)

# =============== SAVE ================
np.save(os.path.join(SAVE_DIR, "speaker_y.npy"), y)
np.save(os.path.join(SAVE_DIR, "speaker_ids.npy"), speaker_ids)

for L in range(n_layers):
    np.save(os.path.join(SAVE_DIR, f"layer_{L:02d}_X.npy"), layer_X[L])

print(f"Saved layerwise TRAIN/DEV features to {SAVE_DIR}")
