import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import WavLMModel, AutoFeatureExtractor
import soundfile as sf
import warnings

# -------------------- CONFIG --------------------
MODEL_NAME = "microsoft/wavlm-large"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SPLIT = "train"  # train / dev / test

BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv_edic/train_balanced_metadata.csv"
OUTPUT_DIR = f"{BASE_DIR}/English/edic_features_npy/wavlm_large/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")

# -------------------- LOAD MODEL --------------------
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
for p in model.parameters(): p.requires_grad = False

hidden_size = model.config.hidden_size  # base-plus = 768
print(f"Loaded WavLM | Hidden size = {hidden_size}\n")

# -------------------- SAFE AUDIO LOADER --------------------
def safe_load(path):
    try:
        return torchaudio.load(path)
    except:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        return torch.tensor(x).unsqueeze(0), sr

# -------------------- FEATURE EXTRACTOR --------------------
def extract_wavlm_feat(path):
    wav, sr = safe_load(path)

    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample to 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    inp = processor(wav.squeeze(), sampling_rate=sr, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model(**inp)
        last_hidden = out.last_hidden_state  # (1, T, D)
        feat = last_hidden.mean(dim=1).cpu().numpy().flatten()  # (D,)

    return feat

# -------------------- LOAD METADATA --------------------
df = pd.read_csv(CSV_PATH)
req_cols = {"audio_path","participant_id","phq8_binary","phq8_score"}
missing = req_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"Loaded {len(df)} segments for {SPLIT}\n")

# -------------------- STORAGE --------------------
segment_X = []
segment_y = []
segment_score = []
segment_speaker = []
segment_index = []

# -------------------- LOOP --------------------
for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {SPLIT}"):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_wavlm_feat(path)
        segment_X.append(feat)
        segment_y.append(label)
        segment_score.append(score)
        segment_speaker.append(pid)
        segment_index.append(i)

    except Exception as e:
        print(f"[Skip] {path} → {e}")

# -------------------- SAVE --------------------
segment_X = np.stack(segment_X)

np.save(f"{OUTPUT_DIR}/segment_X.npy", segment_X)
np.save(f"{OUTPUT_DIR}/segment_y.npy", np.array(segment_y))
np.save(f"{OUTPUT_DIR}/segment_score.npy", np.array(segment_score))
np.save(f"{OUTPUT_DIR}/segment_speaker_id.npy", np.array(segment_speaker))
np.save(f"{OUTPUT_DIR}/segment_index.npy", np.array(segment_index))

print("\n Segment-level WavLM feature extraction DONE!")
print(f"Saved to: {OUTPUT_DIR}")
print(f"Features shape: {segment_X.shape}  (segments × {hidden_size})")
