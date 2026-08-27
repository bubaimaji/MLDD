import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperModel
import soundfile as sf
import warnings

# ---------------- CONFIG ----------------
MODEL_NAME = "openai/whisper-large"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SPLIT = "test"  

BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv_edic/test_segment_metadata.csv"
OUTPUT_DIR = f"{BASE_DIR}/English/edic_features_npy/whisper_large/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")

# ---------------- LOAD MODEL ----------------
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
model = WhisperModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
for p in model.parameters(): p.requires_grad = False

hidden_dim = model.config.d_model  # 1280 for whisper-large
print(f"Loaded Whisper | Hidden dim = {hidden_dim}\n")

# ---------------- SAFE AUDIO LOADER ----------------
def safe_load(path):
    try:
        return torchaudio.load(path)
    except:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        return torch.tensor(x).unsqueeze(0), sr

# ---------------- FEATURE EXTRACTOR ----------------
def extract_whisper_feat(path):
    wav, sr = safe_load(path)

    # mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # resample to 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # ✅ MINIMAL padding only for very short clips (<1 sec)
    if wav.shape[1] < 16000:
        wav = torch.nn.functional.pad(wav, (0, 16000 - wav.shape[1]))

    # (WhisperFeatureExtractor handles padding internally)
    inp = feature_extractor(
        wav.squeeze(),
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        enc = model.encoder(inp)
        h = enc.last_hidden_state              # (1, T, 1280)
        feat = h.mean(dim=1).cpu().numpy().flatten()  # (1280,)

    return feat

# ---------------- LOAD METADATA ----------------
df = pd.read_csv(CSV_PATH)
req = {"audio_path","participant_id","phq8_binary","phq8_score"}
missing = req - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"Loaded {len(df)} EDIC segments from {SPLIT}\n")

# ---------------- STORAGE ----------------
segment_X, segment_y = [], []
segment_score, segment_speaker, segment_index = [], [], []

# ---------------- EXTRACT ----------------
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Whisper {SPLIT}"):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_whisper_feat(path)
        segment_X.append(feat)
        segment_y.append(label)
        segment_score.append(score)
        segment_speaker.append(pid)
        segment_index.append(idx)
    except Exception as e:
        print(f"[Skip] {path} → {e}")

segment_X = np.stack(segment_X)

# ---------------- SAVE ----------------
np.save(f"{OUTPUT_DIR}/segment_X.npy", segment_X)
np.save(f"{OUTPUT_DIR}/segment_y.npy", np.array(segment_y))
np.save(f"{OUTPUT_DIR}/segment_score.npy", np.array(segment_score))
np.save(f"{OUTPUT_DIR}/segment_speaker_id.npy", np.array(segment_speaker))
np.save(f"{OUTPUT_DIR}/segment_index.npy", np.array(segment_index))

print("\n Whisper segment extraction DONE!")
print(f"Saved → {OUTPUT_DIR}")
print(f"Shape: {segment_X.shape}  (segments × {hidden_dim})")
