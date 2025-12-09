import os, warnings
import torch, torchaudio, soundfile as sf
import pandas as pd, numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

# ---------------- CONFIG -----------------
SPLIT = "test"  # same format as wav2vec2 script

BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv_edic/test_segment_metadata.csv"

MODEL_NAME = "facebook/data2vec-audio-base-960h"
OUTPUT_DIR = f"{BASE_DIR}/English/edic_features_npy/data2vec_base/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ---------------- Load CSV -----------------
df = pd.read_csv(CSV_PATH)
req = {"audio_path","participant_id","phq8_binary","phq8_score"}
missing = req - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

print(f"Loaded {len(df)} segments for split = {SPLIT}")

# ---------------- Load Data2Vec -----------------
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

hidden_dim = model.config.hidden_size  # should be 768 for base
print(f"Loaded Data2Vec model | Hidden dim = {hidden_dim}\n")

# ---------------- Safe Audio Loader -----------------
def safe_load(path):
    try:
        return torchaudio.load(path)
    except:
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1: x = x.mean(axis=1)
        return torch.tensor(x).unsqueeze(0), sr

# ---------------- Feature Extractor -----------------
def extract_data2vec_feat(path):
    wav, sr = safe_load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    inp = processor(wav.squeeze(), sampling_rate=16000, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model(**inp)
        hidden = out.last_hidden_state  # (1, T, D)

        feat = hidden.mean(dim=1).cpu().numpy().flatten()  # (D,)
        return feat

# ---------------- Storage -----------------
segment_feats = []
segment_labels = []
segment_scores = []
segment_speakers = []
segment_index = []

# ---------------- Loop -----------------
for idx, row in tqdm(df.iterrows(), total=len(df)):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    audio_file = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_data2vec_feat(audio_file)

        segment_feats.append(feat)
        segment_labels.append(label)
        segment_scores.append(score)
        segment_speakers.append(pid)
        segment_index.append(idx)

    except Exception as e:
        print(f"[Skip] {audio_file} → {e}")

# ---------------- Save -----------------
segment_feats = np.stack(segment_feats)

np.save(f"{OUTPUT_DIR}/segment_X.npy", segment_feats)
np.save(f"{OUTPUT_DIR}/segment_y.npy", np.array(segment_labels))
np.save(f"{OUTPUT_DIR}/segment_score.npy", np.array(segment_scores))
np.save(f"{OUTPUT_DIR}/segment_speaker_id.npy", np.array(segment_speakers))
np.save(f"{OUTPUT_DIR}/segment_index.npy", np.array(segment_index))

print("\n Data2Vec segment-level feature extraction DONE!")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Embeddings shape: {segment_feats.shape}  (segments × {hidden_dim})")
