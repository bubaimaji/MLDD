import os, warnings
import torch, torchaudio, soundfile as sf
import pandas as pd, numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel
import torch.nn as nn

# -------------------- CONFIG --------------------
MODEL_NAME = "facebook/data2vec-audio-base-960h"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
SPLIT = "train"  

CSV_PATH = f"/home/bubai-maji/bubai/English/Processed_csv/train_balanced_metadata_new.csv"
BASE_DIR = "/home/bubai-maji/bubai"
SAVE_DIR = os.path.join(
    "/home/bubai-maji/bubai/English/features_npy",
    MODEL_NAME.replace("/", "-"),
    SPLIT
)
os.makedirs(SAVE_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

# -------------------- LOAD MODEL --------------------
print(f"Loading model: {MODEL_NAME} on {DEVICE}")
processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
for p in model.parameters(): p.requires_grad = False

hidden = getattr(model.config, "hidden_size", 768)
layer_norm = nn.LayerNorm(hidden).to(DEVICE)
print(f"Model Loaded | Hidden Size = {hidden}")

# -------------------- SAFE LOAD --------------------
def safe_load(path):
    try: return torchaudio.load(path)
    except:
        x, sr = sf.read(path)
        if x.ndim > 1: x = x.mean(axis=1)
        return torch.tensor(x).unsqueeze(0).float(), sr

# -------------------- EXTRACT FEATURE --------------------
def extract_feat(path):
    wav, sr = safe_load(path)
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000: wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    inp = processor(wav.squeeze(), sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inp).last_hidden_state.squeeze(0)
        out = layer_norm(out).mean(0)
    return out.cpu().numpy()

# -------------------- LOAD CSV --------------------
df = pd.read_csv(CSV_PATH)
need = {"seg_path","label","speaker_id","utterance_id"}
assert need.issubset(df.columns), f"CSV must contain {need}"

# -------------------- PROCESS --------------------
X, y, spk, utt = [], [], [], []

for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {SPLIT}"):
    full = os.path.join(BASE_DIR, r["seg_path"])
    try:
        feat = extract_feat(full)
        X.append(feat)
        y.append(int(r["label"]))
        spk.append(r["speaker_id"])
        utt.append(r["utterance_id"])
    except Exception as e:
        print(f"Skip: {full} -> {e}")

# -------------------- SAVE --------------------
np.save(f"{SAVE_DIR}/segment_X.npy", np.stack(X))
np.save(f"{SAVE_DIR}/segment_y.npy", np.array(y))
np.save(f"{SAVE_DIR}/segment_speaker.npy", np.array(spk))
np.save(f"{SAVE_DIR}/segment_utter.npy", np.array(utt))

print(f" DONE: {SPLIT} set")
print(f"Segments: {len(X)} | Speakers: {len(set(spk))}")
print(f"Saved to {SAVE_DIR}")
