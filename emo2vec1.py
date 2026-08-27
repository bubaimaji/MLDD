import os
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import torch
from funasr import AutoModel
import warnings

# -------------------- CONFIG --------------------
MODEL_NAME = "iic/emotion2vec_base"  # or "iic/emotion2vec"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
SPLIT = "test"   

BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv_edic/test_segment_metadata.csv"
OUTPUT_DIR = f"{BASE_DIR}/English/edic_features_npy/{MODEL_NAME.replace('/', '-')}/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
print(f"Loading Emotion2Vec model: {MODEL_NAME} on {DEVICE}")

model = AutoModel(model=MODEL_NAME, device=DEVICE)
print(" Model loaded.\n")

# -------------------- SAFE AUDIO LOADER --------------------
def safe_load(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr

# -------------------- FEATURE EXTRACTION --------------------
def extract_emotion2vec_feature(path):
    # Emotion2Vec extracts internally, we pass wav file path
    result = model.generate(
        input=path,
        output_dir=None,
        granularity="utterance",
        extract_embedding=True
    )
    if isinstance(result, list):
        result = result[0]

    if "feats" not in result:
        raise KeyError(f"Unexpected keys in model output: {result.keys()}")

    emb = np.array(result["feats"]).squeeze()  # shape (D,)
    return emb

# -------------------- LOAD CSV --------------------
df = pd.read_csv(CSV_PATH)
required = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain: {required}")

print(f"Total EDIC segments: {len(df)} for split `{SPLIT}`\n")

# -------------------- STORAGE --------------------
segment_X = []
segment_y = []
segment_score = []
segment_speaker_id = []
segment_index = []

# -------------------- LOOP --------------------
for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Emotion2Vec {SPLIT}"):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    file_path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_emotion2vec_feature(file_path)
        segment_X.append(feat)
        segment_y.append(label)
        segment_score.append(score)
        segment_speaker_id.append(pid)
        segment_index.append(idx)

    except Exception as e:
        print(f"[Skip] {file_path} → {e}")

segment_X = np.stack(segment_X)

# -------------------- SAVE --------------------
np.save(f"{OUTPUT_DIR}/segment_X.npy", segment_X)
np.save(f"{OUTPUT_DIR}/segment_y.npy", np.array(segment_y))
np.save(f"{OUTPUT_DIR}/segment_score.npy", np.array(segment_score))
np.save(f"{OUTPUT_DIR}/segment_speaker_id.npy", np.array(segment_speaker_id))
np.save(f"{OUTPUT_DIR}/segment_index.npy", np.array(segment_index))

print("\n Emotion2Vec EDIC segment extraction DONE!")
print(f"Saved to: {OUTPUT_DIR}")
print(f"Shape: {segment_X.shape}  (segments × embedding_dim)")
