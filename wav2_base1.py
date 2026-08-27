import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ---------------- CONFIG -----------------
SPLIT = "train"  

BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv_edic/train_balanced_metadata.csv"

OUTPUT_DIR = f"{BASE_DIR}/English/edic_features_npy/wav2vec2_large/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ------------- Load Metadata -------------
df = pd.read_csv(CSV_PATH)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print(f"Loaded {len(df)} segments for split = {SPLIT}")

# ----------- Load Wav2Vec2 Model ----------
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(DEVICE)
model.eval()

print("\n Loaded pretrained Wav2Vec2\n")

# ------------ Storage Lists -------------
segment_feats = []
segment_labels = []
segment_scores = []
segment_speakers = []
segment_idx = []

# ------------- Feature Extractor -------------
def extract_wav2vec_features(audio_path):
    audio, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        sr = 16000
    
    audio = audio.squeeze().numpy()
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(inputs.input_values.to(DEVICE))
        hidden_states = outputs.last_hidden_state  # shape: (1, T, 768)

    # Mean pool across time → shape: (768,)
    feat = hidden_states.mean(dim=1).cpu().numpy().flatten()
    return feat


# ---------------- Loop Over Segments ----------------
for i, row in tqdm(df.iterrows(), total=len(df)):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    seg_path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_wav2vec_features(seg_path)
        segment_feats.append(feat)
        segment_labels.append(label)
        segment_scores.append(score)
        segment_speakers.append(pid)
        segment_idx.append(i)

    except Exception as e:
        print(f"[Skip] {seg_path} → {e}")

segment_feats = np.stack(segment_feats)
segment_labels = np.array(segment_labels)
segment_scores = np.array(segment_scores)
segment_speakers = np.array(segment_speakers)
segment_idx = np.array(segment_idx)

# ---------------- Save ----------------
np.save(f"{OUTPUT_DIR}/segment_X.npy", segment_feats)
np.save(f"{OUTPUT_DIR}/segment_y.npy", segment_labels)
np.save(f"{OUTPUT_DIR}/segment_score.npy", segment_scores)
np.save(f"{OUTPUT_DIR}/segment_speaker_id.npy", segment_speakers)
np.save(f"{OUTPUT_DIR}/segment_index.npy", segment_idx)

print("\n Segment-level Wav2Vec2 feature extraction DONE!")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Embeddings shape: {segment_feats.shape}  (segments × 768)")
print(f"Saved files: segment_X, segment_y, ...")
