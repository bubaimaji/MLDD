import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# ---------------- CONFIG -----------------
SPLIT = "train"
BASE_DIR = "/home/bubai-maji/bubai"
CSV_PATH = f"{BASE_DIR}/English/Processed_csv_edic/train_balanced_metadata.csv"
OUTPUT_DIR = f"{BASE_DIR}/English/edic_features_npy/wav2vec2_all_layers/{SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ------------- Load Metadata -------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} segments")

# ----------- Load Wav2Vec2 Model ----------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h",
    output_hidden_states=True   
).to(DEVICE)
model.eval()

print("\nLoaded wav2vec2 with hidden_states \n")

# ------------ Storage Lists -------------
segment_feats = []
segment_labels = []
segment_scores = []
segment_speakers = []
segment_idx = []

# ------------- Feature Extractor -------------
def extract_wav2vec_all_layers(audio_path):
    audio, sr = torchaudio.load(audio_path)

    # Convert stereo to mono
    audio = audio.mean(dim=0)

    # Resample to 16KHz
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    inputs = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(inputs.input_values.to(DEVICE))
        hidden_states = outputs.hidden_states  # tuple of 13 layers (0–12), each -> (1, T, 768)

    # Mean pool each layer across time → 13 vectors of size 768
    layers_mean = [hs.mean(dim=1).cpu().numpy().flatten() for hs in hidden_states]

    # Option 1: concat all layer means (best)
    feat = np.concatenate(layers_mean)   # shape = (13*768 = 9984)

    return feat


# ---------------- Loop Over Segments ----------------
for i, row in tqdm(df.iterrows(), total=len(df)):
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = float(row["phq8_score"])
    seg_path = os.path.join(BASE_DIR, row["audio_path"])

    try:
        feat = extract_wav2vec_all_layers(seg_path)
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

print("\n Extracted ALL LAYER embeddings")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Embeddings shape: {segment_feats.shape} (Segments × 9984 dims)")
