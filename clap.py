import os
import torch
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, ClapModel
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_NAME = "laion/clap-htsat-unfused"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

CSV_PATH = "/home/bubai-maji/bubai/English/Processed_csv/dev_segment_metadata.csv"
SAVE_DIR = os.path.join(
    "/home/bubai-maji/bubai/English/output_features_daic/frame_features",
    MODEL_NAME.replace("/", "-")
)
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------- LOAD MODEL --------------------
print(f"\n Loading model: {MODEL_NAME}")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# Freeze model parameters
for p in model.parameters():
    p.requires_grad = False

try:
    # Common in transformers >=4.34
    hidden_size = model.audio_projection.projection.out_features
except AttributeError:
    try:
        # Older versions may expose it directly
        hidden_size = model.audio_projection.out_features
    except AttributeError:
        # Last resort — infer from a dummy forward pass
        print("Falling back to runtime probe for embedding dimension...")
        with torch.no_grad():
            dummy = torch.zeros(1, 48000)
            dummy_inputs = processor(
                audios=[dummy.squeeze().numpy()],
                sampling_rate=48000,
                return_tensors="pt"
            )  
            dummy_inputs = {k: v.to(DEVICE) for k, v in dummy_inputs.items()}
            dummy_emb = model.get_audio_features(**dummy_inputs)
            hidden_size = dummy_emb.shape[-1]

layer_norm = nn.LayerNorm(hidden_size).to(DEVICE)
print(f"Model loaded successfully on {DEVICE} | Embedding dim = {hidden_size}")

df = pd.read_csv(CSV_PATH)
required_cols = {"audio_path", "participant_id", "phq8_binary", "phq8_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

print(f"Loaded {len(df)} audio segments for CLAP feature extraction.\n")

def safe_load(filepath):
    """Safely load audio (handles multiple backends)."""
    try:
        waveform, sr = torchaudio.load(filepath)
    except Exception:
        try:
            waveform_np, sr = sf.read(filepath)
            if len(waveform_np.shape) == 1:
                waveform = torch.tensor(waveform_np).unsqueeze(0)
            else:
                waveform = torch.tensor(waveform_np.mean(axis=1)).unsqueeze(0)
            waveform = waveform.to(torch.float32)
        except Exception as e2:
            raise RuntimeError(f"Failed to read {filepath}: {e2}")
    return waveform, sr

# -------------------- FEATURE EXTRACTION --------------------
def extract_clap_feature(filepath):
    """
    Extract a single global-level CLAP embedding (typically 512-D).
    """
    waveform, sr = safe_load(filepath)

    # Convert stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 48 kHz (CLAP requirement)
    target_sr = 48000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # Prepare input for CLAP
    audio_np = waveform.squeeze(0).numpy()
    inputs = processor(audios=[audio_np], sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        audio_embeds = model.get_audio_features(**inputs)  # shape: (1, hidden_size)
        emb = audio_embeds.squeeze(0)
        emb = layer_norm(emb).cpu()

    return emb

# -------------------- PROCESS FILES --------------------
features, participant_ids, labels, scores, filepaths = [], [], [], [], []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting CLAP embeddings"):
    audio_path = row["audio_path"]
    pid = int(row["participant_id"])
    label = int(row["phq8_binary"])
    score = int(row["phq8_score"])

    try:
        feat = extract_clap_feature(audio_path)
        features.append(feat)
        participant_ids.append(pid)
        labels.append(label)
        scores.append(score)
        filepaths.append(audio_path)
    except Exception as e:
        print(f"[SKIP] {audio_path}: {e}")

# -------------------- SAVE OUTPUT --------------------
save_data = {
    "features": features,         # list of (512,) tensors
    "participant_ids": participant_ids,
    "phq8_binary": labels,
    "phq8_score": scores,
    "filepaths": filepaths,
}

save_path = os.path.join(SAVE_DIR, "dev_features.pt")
torch.save(save_data, save_path)
print(f" Global CLAP embeddings saved to:\n{save_path}")
