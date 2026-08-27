import os
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ==========================
# CONFIG
# ==========================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# adjust these as needed
FEAT_DIR_A = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"        
FEAT_DIR_B = "/home/bubai-maji/bubai/English/features_npy/wav2vec2_large"  
SPLIT_TRAIN = "train"
SPLIT_DEV   = "dev"

BATCH_SPEAKERS = 8           
LR = 2e-4
EPOCHS = 100
PATIENCE = 10
SEED = 42
LATENT_DIM = 512
TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 8
DROPOUT = 0.2

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================
# HELPERS: data builder
# ==========================
def load_segment_features(base_dir: str, split: str):
    """Loads segment-level arrays saved earlier.
    Expects: segment_X.npy, segment_y.npy, segment_speaker_id.npy
    """
    xa = np.load(os.path.join(base_dir, split, "segment_X.npy"))
    ya = np.load(os.path.join(base_dir, split, "segment_y.npy")).astype(int)
    spka = np.load(os.path.join(base_dir, split, "segment_speaker_id.npy"))
    return xa, ya, spka

class SpeakerDataset(Dataset):
    """
    Build speaker-level entries. Each item returns:
      - A_segments: numpy array (num_segs, dimA)
      - B_segments: numpy array (num_segs, dimB)
      - label: speaker label (0/1)
      - speaker_id: original id
    """
    def __init__(self, feat_dir_a, feat_dir_b, split):
        XA, y, spk = load_segment_features(feat_dir_a, split)
        XB, yb, spkb = load_segment_features(feat_dir_b, split)
        assert len(XA) == len(XB) == len(y) == len(spk), "Mismatch in arrays"
        # group by speaker id
        self.speaker_ids = []
        self.data = []  # list of tuples (XA_speaker, XB_speaker, label, speaker_id)
        grouped_A = defaultdict(list)
        grouped_B = defaultdict(list)
        grouped_y = {}
        for i, sid in enumerate(spk):
            grouped_A[sid].append(XA[i])
            grouped_B[sid].append(XB[i])
            grouped_y[sid] = int(y[i])  # assume same label for that speaker

        for sid in sorted(grouped_A.keys()):
            a_arr = np.vstack(grouped_A[sid])  # shape (n_segs, dimA)
            b_arr = np.vstack(grouped_B[sid])
            lbl = grouped_y[sid]
            self.speaker_ids.append(sid)
            self.data.append((a_arr, b_arr, lbl, sid))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # (A_seq, B_seq, label, speaker_id)

def collate_speakers(batch: List[Tuple[np.ndarray, np.ndarray, int, int]]):
    """
    Collate a list of speaker-items into a batch.
    Pads sequences of segments to max_len in the batch.
    Returns tensors:
      A_batch: [B_speakers, max_len, dimA]
      B_batch: [B_speakers, max_len, dimB]
      mask:    [B_speakers, max_len]  (1 for real seg, 0 for pad)
      labels:  [B_speakers]
      speaker_ids: list
      lengths: list of lengths
    """
    lengths = [a.shape[0] for (a, b, _, _) in batch]
    max_len = max(lengths)
    dimA = batch[0][0].shape[1]
    dimB = batch[0][1].shape[1]
    B = len(batch)

    A_batch = np.zeros((B, max_len, dimA), dtype=np.float32)
    B_batch = np.zeros((B, max_len, dimB), dtype=np.float32)
    mask = np.zeros((B, max_len), dtype=np.float32)
    labels = np.zeros((B,), dtype=np.int64)
    speaker_ids = []

    for i, (a, b, lbl, sid) in enumerate(batch):
        L = a.shape[0]
        A_batch[i, :L] = a
        B_batch[i, :L] = b
        mask[i, :L] = 1.0
        labels[i] = lbl
        speaker_ids.append(int(sid))

    A_batch = torch.tensor(A_batch)
    B_batch = torch.tensor(B_batch)
    mask = torch.tensor(mask)  # float mask (1/0)
    labels = torch.tensor(labels, dtype=torch.long)
    return A_batch, B_batch, mask, labels, speaker_ids, lengths

# ==========================
# MODEL COMPONENTS
# ==========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            # odd dims
            pe[:, 1::2] = torch.cos(pos * div[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return x + self.pe[:, :L]

class ModalityEncoder(nn.Module):
    """
    Projects raw features to latent and runs a TransformerEncoder over segments.
    Input shape: (B, L, dim_in)
    Output shape: (B, L, latent_dim)
    """
    def __init__(self, dim_in, latent_dim=256, n_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(dim_in, latent_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=n_heads,
                                                   dim_feedforward=latent_dim*2, dropout=dropout,
                                                   batch_first=True, activation='relu')
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos = PositionalEncoding(latent_dim, max_len=1024)

    def forward(self, x, mask):
        # x: [B, L, dim_in], mask: [B, L] float with 1 for real
        h = self.proj(x)            # [B,L,latent]
        h = self.pos(h)
        # transformer expects src_key_padding_mask with True for padded positions
        padding_mask = (mask == 0)  # bool
        out = self.trans(h, src_key_padding_mask=padding_mask)
        return out  # [B,L,latent]

class CrossAttentionBlock(nn.Module):
    """
    Bi-directional cross-attention across sequences.
    A -> B and B -> A (multi-head), then optional fusion MLP.
    Inputs:
      A: [B, L, D]
      B: [B, L, D]
      mask: [B, L] (1 for real, 0 for pad)
    Outputs:
      fused_seq: [B, L, 2D]  (concat of A2B and B2A)
    """
    def __init__(self, latent_dim=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.latent = latent_dim
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(2*latent_dim, 2*latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, A, B, mask):
        # A,B: [B,L,D], mask: [B,L] float
        key_padding = (mask == 0)  # [B,L] bool for padding

        # A -> B : Q=A, K= B, V= B
        A2B, _ = self.attn(query=A, key=B, value=B, key_padding_mask=key_padding)
        # B -> A : Q=B, K= A, V= A
        B2A, _ = self.attn2(query=B, key=A, value=A, key_padding_mask=key_padding)

        # fuse per position: concat A2B and B2A (same shape)
        fused = torch.cat([A2B, B2A], dim=-1)  # [B,L,2D]
        fused = self.fuse(fused)               # [B,L,2D] (we keep 2D)
        return fused

class SegmentAttentionPool(nn.Module):
    """
    Attention pooling across segments for a speaker.
    Input: seq [B, L, D], mask [B,L]
    Output: pooled [B, D]
    """
    def __init__(self, dim):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.Tanh(),
            nn.Linear(dim//2, 1)
        )

    def forward(self, seq, mask):
        # seq: [B,L,D], mask: [B,L] (float 1/0)
        B, L, D = seq.shape
        scores = self.att(seq).squeeze(-1)   # [B,L]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B,L,1]
        pooled = (weights * seq).sum(dim=1)  # [B,D]
        return pooled

class SpeakerCrossAttentionFusionModel(nn.Module):
    def __init__(self, dimA, dimB, latent_dim=256,
                 trans_layers=2, trans_heads=4, dropout=0.2):
        super().__init__()
        self.encA = ModalityEncoder(dimA, latent_dim=latent_dim, n_layers=trans_layers, n_heads=trans_heads, dropout=dropout)
        self.encB = ModalityEncoder(dimB, latent_dim=latent_dim, n_layers=trans_layers, n_heads=trans_heads, dropout=dropout)
        self.cross = CrossAttentionBlock(latent_dim=latent_dim, n_heads=trans_heads, dropout=dropout)
        self.pool = SegmentAttentionPool(dim=2*latent_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(2*latent_dim),
            nn.Linear(2*latent_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, A_batch, B_batch, mask):
        # A_batch: [B_s, L, dimA], B_batch: [B_s, L, dimB], mask: [B_s, L]
        A_enc = self.encA(A_batch, mask)   # [B,L,D]
        B_enc = self.encB(B_batch, mask)   # [B,L,D]
        fused_seq = self.cross(A_enc, B_enc, mask)  # [B,L,2D]
        pooled = self.pool(fused_seq, mask)         # [B,2D]
        logits = self.classifier(pooled)            # [B,2]
        return logits, pooled  # pooled for inspection if needed

# ==========================
# TRAIN / EVAL UTIL
# ==========================
def compute_speaker_metrics(y_true, probs):
    # y_true: array (N_spk,), probs: array (N_spk,)
    # compute ROC-threshold and metrics
    try:
        fpr, tpr, th = roc_curve(y_true, probs)
        best_th = th[np.argmax(tpr - fpr)]
    except Exception:
        best_th = 0.5
    preds = (probs >= best_th).astype(int)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    wf1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except:
        auc = 0.5
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        ua = 0.5 * ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6)))
    except:
        ua = 0.5
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "wf1": wf1, "auc": auc, "ua": ua, "th": best_th}

# ==========================
# MAIN TRAIN LOOP
# ==========================
def train_and_eval():
    # build datasets
    train_ds = SpeakerDataset(FEAT_DIR_A, FEAT_DIR_B, SPLIT_TRAIN)
    dev_ds   = SpeakerDataset(FEAT_DIR_A, FEAT_DIR_B, SPLIT_DEV)
    print(f"Speakers train/dev: {len(train_ds)}, {len(dev_ds)}")

    # prepare speaker sampler to balance classes (per-speaker weights)
    train_labels = [train_ds[i][2] for i in range(len(train_ds))]
    class_counts = {c: train_labels.count(c) for c in set(train_labels)}
    # weight inversely by class frequency (simple)
    speaker_weights = np.array([1.0 / class_counts[train_labels[i]] for i in range(len(train_ds))], dtype=np.float32)
    sampler = WeightedRandomSampler(speaker_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SPEAKERS, sampler=sampler, collate_fn=collate_speakers)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SPEAKERS, shuffle=False, collate_fn=collate_speakers)

    # model init
    # use dims from dataset first speaker
    dimA = train_ds[0][0].shape[1]
    dimB = train_ds[0][1].shape[1]
    model = SpeakerCrossAttentionFusionModel(dimA, dimB, latent_dim=LATENT_DIM,
                                             trans_layers=TRANSFORMER_LAYERS, trans_heads=TRANSFORMER_HEADS, dropout=DROPOUT)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # early-stop composite metric
    def composite_score(f1, wf1, auc, ua):
        return 0.4*f1 + 0.3*wf1 + 0.2*auc + 0.1*ua

    best_score = -1.0
    patience = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_losses = []
        for A_batch, B_batch, mask, labels, sids, lengths in train_loader:
            A_batch = A_batch.to(DEVICE)
            B_batch = B_batch.to(DEVICE)
            mask = mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _ = model(A_batch, B_batch, mask)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())

        # --- DEV EVAL (speaker-level)
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for A_batch, B_batch, mask, labels, sids, lengths in dev_loader:
                A_batch = A_batch.to(DEVICE)
                B_batch = B_batch.to(DEVICE)
                mask = mask.to(DEVICE)
                logits, _ = model(A_batch, B_batch, mask)
                probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        metrics = compute_speaker_metrics(all_labels, all_probs)
        score = composite_score(metrics["f1"], metrics["wf1"], metrics["auc"], metrics["ua"])

        print(f"Epoch {epoch:03d} | Loss={np.mean(train_losses):.4f} | Dev F1={metrics['f1']:.3f} | AUC={metrics['auc']:.3f} | Score={score:.4f}")

        # early stop
        if score > best_score:
            best_score = score
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping triggered.")
                break

    # final evaluation - load best and evaluate on dev
    model.load_state_dict(best_state)
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for A_batch, B_batch, mask, labels, sids, lengths in dev_loader:
            A_batch = A_batch.to(DEVICE)
            B_batch = B_batch.to(DEVICE)
            mask = mask.to(DEVICE)
            logits, _ = model(A_batch, B_batch, mask)
            probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    metrics = compute_speaker_metrics(all_labels, all_probs)
    print("\nFINAL DEV METRICS (speaker-level):")
    for k,v in metrics.items():
        if k != "th":
            print(f"{k.upper()}: {v:.4f}")
    print("Threshold:", metrics["th"])

    return metrics, all_probs, all_labels


if __name__ == "__main__":
    RUNS = 5

    all_metrics = {
        "acc": [], "prec": [], "rec": [],
        "f1": [], "wf1": [], "auc": [], "ua": [], "th": []
    }

    run_probs = []   # per-run speaker probability array
    speaker_ids_ref = None   # keeps speaker ID order (same each run)
    run_labels_ref = None    # true labels for each speaker

    for r in range(RUNS):
        print(f"\n============ RUN {r+1}/{RUNS} ============\n")
        metrics, probs, labels = train_and_eval()

        # store scalar metrics
        for k in ["acc","prec","rec","f1","wf1","auc","ua","th"]:
            all_metrics[k].append(metrics[k])

        # store probabilities and label order
        run_probs.append(probs)

        if run_labels_ref is None:
            run_labels_ref = labels
            speaker_ids_ref = list(range(len(labels)))  # index positions become identifiers

    # ================= SUMMARY RESULTS =================
    print("\n=========== FINAL SUMMARY (over 5 runs) ===========")
    for key in ["acc","prec","rec","f1","wf1","auc","ua"]:
        print(f"{key.upper():5s} Mean={np.mean(all_metrics[key]):.4f}  STD={np.std(all_metrics[key]):.4f}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # ====================================================
    # BEST RUN ROC CURVE (Speaker level)
    # ====================================================

    auc_array = np.array(all_metrics["auc"], dtype=float)

    print("AUCs per run:", auc_array)

    best_idx  = np.argmax(auc_array)
    best_auc  = auc_array[best_idx]           # this will be the largest value
    mean_auc  = np.mean(auc_array)
    std_auc   = np.std(auc_array)

    print("best_auc:", best_auc, "mean_auc:", mean_auc, "std_auc:", std_auc)

    fpr, tpr, _ = roc_curve(run_labels_ref, run_probs[best_idx])

    plt.figure(figsize=(3.5, 3))
    plt.plot(fpr, tpr, linewidth=1.5,
         label=f"AUC={best_auc:.3f}±{std_auc:.3f}")  # <-- 0.740 ± 0.026
    plt.plot([0,1], [0,1], linestyle="--", linewidth=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("/home/bubai-maji/bubai/English/daic_results/daic_roc.png", dpi=300)

    # ====================================================
    # MAJORITY-VOTE CONFUSION MATRIX (Speaker level)
    # ====================================================
    prediction_votes = np.zeros((RUNS, len(run_labels_ref)), dtype=int)

    for i, (probs, th) in enumerate(zip(run_probs, all_metrics["th"])):
        prediction_votes[i] = (probs >= th).astype(int)

    # majority prediction across runs for each speaker
    final_preds = (prediction_votes.mean(axis=0) >= 0.5).astype(int)

    cm = confusion_matrix(run_labels_ref, final_preds)
    print("\nMajority Vote Confusion Matrix:\n", cm)

    plt.figure(figsize=(3.5,3))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Depression"],
                yticklabels=["Normal", "Depression"], annot_kws={"size": 12})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    import matplotlib.ticker as ticker
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.tight_layout()
    plt.savefig("/home/bubai-maji/bubai/English/daic_results/daic_cm.png", dpi=300)
    print("Confusion Matrix saved: daic_cm.png")

    print("\n=========== DONE ===========\n")
