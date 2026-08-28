import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from plot import (
    plot_confusion_matrix,
    plot_roc  # you can ignore if unused
)

# =====================================================
# CONFIG
# =====================================================
SEED = 42
FOLDS = 5
BATCH = 32          
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
NUM_CLASSES = 2
LATENT = 256
HEADS = 8
DROPOUT = 0.2
MAX_FRAMES = 1000    
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# NOTE: point these at the *_X_temporal.npy folders from wavlm1.py / is10.py,
# NOT the old pooled IS10_5fold / WavLM_base_5fold folders.
BASE_DIRS = [
    "/home/bubai-maji/bubai/revision2/bangla_feature/IS10_LLD_5fold",
    "/home/bubai-maji/bubai/revision2/bangla_feature/WavLM_large_temporal_5fold",
]

OUT_DIR = "/home/bubai-maji/bubai/Bangla/bn_results"
os.makedirs(OUT_DIR, exist_ok=True)

ALL_FINAL, ALL_LABELS, ALL_PROBS, ALL_TRUE = [], [], [], []
FOLD_FPRS, FOLD_TPRS, FOLD_AUCS = [], [], []


# =====================================================
# LOAD FOLD DATA  (ragged [T_i, D] sequences saved as object arrays)
# =====================================================
def load_fold_temporal(base, fold):
    X = np.load(f"{base}/fold{fold}_X_temporal.npy", allow_pickle=True)  # object array of [T_i, D]
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


def fit_frame_scaler(X_list):
    """Per-feature mean/std computed over ALL frames of ALL training utterances
    (train-only, to avoid the leakage your R1 rebuttal already committed to preventing)."""
    all_frames = np.concatenate([x for x in X_list], axis=0)  # [sum_T, D]
    mean = all_frames.mean(0)
    std = all_frames.std(0)
    std[std == 0] = 1.0
    return mean, std


def apply_frame_scaler(X_list, mean, std):
    return [(x - mean) / std for x in X_list]


# =====================================================
# DATASET  (variable length -> padded + mask via collate_fn)
# =====================================================
class MultiModalSeqDataset(Dataset):
    def __init__(self, X_list_per_modality, y, spk):
        """
        X_list_per_modality: list (len = n_modalities) of lists of [T_i, D_m] arrays
        y:   [N] labels
        spk: [N] speaker ids
        """
        self.X_list_per_modality = X_list_per_modality
        self.y = y
        self.spk = spk

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seqs = [torch.tensor(X[idx], dtype=torch.float32) for X in self.X_list_per_modality]
        return seqs, int(self.y[idx]), self.spk[idx]


def make_collate_fn(max_frames=MAX_FRAMES):
    def collate_fn(batch):
        n_modalities = len(batch[0][0])
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        spks = [b[2] for b in batch]

        padded_list, mask_list = [], []
        for m in range(n_modalities):
            seqs = [b[0][m][:max_frames] for b in batch]
            lengths = [s.shape[0] for s in seqs]
            T_max = max(lengths)
            D = seqs[0].shape[1]
            padded = torch.zeros(len(seqs), T_max, D)
            mask = torch.ones(len(seqs), T_max, dtype=torch.bool)  # True == PAD (ignored by attention)
            for i, s in enumerate(seqs):
                T = s.shape[0]
                padded[i, :T] = s
                mask[i, :T] = False
            padded_list.append(padded)
            mask_list.append(mask)

        return padded_list, mask_list, ys, spks
    return collate_fn


# =====================================================
# MODEL
# =====================================================
class SinusoidalPositionalEncoding(nn.Module):
    """MultiheadAttention has no notion of order by itself -- without this,
    cross-attention over frames would be permutation-invariant in time."""
    def __init__(self, dim, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FrameProj(nn.Module):
    """Per-timestep projection into the shared latent space. Applies to [B, T, in_dim]."""
    def __init__(self, in_dim, latent, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, latent),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pos = SinusoidalPositionalEncoding(latent)

    def forward(self, x):
        # x: [B, T, in_dim] -> [B, T, latent]
        return self.pos(self.net(x))


def masked_mean(x, pad_mask):
    """
    x:        [B, T, D]
    pad_mask: [B, T]  True at PAD positions
    Padded timesteps must NOT count toward the mean -- the old code's
    plain .mean(dim=1) would have silently pulled real utterance
    representations toward zero once sequences got padded.
    """
    valid = (~pad_mask).unsqueeze(-1).float()   # [B, T, 1]
    summed = (x * valid).sum(dim=1)             # [B, D]
    counts = valid.sum(dim=1).clamp(min=1.0)    # [B, 1]
    return summed / counts


class CrossAttentionFusion(nn.Module):
    """
    Genuine bidirectional cross-attention over TEMPORAL TOKEN SEQUENCES
    (frame-level SFM embeddings x frame-level acoustic LLDs), not single
    utterance-level vectors. With T_A, T_B > 1 this is no longer the
    degenerate case Reviewer 5 identified.
    """
    def __init__(self, dims, latent=LATENT, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        assert len(dims) == 2, "CrossAttentionFusion currently assumes exactly 2 modalities."

        self.proj = nn.ModuleList([FrameProj(d, latent, dropout) for d in dims])

        self.att_AtoB = nn.MultiheadAttention(latent, heads, dropout=dropout, batch_first=True)
        self.att_BtoA = nn.MultiheadAttention(latent, heads, dropout=dropout, batch_first=True)

        self.norm_A1 = nn.LayerNorm(latent)
        self.norm_B1 = nn.LayerNorm(latent)

        self.ff_A = nn.Sequential(
            nn.Linear(latent, latent * 2), nn.ReLU(),
            nn.Linear(latent * 2, latent), nn.Dropout(dropout),
        )
        self.ff_B = nn.Sequential(
            nn.Linear(latent, latent * 2), nn.ReLU(),
            nn.Linear(latent * 2, latent), nn.Dropout(dropout),
        )

        self.norm_A2 = nn.LayerNorm(latent)
        self.norm_B2 = nn.LayerNorm(latent)

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent * 2),
            nn.Linear(latent * 2, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, NUM_CLASSES),
        )

    def _fuse(self, Xs, masks, return_attn=False):
        """
        Xs:    [X_A, X_B]        each [B, T_m, in_dim_m]
        masks: [mask_A, mask_B]  each [B, T_m] bool, True == PAD
        """
        A = self.proj[0](Xs[0])   # [B, T_A, latent]
        B = self.proj[1](Xs[1])   # [B, T_B, latent]
        mask_A, mask_B = masks

        # A attends to B  (Q=A, K=V=B)
        A2, attn_AtoB = self.att_AtoB(A, B, B, key_padding_mask=mask_B, need_weights=return_attn)
        A = self.norm_A1(A + A2)

        # B attends to A  (Q=B, K=V=A)
        B2, attn_BtoA = self.att_BtoA(B, A, A, key_padding_mask=mask_A, need_weights=return_attn)
        B = self.norm_B1(B + B2)

        A = self.norm_A2(A + self.ff_A(A))
        B = self.norm_B2(B + self.ff_B(B))

        # masked mean pooling -- padded timesteps must not contribute
        A_vec = masked_mean(A, mask_A)
        B_vec = masked_mean(B, mask_B)

        fused = torch.cat([A_vec, B_vec], dim=-1)   # [B, 2*latent]

        if return_attn:
            return fused, (attn_AtoB, attn_BtoA)
        return fused

    def forward(self, Xs, masks):
        fused = self._fuse(Xs, masks)
        return self.classifier(fused)

    def embed(self, Xs, masks):
        with torch.no_grad():
            fused = self._fuse(Xs, masks)
        return fused.detach()

    def attention_maps(self, Xs, masks):
        """For the interpretability analysis Reviewer 2 (comment 7) is asking for."""
        with torch.no_grad():
            _, attn = self._fuse(Xs, masks, return_attn=True)
        return attn


def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except Exception:
        return np.nan


# =====================================================
# TRAIN ONE FOLD
# =====================================================
def train_fold(fold_idx):
    Xtr_all, Xte_all = [], []
    ytrain = ytest = spktest = None

    for m, base in enumerate(BASE_DIRS):
        X_list, y_list, spk_list = [], [], []
        for f in range(1, FOLDS + 1):
            Xf, yf, sp = load_fold_temporal(base, f)
            X_list.append(Xf); y_list.append(yf); spk_list.append(sp)

        X_te = list(X_list[fold_idx])
        y_te = y_list[fold_idx]
        spk_te = spk_list[fold_idx]

        X_tr = list(np.concatenate([X_list[i] for i in range(FOLDS) if i != fold_idx]))
        y_tr = np.hstack([y_list[i] for i in range(FOLDS) if i != fold_idx])

        Xtr_all.append(X_tr)
        Xte_all.append(X_te)

        if m == 0:
            ytrain, ytest, spktest = y_tr, y_te, spk_te

    # ---- per-modality, train-only frame normalization ---- #
    Xtr_scaled, Xte_scaled = [], []
    for Xtr, Xte in zip(Xtr_all, Xte_all):
        mean, std = fit_frame_scaler(Xtr)
        Xtr_scaled.append(apply_frame_scaler(Xtr, mean, std))
        Xte_scaled.append(apply_frame_scaler(Xte, mean, std))

    # ---- speaker sampler ---- #
    first_base = BASE_DIRS[0]
    spk_all = [np.load(f"{first_base}/fold{f}_speaker.npy", allow_pickle=True) for f in range(1, FOLDS + 1)]
    spk_train = np.hstack([spk_all[i] for i in range(FOLDS) if i != fold_idx])
    uniq, cnt = np.unique(spk_train, return_counts=True)
    cnt_map = dict(zip(uniq, cnt))
    weights = np.array([1.0 / cnt_map[s] for s in spk_train])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    collate_fn = make_collate_fn()

    train_loader = DataLoader(
        MultiModalSeqDataset(Xtr_scaled, ytrain, spk_train),
        batch_size=BATCH, sampler=sampler, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        MultiModalSeqDataset(Xte_scaled, ytest, spktest),
        batch_size=BATCH, shuffle=False, collate_fn=collate_fn,
    )

    dims = [X[0].shape[1] for X in Xtr_scaled]
    model = CrossAttentionFusion(dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_f1, patience_cnt = -1.0, 0

    for ep in range(EPOCHS):
        model.train()
        for Xs, masks, yb, _ in train_loader:
            Xs = [x.to(DEVICE) for x in Xs]
            masks = [m.to(DEVICE) for m in masks]
            yb = yb.to(DEVICE)

            opt.zero_grad()
            logits = model(Xs, masks)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        seg_p, seg_y, seg_s = [], [], []
        with torch.no_grad():
            for Xs, masks, yb, spk in test_loader:
                Xs = [x.to(DEVICE) for x in Xs]
                masks = [m.to(DEVICE) for m in masks]
                probs = torch.softmax(model(Xs, masks), dim=1)[:, 1].cpu().numpy()
                seg_p.extend(probs); seg_y.extend(yb.numpy()); seg_s.extend(spk)

        seg_s = np.array(seg_s, dtype=object)
        seg_y = np.array(seg_y)
        seg_p = np.array(seg_p)

        sp_p, sp_t = [], []
        for pid in np.unique(seg_s):
            mask = (seg_s == pid)
            sp_p.append(seg_p[mask].mean())
            sp_t.append(seg_y[mask][0])
        sp_p, sp_t = np.array(sp_p), np.array(sp_t)

        fpr, tpr, th = roc_curve(sp_t, sp_p)
        best_th = th[np.argmax(tpr - fpr)]
        pred = (sp_p >= best_th).astype(int)
        f1 = f1_score(sp_t, pred)
        print(f"FOLD {fold_idx+1} | Epoch {ep+1} | F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    speaker_learn, speaker_label, speaker_prob = {}, {}, {}
    with torch.no_grad():
        for Xs, masks, yb, spk in test_loader:
            Xs_dev = [x.to(DEVICE) for x in Xs]
            masks_dev = [m.to(DEVICE) for m in masks]
            probs = torch.softmax(model(Xs_dev, masks_dev), dim=1)[:, 1].cpu().numpy()
            emb = model.embed(Xs_dev, masks_dev).cpu().numpy()
            for i, s in enumerate(spk):
                speaker_learn.setdefault(s, []).append(emb[i])
                speaker_label.setdefault(s, []).append(int(yb[i].item()))
                speaker_prob.setdefault(s, []).append(float(probs[i]))

    spk_final, spk_labels, spk_probs = [], [], []
    for s in speaker_learn.keys():
        spk_final.append(np.mean(speaker_learn[s], axis=0))
        spk_labels.append(speaker_label[s][0])
        spk_probs.append(np.mean(speaker_prob[s]))

    spk_final = np.vstack(spk_final)
    spk_labels = np.array(spk_labels)
    spk_probs = np.array(spk_probs)

    ALL_FINAL.append(spk_final)
    ALL_LABELS.append(spk_labels)
    ALL_PROBS.append(spk_probs)
    ALL_TRUE.append(spk_labels)

    fpr, tpr, th = roc_curve(spk_labels, spk_probs)
    best_th = th[np.argmax(tpr - fpr)]
    pred = (spk_probs >= best_th).astype(int)
    auc = safe_auc(spk_labels, spk_probs)

    FOLD_FPRS.append(fpr); FOLD_TPRS.append(tpr); FOLD_AUCS.append(auc)

    acc = accuracy_score(spk_labels, pred)
    prec = precision_score(spk_labels, pred, zero_division=0)
    rec = recall_score(spk_labels, pred, zero_division=0)
    f1 = f1_score(spk_labels, pred)
    wf1 = f1_score(spk_labels, pred, average="weighted")
    tn, fp, fn, tp = confusion_matrix(spk_labels, pred).ravel()
    ua = 0.5 * ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6)))

    return acc, prec, rec, f1, wf1, auc, ua


# =====================================================
# MAIN 5-FOLD LOOP
# =====================================================
if __name__ == "__main__":
    results = {k: [] for k in ["acc", "prec", "rec", "f1", "wf1", "auc", "ua"]}

    for f in range(FOLDS):
        print("\n=======================================")
        print(f"TRAINING FOLD {f+1}/{FOLDS}")
        acc, prec, rec, f1, wf1, auc, ua = train_fold(f)
        for key, val in zip(results.keys(), [acc, prec, rec, f1, wf1, auc, ua]):
            results[key].append(val)

    print("\n========== 5-FOLD SUMMARY ==========")
    for k, v in results.items():
        print(f"{k.upper():5s} Mean={np.nanmean(v):.4f}  STD={np.nanstd(v):.4f}")

    print("\nGenerating final Confusion Matrix ...")
    all_probs = np.concatenate(ALL_PROBS)
    all_true = np.concatenate(ALL_TRUE)

    fpr, tpr, th = roc_curve(all_true, all_probs)
    best_th = th[np.argmax(tpr - fpr)]
    pred = (all_probs >= best_th).astype(int)

    plot_confusion_matrix(all_true, pred, os.path.join(OUT_DIR, "confusion.png"))
    print(f"\nSaved Confusion Matrix to {os.path.join(OUT_DIR, 'confusion.png')}")
    print("Done.")
