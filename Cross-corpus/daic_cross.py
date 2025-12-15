import os
import math
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ==============================
# CONFIG
# ==============================
SEED = 42
RUNS = 5
BATCH = 64
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
NUM_CLASSES = 2
LATENT_DIM = 256
HEADS = 8
DROPOUT = 0.2

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================
# PATHS
# ==============================
DAIC_A = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"
DAIC_B = "/home/bubai-maji/bubai/English/features_npy/wavlm_base"

EDAIC_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
EDAIC_B = "/home/bubai-maji/bubai/English/edic_features_npy/wavlm_base"

BN_A = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/IS10_5fold"
BN_B = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/WavLM_base_5fold"

IT_A = "/home/bubai-maji/bubai/Itali/features_npy/IS10"
IT_B = "/home/bubai-maji/bubai/Itali/features_npy/microsoft-wavlm-base"


# ==============================
# DATA LOADING
# ==============================
def load_split(base, split):
    X = np.load(f"{base}/{split}/segment_X.npy")
    y = np.load(f"{base}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{base}/{split}/segment_speaker_id.npy")
    return X, y, spk


def load_fold(base, fold):
    X = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


# ==============================
# DATASET
# ==============================
class SegDataset(Dataset):
    def __init__(self, Xa, Xb, y, spk):
        self.Xa = torch.tensor(Xa, dtype=torch.float32)
        self.Xb = torch.tensor(Xb, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = np.array(spk)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.Xa[idx], self.Xb[idx], self.y[idx], self.spk[idx]


# ==============================
# MODEL: CrossAttentionFusion (Option A)
# ==============================
class ModalityProj(nn.Module):
    def __init__(self, in_dim, latent=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, latent),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    True bidirectional cross-attention fusion for EXACTLY 2 modalities:
      - A attends to B: Q=A, K=B, V=B
      - B attends to A: Q=B, K=A, V=A
    Then updated tokens go through FFN + residual, and finally classifier on [A_fused ; B_fused].
    """
    def __init__(self, dims, latent=LATENT_DIM, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        assert len(dims) == 2, "CrossAttentionFusion assumes exactly 2 modalities."

        # per-modality projection
        self.proj = nn.ModuleList([ModalityProj(d, latent, dropout) for d in dims])

        # cross-attention blocks
        self.att_AtoB = nn.MultiheadAttention(
            embed_dim=latent, num_heads=heads,
            dropout=dropout, batch_first=True
        )
        self.att_BtoA = nn.MultiheadAttention(
            embed_dim=latent, num_heads=heads,
            dropout=dropout, batch_first=True
        )

        # LayerNorm + FFN for each modality
        self.norm_A1 = nn.LayerNorm(latent)
        self.norm_B1 = nn.LayerNorm(latent)

        self.ff_A = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent),
            nn.Dropout(dropout)
        )
        self.ff_B = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent),
            nn.Dropout(dropout)
        )

        self.norm_A2 = nn.LayerNorm(latent)
        self.norm_B2 = nn.LayerNorm(latent)

        # classifier on concatenated [A_fused, B_fused]
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent * 2),
            nn.Linear(latent * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, NUM_CLASSES)
        )

    def _fuse(self, Xs):
        """
        Xs: list [XA, XB]; each shape [B, in_dim]
        returns fused embedding [B, 2*latent]
        """
        A = self.proj[0](Xs[0])  # [B, L]
        B = self.proj[1](Xs[1])

        # treat each modality as a single token
        A = A.unsqueeze(1)  # [B,1,L]
        B = B.unsqueeze(1)  # [B,1,L]

        # A attends to B
        A2, _ = self.att_AtoB(A, B, B)  # Q=A, K=B, V=B
        A = self.norm_A1(A + A2)

        # B attends to A
        B2, _ = self.att_BtoA(B, A, A)  # Q=B, K=A, V=A
        B = self.norm_B1(B + B2)

        # FFN + residual
        A_ff = self.ff_A(A)
        A = self.norm_A2(A + A_ff)

        B_ff = self.ff_B(B)
        B = self.norm_B2(B + B_ff)

        # remove token dimension
        A_vec = A.squeeze(1)  # [B, latent]
        B_vec = B.squeeze(1)  # [B, latent]

        fused = torch.cat([A_vec, B_vec], dim=-1)  # [B, 2*latent]
        return fused

    def forward(self, Xs):
        fused = self._fuse(Xs)
        return self.classifier(fused)


# ==============================
# METRICS
# ==============================
def compute_speaker_metrics(y_true, probs):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    # threshold by Youden's J on ROC
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
    except Exception:
        auc = 0.5

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        ua = 0.5 * ((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))
    except Exception:
        ua = 0.5

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "wf1": wf1,
        "auc": auc,
        "ua": ua,
        "th": best_th,
    }


def evaluate_corpus(model, Xa, Xb, y, spk, batch_size=64):
    """
    Segment-level inference, then speaker-level aggregation.
    """
    ds = SegDataset(Xa, Xb, y, spk)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    all_probs = []
    all_labels = []
    all_spk = []

    with torch.no_grad():
        for xa, xb, yb, sp in dl:
            xa = xa.to(DEVICE)
            xb = xb.to(DEVICE)
            logits = model([xa, xb])
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(yb.numpy())
            all_spk.extend(sp)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_spk = np.array(all_spk)

    # speaker-level aggregation
    spk_ids = np.unique(all_spk)
    spk_probs = []
    spk_labels = []

    for sid in spk_ids:
        idx = np.where(all_spk == sid)[0]
        spk_probs.append(all_probs[idx].mean())
        spk_labels.append(all_labels[idx][0])

    spk_probs = np.array(spk_probs)
    spk_labels = np.array(spk_labels)

    return compute_speaker_metrics(spk_labels, spk_probs)


# ==============================
# MAIN CROSS-CORPUS LOGIC
# ==============================
def run_cross_daic():
    # ----- Load DAIC train/dev -----
    Xa_tr_raw, y_tr, spk_tr = load_split(DAIC_A, "train")
    Xb_tr_raw, _, _ = load_split(DAIC_B, "train")

    Xa_dev_raw, y_dev, spk_dev = load_split(DAIC_A, "dev")
    Xb_dev_raw, _, _ = load_split(DAIC_B, "dev")

    # feature normalization stats from DAIC train
    mA = Xa_tr_raw.mean(0)
    sA = Xa_tr_raw.std(0); sA[sA == 0] = 1
    mB = Xb_tr_raw.mean(0)
    sB = Xb_tr_raw.std(0); sB[sB == 0] = 1

    # class-balanced sampling by speaker (segment weights)
    spk_counts = {s: np.sum(spk_tr == s) for s in np.unique(spk_tr)}
    weights = np.array([1.0 / spk_counts[s] for s in spk_tr], dtype=np.float32)

    dimA = Xa_tr_raw.shape[1]
    dimB = Xb_tr_raw.shape[1]

    metrics_names = ["acc", "prec", "rec", "f1", "wf1", "auc", "ua"]
    results_edaic = []
    results_bn = []
    results_it = []

    for run in range(RUNS):
        print(f"\n========== RUN {run+1}/{RUNS} ==========")

        # normalize DAIC train/dev
        Xa_tr = np.nan_to_num((Xa_tr_raw - mA) / sA)
        Xb_tr = np.nan_to_num((Xb_tr_raw - mB) / sB)
        Xa_dev = np.nan_to_num((Xa_dev_raw - mA) / sA)
        Xb_dev = np.nan_to_num((Xb_dev_raw - mB) / sB)

        train_loader = DataLoader(
            SegDataset(Xa_tr, Xb_tr, y_tr, spk_tr),
            batch_size=BATCH,
            sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        )

        # model
        model = CrossAttentionFusion(dims=[dimA, dimB], latent=LATENT_DIM, heads=HEADS, dropout=DROPOUT).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()

        best_f1 = -1.0
        patience_counter = 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # ----- TRAINING LOOP (DAIC only) -----
        for ep in range(EPOCHS):
            model.train()
            for xa, xb, yb, sp in train_loader:
                xa = xa.to(DEVICE)
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                opt.zero_grad()
                logits = model([xa, xb])
                loss = crit(logits, yb)
                loss.backward()
                opt.step()

            # early stopping on DAIC-dev speaker F1
            dev_metrics = evaluate_corpus(model, Xa_dev, Xb_dev, y_dev, spk_dev, batch_size=BATCH)
            f1_dev = dev_metrics["f1"]
            print(f"  Epoch {ep+1:03d} | DAIC-DEV F1={f1_dev:.4f}")

            if f1_dev > best_f1:
                best_f1 = f1_dev
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("  Early stopping.")
                    break

        # load best
        model.load_state_dict(best_state)
        model.to(DEVICE)
        model.eval()

        # ----- EDAIC TEST -----
        Xa_te_e, y_te_e, spk_te_e = load_split(EDAIC_A, "test")
        Xb_te_e, _, _ = load_split(EDAIC_B, "test")

        Xa_te_e = np.nan_to_num((Xa_te_e - mA) / sA)
        Xb_te_e = np.nan_to_num((Xb_te_e - mB) / sB)

        met_edaic = evaluate_corpus(model, Xa_te_e, Xb_te_e, y_te_e, spk_te_e, batch_size=BATCH)
        results_edaic.append(met_edaic)

        # ----- BENGALI 5-FOLD TEST (average folds) -----
        fold_metrics_bn = []
        for f in range(1, 6):
            X1, y1, sp1 = load_fold(BN_A, f)
            X2, _, _ = load_fold(BN_B, f)

            X1 = np.nan_to_num((X1 - mA) / sA)
            X2 = np.nan_to_num((X2 - mB) / sB)

            m_fold = evaluate_corpus(model, X1, X2, y1, sp1, batch_size=BATCH)
            fold_metrics_bn.append(m_fold)

        avg_bn = {k: np.mean([fm[k] for fm in fold_metrics_bn]) for k in metrics_names}
        results_bn.append(avg_bn)

        # ----- ITALIAN 5-FOLD TEST (average folds) -----
        fold_metrics_it = []
        for f in range(1, 6):
            X1, y1, sp1 = load_fold(IT_A, f)
            X2, _, _ = load_fold(IT_B, f)

            X1 = np.nan_to_num((X1 - mA) / sA)
            X2 = np.nan_to_num((X2 - mB) / sB)

            m_fold = evaluate_corpus(model, X1, X2, y1, sp1, batch_size=BATCH)
            fold_metrics_it.append(m_fold)

        avg_it = {k: np.mean([fm[k] for fm in fold_metrics_it]) for k in metrics_names}
        results_it.append(avg_it)

    # ==========================
    # SUMMARY
    # ==========================
    def print_summary(name, results_list):
        print(f"\n===== DAIC → {name} (over {RUNS} runs) =====")
        for m in metrics_names:
            vals = np.array([r[m] for r in results_list], dtype=float)
            print(f"{m.upper():4s} Mean={np.nanmean(vals):.4f}  STD={np.nanstd(vals):.4f}")

    print_summary("EDAIC", results_edaic)
    print_summary("BENGALI", results_bn)
    print_summary("ITALIAN", results_it)


if __name__ == "__main__":
    run_cross_daic()
