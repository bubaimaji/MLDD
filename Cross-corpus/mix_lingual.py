import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# =========================
# CONFIG
# =========================
SEED = 42
RUNS = 5
FOLDS = 5           # used for BN / IT cross-lingual folds
BATCH = 64
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
LATENT_DIM = 256
HEADS = 8
DROPOUT = 0.2
NUM_CLASSES = 2

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# PATHS
# =========================
DAIC_A = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"
DAIC_B = "/home/bubai-maji/bubai/English/features_npy/wavlm_base"

EDAIC_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
EDAIC_B = "/home/bubai-maji/bubai/English/edic_features_npy/wavlm_base"

BN_A = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/IS10_5fold"
BN_B = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/WavLM_base_5fold"

IT_A = "/home/bubai-maji/bubai/Itali/features_npy/IS10"
IT_B = "/home/bubai-maji/bubai/Itali/features_npy/microsoft-wavlm-base"


# =========================
# LOADERS
# =========================
def load_split(base, split):
    """Load DAIC / EDAIC split: train / dev / test."""
    X = np.load(f"{base}/{split}/segment_X.npy")
    y = np.load(f"{base}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{base}/{split}/segment_speaker_id.npy")
    return X, y, spk


def load_fold(base, fold):
    """Load language-specific k-fold data (Bangla / Italian)."""
    X = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


# =========================
# DATASET
# =========================
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


# =========================
# MODEL — CrossAttentionFusion
# =========================
class ModalityProj(nn.Module):
    def __init__(self, in_dim, latent=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, latent),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    True bidirectional cross-attention fusion for EXACTLY 2 modalities:
      - A attends to B (Q=A, K=B, V=B)
      - B attends to A (Q=B, K=A, V=A)
    Then per-modality FFN + residual, and classifier on [A_fused ; B_fused].
    """
    def __init__(self, dims, latent=LATENT_DIM, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        assert len(dims) == 2, "CrossAttentionFusion assumes 2 modalities."
        self.proj = nn.ModuleList([ModalityProj(d, latent, dropout) for d in dims])

        self.att_AtoB = nn.MultiheadAttention(latent, heads, dropout, batch_first=True)
        self.att_BtoA = nn.MultiheadAttention(latent, heads, dropout, batch_first=True)

        self.norm_A1 = nn.LayerNorm(latent)
        self.norm_A2 = nn.LayerNorm(latent)
        self.norm_B1 = nn.LayerNorm(latent)
        self.norm_B2 = nn.LayerNorm(latent)

        self.ff_A = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent),
            nn.Dropout(dropout),
        )
        self.ff_B = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent),
            nn.Dropout(dropout),
        )

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
            nn.Linear(32, NUM_CLASSES),
        )

    def _fuse(self, Xs):
        # Xs[0], Xs[1]: [B, D]
        A = self.proj[0](Xs[0]).unsqueeze(1)  # [B,1,L]
        B = self.proj[1](Xs[1]).unsqueeze(1)  # [B,1,L]

        # A attends to B
        A2, _ = self.att_AtoB(A, B, B)
        A = self.norm_A1(A + A2)
        A = self.norm_A2(A + self.ff_A(A))

        # B attends to A
        B2, _ = self.att_BtoA(B, A, A)
        B = self.norm_B1(B + B2)
        B = self.norm_B2(B + self.ff_B(B))

        A_vec = A.squeeze(1)
        B_vec = B.squeeze(1)
        return torch.cat([A_vec, B_vec], dim=-1)

    def forward(self, Xs):
        fused = self._fuse(Xs)
        return self.classifier(fused)


# =========================
# METRICS
# =========================
def compute_speaker_metrics(y_true, probs):
    """
    Aggregate at speaker-level:
      - threshold from best (TPR - FPR)
      - compute ACC, PREC, REC, F1, weighted-F1, AUC, UA
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    try:
        fpr, tpr, th = roc_curve(y_true, probs)
        th_best = th[np.argmax(tpr - fpr)]
    except Exception:
        th_best = 0.5

    preds = (probs >= th_best).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    wf1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    ua = 0.5 * ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6)))

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, wf1=wf1, auc=auc, ua=ua)


def evaluate(model, Xa, Xb, y, spk, batch_size=BATCH):
    """
    Segment-level inference → speaker-level average probabilities → metrics.
    """
    ds = SegDataset(Xa, Xb, y, spk)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    seg_probs = []
    seg_labels = []
    seg_spk = []

    with torch.no_grad():
        for xa, xb, yb, sp in dl:
            xa = xa.to(DEVICE)
            xb = xb.to(DEVICE)
            logits = model([xa, xb])
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            seg_probs.extend(probs)
            seg_labels.extend(yb.numpy())
            seg_spk.extend(sp)

    seg_probs = np.array(seg_probs)
    seg_labels = np.array(seg_labels)
    seg_spk = np.array(seg_spk)

    spk_ids = np.unique(seg_spk)
    spk_probs = []
    spk_labels = []

    for sid in spk_ids:
        idx = np.where(seg_spk == sid)[0]
        spk_probs.append(seg_probs[idx].mean())
        spk_labels.append(seg_labels[idx][0])

    return compute_speaker_metrics(np.array(spk_labels), np.array(spk_probs))


# =========================
# MAIN MIX-LINGUAL EXP
# =========================
def run_mixlingual():
    metrics_keys = ["acc", "prec", "rec", "f1", "wf1", "auc", "ua"]

    # per-run averaged results (after folding)
    results_daic = []
    results_edaic = []
    results_bn = []
    results_it = []

    # Preload DAIC dev raw once (for val + test)
    XA_dev_raw, y_dev, spk_dev = load_split(DAIC_A, "dev")
    XB_dev_raw, _, _ = load_split(DAIC_B, "dev")

    for run in range(RUNS):
        print(f"\n========== MIX-LINGUAL RUN {run + 1}/{RUNS} ==========")

        fold_daic = []
        fold_edaic = []
        fold_bn = []
        fold_it = []

        for fold in range(1, FOLDS + 1):
            print(f"\n----- Mix-Fold {fold}/{FOLDS} -----")

            # =========================
            # 1. BUILD TRAIN SETS
            # =========================
            # EDAIC train (English) — ONLY this for English training
            XA_edaic_tr, y_edaic_tr, spk_edaic_tr = load_split(EDAIC_A, "train")
            XB_edaic_tr, _, _ = load_split(EDAIC_B, "train")

            # Bengali train (all folds except current)
            XA_bn_tr_list, XB_bn_tr_list = [], []
            y_bn_tr_list, spk_bn_tr_list = [], []

            for f in range(1, FOLDS + 1):
                if f == fold:
                    continue
                Xa_b, y_b, spk_b = load_fold(BN_A, f)
                Xb_b, _, _ = load_fold(BN_B, f)
                XA_bn_tr_list.append(Xa_b)
                XB_bn_tr_list.append(Xb_b)
                y_bn_tr_list.append(y_b)
                spk_bn_tr_list.append(spk_b)

            XA_bn_tr = np.vstack(XA_bn_tr_list)
            XB_bn_tr = np.vstack(XB_bn_tr_list)
            y_bn_tr = np.hstack(y_bn_tr_list)
            spk_bn_tr = np.hstack(spk_bn_tr_list)

            # Italian train (all folds except current)
            XA_it_tr_list, XB_it_tr_list = [], []
            y_it_tr_list, spk_it_tr_list = [], []

            for f in range(1, FOLDS + 1):
                if f == fold:
                    continue
                Xa_i, y_i, spk_i = load_fold(IT_A, f)
                Xb_i, _, _ = load_fold(IT_B, f)
                XA_it_tr_list.append(Xa_i)
                XB_it_tr_list.append(Xb_i)
                y_it_tr_list.append(y_i)
                spk_it_tr_list.append(spk_i)

            XA_it_tr = np.vstack(XA_it_tr_list)
            XB_it_tr = np.vstack(XB_it_tr_list)
            y_it_tr = np.hstack(y_it_tr_list)
            spk_it_tr = np.hstack(spk_it_tr_list)

            # Concatenate ALL training data (EDAIC + BN + IT)
            XA_tr_raw = np.vstack([XA_edaic_tr, XA_bn_tr, XA_it_tr])
            XB_tr_raw = np.vstack([XB_edaic_tr, XB_bn_tr, XB_it_tr])
            y_tr = np.hstack([y_edaic_tr, y_bn_tr, y_it_tr])
            spk_tr = np.hstack([spk_edaic_tr, spk_bn_tr, spk_it_tr])

            # =========================
            # 2. NORMALIZATION
            # =========================
            mA = XA_tr_raw.mean(0)
            sA = XA_tr_raw.std(0)
            sA[sA == 0] = 1.0

            mB = XB_tr_raw.mean(0)
            sB = XB_tr_raw.std(0)
            sB[sB == 0] = 1.0

            XA_tr = np.nan_to_num((XA_tr_raw - mA) / sA)
            XB_tr = np.nan_to_num((XB_tr_raw - mB) / sB)

            # DAIC dev normalized (for validation + later reused as DAIC test)
            XA_dev = np.nan_to_num((XA_dev_raw - mA) / sA)
            XB_dev = np.nan_to_num((XB_dev_raw - mB) / sB)

            # =========================
            # 3. SPEAKER-BALANCED SAMPLER
            # =========================
            spk_counts = {s: np.sum(spk_tr == s) for s in np.unique(spk_tr)}
            weights = np.array([1.0 / spk_counts[s] for s in spk_tr], dtype=np.float32)

            train_loader = DataLoader(
                SegDataset(XA_tr, XB_tr, y_tr, spk_tr),
                batch_size=BATCH,
                sampler=WeightedRandomSampler(
                    weights,
                    num_samples=len(weights),
                    replacement=True
                ),
            )

            # =========================
            # 4. MODEL + TRAINING (early stopping on DAIC-DEV)
            # =========================
            dimA = XA_tr.shape[1]
            dimB = XB_tr.shape[1]
            model = CrossAttentionFusion([dimA, dimB]).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            crit = nn.CrossEntropyLoss()

            best_f1 = -1.0
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            for ep in range(EPOCHS):
                model.train()
                epoch_loss = 0.0

                for xa, xb, yb, _ in train_loader:
                    xa = xa.to(DEVICE)
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)

                    opt.zero_grad()
                    logits = model([xa, xb])
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()

                    epoch_loss += loss.item() * xa.size(0)

                epoch_loss /= len(train_loader.dataset)

                dev_metrics = evaluate(model, XA_dev, XB_dev, y_dev, spk_dev)
                print(
                    f"  Fold {fold} | Epoch {ep + 1:03d} "
                    f"| Loss={epoch_loss:.4f} | DAIC-DEV F1={dev_metrics['f1']:.4f}"
                )

                if dev_metrics["f1"] > best_f1:
                    best_f1 = dev_metrics["f1"]
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print("   Early stopping triggered.")
                        break

            model.load_state_dict(best_state)
            model.to(DEVICE)
            model.eval()

            # =========================
            # 5. BUILD TEST SETS (this mix-fold)
            # =========================

            # DAIC-test (using DAIC-dev as test set, as requested)
            XA_daic_te = XA_dev.copy()
            XB_daic_te = XB_dev.copy()
            y_daic_te = y_dev.copy()
            spk_daic_te = spk_dev.copy()
            m_daic = evaluate(model, XA_daic_te, XB_daic_te, y_daic_te, spk_daic_te)
            fold_daic.append(m_daic)

            # EDAIC-test (true test split)
            XA_edaic_te, y_edaic_te, spk_edaic_te = load_split(EDAIC_A, "test")
            XB_edaic_te, _, _ = load_split(EDAIC_B, "test")
            XA_edaic_te = np.nan_to_num((XA_edaic_te - mA) / sA)
            XB_edaic_te = np.nan_to_num((XB_edaic_te - mB) / sB)
            m_edaic = evaluate(model, XA_edaic_te, XB_edaic_te, y_edaic_te, spk_edaic_te)
            fold_edaic.append(m_edaic)

            # Bengali test: current fold
            XA_bn_te, y_bn_te, spk_bn_te = load_fold(BN_A, fold)
            XB_bn_te, _, _ = load_fold(BN_B, fold)
            XA_bn_te = np.nan_to_num((XA_bn_te - mA) / sA)
            XB_bn_te = np.nan_to_num((XB_bn_te - mB) / sB)
            m_bn = evaluate(model, XA_bn_te, XB_bn_te, y_bn_te, spk_bn_te)
            fold_bn.append(m_bn)

            # Italian test: current fold
            XA_it_te, y_it_te, spk_it_te = load_fold(IT_A, fold)
            XB_it_te, _, _ = load_fold(IT_B, fold)
            XA_it_te = np.nan_to_num((XA_it_te - mA) / sA)
            XB_it_te = np.nan_to_num((XB_it_te - mB) / sB)
            m_it = evaluate(model, XA_it_te, XB_it_te, y_it_te, spk_it_te)
            fold_it.append(m_it)

        # =========================
        # 6. AVERAGE OVER 5 MIX-FOLDS → RUN-LEVEL METRICS
        # =========================
        def avg_over_folds(fold_list):
            return {k: np.mean([m[k] for m in fold_list]) for k in metrics_keys}

        results_daic.append(avg_over_folds(fold_daic))
        results_edaic.append(avg_over_folds(fold_edaic))
        results_bn.append(avg_over_folds(fold_bn))
        results_it.append(avg_over_folds(fold_it))

    # =========================
    # FINAL SUMMARY over RUNS
    # =========================
    def print_summary(name, res_list):
        print(f"\n===== MIX-LINGUAL → {name} (over {RUNS} runs) =====")
        for k in metrics_keys:
            vals = np.array([r[k] for r in res_list], dtype=float)
            print(f"{k.upper():4s} Mean={np.nanmean(vals):.4f}  STD={np.nanstd(vals):.4f}")

    print_summary("DAIC (DEV as TEST)", results_daic)
    print_summary("EDAIC", results_edaic)
    print_summary("BENGALI", results_bn)
    print_summary("ITALIAN", results_it)


if __name__ == "__main__":
    run_mixlingual()
