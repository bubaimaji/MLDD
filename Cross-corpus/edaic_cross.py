import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# =============================
# CONFIG
# =============================
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

# =============================
# PATHS
# =============================
DAIC_A = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"
DAIC_B = "/home/bubai-maji/bubai/English/features_npy/data2vec_large"

EDAIC_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
EDAIC_B = "/home/bubai-maji/bubai/English/edic_features_npy/data2vec_large"

BN_A = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/IS10_5fold"
BN_B = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/Data2Vec_large_5fold"

IT_A = "/home/bubai-maji/bubai/Itali/features_npy/IS10"
IT_B = "/home/bubai-maji/bubai/Itali/features_npy/facebook-data2vec-audio-large-960h"


# =============================
# DATA LOADING
# =============================
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


# =============================
# DATASET
# =============================
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


# =============================
# MODEL — CrossAttentionFusion
# =============================
class ModalityProj(nn.Module):
    def __init__(self, in_dim, latent=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, latent),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dims, latent=LATENT_DIM, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.ModuleList([ModalityProj(d, latent) for d in dims])

        self.att_AtoB = nn.MultiheadAttention(latent, heads, dropout, batch_first=True)
        self.att_BtoA = nn.MultiheadAttention(latent, heads, dropout, batch_first=True)

        self.norm_A1 = nn.LayerNorm(latent)
        self.norm_B1 = nn.LayerNorm(latent)
        self.norm_A2 = nn.LayerNorm(latent)
        self.norm_B2 = nn.LayerNorm(latent)

        self.ff_A = nn.Sequential(
            nn.Linear(latent, latent * 2), nn.ReLU(),
            nn.Linear(latent * 2, latent), nn.Dropout(dropout)
        )
        self.ff_B = nn.Sequential(
            nn.Linear(latent, latent * 2), nn.ReLU(),
            nn.Linear(latent * 2, latent), nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent * 2),
            nn.Linear(latent * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, NUM_CLASSES)
        )

    def _fuse(self, Xs):
        A = self.proj[0](Xs[0]).unsqueeze(1)
        B = self.proj[1](Xs[1]).unsqueeze(1)

        A2, _ = self.att_AtoB(A, B, B)
        A = self.norm_A1(A + A2)
        A = self.norm_A2(A + self.ff_A(A))

        B2, _ = self.att_BtoA(B, A, A)
        B = self.norm_B1(B + B2)
        B = self.norm_B2(B + self.ff_B(B))

        return torch.cat([A.squeeze(1), B.squeeze(1)], dim=-1)

    def forward(self, Xs):
        fused = self._fuse(Xs)
        return self.classifier(fused)


# =============================
# METRICS
# =============================
def compute_speaker_metrics(y_true, probs):
    y_true, probs = np.array(y_true), np.array(probs)
    try:
        fpr, tpr, th = roc_curve(y_true, probs)
        best_th = th[np.argmax(tpr - fpr)]
    except: best_th = 0.5

    preds = (probs >= best_th).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    wf1 = f1_score(y_true, preds, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    ua = 0.5*((tp/(tp+fn+1e-6))+(tn/(tn+fp+1e-6)))

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, wf1=wf1, auc=auc, ua=ua)


def evaluate(model, Xa, Xb, y, spk):
    ds = SegDataset(Xa, Xb, y, spk)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False)

    model.eval()
    seg_prob, seg_y, seg_s = [], [], []

    with torch.no_grad():
        for xa, xb, yb, sp in dl:
            xa = xa.to(DEVICE); xb = xb.to(DEVICE)
            probs = torch.softmax(model([xa, xb]), 1)[:, 1].cpu().numpy()

            seg_prob.extend(probs); seg_y.extend(yb.numpy()); seg_s.extend(sp)

    seg_prob, seg_y, seg_s = map(np.array, [seg_prob, seg_y, seg_s])

    spk_ids = np.unique(seg_s)
    sp_p, sp_t = [], []

    for sid in spk_ids:
        idx = np.where(seg_s == sid)[0]
        sp_p.append(seg_prob[idx].mean())
        sp_t.append(seg_y[idx][0])

    return compute_speaker_metrics(np.array(sp_t), np.array(sp_p))


# =============================
# TRAINING LOOP
# =============================
def run_cross_edaic():
    # Load EDAIC train/dev
    Xa_tr_raw, y_tr, spk_tr = load_split(EDAIC_A, "train")
    Xb_tr_raw, _, _ = load_split(EDAIC_B, "train")

    Xa_dev_raw, y_dev, spk_dev = load_split(EDAIC_A, "test")
    Xb_dev_raw, _, _ = load_split(EDAIC_B, "test")

    # DAIC normalization source = EDAIC train
    mA, sA = Xa_tr_raw.mean(0), Xa_tr_raw.std(0); sA[sA == 0] = 1
    mB, sB = Xb_tr_raw.mean(0), Xb_tr_raw.std(0); sB[sB == 0] = 1

    spk_counts = {s: np.sum(spk_tr == s) for s in np.unique(spk_tr)}
    weights = np.array([1/spk_counts[s] for s in spk_tr], dtype=np.float32)

    dims = [Xa_tr_raw.shape[1], Xb_tr_raw.shape[1]]

    metrics_list = ["acc", "prec", "rec", "f1", "wf1", "auc", "ua"]
    res_daic, res_bn, res_it = [], [], []

    for run in range(RUNS):
        print(f"\n========== RUN {run+1}/{RUNS} ==========")

        Xa_tr, Xb_tr = (Xa_tr_raw-mA)/sA, (Xb_tr_raw-mB)/sB
        Xa_dev, Xb_dev = (Xa_dev_raw-mA)/sA, (Xb_dev_raw-mB)/sB

        train_loader = DataLoader(
            SegDataset(Xa_tr, Xb_tr, y_tr, spk_tr),
            batch_size=BATCH,
            sampler=WeightedRandomSampler(weights, len(weights), replacement=True)
        )

        model = CrossAttentionFusion(dims).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()

        best_f1 = -1; patience = 0
        best_state = {k: v.cpu() for k,v in model.state_dict().items()}

        # ---- TRAIN (EDAIC) ----
        for ep in range(EPOCHS):
            model.train()
            for xa,xb,yb,_ in train_loader:
                xa,xb,yb = xa.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); loss = crit(model([xa,xb]), yb); loss.backward(); opt.step()

            dev_m = evaluate(model, Xa_dev, Xb_dev, y_dev, spk_dev)
            print(f" Epoch {ep+1:03d} | EDAIC-DEV F1={dev_m['f1']:.4f}")

            if dev_m["f1"] > best_f1:
                best_f1 = dev_m["f1"]
                best_state = {k:v.cpu() for k,v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

        model.load_state_dict(best_state)
        model.to(DEVICE)

        # ---- TEST: DAIC ----
        Xa_t, y_t, spk_t = load_split(DAIC_A, "dev")
        Xb_t, _, _ = load_split(DAIC_B, "dev")
        Xa_t, Xb_t = (Xa_t-mA)/sA, (Xb_t-mB)/sB
        res_daic.append(evaluate(model, Xa_t, Xb_t, y_t, spk_t))

        # ---- TEST: Bengali ----
        fold_bn = []
        for f in range(1,6):
            XA,yT,spT = load_fold(BN_A, f)
            XB,_,_ = load_fold(BN_B, f)
            fold_bn.append(evaluate(model, (XA-mA)/sA, (XB-mB)/sB, yT, spT))
        res_bn.append({k:np.mean([m[k] for m in fold_bn]) for k in metrics_list})

        # ---- TEST: Italian ----
        fold_it = []
        for f in range(1,6):
            XA,yT,spT = load_fold(IT_A, f)
            XB,_,_ = load_fold(IT_B, f)
            fold_it.append(evaluate(model, (XA-mA)/sA, (XB-mB)/sB, yT, spT))
        res_it.append({k:np.mean([m[k] for m in fold_it]) for k in metrics_list})

    def summary(name, R):
        print(f"\n==== EDAIC → {name} ====")
        for k in metrics_list:
            vals = np.array([v[k] for v in R])
            print(f"{k.upper()}: Mean={vals.mean():.4f} STD={vals.std():.4f}")

    summary("DAIC", res_daic)
    summary("BENGALI", res_bn)
    summary("ITALIAN", res_it)


if __name__ == "__main__":
    run_cross_edaic()
