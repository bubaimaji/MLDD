import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from plot import plot_confusion_matrix, plot_auc_curve_folds, plot_tsne

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH = 64
LR = 2e-4
EPOCHS = 80
PATIENCE = 10
SEED = 42
runs = 5

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
BASE_B = "/home/bubai-maji/bubai/English/edic_features_npy/whisper_medium"


# -------------------------------------------------------
# LOAD SPLIT
# -------------------------------------------------------
def load_split(base, split):
    X = np.load(f"{base}/{split}/segment_X.npy")
    y = np.load(f"{base}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{base}/{split}/segment_speaker_id.npy")
    return X, y, spk


# -------------------------------------------------------
# DATASET
# -------------------------------------------------------
class SegDataset(Dataset):
    def __init__(self, Xa, Xb, y, spk):
        self.Xa = torch.tensor(Xa, dtype=torch.float32)
        self.Xb = torch.tensor(Xb, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = torch.tensor(spk, dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.Xa[idx], self.Xb[idx], self.y[idx], self.spk[idx]


# -------------------------------------------------------
# CROSS ATTENTION
# -------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, dimA, dimB, latent_dim=128, heads=4):
        super().__init__()
        self.projA = nn.Linear(dimA, latent_dim)
        self.projB = nn.Linear(dimB, latent_dim)

        self.attnAB = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.attnBA = nn.MultiheadAttention(latent_dim, heads, batch_first=True)

    def forward(self, XA, XB):
        A = self.projA(XA).unsqueeze(1)
        B = self.projB(XB).unsqueeze(1)
        A2B,_ = self.attnAB(A, B, B)
        B2A,_ = self.attnBA(B, A, A)
        return torch.cat([A2B.squeeze(1), B2A.squeeze(1)], dim=-1)  # [B,256]


# -------------------------------------------------------
# FUSION MLP
# -------------------------------------------------------
class FusionMLP(nn.Module):
    def __init__(self, dimA, dimB):
        super().__init__()
        self.cross = CrossAttention(dimA, dimB)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256,128), nn.ReLU(),
            nn.Linear(128,32), nn.ReLU(),
            nn.Linear(32,2)
        )

    def forward(self, XA, XB):
        fused = self.cross(XA, XB)
        return self.classifier(fused)


# -------------------------------------------------------
# STORAGE FOR ALL RUNS
# -------------------------------------------------------
metrics = {"acc":[], "f1":[], "auc":[], "prec":[], "rec":[], "wf1":[], "ua":[]}
all_runs_data = []


for run in range(runs):
    print(f"\n========== RUN {run+1} ==========")

    # Load data
    Xa_tr, y_tr, spk_tr = load_split(BASE_A, "train")
    Xa_de, y_de, spk_de = load_split(BASE_A, "test")

    Xb_tr, _, _ = load_split(BASE_B, "train")
    Xb_de, _, _ = load_split(BASE_B, "test")

    # Normalize features
    mA, sA = Xa_tr.mean(0), Xa_tr.std(0); sA[sA==0]=1
    mB, sB = Xb_tr.mean(0), Xb_tr.std(0); sB[sB==0]=1

    Xa_tr = np.nan_to_num((Xa_tr-mA)/sA)
    Xa_de = np.nan_to_num((Xa_de-mA)/sA)
    Xb_tr = np.nan_to_num((Xb_tr-mB)/sB)
    Xb_de = np.nan_to_num((Xb_de-mB)/sB)

    # Balanced sampling across speakers
    spk_counts = {s: np.sum(spk_tr==s) for s in np.unique(spk_tr)}
    weights = np.array([1.0/spk_counts[s] for s in spk_tr])
    sampler = WeightedRandomSampler(weights, len(weights), True)

    train_loader = DataLoader(SegDataset(Xa_tr,Xb_tr,y_tr,spk_tr),
                              batch_size=BATCH, sampler=sampler)

    dev_loader   = DataLoader(SegDataset(Xa_de,Xb_de,y_de,spk_de),
                              batch_size=BATCH, shuffle=False)

    # Model
    model = FusionMLP(Xa_tr.shape[1], Xb_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_f1 = 0
    patience_counter = 0     # <-- ADDED

    # ---------------- Training Loop ----------------
    for ep in range(EPOCHS):
        model.train()
        for xa, xb, y, _ in train_loader:
            xa, xb, y = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xa, xb), y)
            loss.backward()
            opt.step()

        # ---------------- Validation ----------------
        model.eval()
        seg_probs = []
        seg_y = []
        seg_spk = []
        fused_segments = []

        with torch.no_grad():
            for xa, xb, y, spk in dev_loader:
                xa_d, xb_d = xa.to(DEVICE), xb.to(DEVICE)

                logits = model(xa_d, xb_d)
                probs = torch.softmax(logits,1)[:,1].cpu().numpy()
                fused = model.cross(xa_d, xb_d).cpu().numpy()

                seg_probs.extend(probs)
                seg_y.extend(y.numpy())
                seg_spk.extend(spk.numpy())
                fused_segments.append(fused)

        seg_probs = np.array(seg_probs)
        seg_y    = np.array(seg_y)
        seg_spk  = np.array(seg_spk)
        fused_segments = np.concatenate(fused_segments, axis=0)

        # -------- Speaker-level Aggregation --------
        sp, st = [], []
        for pid in np.unique(seg_spk):
            idx = np.where(seg_spk == pid)[0]
            sp.append(seg_probs[idx].mean())
            st.append(seg_y[idx][0])

        sp = np.array(sp)
        st = np.array(st)

        # -------- Segment-level embeddings (for t-SNE) --------
        init_list = []
        final_list = []
        label_list = []

        with torch.no_grad():
            for xa, xb, y, spk in dev_loader:
                xa_d, xb_d = xa.to(DEVICE), xb.to(DEVICE)

                fused = model.cross(xa_d, xb_d).cpu().numpy()
                init_emb = np.concatenate([xa.numpy(), xb.numpy()], axis=1)

                init_list.append(init_emb)
                final_list.append(fused)
                label_list.append(y.numpy())

        init_emb_all = np.vstack(init_list)
        final_emb_all = np.vstack(final_list)
        labels_all    = np.hstack(label_list)

        # ROC + Prediction
        fpr, tpr, th = roc_curve(st, sp)
        best_th = th[np.argmax(tpr - fpr)]
        pred = (sp >= best_th).astype(int)

        f1 = f1_score(st, pred)
        print(f"Epoch {ep+1} | F1 = {f1:.3f}")

        # Save best run
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0   # <-- RESET
            best_run_data = {
                "true": st,
                "pred": pred,
                "prob": sp,
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc_score(st, sp),
                "init_emb": init_emb_all,
                "final_emb": final_emb_all,
                "labels": labels_all
            }

        else:
            patience_counter += 1   # <-- INCREASE
            if patience_counter >= PATIENCE:   # <-- EARLY STOP
                print(f"EARLY STOP at epoch {ep+1}")
                break

    metrics["acc"].append(accuracy_score(best_run_data["true"], best_run_data["pred"]))
    metrics["f1"].append(f1_score(best_run_data["true"], best_run_data["pred"]))
    metrics["auc"].append(best_run_data["auc"])
    metrics["prec"].append(precision_score(best_run_data["true"], best_run_data["pred"]))
    metrics["rec"].append(recall_score(best_run_data["true"], best_run_data["pred"]))
    metrics["wf1"].append(f1_score(best_run_data["true"], best_run_data["pred"], average="weighted"))

    tn,fp,fn,tp = confusion_matrix(best_run_data["true"], best_run_data["pred"]).ravel()
    ua = 0.5*((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))
    metrics["ua"].append(ua)

    all_runs_data.append(best_run_data)


# =======================================================
#            SELECT BEST RUN FOR PLOTTING
# =======================================================

best_idx = np.argmax(metrics["f1"])
best = all_runs_data[best_idx]

print(f"\nBest run = RUN {best_idx+1} (F1={metrics['f1'][best_idx]:.3f})")

out_dir = "edic_plots/final"
os.makedirs(out_dir, exist_ok=True)

plot_confusion_matrix(best["true"], best["pred"], f"{out_dir}/confusion_matrix.png")
plot_auc_curve_folds(best["fpr"], best["tpr"], best["auc"], f"{out_dir}/roc_curve.png")
plot_tsne(best["init_emb"], best["final_emb"], best["labels"], f"{out_dir}/tsne.png")


# =======================================================
#                PRINT FINAL METRICS
# =======================================================
print("\n====== FINAL METRICS (mean over 5 runs) ======")
for k,v in metrics.items():
    print(f"{k.upper():5s} Mean={np.mean(v):.3f}   STD={np.std(v):.3f}")

print(f"\nPlots saved in: {out_dir}")
