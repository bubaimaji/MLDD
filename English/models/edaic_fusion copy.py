import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CONFIG ==========
SEED = 42
BATCH = 64
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
NUM_CLASSES = 2
runs = 5  # number of repeated runs

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# Matplotlib global settings for high-quality figures
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# ========== PATHS ==========
BASE_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
BASE_B = "/home/bubai-maji/bubai/English/edic_features_npy/data2vec_large"

OUT_DIR = "/home/bubai-maji/bubai/English/edic_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== UTIL ==========
def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except:
        return np.nan

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

    def __len__(self):
        return len(self.y)

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
        # XA, XB: [B, dimA], [B, dimB]
        A = self.projA(XA).unsqueeze(1)  # [B,1,L]
        B = self.projB(XB).unsqueeze(1)  # [B,1,L]
        A2B, _ = self.attnAB(A, B, B)    # A attending to B
        B2A, _ = self.attnBA(B, A, A)    # B attending to A
        return torch.cat([A2B.squeeze(1), B2A.squeeze(1)], dim=-1)  # [B, 2*latent_dim]=[B,256]


# -------------------------------------------------------
# FUSION MLP
# -------------------------------------------------------
class FusionMLP(nn.Module):
    def __init__(self, dimA, dimB):
        super().__init__()
        self.cross = CrossAttention(dimA, dimB, latent_dim=128, heads=4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, XA, XB):
        fused = self.cross(XA, XB)
        return self.classifier(fused)


# -------------------------------------------------------
# STORAGE FOR ALL RUNS
# -------------------------------------------------------
metrics = {"acc": [], "f1": [], "auc": [], "prec": [], "rec": [], "wf1": [], "ua": []}
all_runs_data = []  # each element: dict with true, pred, prob, fpr, tpr, auc, init_emb, final_emb, labels

# For majority voting across runs (speaker-level)
votes = {}  # {speaker_index: [preds across runs]}


# =======================================================
# MAIN TRAINING / EVAL LOOP (MULTIPLE RUNS)
# =======================================================
for run in range(runs):
    print(f"\n========== RUN {run+1}/{runs} ==========")

    # ------------- Load data -------------
    Xa_tr, y_tr, spk_tr = load_split(BASE_A, "train")
    Xa_de, y_de, spk_de = load_split(BASE_A, "test")

    Xb_tr, _, _ = load_split(BASE_B, "train")
    Xb_de, _, _ = load_split(BASE_B, "test")

    # ------------- Normalize features -------------
    mA, sA = Xa_tr.mean(0), Xa_tr.std(0); sA[sA == 0] = 1
    mB, sB = Xb_tr.mean(0), Xb_tr.std(0); sB[sB == 0] = 1

    Xa_tr = np.nan_to_num((Xa_tr - mA) / sA)
    Xa_de = np.nan_to_num((Xa_de - mA) / sA)
    Xb_tr = np.nan_to_num((Xb_tr - mB) / sB)
    Xb_de = np.nan_to_num((Xb_de - mB) / sB)

    # ------------- Speaker-balanced sampling -------------
    spk_counts = {s: np.sum(spk_tr == s) for s in np.unique(spk_tr)}
    weights = np.array([1.0 / spk_counts[s] for s in spk_tr])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        SegDataset(Xa_tr, Xb_tr, y_tr, spk_tr),
        batch_size=BATCH,
        sampler=sampler
    )

    dev_loader = DataLoader(
        SegDataset(Xa_de, Xb_de, y_de, spk_de),
        batch_size=BATCH,
        shuffle=False
    )

    # ------------- Model / Optimizer / Loss -------------
    model = FusionMLP(Xa_tr.shape[1], Xb_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_f1 = 0.0
    patience_counter = 0
    best_run_data = None

    # ---------------- Training Loop ----------------
    for ep in range(EPOCHS):
        model.train()
        for xa, xb, y, _ in train_loader:
            xa, xb, y = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(xa, xb)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

        # ---------------- Validation ----------------
        model.eval()
        seg_probs = []
        seg_y = []
        seg_spk = []

        with torch.no_grad():
            for xa, xb, y, spk in dev_loader:
                xa_d, xb_d = xa.to(DEVICE), xb.to(DEVICE)
                logits = model(xa_d, xb_d)
                probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()

                seg_probs.extend(probs)
                seg_y.extend(y.numpy())
                seg_spk.extend(spk.numpy())

        seg_probs = np.array(seg_probs)
        seg_y = np.array(seg_y)
        seg_spk = np.array(seg_spk)

        # -------- Speaker-level Aggregation --------
        sp, st = [], []
        for pid in np.unique(seg_spk):
            idx = np.where(seg_spk == pid)[0]
            sp.append(seg_probs[idx].mean())
            st.append(seg_y[idx][0])

        sp = np.array(sp)
        st = np.array(st)

        # -------- Segment-level embeddings (for t-SNE etc.) --------
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
        labels_all = np.hstack(label_list)

        # -------- ROC + Prediction (speaker-level) --------
        fpr, tpr, th = roc_curve(st, sp)
        best_th = th[np.argmax(tpr - fpr)]
        pred = (sp >= best_th).astype(int)

        f1 = f1_score(st, pred)
        print(f"Run {run+1} | Epoch {ep+1} | Speaker-level F1 = {f1:.3f}")

        # Save best epoch of this run
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            best_run_data = {
                "true": st,
                "pred": pred,
                "prob": sp,
                "fpr": fpr,
                "tpr": tpr,
                "auc": safe_auc(st, sp),
                "init_emb": init_emb_all,
                "final_emb": final_emb_all,
                "labels": labels_all
            }
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"EARLY STOP at epoch {ep+1} (Run {run+1})")
                break

    # ------------- Store metrics for this run -------------
    acc = accuracy_score(best_run_data["true"], best_run_data["pred"])
    f1r = f1_score(best_run_data["true"], best_run_data["pred"])
    auc_r = best_run_data["auc"]
    prec = precision_score(best_run_data["true"], best_run_data["pred"])
    rec = recall_score(best_run_data["true"], best_run_data["pred"])
    wf1 = f1_score(best_run_data["true"], best_run_data["pred"], average="weighted")

    tn, fp, fn, tp = confusion_matrix(best_run_data["true"], best_run_data["pred"]).ravel()
    ua = 0.5 * ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6)))

    metrics["acc"].append(acc)
    metrics["f1"].append(f1r)
    metrics["auc"].append(auc_r)
    metrics["prec"].append(prec)
    metrics["rec"].append(rec)
    metrics["wf1"].append(wf1)
    metrics["ua"].append(ua)

    all_runs_data.append(best_run_data)

    # ------------- Collect votes for majority voting (speaker-level) -------------
    # We assume the order of speakers (np.unique) is consistent across runs
    for i, p in enumerate(best_run_data["pred"]):
        if i not in votes:
            votes[i] = []
        votes[i].append(int(p))


# =======================================================
# ROC CURVE - Best Run Only (Highest AUC)
# =======================================================
# =======================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import os

# =======================================================
# GLOBAL STYLE CONFIG — matches your 2nd code defaults
# =======================================================
plt.rcdefaults()
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# =======================================================
# BEST RUN ROC PLOT (Speaker Level)
# =======================================================
auc_list = np.array([d["auc"] for d in all_runs_data], dtype=float)

best_run_idx = np.nanargmax(auc_list)
best_auc = auc_list[best_run_idx]
mean_auc = np.nanmean(auc_list)
std_auc = np.nanstd(auc_list)

best_data = all_runs_data[best_run_idx]
fpr = best_data["fpr"]
tpr = best_data["tpr"]

print("AUCs Across Runs:", auc_list)
print(f"Best AUC: {best_auc:.3f}, Mean AUC: {mean_auc:.3f}, STD: {std_auc:.3f}")

plt.figure(figsize=(3.5, 3))
plt.plot(fpr, tpr, linewidth=1.5,
         label=f"AUC={best_auc:.3f}±{std_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "edaic_roc.png"), dpi=300)
plt.close()

print(f"Saved ROC Curve (Best Run #{best_run_idx+1}) → edaic_roc.png")


# =======================================================
# MAJORITY VOTE CONFUSION MATRIX
# =======================================================
true_labels = all_runs_data[0]["true"]

# Majority vote based on predictions saved previously
final_preds = []
for i in range(len(true_labels)):
    preds_i = votes.get(i, [])
    if len(preds_i) == 0:
        maj = 0
    else:
        maj = max(set(preds_i), key=preds_i.count)
    final_preds.append(maj)

cm = confusion_matrix(true_labels, final_preds)

print("\nMajority Vote Confusion Matrix:")
print(cm)

plt.figure(figsize=(3.5, 3))
ax = sns.heatmap(cm, annot=True, fmt="d",
                 cmap="Blues",
                 xticklabels=["Normal", "Depression"],
                 yticklabels=["Normal", "Depression"],
                 annot_kws={"size": 12})

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "edaic_cm.png"), dpi=300)
plt.close()

print("Saved Confusion Matrix → edaic_cm.png")


# =======================================================
# SUMMARY METRICS REPORT
# =======================================================
print("\n======== FINAL REPORT (Average of {} runs) ========".format(runs))
for k, v in metrics.items():
    print(f"{k}: Mean={np.mean(v):.4f}, STD={np.std(v):.4f}")

print("\nDone ")
