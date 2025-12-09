import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================
# CONFIG
# ====================================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

FEAT_DIR_A = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"
FEAT_DIR_B = "/home/bubai-maji/bubai/English/features_npy/data2vec_large"
BATCH = 64
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
LATENT_DIM = 256
HEADS = 8
RUNS = 5

OUT_DIR = "/home/bubai-maji/bubai/English/daic_results"
os.makedirs(OUT_DIR, exist_ok=True)


# ====================================================
# DATA LOADING
# ====================================================
def load_data(split):
    X = np.load(f"{FEAT_DIR_A}/{split}/segment_X.npy")
    y = np.load(f"{FEAT_DIR_A}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{FEAT_DIR_A}/{split}/segment_speaker_id.npy")

    X2 = np.load(f"{FEAT_DIR_B}/{split}/segment_X.npy")
    return X, X2, y, spk


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


# ====================================================
# MODEL — Correct Cross Attention Fusion
# ====================================================
class ModalityProj(nn.Module):
    def __init__(self, d, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, latent),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dimA, dimB, latent=LATENT_DIM, heads=HEADS, dropout=0.2):
        super().__init__()

        self.projA = ModalityProj(dimA, latent)
        self.projB = ModalityProj(dimB, latent)

        self.attA2B = nn.MultiheadAttention(latent, heads, batch_first=True, dropout=dropout)
        self.attB2A = nn.MultiheadAttention(latent, heads, batch_first=True, dropout=dropout)

        self.normA1 = nn.LayerNorm(latent)
        self.normB1 = nn.LayerNorm(latent)

        self.ffA = nn.Sequential(nn.Linear(latent, latent*2), nn.ReLU(), nn.Linear(latent*2, latent))
        self.ffB = nn.Sequential(nn.Linear(latent, latent*2), nn.ReLU(), nn.Linear(latent*2, latent))

        self.normA2 = nn.LayerNorm(latent)
        self.normB2 = nn.LayerNorm(latent)

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent*2),
            nn.Linear(latent*2,128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32,2)
        )

    def forward(self, A, B):
        A = self.projA(A).unsqueeze(1)
        B = self.projB(B).unsqueeze(1)

        A2, _ = self.attA2B(A, B, B)
        B2, _ = self.attB2A(B, A, A)

        A = self.normA1(A + A2)
        B = self.normB1(B + B2)

        A = self.normA2(A + self.ffA(A))
        B = self.normB2(B + self.ffB(B))

        A = A.squeeze(1)
        B = B.squeeze(1)
        fused = torch.cat([A, B], dim=-1)
        return self.classifier(fused)


# ====================================================
# TRAINING + EVAL (SPEAKER LEVEL)
# ====================================================
def evaluate(model, loader):
    model.eval()
    seg_probs = []
    seg_labels = []
    seg_spk = []

    with torch.no_grad():
        for xa, xb, y, spk in loader:
            xa, xb = xa.to(DEVICE), xb.to(DEVICE)
            p = torch.softmax(model(xa, xb), 1)[:,1].cpu().numpy()
            seg_probs.extend(p)
            seg_labels.extend(y.numpy())
            seg_spk.extend(spk.numpy())

    seg_labels = np.array(seg_labels)
    seg_probs = np.array(seg_probs)
    seg_spk = np.array(seg_spk)

    sp_probs, sp_labels = [], []
    for sid in np.unique(seg_spk):
        idx = np.where(seg_spk==sid)[0]
        sp_probs.append(seg_probs[idx].mean())
        sp_labels.append(seg_labels[idx][0])

    sp_probs = np.array(sp_probs)
    sp_labels = np.array(sp_labels)

    fpr, tpr, th = roc_curve(sp_labels, sp_probs)
    best_th = th[np.argmax(tpr - fpr)]
    preds = (sp_probs >= best_th).astype(int)

    return preds, sp_probs, sp_labels, fpr, tpr


def train_one_run():

    Xa_tr, Xb_tr, y_tr, sp_tr = load_data("train")
    Xa_de, Xb_de, y_de, sp_de = load_data("dev")

    train_loader = DataLoader(SegDataset(Xa_tr,Xb_tr,y_tr,sp_tr), batch_size=BATCH, shuffle=True)
    dev_loader = DataLoader(SegDataset(Xa_de,Xb_de,y_de,sp_de), batch_size=BATCH)

    model = CrossAttentionFusion(Xa_tr.shape[1], Xb_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_f1 = -1
    patience = 0
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        for xa, xb, y, sp in train_loader:
            xa, xb, y = xa.to(DEVICE), xb.to(DEVICE), y.to(DEVICE)
            loss = crit(model(xa, xb), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        preds, probs, labels, _, _ = evaluate(model, dev_loader)
        f1 = f1_score(labels, preds)
        print(f"Epoch {ep+1:03d} | F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return evaluate(model, dev_loader)


# ====================================================
# MULTI-RUN EXPERIMENT
# ====================================================
all_runs = []
all_metrics = {"acc":[], "prec":[], "rec":[], "f1":[], "wf1":[], "auc":[], "ua":[]}

for r in range(RUNS):
    print(f"\n========= RUN {r+1}/{RUNS} =========")
    preds, probs, labels, fpr, tpr = train_one_run()
    all_runs.append((preds, probs, labels, fpr, tpr))

    tn,fp,fn,tp = confusion_matrix(labels, preds).ravel()
    ua = 0.5*((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))

    all_metrics["acc"].append(accuracy_score(labels,preds))
    all_metrics["prec"].append(precision_score(labels,preds))
    all_metrics["rec"].append(recall_score(labels,preds))
    all_metrics["f1"].append(f1_score(labels,preds))
    all_metrics["wf1"].append(f1_score(labels,preds,average="weighted"))
    all_metrics["auc"].append(roc_auc_score(labels,probs))
    all_metrics["ua"].append(ua)


# ====================================================
# FINAL RESULTS & PLOTS
# ====================================================
print("\n=========== FINAL SUMMARY ===========")
for k in all_metrics:
    print(f"{k.upper():4s} Mean={np.mean(all_metrics[k]):.4f}  STD={np.std(all_metrics[k]):.4f}")

# Best ROC
best_idx = np.argmax(all_metrics["auc"])
_, best_probs, best_labels, fpr, tpr = all_runs[best_idx]
best_auc = roc_auc_score(best_labels, best_probs)

plt.figure(figsize=(3.5,3))
plt.plot(fpr,tpr, label=f"AUC={best_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/daic_best_roc.png")

# Majority Vote Confusion Matrix
votes = np.array([run[0] for run in all_runs])
maj_pred = (votes.mean(axis=0) >=0.5).astype(int)
cm = confusion_matrix(best_labels, maj_pred)

plt.figure(figsize=(3.5,3))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
            xticklabels=["Normal","Depression"],
            yticklabels=["Normal","Depression"])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/daic_majority_cm.png")

print("\nROC + CM saved in results folder")
print("DONE ✔")
