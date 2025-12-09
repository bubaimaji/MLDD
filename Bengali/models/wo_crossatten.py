import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ==========================================================
# CONFIG
# ==========================================================
SEED = 42
FOLDS = 5
BATCH = 64
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
NUM_CLASSES = 2

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

IS10_DIR = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/IS10_5fold"
PRETRAIN_DIR = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/Whisper_large_5fold"

# ==========================================================
# UTILS
# ==========================================================
def safe_auc(y, p):
    try: return roc_auc_score(y, p)
    except: return np.nan


# ==========================================================
# DATA LOADING
# ==========================================================
def load_fold(base, fold):
    X = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


class MultiSegDataset(Dataset):
    def __init__(self, X, y, spk=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = spk

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        if self.spk is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.spk[idx]


# ==========================================================
# MODEL
# ==========================================================
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim,256),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,32),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(32,NUM_CLASSES)
        )
    def forward(self, x):
        return self.net(x)


# ==========================================================
# TRAIN CONCAT MLP
# ==========================================================
def train_concat_mlp(X_tr_list, y_tr, X_te_list, y_te, spk_tr, spk_te):

    X_tr = np.concatenate(X_tr_list, axis=1)
    X_te = np.concatenate(X_te_list, axis=1)

    model = MLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    uniq, cnt = np.unique(spk_tr, return_counts=True)
    weights = np.array([1.0 / cnt[np.where(uniq==s)[0][0]] for s in spk_tr])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    tr_loader = DataLoader(MultiSegDataset(X_tr, y_tr, spk_tr), batch_size=BATCH, sampler=sampler)
    te_loader = DataLoader(MultiSegDataset(X_te, y_te, spk_te), batch_size=BATCH, shuffle=False)

    best_state = None
    best_f1 = -1
    wait = 0

    print("Training concat-MLP...")

    for ep in range(EPOCHS):
        model.train()
        for xb, yb, _ in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # Speaker-level validation
        model.eval()
        seg_p, seg_y, seg_s = [], [], []
        with torch.no_grad():
            for xb, yb, spk in te_loader:
                p = torch.softmax(model(xb.to(DEVICE)),1)[:,1].cpu().numpy()
                seg_p.extend(p); seg_y.extend(yb.numpy()); seg_s.extend(spk)

        seg_p = np.array(seg_p); seg_y = np.array(seg_y); seg_s = np.array(seg_s, dtype=object)

        sp_p = []; sp_t = []
        for pid in np.unique(seg_s):
            m = (seg_s == pid)
            sp_p.append(seg_p[m].mean())
            sp_t.append(seg_y[m][0])
        sp_p = np.array(sp_p); sp_t = np.array(sp_t)

        fpr,tpr,th = roc_curve(sp_t,sp_p)
        best_th = th[np.argmax(tpr - fpr)]
        pred = (sp_p >= best_th).astype(int)
        f1 = f1_score(sp_t,pred)

        print(f"Epoch {ep+1:03d}/{EPOCHS} | Dev F1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("EARLY STOPPING")
                break

    model.load_state_dict(best_state)
    model.eval()

    seg_p, seg_y, seg_s = [], [], []
    with torch.no_grad():
        for xb, yb, spk in te_loader:
            p = torch.softmax(model(xb.to(DEVICE)),1)[:,1].cpu().numpy()
            seg_p.extend(p); seg_y.extend(yb.numpy()); seg_s.extend(spk)

    seg_p = np.array(seg_p); seg_y = np.array(seg_y); seg_s = np.array(seg_s, dtype=object)

    sp_p = []; sp_t = []
    for pid in np.unique(seg_s):
        mask=(seg_s==pid)
        sp_p.append(seg_p[mask].mean())
        sp_t.append(seg_y[mask][0])

    sp_p=np.array(sp_p); sp_t=np.array(sp_t)
    fpr,tpr,th=roc_curve(sp_t,sp_p)
    auc=safe_auc(sp_t,sp_p)

    return sp_t, sp_p, auc


# ==========================================================
# MAIN FOLD LOOP
# ==========================================================
fold_metrics = {
    "acc": [], "prec": [], "rec": [], "f1": [], "wf1": [], "auc": [], "ua": []
}

fusion_all_true = []
fusion_all_prob = []

for f in range(1, FOLDS+1):
    print(f"\n========== FOLD {f}/{FOLDS} ==========")

    X_tr_list=[]; X_te_list=[]
    spk_tr=None; spk_te=None

    for base in [IS10_DIR, PRETRAIN_DIR]:
        X_list=[]; y_list=[]; spk_list=[]
        for k in range(1,FOLDS+1):
            X,y,spk=load_fold(base,k)
            X_list.append(X); y_list.append(y); spk_list.append(spk)

        X_te=X_list[f-1]
        y_te=y_list[f-1]
        spk_te=spk_list[f-1]

        X_tr=np.vstack([X_list[i] for i in range(FOLDS) if i!=(f-1)])
        y_tr=np.hstack([y_list[i] for i in range(FOLDS) if i!=(f-1)])
        if spk_tr is None:
            spk_tr=np.hstack([spk_list[i] for i in range(FOLDS) if i!=(f-1)])

        m=X_tr.mean(0)
        s=X_tr.std(0); s[s==0]=1
        X_tr_list.append((X_tr-m)/s)
        X_te_list.append((X_te-m)/s)

    sp_true, sp_prob, auc = train_concat_mlp(X_tr_list, y_tr, X_te_list, y_te, spk_tr, spk_te)

    # Fold-level metrics
    fpr,tpr,th=roc_curve(sp_true,sp_prob)
    best_th = th[np.argmax(tpr - fpr)]
    pred = (sp_prob >= best_th).astype(int)

    fold_metrics["acc"].append(accuracy_score(sp_true,pred))
    fold_metrics["prec"].append(precision_score(sp_true,pred,zero_division=0))
    fold_metrics["rec"].append(recall_score(sp_true,pred,zero_division=0))
    fold_metrics["f1"].append(f1_score(sp_true,pred,zero_division=0))
    fold_metrics["wf1"].append(f1_score(sp_true,pred,average="weighted",zero_division=0))
    fold_metrics["auc"].append(auc)
    tn,fp,fn,tp=confusion_matrix(sp_true,pred).ravel()
    ua=0.5*((tp/(tp+fn+1e-6))+(tn/(tn+fp+1e-6)))
    fold_metrics["ua"].append(ua)

    fusion_all_true.extend(sp_true)
    fusion_all_prob.extend(sp_prob)


# ==========================================================
# FINAL RESULTS
# ==========================================================
fusion_all_true=np.array(fusion_all_true)
fusion_all_prob=np.array(fusion_all_prob)

print("\n============= CROSS-FOLD PERFORMANCE SUMMARY (SPEAKER LEVEL) =============")
for m in fold_metrics.keys():
    vals=np.array(fold_metrics[m],dtype=float)
    print(f"{m.upper():4s}: Mean={np.nanmean(vals):.4f}  STD={np.nanstd(vals):.4f}")

# Final confusion matrix on all speakers
fpr,tpr,th=roc_curve(fusion_all_true,fusion_all_prob)
best_th = th[np.argmax(tpr - fpr)]
pred = (fusion_all_prob >= best_th).astype(int)

print("\n============= CONFUSION MATRIX (ALL SPEAKERS) =============")
print(confusion_matrix(fusion_all_true,pred))

print("\nCompleted Successfully ")
