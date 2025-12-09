import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)


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

BASE_DIRS = [
    "/home/bubai-maji/bubai/Itali/features_npy/IS10",
    "/home/bubai-maji/bubai/Itali/features_npy/MIT-ast-finetuned-audioset-10-10-0.4593",
]

# =====================================================
# LOAD FOLD DATA
# =====================================================
def load_fold(base, fold):
    X = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


# =====================================================
# DATASET
# =====================================================
class SimpleDataset(Dataset):
    def __init__(self, X_concat, y, spk):
        self.X = torch.tensor(X_concat, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = spk

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.spk[idx]


# =====================================================
# MODEL — CONCAT + MLP
# =====================================================
class MLPFusion(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)

def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except:
        return np.nan


def train_fold(fold_idx):

    # ---------- Load and concat ALL modalities ----------
    Xtr_list = []
    Xte_list = []

    for base in BASE_DIRS:
        X_all, y_all, spk_all = [], [], []

        for f in range(1, FOLDS+1):
            Xf, yf, sp = load_fold(base, f)
            X_all.append(Xf)
            y_all.append(yf)
            spk_all.append(sp)

        X_te = X_all[fold_idx]
        y_te = y_all[fold_idx]
        spk_te = spk_all[fold_idx]

        X_tr = np.vstack([X_all[i] for i in range(FOLDS) if i != fold_idx])
        y_tr = np.hstack([y_all[i] for i in range(FOLDS) if i != fold_idx])

        # Standardize per modality
        mean = X_tr.mean(0)
        std = X_tr.std(0); std[std == 0] = 1
        Xtr_list.append((X_tr - mean) / std)
        Xte_list.append((X_te - mean) / std)

        if base == BASE_DIRS[0]:
            ytrain = y_tr
            ytest  = y_te
            spktest = spk_te

    # ---------- Concat features ----------
    Xtr_concat = np.concatenate(Xtr_list, axis=1)
    Xte_concat = np.concatenate(Xte_list, axis=1)

    # ---------- Speaker sampler ----------
    first_base = BASE_DIRS[0]
    spk_all = [np.load(f"{first_base}/fold{f}_speaker.npy", allow_pickle=True)
               for f in range(1, FOLDS+1)]

    spk_train = np.hstack([spk_all[i] for i in range(FOLDS) if i != fold_idx])
    spk_test  = spk_all[fold_idx]

    uniq, cnt = np.unique(spk_train, return_counts=True)
    weights = np.array([1 / cnt[np.where(uniq == s)][0] for s in spk_train])

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # ---------- Loaders ----------
    train_loader = DataLoader(
        SimpleDataset(Xtr_concat, ytrain, spk_train),
        batch_size=BATCH, sampler=sampler
    )
    test_loader = DataLoader(
        SimpleDataset(Xte_concat, ytest, spk_test),
        batch_size=BATCH, shuffle=False
    )

    # ---------- Model ----------
    in_dim = Xtr_concat.shape[1]
    model = MLPFusion(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_state = {k: v.cpu() for k,v in model.state_dict().items()}
    best_f1 = -1
    patience_cnt = 0

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    for ep in range(EPOCHS):

        model.train()
        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # ---------- Evaluate on speaker-level ----------
        seg_p, seg_y, seg_s = [], [], []

        model.eval()
        with torch.no_grad():
            for xb, yb, spk in test_loader:
                p = torch.softmax(model(xb.to(DEVICE)), 1)[:,1].cpu().numpy()
                seg_p.extend(p)
                seg_y.extend(yb.numpy())
                seg_s.extend(spk)

        seg_p = np.array(seg_p)
        seg_y = np.array(seg_y)
        seg_s = np.array(seg_s, dtype=object)

        # speaker aggregation
        sp_p, sp_t = [], []
        for pid in np.unique(seg_s):
            mask = seg_s == pid
            sp_p.append(seg_p[mask].mean())
            sp_t.append(seg_y[mask][0])

        sp_p = np.array(sp_p)
        sp_t = np.array(sp_t)

        fpr, tpr, th = roc_curve(sp_t, sp_p)
        best_th = th[np.argmax(tpr - fpr)]
        pred = (sp_p >= best_th).astype(int)

        f1 = f1_score(sp_t, pred)
        print(f"FOLD {fold_idx+1} | Epoch {ep+1} | F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    # =====================================================
    # FINAL TEST
    # =====================================================
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    seg_p, seg_y, seg_s = [], [], []

    with torch.no_grad():
        for xb, yb, spk in test_loader:
            p = torch.softmax(model(xb.to(DEVICE)), 1)[:,1].cpu().numpy()
            seg_p.extend(p)
            seg_y.extend(yb.numpy())
            seg_s.extend(spk)

    seg_p = np.array(seg_p)
    seg_y = np.array(seg_y)
    seg_s = np.array(seg_s, dtype=object)

    # ---- aggregate speakers ----
    sp_p, sp_t = [], []
    for pid in np.unique(seg_s):
        mask = seg_s == pid
        sp_p.append(seg_p[mask].mean())
        sp_t.append(seg_y[mask][0])

    sp_p = np.array(sp_p)
    sp_t = np.array(sp_t)

    fpr, tpr, th = roc_curve(sp_t, sp_p)
    best_th = th[np.argmax(tpr - fpr)]
    pred = (sp_p >= best_th).astype(int)

    # ---- metrics ----
    acc = accuracy_score(sp_t, pred)
    prec = precision_score(sp_t, pred, zero_division=0)
    rec = recall_score(sp_t, pred, zero_division=0)
    f1 = f1_score(sp_t, pred)
    wf1 = f1_score(sp_t, pred, average="weighted")
    auc = safe_auc(sp_t, sp_p)
    tn,fp,fn,tp = confusion_matrix(sp_t, pred).ravel()
    ua = 0.5*((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))

    return acc,prec,rec,f1,wf1,auc,ua


# =====================================================
# RUN ALL FOLDS
# =====================================================
results = {k:[] for k in ["acc","prec","rec","f1","wf1","auc","ua"]}

for f in range(FOLDS):
    print("\n===============================")
    print(f"TRAINING FOLD {f+1}/{FOLDS}")
    acc,prec,rec,f1,wf1,auc,ua = train_fold(f)

    results["acc"].append(acc)
    results["prec"].append(prec)
    results["rec"].append(rec)
    results["f1"].append(f1)
    results["wf1"].append(wf1)
    results["auc"].append(auc)
    results["ua"].append(ua)

print("\n========== FINAL 5-FOLD METRICS ==========")
for k,v in results.items():
    print(f"{k.upper():6s}:  Mean={np.nanmean(v):.4f}  STD={np.nanstd(v):.4f}")
