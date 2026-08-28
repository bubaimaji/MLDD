import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ========== CONFIG ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 64
LR = 2e-4
EPOCHS = 100
PATIENCE = 10
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"

# ========== DATA LOADING FUNCTION ==========
def load_split(split):
    X = np.load(f"{BASE}/{split}/segment_X.npy")
    y = np.load(f"{BASE}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{BASE}/{split}/segment_speaker_id.npy")
    return X, y, spk

# ========== RESULTS STORAGE ==========
runs = 5
all_metrics = {"acc":[], "prec":[], "rec":[], "f1":[], "auc":[], "ua":[]}

for run in range(runs):
    print(f"\n============= RUN {run+1}/{runs} =============")

    # ========== LOAD DATA ==========
    X_train, y_train, spk_train = load_split("train")
    X_dev,   y_dev,   spk_dev   = load_split("dev")

    print(f"Train segments: {X_train.shape}, Dev segments: {X_dev.shape}")

    # ========== NORMALIZATION ==========
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_dev   = (X_dev   - mean) / std

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=5.0, neginf=-5.0)
    X_dev   = np.nan_to_num(X_dev,   nan=0.0, posinf=5.0, neginf=-5.0)

    # ========== DATASET ==========
    class SegDataset(Dataset):
        def __init__(self, X, y, spk):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            self.spk = torch.tensor(spk, dtype=torch.long)
        def __len__(self): return len(self.y)
        def __getitem__(self, idx): return self.X[idx], self.y[idx], self.spk[idx]

    # Speaker balanced sampler
    spk_counts = {spk: np.sum(spk_train == spk) for spk in np.unique(spk_train)}
    weights = np.array([1.0 / spk_counts[s] for s in spk_train])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(SegDataset(X_train,y_train,spk_train), batch_size=BATCH, sampler=sampler)
    dev_loader   = DataLoader(SegDataset(X_dev,y_dev,spk_dev), batch_size=BATCH, shuffle=False)

    # ========== MLP ==========
    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim,128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128,32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32,2)
            )
        def forward(self, x): return self.net(x)

    model = MLP(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ========== TRAIN ==========
    best_f1, patience = 0, 0
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # ==== DEV ====
        model.eval()
        seg_probs, seg_y, seg_spk = [], [], []
        with torch.no_grad():
            for xb, yb, spk in dev_loader:
                xb = xb.to(DEVICE)
                prob = torch.softmax(model(xb),1)[:,1].cpu().numpy()
                seg_probs.extend(prob)
                seg_y.extend(yb.numpy())
                seg_spk.extend(spk.numpy())

        seg_probs = np.array(seg_probs)
        seg_y = np.array(seg_y)
        seg_spk = np.array(seg_spk)

        # Speaker aggregation
        spk_probs, spk_true = [], []
        for pid in np.unique(seg_spk):
            idx = np.where(seg_spk == pid)[0]
            spk_probs.append(seg_probs[idx].mean())
            spk_true.append(seg_y[idx][0])

        spk_probs = np.array(spk_probs)
        spk_true = np.array(spk_true)

        # threshold
        fpr, tpr, th = roc_curve(spk_true, spk_probs)
        best_th = th[np.argmax(tpr - fpr)]

        spk_pred = (spk_probs >= best_th).astype(int)
        f1 = f1_score(spk_true, spk_pred)

        print(f"Epoch {epoch+1:03d} | Loss={np.mean(losses):.4f} | F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # ========== FINAL EVAL ==========
    model.load_state_dict(best_state)
    model.eval()

    seg_probs, seg_y, seg_spk = [], [], []
    with torch.no_grad():
        for xb, yb, spk in dev_loader:
            xb = xb.to(DEVICE)
            seg_probs.extend(torch.softmax(model(xb),1)[:,1].cpu().numpy())
            seg_y.extend(yb.numpy())
            seg_spk.extend(spk.numpy())

    seg_probs = np.array(seg_probs)
    seg_y = np.array(seg_y)
    seg_spk = np.array(seg_spk)

    spk_probs, spk_true = [], []
    for pid in np.unique(seg_spk):
        idx = np.where(seg_spk == pid)[0]
        spk_probs.append(seg_probs[idx].mean())
        spk_true.append(seg_y[idx][0])

    spk_probs = np.array(spk_probs)
    spk_true = np.array(spk_true)

    fpr, tpr, th = roc_curve(spk_true, spk_probs)
    best_th = th[np.argmax(tpr - fpr)]
    spk_pred = (spk_probs >= best_th).astype(int)

    acc = accuracy_score(spk_true, spk_pred)
    prec = precision_score(spk_true, spk_pred, zero_division=0)
    rec = recall_score(spk_true, spk_pred)
    f1 = f1_score(spk_true, spk_pred)
    auc = roc_auc_score(spk_true, spk_probs)

    tn, fp, fn, tp = confusion_matrix(spk_true, spk_pred).ravel()
    ua = 0.5 * ((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))

    print(f"SPEAKER PERFORMANCE (Run {run+1}) "
          f"Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}, UA={ua:.3f}")

    all_metrics["acc"].append(acc)
    all_metrics["prec"].append(prec)
    all_metrics["rec"].append(rec)
    all_metrics["f1"].append(f1)
    all_metrics["auc"].append(auc)
    all_metrics["ua"].append(ua)

# ========== FINAL STD RESULTS ==========
print("\n============== FINAL RESULTS OVER 5 RUNS ==============")
for k,v in all_metrics.items():
    print(f"{k.upper()} Mean={np.mean(v):.3f}  STD={np.std(v):.3f}")
