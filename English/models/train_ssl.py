import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ========================= CONFIG =========================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
N_RUNS = 5
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = "/home/bubai-maji/bubai/English/features_npy/facebook-data2vec-audio-large-960h1"
TRAIN_DIR = os.path.join(BASE, "train")
DEV_DIR   = os.path.join(BASE, "dev")

# ========================= DATASET =========================
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ========================= MLP MODEL ======================
class SSLMLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim,256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,2)
        )
    def forward(self, x): return self.net(x)

# ========================= LOAD DATA ======================
X_train = np.load(f"{TRAIN_DIR}/speaker_X.npy")
y_train = np.load(f"{TRAIN_DIR}/speaker_y.npy").astype(int)
X_dev   = np.load(f"{DEV_DIR}/speaker_X.npy")
y_dev   = np.load(f"{DEV_DIR}/speaker_y.npy").astype(int)

# Binary
y_train[y_train>1] = 1
y_dev[y_dev>1] = 1

print("Train:", X_train.shape, "| Dev:", X_dev.shape)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)

# Weighted sampler
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[y] for y in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(FeatureDataset(X_train,y_train), batch_size=BATCH_SIZE, sampler=sampler)
dev_loader   = DataLoader(FeatureDataset(X_dev,y_dev), batch_size=BATCH_SIZE)

# ========================= RESULTS STORAGE ======================
results = {"acc":[], "prec":[], "rec":[], "f1":[], "wf1":[], "auc":[], "ua":[]}

# ========================= TRAINING LOOP ========================
for run in range(N_RUNS):
    print(f"\n========== RUN {run+1}/{N_RUNS} ==========")

    # Init model, optimizer, loss
    model = SSLMLPClassifier(X_train.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1
    patience = 0

    # -------- Train --------
    for epoch in range(1, EPOCHS+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        # -------- Dev eval (during training) --------
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.softmax(logits,1)[:,1].cpu().numpy()
                y_true.extend(yb.numpy())
                y_prob.extend(probs)

        y_true = np.array(y_true); y_prob = np.array(y_prob)
        f1 = f1_score(y_true, (y_prob >= 0.5).astype(int))

        print(f"Epoch {epoch:03d} | F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE: break

    model.load_state_dict(best_state)
    model.eval()

    # ========= Final Evaluation =========
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            y_prob.extend(torch.softmax(logits,1)[:,1].cpu().numpy())
            y_true.extend(yb.numpy())

    y_true = np.array(y_true); y_prob = np.array(y_prob)

    # Best threshold (TPR-FPR)
    fpr, tpr, th = roc_curve(y_true, y_prob)
    best_th = th[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= best_th).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    ua = 0.5 * ((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))

    print(f"RESULT (Run {run+1}): Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, "
          f"F1={f1:.3f}, WF1={wf1:.3f}, AUC={auc:.3f}, UA={ua:.3f}")

    results["acc"].append(acc)
    results["prec"].append(prec)
    results["rec"].append(rec)
    results["f1"].append(f1)
    results["wf1"].append(wf1)
    results["auc"].append(auc)
    results["ua"].append(ua)

# ========================= FINAL SUMMARY =========================
print("\n========== FINAL RESULTS OVER 5 RUNS ==========")
for k,v in results.items():
    print(f"{k.upper()} Mean={np.mean(v):.3f}  STD={np.std(v):.3f}")
