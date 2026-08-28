import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# =====================================
# Config
# =====================================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-4
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_DIR = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"

TRAIN_DIR = os.path.join(MODEL_DIR, "train")
DEV_DIR   = os.path.join(MODEL_DIR, "dev")

class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# =====================================
# MLP Classifier (Flexible Architecture)
# =====================================
class HandcfratedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], num_classes=2, dropout=0.1):
        super().__init__()
        layers = [nn.LayerNorm(input_dim)]
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# =====================================
# Class-Balanced Focal Loss
# =====================================
class BalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.softmax(logits, dim=1)
        pt = pt[torch.arange(len(targets)), targets]
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce_loss).mean()

# =====================================
# Load Data
# =====================================
print("Loading speaker-level features...")
X_train = np.load(os.path.join(TRAIN_DIR, "speaker_X.npy"))
y_train = np.load(os.path.join(TRAIN_DIR, "speaker_y.npy")).astype(int)
X_dev   = np.load(os.path.join(DEV_DIR, "speaker_X.npy"))
y_dev   = np.load(os.path.join(DEV_DIR, "speaker_y.npy")).astype(int)

y_train[y_train > 1] = 1
y_dev[y_dev > 1] = 1

print(f"Train: {X_train.shape}, Dev: {X_dev.shape}")
print("Label distribution (train):", np.bincount(y_train))

# Normalize
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_dev   = scaler.transform(X_dev)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0) + 1e-8  # to avoid division by zero
X_train = (X_train - mean) / std
X_dev   = (X_dev - mean) / std

# =====================================
# Weighted Sampling to Fix Imbalance
# =====================================
class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = [class_weights[y] for y in y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(FeatureDataset(X_train, y_train), batch_size=BATCH_SIZE, sampler=sampler)
dev_loader   = DataLoader(FeatureDataset(X_dev, y_dev), batch_size=BATCH_SIZE)

# =====================================
# Model, Optimizer, Loss, Scheduler
# =====================================
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

model = HandcfratedClassifier(input_dim=X_train.shape[1]).to(DEVICE)
criterion = BalancedFocalLoss(alpha=weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)


N_RUNS = 5  # number of runs for averaging
results = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'auc': [], 'wa': [], 'ua': []}

for run in range(N_RUNS):
    print(f"\n===== Run {run+1}/{N_RUNS} =====")
    seed = SEED + run
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Reinitialize model, optimizer, etc.
    model = HandcfratedClassifier(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = BalancedFocalLoss(alpha=weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_f1, best_state, patience_counter = 0, None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_true.extend(yb.numpy())
                y_prob.extend(probs)

        y_true, y_prob = np.array(y_true), np.array(y_prob)
        preds = (y_prob >= 0.5).astype(int)
        f1 = f1_score(y_true, preds)
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Final Evaluation
    model.load_state_dict(best_state)
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            y_true.extend(yb.numpy())
            y_prob.extend(probs)

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_th = thresholds[np.argmax(tpr - fpr)]
    y_pred = (y_prob >= best_th).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    wa = acc
    ua = 0.5 * ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6)))

    # Store metrics
    results['acc'].append(acc)
    results['prec'].append(prec)
    results['rec'].append(rec)
    results['f1'].append(f1)
    results['auc'].append(auc)
    results['wa'].append(wa)
    results['ua'].append(ua)

# =====================================
# Print Final Results with Mean ± Std
# =====================================
print("\n=== Final Averaged Dev Set Results ===")
for metric, vals in results.items():
    mean = np.mean(vals)
    std = np.std(vals)
    print(f"{metric.upper():<8}: {mean:.3f} ± {std:.3f}")
