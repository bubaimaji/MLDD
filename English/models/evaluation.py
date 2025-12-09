import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from English.models.ediac_model import MLPBranch, EarlyConcatFusion, AttentionFusion

# ===================================================
# Config
# ===================================================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
FREEZE_EPOCHS = 3
PATIENCE = 10
LR_BASE = 1e-4
LR_FUSION = 3e-5
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

BASE = "/home/bubai-maji/bubai/English/features_npy"
OUTDIR = "/home/bubai-maji/bubai/English/outputs_fusion"
os.makedirs(OUTDIR, exist_ok=True)

FEAT_A = os.path.join(BASE, "ComParE_functionals")
FEAT_B = os.path.join(BASE, "microsoft-wavlm-base-plus1")

# ===================================================
# Dataset
# ===================================================
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class DualFeatureDataset(Dataset):
    def __init__(self, XA, XB, y):
        self.XA = torch.tensor(XA, dtype=torch.float32)
        self.XB = torch.tensor(XB, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.XA[i], self.XB[i], self.y[i]

# ===================================================
# Load Data
# ===================================================
def load_features(base_dir, split):
    X = np.load(os.path.join(base_dir, split, "speaker_X.npy"))
    y = np.load(os.path.join(base_dir, split, "speaker_y.npy")).astype(int)
    y[y > 1] = 1
    return X, y

XA_tr, y_tr = load_features(FEAT_A, "train")
XA_te, y_te = load_features(FEAT_A, "dev")
XB_tr, _ = load_features(FEAT_B, "train")
XB_te, _ = load_features(FEAT_B, "dev")

scA, scB = StandardScaler(), StandardScaler()
XA_tr, XB_tr = scA.fit_transform(XA_tr), scB.fit_transform(XB_tr)
XA_te, XB_te = scA.transform(XA_te), scB.transform(XB_te)

print(f"Train shapes: A={XA_tr.shape}, B={XB_tr.shape}")
print("Label dist:", np.bincount(y_tr))

# ===================================================
# Weighted Sampling
# ===================================================
cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
cw = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
class_counts = np.bincount(y_tr)
sample_weights = [1. / class_counts[y] for y in y_tr]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

dl_A_tr = DataLoader(FeatureDataset(XA_tr, y_tr), BATCH_SIZE, sampler = None, shuffle = True)
dl_B_tr = DataLoader(FeatureDataset(XB_tr, y_tr), BATCH_SIZE, sampler = None, shuffle = True)
dl_dual_tr = DataLoader(DualFeatureDataset(XA_tr, XB_tr, y_tr), BATCH_SIZE, sampler = None, shuffle = True)
dl_A_te = DataLoader(FeatureDataset(XA_te, y_te), BATCH_SIZE)
dl_B_te = DataLoader(FeatureDataset(XB_te, y_te), BATCH_SIZE)
dl_dual_te = DataLoader(DualFeatureDataset(XA_te, XB_te, y_te), BATCH_SIZE)

# ===================================================
# Balanced Focal Loss
# ===================================================
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

loss_fn = BalancedFocalLoss(alpha=cw)

# ===================================================
# Training + Evaluation
# ===================================================
def freeze_encoder(model):
    if hasattr(model, "enc_a") and hasattr(model, "enc_b"):
        for p in model.enc_a.parameters():
            p.requires_grad = False
        for p in model.enc_b.parameters():
            p.requires_grad = False

def unfreeze_encoder(model):
    if hasattr(model, "enc_a") and hasattr(model, "enc_b"):
        for p in model.enc_a.parameters():
            p.requires_grad = True
        for p in model.enc_b.parameters():
            p.requires_grad = True

def evaluate(model, dev_loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for batch in dev_loader:
            if len(batch) == 2:
                xb, yb = batch
                logits = model(xb.to(DEVICE))
            else:
                xa, xb, yb = batch
                logits = model(xa.to(DEVICE), xb.to(DEVICE))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            yt.extend(yb.numpy())
            yp.extend(probs)
    yt, yp = np.array(yt), np.array(yp)
    fpr, tpr, th = roc_curve(yt, yp)
    j = tpr - fpr
    best_th = th[np.argmax(j)]
    preds = (yp >= best_th).astype(int)
    cm = confusion_matrix(yt, preds)
    tn, fp, fn, tp = cm.ravel()
    wa = accuracy_score(yt, preds)
    ua = 0.5 * ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6)))
    return {
        "acc": wa,
        "prec": precision_score(yt, preds),
        "rec": recall_score(yt, preds),
        "f1": f1_score(yt, preds),
        "auc": roc_auc_score(yt, yp),
        "wa": wa,
        "ua": ua,
        "best_th": best_th
    }

def train_and_eval(model, train_loader, dev_loader, name):
    lr = LR_FUSION if "Concat" in name or "Attention" in name else LR_BASE
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=10)
    best_f1, patience = 0, 0
    freeze = "Attention" in name or "Concat" in name
    if freeze: freeze_encoder(model)

    print(f"\n--- Training {name} (lr={lr}) ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            opt.zero_grad()
            if len(batch) == 2:
                xb, yb = batch
                logits = model(xb.to(DEVICE))
            else:
                xa, xb, yb = batch
                logits = model(xa.to(DEVICE), xb.to(DEVICE))
            loss = loss_fn(logits, yb.to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        # Unfreeze after N epochs
        if freeze and epoch == FREEZE_EPOCHS:
            unfreeze_encoder(model)

        # Evaluate each epoch
        res = evaluate(model, dev_loader)
        sched.step(res["f1"])

        print(f"Epoch {epoch:03d} | Loss={total_loss/len(train_loader):.4f} | "
              f"F1={res['f1']:.3f} | AUC={res['auc']:.3f} | WA={res['wa']:.3f} | UA={res['ua']:.3f}")

        if res["f1"] > best_f1:
            best_f1, best_state, patience = res["f1"], model.state_dict(), 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    final_res = evaluate(model, dev_loader)
    print(f"\n{name} → Best F1={final_res['f1']:.3f} | AUC={final_res['auc']:.3f}\n")
    return final_res

# ===================================================
# Run all configurations
# ===================================================
results = {}

results["ComParE-only"] = train_and_eval(
    MLPBranch(XA_tr.shape[1], dropout=0.1).to(DEVICE),
    dl_A_tr, dl_A_te, "ComParE"
)

results["SSL-only"] = train_and_eval(
    MLPBranch(XB_tr.shape[1], dropout=0.3).to(DEVICE),
    dl_B_tr, dl_B_te, "SSL"
)

results["Concat"] = train_and_eval(
    EarlyConcatFusion(XA_tr.shape[1], XB_tr.shape[1]).to(DEVICE),
    dl_dual_tr, dl_dual_te, "Concat"
)

results["AttentionFusion"] = train_and_eval(
    AttentionFusion(XA_tr.shape[1], XB_tr.shape[1]).to(DEVICE),
    dl_dual_tr, dl_dual_te, "AttentionFusion"
)

# ===================================================
# Save results
# ===================================================
out_path = os.path.join(OUTDIR, "fusion_metrics_final.csv")
with open(out_path, "w") as f:
    f.write("Model,Accuracy,Precision,Recall,F1,AUC,WA,UA\n")
    for k, v in results.items():
        f.write(f"{k},{v['acc']:.4f},{v['prec']:.4f},{v['rec']:.4f},{v['f1']:.4f},{v['auc']:.4f},{v['wa']:.4f},{v['ua']:.4f}\n")
print(f"\n Results saved to {out_path}")
