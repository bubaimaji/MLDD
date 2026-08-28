import os
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

FEAT_DIR_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
FEAT_DIR_B = "/home/bubai-maji/bubai/English/edic_features_npy/data2vec_large"

SPLIT_TRAIN = "train"
SPLIT_DEV = "test"

LR = 2e-4
BATCH = 64
EPOCHS = 100
PATIENCE = 10
RUNS = 5
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# =====================
# Load data & group by speaker
# =====================
def load_split_feats(feat_dir, split):
    X = np.load(os.path.join(feat_dir, split, "segment_X.npy"))
    y = np.load(os.path.join(feat_dir, split, "segment_y.npy")).astype(int)
    speakers = np.load(os.path.join(feat_dir, split, "segment_speaker_id.npy"))
    return X, y, speakers


def build_speaker_data():
    XtrA, ytr, spk_tr = load_split_feats(FEAT_DIR_A, SPLIT_TRAIN)
    XtrB, _, _ = load_split_feats(FEAT_DIR_B, SPLIT_TRAIN)

    XdvA, ydv, spk_dv = load_split_feats(FEAT_DIR_A, SPLIT_DEV)
    XdvB, _, _ = load_split_feats(FEAT_DIR_B, SPLIT_DEV)

    def avg_by_speaker(X, y, spk):
        grouped = defaultdict(list)
        labels = {}
        for i, s in enumerate(spk):
            grouped[s].append(X[i])
            labels[s] = int(y[i])
        speakers = sorted(grouped.keys())
        X_sp = np.vstack([np.mean(grouped[s], axis=0) for s in speakers])
        y_sp = np.array([labels[s] for s in speakers])
        return X_sp, y_sp, speakers

    XtrA_sp, ytr_sp, spk_tr_sp = avg_by_speaker(XtrA, ytr, spk_tr)
    XtrB_sp, _, _ = avg_by_speaker(XtrB, ytr, spk_tr)

    XdvA_sp, ydv_sp, spk_dv_sp = avg_by_speaker(XdvA, ydv, spk_dv)
    XdvB_sp, _, _ = avg_by_speaker(XdvB, ydv, spk_dv)

    # Standardize from train only
    meanA, stdA = XtrA_sp.mean(0), XtrA_sp.std(0); stdA[stdA==0]=1
    meanB, stdB = XtrB_sp.mean(0), XtrB_sp.std(0); stdB[stdB==0]=1

    Xtr = np.concatenate([(XtrA_sp-meanA)/stdA, (XtrB_sp-meanB)/stdB], axis=1)
    Xdv = np.concatenate([(XdvA_sp-meanA)/stdA, (XdvB_sp-meanB)/stdB], axis=1)

    return Xtr, ytr_sp, Xdv, ydv_sp


# =====================
# Model
# =====================
class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,2)
        )
    def forward(self, x): return self.net(x)


# =====================
# Train + Eval
# =====================
def train_eval():
    Xtr, ytr, Xdv, ydv = build_speaker_data()

    class_counts = {c: list(ytr).count(c) for c in set(ytr)}
    weights = np.array([1.0/class_counts[y] for y in ytr])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(list(zip(Xtr,ytr)), batch_size=BATCH, sampler=sampler)
    dev_loader = DataLoader(list(zip(Xdv,ydv)), batch_size=BATCH, shuffle=False)

    model = MLP(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, wait = -1, 0
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb = xb.float().to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        # Dev evaluation
        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.float().to(DEVICE)
                all_probs.extend(torch.softmax(model(xb),1)[:,1].cpu().numpy())
                all_y.extend(yb.numpy())

        probs, labels = np.array(all_probs), np.array(all_y)

        # speaker-level metrics
        fpr,tpr,th=roc_curve(labels,probs)
        th_best = th[np.argmax(tpr - fpr)]
        pred = (probs >= th_best).astype(int)

        f1 = f1_score(labels,pred)
        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break

    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()

    all_probs, all_y = [], []
    with torch.no_grad():
        for xb, yb in dev_loader:
            xb = xb.float().to(DEVICE)
            all_probs.extend(torch.softmax(model(xb),1)[:,1].cpu().numpy())
            all_y.extend(yb.numpy())

    probs, labels = np.array(all_probs), np.array(all_y)
    fpr,tpr,th = roc_curve(labels,probs)
    th_best = th[np.argmax(tpr-fpr)]
    pred = (probs>=th_best).astype(int)

    metrics = {
        "acc": accuracy_score(labels,pred),
        "prec": precision_score(labels,pred,zero_division=0),
        "rec": recall_score(labels,pred,zero_division=0),
        "f1": f1_score(labels,pred,zero_division=0),
        "wf1": f1_score(labels,pred,average="weighted",zero_division=0),
        "auc": roc_auc_score(labels,probs),
    }
    tn,fp,fn,tp = confusion_matrix(labels,pred).ravel()
    metrics["ua"] = 0.5*((tp/(tp+fn+1e-6))+(tn/(tn+fp+1e-6)))

    return metrics


# =====================
# MAIN MULTI RUN EVAL
# =====================
all_metrics = {k: [] for k in ["acc","prec","rec","f1","wf1","auc","ua"]}

for r in range(RUNS):
    print(f"\n===== RUN {r+1}/{RUNS} =====")
    m = train_eval()
    for k,v in m.items(): all_metrics[k].append(v)

print("\n======= FINAL SUMMARY (Speaker Level, 5 runs) =======")
for k in all_metrics.keys():
    vals = np.array(all_metrics[k])
    print(f"{k.upper():4s} Mean={vals.mean():.4f}  STD={vals.std():.4f}")

print("\nDone")
