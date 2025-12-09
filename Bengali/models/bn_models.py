# model.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from plot import (
    plot_confusion_matrix,
    plot_roc
)

# =====================================================
# CONFIG
# =====================================================
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
    "/home/bubai-maji/bubai/Bangla/bangla_features_npy/IS10_5fold",
    "/home/bubai-maji/bubai/Bangla/bangla_features_npy/WavLM_base_5fold",
]

os.makedirs("/home/bubai-maji/bubai/Bangla/bn_results", exist_ok=True)

# global speaker-wise buffers (only 1 point per speaker)
ALL_INITIAL = []
ALL_FINAL   = []
ALL_LABELS  = []
ALL_PROBS   = []
ALL_TRUE    = []

# fold-wise ROC for mean curve
FOLD_FPRS = []
FOLD_TPRS = []
FOLD_AUCS = []


# =====================================================
# LOAD FOLD DATA
# =====================================================
def load_fold(base, fold):
    X = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    #utt = np.load(f"{base}/fold{fold}_utterance.npy", allow_pickle=True)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


# =====================================================
# DATASET
# =====================================================
class MultiSegDataset(Dataset):
    def __init__(self, X_list, y, spk):
        self.X_list = [torch.tensor(X, dtype=torch.float32) for X in X_list]
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = spk

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [X[idx] for X in self.X_list], self.y[idx], self.spk[idx]


# =====================================================
# MODEL
# =====================================================
class ModalityProj(nn.Module):
    def __init__(self, in_dim, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, latent),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dims, latent=256, heads=8, dropout=0.2):
        super().__init__()
        self.proj = nn.ModuleList([ModalityProj(d, latent) for d in dims])

        self.att = nn.MultiheadAttention(
            latent, heads, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(latent, latent*2),
            nn.ReLU(),
            nn.Linear(latent*2, latent),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(latent)
        self.norm2 = nn.LayerNorm(latent)

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.Dropout(0.2),
            nn.Linear(32, NUM_CLASSES)
        )

    def forward(self, Xs):
        toks = [self.proj[i](Xs[i]) for i in range(len(Xs))]
        toks = torch.stack(toks, dim=1)

        att_out, _ = self.att(toks, toks, toks)
        x = self.norm1(att_out + toks)

        ff_out = self.ff(x)
        x = self.norm2(ff_out + x)

        scores = x.mean(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)

        pooled = (x * w).sum(1)
        return self.classifier(pooled)

    def embed(self, Xs):
        toks = [self.proj[i](Xs[i]) for i in range(len(Xs))]
        toks = torch.stack(toks, dim=1)

        att_out, _ = self.att(toks, toks, toks)
        x = self.norm1(att_out + toks)

        ff_out = self.ff(x)
        x = self.norm2(ff_out + x)

        scores = x.mean(-1)
        w = torch.softmax(scores, dim=1).unsqueeze(-1)

        pooled = (x * w).sum(1)
        return pooled.detach()


# =====================================================
# SAFE AUC
# =====================================================
def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except:
        return np.nan


# =====================================================
# TRAIN ONE FOLD
# =====================================================
def train_fold(fold_idx):

    Xtr_all, Xte_all = [], []

    # ------------ Load all modalities ------------
    for m, base in enumerate(BASE_DIRS):

        X_list, y_list, spk_list = [], [], []

        for f in range(1, FOLDS+1):
            Xf, yf, sp = load_fold(base, f)
            X_list.append(Xf)
            y_list.append(yf)
            spk_list.append(sp)

        X_te = X_list[fold_idx]
        y_te = y_list[fold_idx]
        spk_te = spk_list[fold_idx]

        X_tr = np.vstack([X_list[i] for i in range(FOLDS) if i != fold_idx])
        y_tr = np.hstack([y_list[i] for i in range(FOLDS) if i != fold_idx])

        Xtr_all.append(X_tr)
        Xte_all.append(X_te)

        if m == 0:
            ytrain = y_tr
            ytest = y_te
            spktest = spk_te

    # ------------ Standardize ------------
    Xtr_scaled, Xte_scaled = [], []
    for Xtr, Xte in zip(Xtr_all, Xte_all):
        mean = Xtr.mean(0)
        std = Xtr.std(0); std[std == 0] = 1
        Xtr_scaled.append((Xtr - mean) / std)
        Xte_scaled.append((Xte - mean) / std)

    # ------------ Speaker Sampler ------------
    first_base = BASE_DIRS[0]
    spk_all = [np.load(f"{first_base}/fold{f}_speaker.npy", allow_pickle=True)
               for f in range(1, FOLDS+1)]

    spk_train = np.hstack([spk_all[i] for i in range(FOLDS) if i != fold_idx])
    spk_test  = spk_all[fold_idx]

    uniq, cnt = np.unique(spk_train, return_counts=True)
    cnt_map = dict(zip(uniq, cnt))
    weights = np.array([1.0 / cnt_map[s] for s in spk_train])

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        MultiSegDataset(Xtr_scaled, ytrain, spk_train),
        batch_size=BATCH, sampler=sampler
    )
    test_loader = DataLoader(
        MultiSegDataset(Xte_scaled, ytest, spk_test),
        batch_size=BATCH, shuffle=False
    )

    # ------------ Model ------------
    dims = [X.shape[1] for X in Xtr_scaled]
    model = CrossAttentionFusion(dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    # initialize best_state to avoid later undefined var
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_f1 = -1
    patience_cnt = 0

    # =====================================================
    # TRAINING LOOP
    # =====================================================
    for ep in range(EPOCHS):

        model.train()
        for Xs, yb, _ in train_loader:
            Xs = [x.to(DEVICE) for x in Xs]
            yb = yb.to(DEVICE)

            opt.zero_grad()
            loss = crit(model(Xs), yb)
            loss.backward()
            opt.step()

        # ------------ DEV EVAL ------------
        model.eval()
        seg_p, seg_y, seg_s = [], [], []

        with torch.no_grad():
            for Xs, yb, spk in test_loader:
                Xs = [x.to(DEVICE) for x in Xs]
                p = torch.softmax(model(Xs), 1)[:,1].cpu().numpy()
                seg_p.extend(p)
                seg_y.extend(yb.numpy())
                seg_s.extend(spk)

        seg_s = np.array(seg_s, dtype=object)
        seg_y = np.array(seg_y)
        seg_p = np.array(seg_p)

        # speaker aggregation
        sp_p, sp_t = [], []
        for pid in np.unique(seg_s):
            mask = (seg_s == pid)
            sp_p.append(seg_p[mask].mean())
            sp_t.append(seg_y[mask][0])

        sp_p = np.array(sp_p)
        sp_t = np.array(sp_t)

        fpr, tpr, th = roc_curve(sp_t, sp_p)
        best_th = th[np.argmax(tpr - fpr)]
        pred = (sp_p >= best_th).astype(int)

        f1 = f1_score(sp_t, pred)
        print(f"FOLD {fold_idx+1} | Epoch {ep+1} | F1={f1:.4f}")

        # early stopping
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping.")
                break

    # =====================================================
    # FINAL EVALUATION
    # =====================================================
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    # ---- Aggregate speaker embeddings ----
    speaker_raw = {}
    speaker_learn = {}
    speaker_label = {}
    speaker_prob = {}

    with torch.no_grad():
        for Xs, yb, spk in test_loader:
            Xs = [x.to(DEVICE) for x in Xs]
            probs = torch.softmax(model(Xs), 1)[:,1].cpu().numpy()
            raw = np.concatenate([x.cpu().numpy() for x in Xs], axis=1)
            emb = model.embed(Xs).cpu().numpy()

            for i, s in enumerate(spk):
                speaker_raw.setdefault(s, []).append(raw[i])
                speaker_learn.setdefault(s, []).append(emb[i])
                speaker_label.setdefault(s, []).append(int(yb[i].item()))
                speaker_prob.setdefault(s, []).append(float(probs[i]))

    # ---- Convert to 1-per-speaker ----
    spk_initial = []
    spk_final = []
    spk_labels = []
    spk_probs = []

    for s in speaker_raw.keys():
        spk_initial.append(np.mean(speaker_raw[s], axis=0))
        spk_final.append(np.mean(speaker_learn[s], axis=0))
        spk_labels.append(speaker_label[s][0])
        spk_probs.append(np.mean(speaker_prob[s]))

    spk_initial = np.vstack(spk_initial)
    spk_final   = np.vstack(spk_final)
    spk_labels  = np.array(spk_labels)
    spk_probs   = np.array(spk_probs)

    # store globally
    ALL_INITIAL.append(spk_initial)
    ALL_FINAL.append(spk_final)
    ALL_LABELS.append(spk_labels)
    ALL_PROBS.append(spk_probs)
    ALL_TRUE.append(spk_labels)

    # ---- speaker-level metrics ----
    fpr, tpr, th = roc_curve(spk_labels, spk_probs)
    best_th = th[np.argmax(tpr - fpr)]
    pred = (spk_probs >= best_th).astype(int)

    auc = safe_auc(spk_labels, spk_probs)

    FOLD_FPRS.append(fpr)
    FOLD_TPRS.append(tpr)
    FOLD_AUCS.append(auc)

    acc = accuracy_score(spk_labels, pred)
    prec = precision_score(spk_labels, pred, zero_division=0)
    rec = recall_score(spk_labels, pred, zero_division=0)
    f1 = f1_score(spk_labels, pred)
    wf1 = f1_score(spk_labels, pred, average="weighted")
    tn,fp,fn,tp = confusion_matrix(spk_labels, pred).ravel()
    ua = 0.5 * ((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))

    return acc,prec,rec,f1,wf1,auc,ua


results = {k:[] for k in ["acc","prec","rec","f1","wf1","auc","ua"]}

for f in range(FOLDS):
    print("\n=======================================")
    print(f"TRAINING FOLD {f+1}/{FOLDS}")

    acc,prec,rec,f1,wf1,auc,ua = train_fold(f)

    for key, val in zip(results.keys(), [acc,prec,rec,f1,wf1,auc,ua]):
        results[key].append(val)

print("\n========== 5-FOLD SUMMARY ==========")
for k,v in results.items():
    print(f"{k.upper():5s} Mean={np.nanmean(v):.4f}  STD={np.nanstd(v):.4f}")


print("\nGenerating final TSNE, ROC, Confusion Matrix ...")

# ---- 1. t-SNE ----------
initial_emb = np.vstack(ALL_INITIAL)
final_emb   = np.vstack(ALL_FINAL)
labels      = np.hstack(ALL_LABELS)


# ---- 3. Confusion Matrix (Speaker-level) ----
all_probs = np.concatenate(ALL_PROBS)
all_true  = np.concatenate(ALL_TRUE)

fpr, tpr, th = roc_curve(all_true, all_probs)
best_th = th[np.argmax(tpr - fpr)]
pred = (all_probs >= best_th).astype(int)

plot_confusion_matrix(
    all_true,
    pred,
    "/home/bubai-maji/bubai/Bangla/bn_results/confusion.png"
)

print("\n Saved Confusion Matrix to bn_results/")
