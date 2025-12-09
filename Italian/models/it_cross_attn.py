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
    plot_three_model_roc   # you can ignore if unused
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
    "/home/bubai-maji/bubai/Itali/features_npy/IS10",
    "/home/bubai-maji/bubai/Itali/features_npy/openai-whisper-large",
]

OUT_DIR = "/home/bubai-maji/bubai/Itali/it_results"
os.makedirs(OUT_DIR, exist_ok=True)

# global speaker-wise buffers (only 1 point per speaker)
ALL_INITIAL = []   # raw concatenated features
ALL_FINAL   = []   # learned fused embedding
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
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


# =====================================================
# DATASET
# =====================================================
class MultiSegDataset(Dataset):
    def __init__(self, X_list, y, spk):
        """
        X_list: list of numpy arrays, one per modality
                all with same first dimension (#segments)
        y:      numpy array of labels per segment
        spk:    numpy array of speaker ids per segment
        """
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
    """
    True bidirectional cross-attention fusion for EXACTLY 2 modalities:
      - Modality A attends to B: Q=A, K=B, V=B
      - Modality B attends to A: Q=B, K=A, V=A
    Then their updated representations are concatenated and classified.
    """
    def __init__(self, dims, latent=256, heads=8, dropout=0.2):
        super().__init__()
        assert len(dims) == 2, "CrossAttentionFusion currently assumes exactly 2 modalities."

        # per-modality projection to common latent space
        self.proj = nn.ModuleList([ModalityProj(d, latent) for d in dims])

        # cross-attention (A->B and B->A)
        self.att_AtoB = nn.MultiheadAttention(
            embed_dim=latent,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.att_BtoA = nn.MultiheadAttention(
            embed_dim=latent,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm + FFN per modality
        self.norm_A1 = nn.LayerNorm(latent)
        self.norm_B1 = nn.LayerNorm(latent)

        self.ff_A = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent),
            nn.Dropout(dropout)
        )
        self.ff_B = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.ReLU(),
            nn.Linear(latent * 2, latent),
            nn.Dropout(dropout)
        )

        self.norm_A2 = nn.LayerNorm(latent)
        self.norm_B2 = nn.LayerNorm(latent)

        # final classifier on concatenated [A_fused ; B_fused]
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent * 2),
            nn.Linear(latent * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, NUM_CLASSES)
        )

    def _fuse(self, Xs):
        """
        Internal: given list Xs = [X_A, X_B] (each [B, in_dim]),
        return fused embedding [B, 2*latent]
        """
        # project each modality: [B, in_dim] -> [B, latent]
        A = self.proj[0](Xs[0])
        B = self.proj[1](Xs[1])

        # treat each modality as a single token: [B, 1, latent]
        A = A.unsqueeze(1)
        B = B.unsqueeze(1)

        # === Cross Attention: A attends to B ===
        # Q=A, K=B, V=B  -> A receives info from B
        A2, _ = self.att_AtoB(A, B, B)
        A = self.norm_A1(A + A2)

        # === Cross Attention: B attends to A ===
        # Q=B, K=A, V=A  -> B receives info from A
        B2, _ = self.att_BtoA(B, A, A)
        B = self.norm_B1(B + B2)

        # === Feed-forward per modality ===
        A_ff = self.ff_A(A)
        A = self.norm_A2(A + A_ff)

        B_ff = self.ff_B(B)
        B = self.norm_B2(B + B_ff)

        # remove the token dimension: [B, 1, latent] -> [B, latent]
        A_vec = A.squeeze(1)
        B_vec = B.squeeze(1)

        # fused representation: concatenate
        fused = torch.cat([A_vec, B_vec], dim=-1)   # [B, 2*latent]
        return fused

    def forward(self, Xs):
        fused = self._fuse(Xs)
        return self.classifier(fused)

    def embed(self, Xs):
        """
        Returns fused embedding [B, 2*latent] (detached).
        Used for analysis/TSNE.
        """
        with torch.no_grad():
            fused = self._fuse(Xs)
        return fused.detach()


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

    # ------------ Load all modalities ------------ #
    for m, base in enumerate(BASE_DIRS):

        X_list, y_list, spk_list = [], [], []

        for f in range(1, FOLDS+1):
            Xf, yf, sp = load_fold(base, f)
            X_list.append(Xf)
            y_list.append(yf)
            spk_list.append(sp)

        # test fold
        X_te = X_list[fold_idx]
        y_te = y_list[fold_idx]
        spk_te = spk_list[fold_idx]

        # train folds (all others)
        X_tr = np.vstack([X_list[i] for i in range(FOLDS) if i != fold_idx])
        y_tr = np.hstack([y_list[i] for i in range(FOLDS) if i != fold_idx])

        Xtr_all.append(X_tr)
        Xte_all.append(X_te)

        if m == 0:
            ytrain = y_tr
            ytest = y_te
            spktest = spk_te

    # ------------ Standardize per modality ------------ #
    Xtr_scaled, Xte_scaled = [], []
    for Xtr, Xte in zip(Xtr_all, Xte_all):
        mean = Xtr.mean(0)
        std = Xtr.std(0)
        std[std == 0] = 1.0
        Xtr_scaled.append((Xtr - mean) / std)
        Xte_scaled.append((Xte - mean) / std)

    # ------------ Speaker Sampler (from first modality) ------------ #
    first_base = BASE_DIRS[0]
    spk_all = [
        np.load(f"{first_base}/fold{f}_speaker.npy", allow_pickle=True)
        for f in range(1, FOLDS+1)
    ]

    spk_train = np.hstack([spk_all[i] for i in range(FOLDS) if i != fold_idx])
    spk_test  = spk_all[fold_idx]

    uniq, cnt = np.unique(spk_train, return_counts=True)
    cnt_map = dict(zip(uniq, cnt))
    weights = np.array([1.0 / cnt_map[s] for s in spk_train])

    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        MultiSegDataset(Xtr_scaled, ytrain, spk_train),
        batch_size=BATCH,
        sampler=sampler
    )
    test_loader = DataLoader(
        MultiSegDataset(Xte_scaled, ytest, spk_test),
        batch_size=BATCH,
        shuffle=False
    )

    # ------------ Model ------------ #
    dims = [X.shape[1] for X in Xtr_scaled]
    model = CrossAttentionFusion(dims).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_f1 = -1.0
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
            logits = model(Xs)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        # ------------ DEV EVAL (speaker-level) ------------ #
        model.eval()
        seg_p, seg_y, seg_s = [], [], []

        with torch.no_grad():
            for Xs, yb, spk in test_loader:
                Xs = [x.to(DEVICE) for x in Xs]
                probs = torch.softmax(model(Xs), dim=1)[:, 1].cpu().numpy()
                seg_p.extend(probs)
                seg_y.extend(yb.numpy())
                seg_s.extend(spk)

        seg_s = np.array(seg_s, dtype=object)
        seg_y = np.array(seg_y)
        seg_p = np.array(seg_p)

        # speaker aggregation: mean probability per speaker
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

    # ---- Aggregate speaker embeddings ---- #
    speaker_raw = {}
    speaker_learn = {}
    speaker_label = {}
    speaker_prob = {}

    with torch.no_grad():
        for Xs, yb, spk in test_loader:
            Xs_dev = [x.to(DEVICE) for x in Xs]
            probs = torch.softmax(model(Xs_dev), dim=1)[:, 1].cpu().numpy()
            raw = np.concatenate([x.cpu().numpy() for x in Xs_dev], axis=1)
            emb = model.embed(Xs_dev).cpu().numpy()   # fused embedding

            for i, s in enumerate(spk):
                speaker_raw.setdefault(s, []).append(raw[i])
                speaker_learn.setdefault(s, []).append(emb[i])
                speaker_label.setdefault(s, []).append(int(yb[i].item()))
                speaker_prob.setdefault(s, []).append(float(probs[i]))

    # ---- One entry per speaker ---- #
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

    # store globally for all folds
    ALL_INITIAL.append(spk_initial)
    ALL_FINAL.append(spk_final)
    ALL_LABELS.append(spk_labels)
    ALL_PROBS.append(spk_probs)
    ALL_TRUE.append(spk_labels)

    # ---- speaker-level metrics ---- #
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
    tn, fp, fn, tp = confusion_matrix(spk_labels, pred).ravel()
    ua = 0.5 * ((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))

    return acc, prec, rec, f1, wf1, auc, ua


# =====================================================
# MAIN 5-FOLD LOOP
# =====================================================
results = {k: [] for k in ["acc", "prec", "rec", "f1", "wf1", "auc", "ua"]}

for f in range(FOLDS):
    print("\n=======================================")
    print(f"TRAINING FOLD {f+1}/{FOLDS}")

    acc, prec, rec, f1, wf1, auc, ua = train_fold(f)

    for key, val in zip(results.keys(), [acc, prec, rec, f1, wf1, auc, ua]):
        results[key].append(val)

print("\n========== 5-FOLD SUMMARY ==========")
for k, v in results.items():
    print(f"{k.upper():5s} Mean={np.nanmean(v):.4f}  STD={np.nanstd(v):.4f}")

print("\nGenerating final Confusion Matrix ...")

# ---- Confusion Matrix (Speaker-level over all folds) ---- #
all_probs = np.concatenate(ALL_PROBS)
all_true  = np.concatenate(ALL_TRUE)

fpr, tpr, th = roc_curve(all_true, all_probs)
best_th = th[np.argmax(tpr - fpr)]
pred = (all_probs >= best_th).astype(int)

plot_confusion_matrix(
    all_true,
    pred,
    os.path.join(OUT_DIR, "confusion.png")
)

print(f"\nSaved Confusion Matrix to {os.path.join(OUT_DIR, 'confusion.png')}")
print("Done.")
