import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

# ========== CONFIG ============
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

IS10_DIR = "/home/bubai-maji/bubai/Itali/features_npy/IS10"
PRETRAIN_DIR = "/home/bubai-maji/bubai/Itali/features_npy/MIT-ast-finetuned-audioset-10-10-0.4593"
OUT_DIR = "/home/bubai-maji/bubai/Itali/it_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== UTIL ==========

def safe_auc(y, p):
    try:
        return roc_auc_score(y, p)
    except:
        return np.nan

# ========== DATA HELPERS ==========

def load_fold(base, fold):
    X = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk

class MultiSegDataset(Dataset):
    def __init__(self, X_list, y, spk=None):
        # X_list: list of numpy arrays or single numpy array
        if isinstance(X_list, list):
            self.X_list = [torch.tensor(X, dtype=torch.float32) for X in X_list]
        else:
            self.X_list = [torch.tensor(X_list, dtype=torch.float32)]
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = spk

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        Xs = [X[idx] for X in self.X_list]
        if self.spk is None:
            return Xs, self.y[idx]
        return Xs, self.y[idx], self.spk[idx]

# ========== MODELS ==========
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, NUM_CLASSES)
        )
    def forward(self, x):
        return self.net(x)

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
        self.att = nn.MultiheadAttention(latent, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(latent, latent*2), nn.ReLU(), nn.Linear(latent*2, latent), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(latent)
        self.norm2 = nn.LayerNorm(latent)
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 32), nn.Dropout(0.2),
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

# ========== TRAIN / EVAL HELPERS ==========

def train_mlp(X_tr, y_tr, X_te, y_te, spk_te, device=DEVICE):
    # X_tr, X_te are numpy arrays already standardized
    model = MLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    tr_loader = DataLoader(MultiSegDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(MultiSegDataset(X_te, y_te, spk_te), batch_size=BATCH, shuffle=False)

    for ep in range(EPOCHS):
        model.train()
        for Xs, yb in tr_loader:
            xb = Xs[0].to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    # evaluate at speaker level
    model.eval()
    seg_p, seg_y, seg_s = [], [], []
    with torch.no_grad():
        for Xs, yb, spk in te_loader:
            xb = Xs[0].to(device)
            p = torch.softmax(model(xb), 1)[:,1].cpu().numpy()
            seg_p.extend(p); seg_y.extend(yb.numpy()); seg_s.extend(spk)

    seg_p = np.array(seg_p); seg_y = np.array(seg_y); seg_s = np.array(seg_s, dtype=object)
    sp_p, sp_t = [], []
    for pid in np.unique(seg_s):
        mask = (seg_s == pid)
        sp_p.append(seg_p[mask].mean())
        sp_t.append(seg_y[mask][0])
    sp_p = np.array(sp_p); sp_t = np.array(sp_t)

    fpr, tpr, th = roc_curve(sp_t, sp_p)
    auc = safe_auc(sp_t, sp_p)
    return fpr, tpr, auc, sp_t, sp_p


def train_fusion(X_tr_list, y_tr, X_te_list, y_te, spk_tr, spk_te, device=DEVICE):
    dims = [X.shape[1] for X in X_tr_list]
    model = CrossAttentionFusion(dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    # speaker-balanced sampler using first modality speaker ids
    uniq, cnt = np.unique(spk_tr, return_counts=True)
    cnt_map = dict(zip(uniq, cnt))
    weights = np.array([1.0 / cnt_map[s] for s in spk_tr])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    tr_loader = DataLoader(MultiSegDataset(X_tr_list, y_tr, spk_tr), batch_size=BATCH, sampler=sampler)
    te_loader = DataLoader(MultiSegDataset(X_te_list, y_te, spk_te), batch_size=BATCH, shuffle=False)

    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_f1 = -1
    patience_cnt = 0

    for ep in range(EPOCHS):
        model.train()
        for Xs, yb, _ in tr_loader:
            Xs = [x.to(device) for x in Xs]
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(Xs), yb)
            loss.backward()
            opt.step()

        # dev eval on speaker level
        model.eval()
        seg_p, seg_y, seg_s = [], [], []
        with torch.no_grad():
            for Xs, yb, spk in te_loader:
                Xs = [x.to(device) for x in Xs]
                p = torch.softmax(model(Xs), 1)[:,1].cpu().numpy()
                seg_p.extend(p); seg_y.extend(yb.numpy()); seg_s.extend(spk)

        seg_p = np.array(seg_p); seg_y = np.array(seg_y); seg_s = np.array(seg_s, dtype=object)
        sp_p, sp_t = [], []
        for pid in np.unique(seg_s):
            mask = (seg_s == pid)
            sp_p.append(seg_p[mask].mean())
            sp_t.append(seg_y[mask][0])
        sp_p = np.array(sp_p); sp_t = np.array(sp_t)

        fpr, tpr, th = roc_curve(sp_t, sp_p)
        best_th = th[np.argmax(tpr - fpr)]
        pred = (sp_p >= best_th).astype(int)
        f1 = f1_score(sp_t, pred)
        print(f"Fusion: Epoch {ep+1} | Dev F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Fusion: Early stopping.")
                break

    # final evaluation load best
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    speaker_raw = {}
    speaker_learn = {}
    speaker_label = {}
    speaker_prob = {}

    with torch.no_grad():
        for Xs, yb, spk in te_loader:
            Xs = [x.to(device) for x in Xs]
            probs = torch.softmax(model(Xs), 1)[:,1].cpu().numpy()
            emb = model.embed(Xs).cpu().numpy()
            raw = np.concatenate([x.cpu().numpy() for x in Xs], axis=1)
            for i, s in enumerate(spk):
                speaker_raw.setdefault(s, []).append(raw[i])
                speaker_learn.setdefault(s, []).append(emb[i])
                speaker_label.setdefault(s, []).append(int(yb[i].item()))
                speaker_prob.setdefault(s, []).append(float(probs[i]))

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
    spk_final = np.vstack(spk_final)
    spk_labels = np.array(spk_labels)
    spk_probs = np.array(spk_probs)

    fpr, tpr, th = roc_curve(spk_labels, spk_probs)
    auc = safe_auc(spk_labels, spk_probs)

    return fpr, tpr, auc, spk_labels, spk_probs

# ========== MAIN FOLD LOOP ==========

# containers for ROC per-model across folds
model_names = ["FUSION"]
fold_rocs = {name: [] for name in model_names}  # list of (fpr,tpr,auc)

# for confusion matrix we only keep fusion's aggregated true/pred across all folds
fusion_all_true = []
fusion_all_pred = []

for fold_idx in range(1, FOLDS+1):
    print(f"\n===== Fold {fold_idx}/{FOLDS} =====")

    # load modality data
    X1_tr_list, X1_te_list, spk_list = [], [], []
    bases = [IS10_DIR, PRETRAIN_DIR]

    X_tr_list = []
    X_te_list = []
    spk_tr = None
    spk_te = None

    for base in bases:
        X_list = []
        y_list = []
        spk_this = []
        for f in range(1, FOLDS+1):
            Xf, yf, spf = load_fold(base, f)
            X_list.append(Xf); y_list.append(yf); spk_this.append(spf)
        X_te = X_list[fold_idx-1]
        y_te = y_list[fold_idx-1]
        spk_te = spk_this[fold_idx-1]
        X_tr = np.vstack([X_list[i] for i in range(FOLDS) if i != (fold_idx-1)])
        y_tr = np.hstack([y_list[i] for i in range(FOLDS) if i != (fold_idx-1)])
        if spk_tr is None:
            spk_tr = np.hstack([spk_this[i] for i in range(FOLDS) if i != (fold_idx-1)])
        # standardize per-modality
        mean = X_tr.mean(0)
        std = X_tr.std(0); std[std == 0] = 1
        X_tr_scaled = (X_tr - mean) / std
        X_te_scaled = (X_te - mean) / std

        X_tr_list.append(X_tr_scaled)
        X_te_list.append(X_te_scaled)

    # ytrain and ytest from first base
    ytrain = np.hstack([np.load(f"{bases[0]}/fold{f}_y.npy").astype(int) for f in range(1, FOLDS+1) if f != fold_idx])
    ytest = np.load(f"{bases[0]}/fold{fold_idx}_y.npy").astype(int)

    # ---- Train IS10 MLP (single modality 0)
    #fpr0, tpr0, auc0, sp_t0, sp_p0 = train_mlp(X_tr_list[0], ytrain, X_te_list[0], ytest, spk_te)
    #fold_rocs["IS10"].append((fpr0, tpr0, auc0))

    # ---- Train PRETRAIN MLP (single modality 1)
    #fpr1, tpr1, auc1, sp_t1, sp_p1 = train_mlp(X_tr_list[1], ytrain, X_te_list[1], ytest, spk_te)
    #fold_rocs["PRETRAIN"].append((fpr1, tpr1, auc1))

    # ---- Train FUSION model (both modalities)
    fpr2, tpr2, auc2, sp_t2, sp_p2 = train_fusion(X_tr_list, ytrain, X_te_list, ytest, spk_tr, spk_te)
    fold_rocs["FUSION"].append((fpr2, tpr2, auc2))

    # collect fusion confusion info: compute best threshold on fold and store predictions
    ths = roc_curve(sp_t2, sp_p2)[2]
    best_th = ths[np.argmax(tpr2 - fpr2)]
    pred = (sp_p2 >= best_th).astype(int)
    fusion_all_true.append(sp_t2)
    fusion_all_pred.append(pred)

# ========== AGGREGATE & PLOT ROC ===========
# compute mean ROC curve per model by interpolating tprs on common fpr grid
import seaborn as sns

# ========== AGGREGATE & PLOT ROC ===========    
mean_fpr = np.linspace(0, 1, 200)
plt.figure(figsize=(3.5,3))

for name in model_names:
    tprs = []
    aucs = []

    for fpr, tpr, auc in fold_rocs[name]:
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.nanmean(aucs)
    std_auc = np.nanstd(aucs)
    plt.plot(mean_fpr, mean_tpr, label=f"AUC={mean_auc:.3f}±{std_auc:.3f}")

plt.plot([0,1],[0,1], linestyle='--', linewidth=0.7)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, 'it_roc.png'), dpi=300)
print(f"Saved ROC plot to {os.path.join(OUT_DIR, 'mean_roc_all_models.png')}")

# ========== FUSION CONFUSION MATRIX (aggregated across folds) ========
fusion_true_all = np.concatenate(fusion_all_true)
fusion_pred_all = np.concatenate(fusion_all_pred)
cm = confusion_matrix(fusion_true_all, fusion_pred_all)
print("Fusion speaker-level confusion matrix:\n", cm)

plt.figure(figsize=(3.5, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Depression"],
            yticklabels=["Normal", "Depression"], annot_kws={"size": 12})
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

plt.savefig(os.path.join(OUT_DIR, "it_cm.png"), dpi=300)
print(f"Saved Fusion confusion matrix to {os.path.join(OUT_DIR, 'fusion_confusion_matrix.png')}")

# ========== SUMMARY STATS ==========
for name in model_names:
    aucs = [item[2] for item in fold_rocs[name]]
    print(f"{name}: Mean AUC = {np.nanmean(aucs):.4f}  STD = {np.nanstd(aucs):.4f}")

print("\nDone.")
