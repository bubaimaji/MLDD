import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import collections

# ======================= CONFIG =======================
SEED = 42
FOLDS = 5
BATCH = 64
EPOCHS = 100
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Single modality path (ComParE functionals) ----
FEAT_DIR = "/home/bubai-maji/bubai/Itali/features_npy/MIT-ast-finetuned-audioset-10-10-0.4593"

NUM_CLASSES = 2

# ===================== DATASET ========================
class SingleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): 
        return len(self.y)
    def __getitem__(self, idx): 
        return self.X[idx], self.y[idx]

# ====================== MODEL =========================
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

# ===================== SAFE AUC =======================
def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

# ===================== LOAD FOLDS =====================
folds = []
for i in range(1, FOLDS+1):
    X = np.load(f"{FEAT_DIR}/fold{i}_X.npy")
    y = np.load(f"{FEAT_DIR}/fold{i}_y.npy")
    utt = np.load(f"{FEAT_DIR}/fold{i}_utterance.npy", allow_pickle=True)
    spk = np.load(f"{FEAT_DIR}/fold{i}_speaker.npy", allow_pickle=True)
    folds.append((X, y, utt, spk))

print("Loaded", len(folds), "folds")

# ===================== RESULTS STORAGE =====================
results = {"segment": [], "utterance": [], "speaker": []}

# ================== CROSS-VALIDATION =======================
for f in range(FOLDS):
    print(f"\n=========== Fold {f+1} ===========")

    X_te, y_te, utt_te, spk_te = folds[f]
    X_tr = np.vstack([folds[j][0] for j in range(FOLDS) if j != f])
    y_tr = np.hstack([folds[j][1] for j in range(FOLDS) if j != f])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)

    train_loader = DataLoader(SingleDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(SingleDataset(X_te, y_te), batch_size=BATCH)

    model = MLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    # -------- TRAIN --------
    model.train()
    for ep in range(EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    # -------- TEST --------
    model.eval()
    ys, yp, ps = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            prob = torch.softmax(logits,1)[:,1].cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
            ys.extend(yb.numpy()); yp.extend(pred); ps.extend(prob)

    ys, yp, ps = map(np.array, [ys, yp, ps])

    # ===== Segment Metrics =====
    seg_WA   = accuracy_score(ys, yp)
    seg_UA   = 0.5*(recall_score(ys,yp,pos_label=0)+recall_score(ys,yp,pos_label=1))
    seg_prec = precision_score(ys, yp, zero_division=0)
    seg_rec  = recall_score(ys, yp, zero_division=0)
    seg_F1   = f1_score(ys, yp)
    seg_WF1  = f1_score(ys, yp, average="weighted")
    seg_AUC  = safe_auc(ys, ps)

    results["segment"].append([seg_WA, seg_UA, seg_prec, seg_rec, seg_F1, seg_WF1, seg_AUC])

    print(f"Segment | WA={seg_WA:.3f} UA={seg_UA:.3f} P={seg_prec:.3f} R={seg_rec:.3f} "
          f"F1={seg_F1:.3f} WF1={seg_WF1:.3f} AUC={seg_AUC:.3f}")

    # ===== Utterance Metrics =====
    ut, up, pp = [], [], []
    for u in np.unique(utt_te):
        idx = np.where(utt_te == u)[0]
        up.append(np.bincount(yp[idx]).argmax())
        ut.append(ys[idx][0])
        pp.append(ps[idx].mean())
    ut, up, pp = map(np.array, [ut, up, pp])

    utt_WA   = accuracy_score(ut, up)
    utt_UA   = 0.5*(recall_score(ut,up,pos_label=0)+recall_score(ut,up,pos_label=1))
    utt_prec = precision_score(ut, up, zero_division=0)
    utt_rec  = recall_score(ut, up, zero_division=0)
    utt_F1   = f1_score(ut, up)
    utt_WF1  = f1_score(ut, up, average="weighted")
    utt_AUC  = safe_auc(ut, pp)

    results["utterance"].append([utt_WA,utt_UA,utt_prec,utt_rec,utt_F1,utt_WF1,utt_AUC])

    print(f"Utterance | WA={utt_WA:.3f} UA={utt_UA:.3f} P={utt_prec:.3f} R={utt_rec:.3f} "
          f"F1={utt_F1:.3f} WF1={utt_WF1:.3f} AUC={utt_AUC:.3f}")

    # ===== Speaker Metrics =====
    st, sp, pp = [], [], []
    for s in np.unique(spk_te):
        idx = np.where(spk_te == s)[0]
        sp.append(np.bincount(yp[idx]).argmax())
        st.append(ys[idx][0])
        pp.append(ps[idx].mean())
    st, sp, pp = map(np.array, [st, sp, pp])

    spk_WA   = accuracy_score(st, sp)
    spk_UA   = 0.5*(recall_score(st,sp,pos_label=0)+recall_score(st,sp,pos_label=1))
    spk_prec = precision_score(st, sp, zero_division=0)
    spk_rec  = recall_score(st, sp, zero_division=0)
    spk_F1   = f1_score(st, sp)
    spk_WF1  = f1_score(st, sp, average="weighted")
    spk_AUC  = safe_auc(st, pp)

    results["speaker"].append([spk_WA,spk_UA,spk_prec,spk_rec,spk_F1,spk_WF1,spk_AUC])

    print(f"Speaker | WA={spk_WA:.3f} UA={spk_UA:.3f} P={spk_prec:.3f} R={spk_rec:.3f} "
          f"F1={spk_F1:.3f} WF1={spk_WF1:.3f} AUC={spk_AUC:.3f}")

# ==================== SUMMARY ==========================
def summary(level):
    arr = np.array(results[level], float)
    print(f"\n---- {level.upper()} 5-Fold Avg ----")
    print(f"WA   : {np.nanmean(arr[:,0]):.3f} ± {np.nanstd(arr[:,0]):.3f}")
    print(f"UA   : {np.nanmean(arr[:,1]):.3f} ± {np.nanstd(arr[:,1]):.3f}")
    print(f"Prec : {np.nanmean(arr[:,2]):.3f} ± {np.nanstd(arr[:,2]):.3f}")
    print(f"Rec  : {np.nanmean(arr[:,3]):.3f} ± {np.nanstd(arr[:,3]):.3f}")
    print(f"F1   : {np.nanmean(arr[:,4]):.3f} ± {np.nanstd(arr[:,4]):.3f}")
    print(f"WF1  : {np.nanmean(arr[:,5]):.3f} ± {np.nanstd(arr[:,5]):.3f}")
    print(f"AUC  : {np.nanmean(arr[:,6]):.3f} ± {np.nanstd(arr[:,6]):.3f}")

summary("segment")
summary("utterance")
summary("speaker")
