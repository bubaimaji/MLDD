import os, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
SEED = 42
FOLDS = 5
BATCH = 64
EPOCHS = 100  
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

FEAT_DIR = "/home/bubai-maji/bubai/Itali/features_npy/IS10"
NUM_CLASSES = 2

# ---------------- DATASET ----------------
class SingleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ---------------- MODEL ----------------
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
    def forward(self, x): return self.net(x)

# ---------------- AUC CHECK --------------
def safe_auc(y, p):
    if len(np.unique(y)) < 2: return np.nan
    return roc_auc_score(y, p)

# ---------------- LOAD FOLDS ------------
folds = []
for i in range(1, FOLDS+1):
    X = np.load(f"{FEAT_DIR}/fold{i}_X.npy")
    y = np.load(f"{FEAT_DIR}/fold{i}_y.npy")
    utt = np.load(f"{FEAT_DIR}/fold{i}_utterance.npy", allow_pickle=True)
    spk = np.load(f"{FEAT_DIR}/fold{i}_speaker.npy", allow_pickle=True)
    folds.append((X, y, utt, spk))

print("Loaded", len(folds), "folds")

# -------------- RESULTS SEPARATELY -----------
results_MLP = {"segment": [], "utterance": [], "speaker": []}
results_SVM = {"segment": [], "utterance": [], "speaker": []}
results_RF  = {"segment": [], "utterance": [], "speaker": []}

# -------- Helper to compute all 3 levels ----------
def evaluate_levels(y_true, y_pred, y_prob, utt, spk, res_dict, tag=""):
    # Segment level
    seg_WA   = accuracy_score(y_true, y_pred)
    seg_UA   = 0.5*(recall_score(y_true,y_pred,pos_label=0)+recall_score(y_true,y_pred,pos_label=1))
    seg_prec = precision_score(y_true, y_pred, zero_division=0)
    seg_rec  = recall_score(y_true, y_pred, zero_division=0)
    seg_F1   = f1_score(y_true, y_pred)
    seg_WF1  = f1_score(y_true, y_pred, average="weighted")
    seg_AUC  = safe_auc(y_true, y_prob)
    res_dict["segment"].append([seg_WA, seg_UA, seg_prec, seg_rec, seg_F1, seg_WF1, seg_AUC])
    print(f"{tag} Segment | WA={seg_WA:.3f}, UA={seg_UA:.3f}, F1={seg_F1:.3f}, AUC={seg_AUC:.3f}")

    # Utterance level
    ut, up, ps = [], [], []
    for u in np.unique(utt):
        idx = np.where(utt == u)[0]
        up.append(np.bincount(y_pred[idx]).argmax())
        ut.append(y_true[idx][0])
        ps.append(y_prob[idx].mean())
    ut, up, ps = map(np.array, [ut, up, ps])
    utt_WA   = accuracy_score(ut, up)
    utt_UA   = 0.5*(recall_score(ut,up,pos_label=0)+recall_score(ut,up,pos_label=1))
    utt_F1   = f1_score(ut, up)
    utt_AUC  = safe_auc(ut, ps)
    res_dict["utterance"].append([utt_WA, utt_UA, 0, 0, utt_F1, 0, utt_AUC])
    print(f"{tag} Utterance | WA={utt_WA:.3f}, UA={utt_UA:.3f}, F1={utt_F1:.3f}, AUC={utt_AUC:.3f}")

    # Speaker level
    st, sp, pp = [], [], []
    for s in np.unique(spk):
        idx = np.where(spk == s)[0]
        sp.append(np.bincount(y_pred[idx]).argmax())
        st.append(y_true[idx][0])
        pp.append(y_prob[idx].mean())
    st, sp, pp = map(np.array, [st, sp, pp])
    spk_WA   = accuracy_score(st, sp)
    spk_UA   = 0.5*(recall_score(st,sp,pos_label=0)+recall_score(st,sp,pos_label=1))
    spk_F1   = f1_score(st, sp)
    spk_AUC  = safe_auc(st, pp)
    res_dict["speaker"].append([spk_WA, spk_UA, 0, 0, spk_F1, 0, spk_AUC])
    print(f"{tag} Speaker | WA={spk_WA:.3f}, UA={spk_UA:.3f}, F1={spk_F1:.3f}, AUC={spk_AUC:.3f}")


# ====================== 5-FOLD LOOP ============================
for f in range(FOLDS):
    print(f"\n================ Fold {f+1}/{FOLDS} ================")

    X_te, y_te, utt_te, spk_te = folds[f]
    X_tr = np.vstack([folds[j][0] for j in range(FOLDS) if j != f])
    y_tr = np.hstack([folds[j][1] for j in range(FOLDS) if j != f])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # ===== MLP training =====
    train_loader = DataLoader(SingleDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(SingleDataset(X_te, y_te), batch_size=BATCH)
    model = MLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    # ===== MLP test =====
    model.eval()
    ys, yp, ps = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            prob = torch.softmax(logits,1)[:,1].cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
            ys.extend(yb.numpy()); yp.extend(pred); ps.extend(prob)
    print("\n--- MLP Evaluation ---")
    evaluate_levels(np.array(ys), np.array(yp), np.array(ps), utt_te, spk_te, results_MLP, "MLP")

    # ===== SVM =====
    print("\n--- SVM Evaluation ---")
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)
    svm.fit(X_tr, y_tr)
    svm_pred = svm.predict(X_te)
    svm_prob = svm.predict_proba(X_te)[:,1]
    evaluate_levels(y_te, svm_pred, svm_prob, utt_te, spk_te, results_SVM, "SVM")

    # ===== Random Forest =====
    print("\n--- Random Forest Evaluation ---")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_prob = rf.predict_proba(X_te)[:,1]
    evaluate_levels(y_te, rf_pred, rf_prob, utt_te, spk_te, results_RF, "RF")


# ==================== SUMMARY PRINTING ======================
def print_summary(res, name):
    print(f"\n========== {name} PERFORMANCE (5-fold) ==========")
    for level in ["segment", "utterance", "speaker"]:
        arr = np.array(res[level], float)
        print(f"\n>> {level.upper()}")
        print(f"WA   = {np.nanmean(arr[:,0]):.3f} ± {np.nanstd(arr[:,0]):.3f}")
        print(f"UA   = {np.nanmean(arr[:,1]):.3f} ± {np.nanstd(arr[:,1]):.3f}")
        print(f"F1   = {np.nanmean(arr[:,4]):.3f} ± {np.nanstd(arr[:,4]):.3f}")
        print(f"AUC  = {np.nanmean(arr[:,6]):.3f} ± {np.nanstd(arr[:,6]):.3f}")

print_summary(results_MLP, "MLP")
print_summary(results_SVM, "SVM")
print_summary(results_RF, "RF")

print("\n========== DONE ==========\n")
