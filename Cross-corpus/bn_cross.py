import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# =============================
# CONFIG
# =============================
SEED = 42
RUNS = 5
FOLDS = 5
BATCH = 64
EPOCHS = 100
PATIENCE = 10
LR = 2e-4
LATENT_DIM = 256
HEADS = 8
DROPOUT = 0.2
NUM_CLASSES = 2

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================
# PATHS
# =============================
DAIC_A = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"
DAIC_B = "/home/bubai-maji/bubai/English/features_npy/wavlm_base"

EDAIC_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
EDAIC_B = "/home/bubai-maji/bubai/English/edic_features_npy/wavlm_base"

BN_A = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/IS10_5fold"
BN_B = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/WavLM_base_5fold"

IT_A = "/home/bubai-maji/bubai/Itali/features_npy/IS10"
IT_B = "/home/bubai-maji/bubai/Itali/features_npy/microsoft-wavlm-base"

# =============================
# LOADERS
# =============================
def load_split(base, split):
    X = np.load(f"{base}/{split}/segment_X.npy")
    y = np.load(f"{base}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{base}/{split}/segment_speaker_id.npy")
    return X, y, spk

def load_fold(base, fold):
    X = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return X, y, spk


# =============================
# DATASET
# =============================
class SegDataset(Dataset):
    def __init__(self, Xa, Xb, y, spk):
        self.Xa = torch.tensor(Xa, dtype=torch.float32)
        self.Xb = torch.tensor(Xb, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = np.array(spk)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.Xa[idx], self.Xb[idx], self.y[idx], self.spk[idx]


# =============================
# MODEL — CrossAttentionFusion (Full)
# =============================
class ModalityProj(nn.Module):
    def __init__(self, in_dim, latent=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, latent),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dims, latent=LATENT_DIM, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.ModuleList([ModalityProj(d, latent) for d in dims])

        self.att_AtoB = nn.MultiheadAttention(latent, heads, dropout, batch_first=True)
        self.att_BtoA = nn.MultiheadAttention(latent, heads, dropout, batch_first=True)

        self.norm_A1 = nn.LayerNorm(latent)
        self.norm_A2 = nn.LayerNorm(latent)
        self.norm_B1 = nn.LayerNorm(latent)
        self.norm_B2 = nn.LayerNorm(latent)

        self.ff_A = nn.Sequential(
            nn.Linear(latent, latent*2), nn.ReLU(),
            nn.Linear(latent*2, latent), nn.Dropout(dropout)
        )
        self.ff_B = nn.Sequential(
            nn.Linear(latent, latent*2), nn.ReLU(),
            nn.Linear(latent*2, latent), nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent*2),
            nn.Linear(latent*2,256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, NUM_CLASSES)
        )

    def _fuse(self, Xs):
        A = self.proj[0](Xs[0]).unsqueeze(1)
        B = self.proj[1](Xs[1]).unsqueeze(1)

        A2,_ = self.att_AtoB(A,B,B)
        A = self.norm_A1(A + A2)
        A = self.norm_A2(A + self.ff_A(A))

        B2,_ = self.att_BtoA(B,A,A)
        B = self.norm_B1(B + B2)
        B = self.norm_B2(B + self.ff_B(B))

        return torch.cat([A.squeeze(1), B.squeeze(1)], dim=-1)

    def forward(self, Xs):
        fused = self._fuse(Xs)
        return self.classifier(fused)


# =============================
# METRICS — Speaker-level
# =============================
def compute_speaker_metrics(y_true, probs):
    y_true, probs = np.array(y_true), np.array(probs)
    try:
        fpr, tpr, th = roc_curve(y_true, probs)
        th_best = th[np.argmax(tpr - fpr)]
    except:
        th_best = 0.5

    preds = (probs >= th_best).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    wf1 = f1_score(y_true, preds, average="weighted", zero_division=0)
    auc = roc_auc_score(y_true, probs) if len(np.unique(y_true))>1 else 0.5
    tn,fp,fn,tp = confusion_matrix(y_true,preds).ravel()
    ua = 0.5*((tp/(tp+fn+1e-6))+(tn/(tn+fp+1e-6)))

    return dict(acc=acc,prec=prec,rec=rec,f1=f1,wf1=wf1,auc=auc,ua=ua)


def evaluate(model, Xa, Xb, y, spk):
    dl = DataLoader(SegDataset(Xa,Xb,y,spk), batch_size=BATCH, shuffle=False)
    seg_prob=[]; seg_y=[]; seg_s=[]
    model.eval()
    with torch.no_grad():
        for xa,xb,yb,sp in dl:
            prob = torch.softmax(model([xa.to(DEVICE),xb.to(DEVICE)]),1)[:,1].cpu().numpy()
            seg_prob.extend(prob); seg_y.extend(yb.numpy()); seg_s.extend(sp)
    seg_prob,seg_y,seg_s = map(np.array,[seg_prob,seg_y,seg_s])

    spk_ids = np.unique(seg_s)
    sp_p=[]; sp_t=[]
    for sid in spk_ids:
        idx=np.where(seg_s==sid)[0]
        sp_p.append(seg_prob[idx].mean())
        sp_t.append(seg_y[idx][0])

    return compute_speaker_metrics(np.array(sp_t), np.array(sp_p))


# =============================
# MAIN CROSS-CVP
# =============================
def run_cross_bengali():

    metrics = ["acc","prec","rec","f1","wf1","auc","ua"]
    results_daic=[]; results_edaic=[]; results_it=[]

    # ----- 5 RUNS -----
    for run in range(RUNS):
        print(f"\n========= RUN {run+1}/{RUNS} =========")

        fold_results_daic=[]; fold_results_edaic=[]; fold_results_it=[]

        # ----- 5-fold Train/Test -----
        for fold in range(1,FOLDS+1):

            print(f"\n--- Fold {fold}/{FOLDS} ---")

            # Test fold
            Xte_A, y_te, spk_te = load_fold(BN_A, fold)
            Xte_B, _, _ = load_fold(BN_B, fold)

            # Train folds = all except current
            Xtr_A=[]; Xtr_B=[]; y_tr=[]; spk_tr=[]
            for f in range(1,FOLDS+1):
                if f == fold: continue
                Xa,y_a,sp_a = load_fold(BN_A, f)
                Xb,_,_ = load_fold(BN_B, f)
                Xtr_A.append(Xa); Xtr_B.append(Xb)
                y_tr.append(y_a); spk_tr.append(sp_a)

            Xtr_A=np.vstack(Xtr_A); Xtr_B=np.vstack(Xtr_B)
            y_tr=np.hstack(y_tr); spk_tr=np.hstack(spk_tr)

            # Normalization stats from train folds ONLY
            mA,sA = Xtr_A.mean(0), Xtr_A.std(0); sA[sA==0]=1
            mB,sB = Xtr_B.mean(0), Xtr_B.std(0); sB[sB==0]=1

            Xtr_A=(Xtr_A-mA)/sA; Xtr_B=(Xtr_B-mB)/sB
            Xte_A=(Xte_A-mA)/sA; Xte_B=(Xte_B-mB)/sB

            # Speaker-balanced train sampling
            spk_counts={s:np.sum(spk_tr==s) for s in np.unique(spk_tr)}
            weights=np.array([1/spk_counts[s] for s in spk_tr],dtype=np.float32)

            dimA,dimB = Xtr_A.shape[1], Xtr_B.shape[1]
            model = CrossAttentionFusion([dimA,dimB]).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=LR)
            loss_fn = nn.CrossEntropyLoss()

            train_loader = DataLoader(
                SegDataset(Xtr_A,Xtr_B,y_tr,spk_tr),
                batch_size=BATCH,
                sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
            )

            # ---- Train ----
            best_f1=-1; patience=0
            best_state={k:v.cpu() for k,v in model.state_dict().items()}

            for ep in range(EPOCHS):
                model.train()
                for xa,xb,yb,_ in train_loader:
                    xa,xb,yb = xa.to(DEVICE), xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    yhat = model([xa,xb])
                    loss = loss_fn(yhat,yb)
                    loss.backward()
                    opt.step()

                # validation = Bengali test fold (speaker-level)
                met_val = evaluate(model,Xte_A,Xte_B,y_te,spk_te)
                print(f" Fold{fold} Ep{ep+1:03d} | Val F1={met_val['f1']:.4f}")
                if met_val["f1"]>best_f1:
                    best_f1=met_val["f1"]
                    best_state={k:v.cpu() for k,v in model.state_dict().items()}
                    patience=0
                else:
                    patience+=1
                    if patience>=PATIENCE:
                        break

            model.load_state_dict(best_state)
            model.to(DEVICE)

            # ---------------------------------
            # CROSS-CORPUS TESTS
            # ---------------------------------

            # DAIC test
            Xa,y,s = load_split(DAIC_A,"dev")
            Xb,_,_ = load_split(DAIC_B,"dev")
            fold_results_daic.append(
                evaluate(model,(Xa-mA)/sA,(Xb-mB)/sB,y,s)
            )

            # EDAIC test
            Xa,y,s = load_split(EDAIC_A,"test")
            Xb,_,_ = load_split(EDAIC_B,"test")
            fold_results_edaic.append(
                evaluate(model,(Xa-mA)/sA,(Xb-mB)/sB,y,s)
            )

            # Italian 5-fold avg
            it_fold=[]
            for f2 in range(1,6):
                Xa,y,s = load_fold(IT_A,f2)
                Xb,_,_ = load_fold(IT_B,f2)
                it_fold.append(
                    evaluate(model,(Xa-mA)/sA,(Xb-mB)/sB,y,s)
                )
            fold_results_it.append({k:np.mean([d[k] for d in it_fold]) for k in metrics})

        # Average FOLD results → RUN result
        results_daic.append({k:np.mean([d[k] for d in fold_results_daic]) for k in metrics})
        results_edaic.append({k:np.mean([d[k] for d in fold_results_edaic]) for k in metrics})
        results_it.append({k:np.mean([d[k] for d in fold_results_it]) for k in metrics})

    # ==============================
    # REPORT
    # ==============================
    def report(name,data):
        print(f"\n==== Bengali → {name} (5 runs avg) ====")
        for k in metrics:
            arr=np.array([d[k] for d in data])
            print(f"{k.upper():5s} Mean={arr.mean():.4f} STD={arr.std():.4f}")

    report("DAIC",results_daic)
    report("EDAIC",results_edaic)
    report("ITALIAN",results_it)


if __name__ == "__main__":
    run_cross_bengali()
