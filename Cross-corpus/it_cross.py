import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ==================================
# CONFIG
# ==================================
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

# ==================================
# PATHS
# ==================================
DAIC_A = "/home/bubai-maji/bubai/English/features_npy/IS10_paraling"
DAIC_B = "/home/bubai-maji/bubai/English/features_npy/wavlm_base"

EDAIC_A = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"
EDAIC_B = "/home/bubai-maji/bubai/English/edic_features_npy/wavlm_base"

BN_A = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/IS10_5fold"
BN_B = "/home/bubai-maji/bubai/Bangla/bangla_features_npy/WavLM_base_5fold"

IT_A = "/home/bubai-maji/bubai/Itali/features_npy/IS10"
IT_B = "/home/bubai-maji/bubai/Itali/features_npy/microsoft-wavlm-base"


# ==================================
# LOADERS
# ==================================
def load_split(base, split):
    Xa = np.load(f"{base}/{split}/segment_X.npy")
    y = np.load(f"{base}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{base}/{split}/segment_speaker_id.npy")
    return Xa, y, spk

def load_fold(base, fold):
    Xa = np.load(f"{base}/fold{fold}_X.npy")
    y = np.load(f"{base}/fold{fold}_y.npy").astype(int)
    spk = np.load(f"{base}/fold{fold}_speaker.npy", allow_pickle=True)
    return Xa, y, spk


# ==================================
# DATASET
# ==================================
class SegDataset(Dataset):
    def __init__(self, Xa, Xb, y, sp):
        self.Xa = torch.tensor(Xa, dtype=torch.float32)
        self.Xb = torch.tensor(Xb, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.spk = np.array(sp)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.Xa[i], self.Xb[i], self.y[i], self.spk[i]


# ==================================
# MODEL — TRUE CrossAttentionFusion
# ==================================
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
            nn.Linear(32,NUM_CLASSES)
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

        return torch.cat([A.squeeze(1), B.squeeze(1)], -1)

    def forward(self, Xs):
        return self.classifier(self._fuse(Xs))


# ==================================
# METRICS
# ==================================
def speaker_metrics(y, p):
    y,p = np.array(y),np.array(p)
    try:
        f,t,th = roc_curve(y,p)
        th = th[np.argmax(t - f)]
    except:
        th = 0.5

    pr = (p>=th).astype(int)

    acc = accuracy_score(y,pr)
    prec = precision_score(y,pr,zero_division=0)
    rec = recall_score(y,pr,zero_division=0)
    f1 = f1_score(y,pr)
    wf1 = f1_score(y,pr,average="weighted")
    auc = roc_auc_score(y,p) if len(np.unique(y))>1 else 0.5
    tn,fp,fn,tp = confusion_matrix(y,pr).ravel()
    ua = .5*((tp/(tp+fn+1e-6))+(tn/(tn+fp+1e-6)))
    return dict(acc=acc,prec=prec,rec=rec,f1=f1,wf1=wf1,auc=auc,ua=ua)


def evaluate(model, Xa,Xb,y,sp):
    dl = DataLoader(SegDataset(Xa,Xb,y,sp), batch_size=BATCH, shuffle=False)
    prob=[]; Y=[]; S=[]
    model.eval()
    with torch.no_grad():
        for a,b,yb,spk in dl:
            pr = torch.softmax(model([a.to(DEVICE),b.to(DEVICE)]),1)[:,1].cpu().numpy()
            prob.extend(pr); Y.extend(yb.numpy()); S.extend(spk)
    prob,Y,S = map(np.array,[prob,Y,S])
    sp_ids = np.unique(S)
    sp_p=[]; sp_y=[]
    for sid in sp_ids:
        idx = np.where(S==sid)[0]
        sp_p.append(prob[idx].mean())
        sp_y.append(Y[idx][0])
    return speaker_metrics(sp_y,sp_p)


# ==================================
# MAIN
# ==================================
def run_cross_italian():

    metrics = ["acc","prec","rec","f1","wf1","auc","ua"]
    res_daic,res_edaic,res_bn = [],[],[]

    for run in range(RUNS):
        print(f"\n=== RUN {run+1}/{RUNS} ===")

        fold_daic=[]; fold_edaic=[]; fold_bn=[]

        for fold in range(1,FOLDS+1):
            print(f"\n-- Fold {fold}/{FOLDS} --")

            # TEST = current fold
            X_t1,y_te,sp_te = load_fold(IT_A,fold)
            X_t2,_,_ = load_fold(IT_B,fold)

            # TRAIN = remaining folds
            XA=[]; XB=[]; Y=[]; S=[]
            for f in range(1,FOLDS+1):
                if f==fold: continue
                x1,y1,sp1 = load_fold(IT_A,f)
                x2,_,_ = load_fold(IT_B,f)
                XA.append(x1); XB.append(x2)
                Y.append(y1); S.append(sp1)
            XA=np.vstack(XA); XB=np.vstack(XB)
            Y=np.hstack(Y); S=np.hstack(S)

            # Normalize from Italian TRAIN only
            mA,sA = XA.mean(0),XA.std(0); sA[sA==0]=1
            mB,sB = XB.mean(0),XB.std(0); sB[sB==0]=1

            XA=(XA-mA)/sA; XB=(XB-mB)/sB
            X_t1=(X_t1-mA)/sA; X_t2=(X_t2-mB)/sB

            # Speaker-balanced sampler
            cnt={s:np.sum(S==s) for s in np.unique(S)}
            wts=np.array([1/cnt[s] for s in S],float)

            dims = [XA.shape[1],XB.shape[1]]
            model = CrossAttentionFusion(dims).to(DEVICE)
            opt=torch.optim.AdamW(model.parameters(),lr=LR)
            loss_fn = nn.CrossEntropyLoss()

            tl = DataLoader(
                SegDataset(XA,XB,Y,S),
                batch_size=BATCH,
                sampler=WeightedRandomSampler(wts,len(wts),True)
            )

            best_f1=-1; pat=0
            best={k:v.cpu() for k,v in model.state_dict().items()}

            for ep in range(EPOCHS):
                model.train()
                for a,b,yb,_ in tl:
                    a,b,yb = a.to(DEVICE),b.to(DEVICE),yb.to(DEVICE)
                    opt.zero_grad()
                    loss = loss_fn(model([a,b]),yb)
                    loss.backward()
                    opt.step()

                val = evaluate(model,X_t1,X_t2,y_te,sp_te)
                print(f" Fold{fold} Ep{ep+1:03d} | ValF1={val['f1']:.4f}")

                if val["f1"]>best_f1:
                    best_f1=val["f1"]
                    best={k:v.cpu() for k,v in model.state_dict().items()}
                    pat=0
                else:
                    pat+=1
                    if pat>=PATIENCE: break

            model.load_state_dict(best)
            model.to(DEVICE)

            # ===== CROSS-CORPUS TESTING =====
            # DAIC
            Xa,y,sp = load_split(DAIC_A,"dev")
            Xb,_,_= load_split(DAIC_B,"dev")
            fold_daic.append(
                evaluate(model,(Xa-mA)/sA,(Xb-mB)/sB,y,sp)
            )

            # EDAIC
            Xa,y,sp = load_split(EDAIC_A,"test")
            Xb,_,_= load_split(EDAIC_B,"test")
            fold_edaic.append(
                evaluate(model,(Xa-mA)/sA,(Xb-mB)/sB,y,sp)
            )

            # Bengali (avg over 5 folds)
            tmp=[]
            for bf in range(1,6):
                Xa,y,sp = load_fold(BN_A,bf)
                Xb,_,_ = load_fold(BN_B,bf)
                tmp.append(
                    evaluate(model,(Xa-mA)/sA,(Xb-mB)/sB,y,sp)
                )
            fold_bn.append({k:np.mean([d[k] for d in tmp]) for k in metrics})

        res_daic.append({k:np.mean([d[k] for d in fold_daic]) for k in metrics})
        res_edaic.append({k:np.mean([d[k] for d in fold_edaic]) for k in metrics})
        res_bn.append({k:np.mean([d[k] for d in fold_bn]) for k in metrics})

    # REPORT
    def show(name,D):
        print(f"\n=== Italian → {name} ===")
        for k in metrics:
            arr=np.array([r[k] for r in D])
            print(f"{k.upper():4s}: Mean={arr.mean():.4f} STD={arr.std():.4f}")

    show("DAIC",res_daic)
    show("EDAIC",res_edaic)
    show("BENGALI",res_bn)


if __name__ == "__main__":
    run_cross_italian()
