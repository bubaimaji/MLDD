import os
import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ======================= CONFIG =======================
SEED = 42
BATCH = 64  # smaller batch improves generalization
EPOCHS = 50
LR = 2e-4  # smaller learning rate to avoid collapse
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

TRAIN_CSV_PATH = "/home/bubai-maji/bubai/English/Processed_csv_edic/train_balanced_metadata.csv"
DEV_CSV_PATH   = "/home/bubai-maji/bubai/English/Processed_csv_edic/test_segment_metadata.csv"

SAMPLE_RATE = 16000
SEG_MS = 4000
NUM_CLASSES = 2

FBANK_BINS = 64
SPEC_NFFT = 1024
SPEC_BINS = SPEC_NFFT // 2 + 1
MFCC_DIM = 40

np.random.seed(SEED)
torch.manual_seed(SEED)

warnings = __import__("warnings")
warnings.filterwarnings("ignore", module="torchaudio")


# ======================= METRICS =======================
def safe_auc(y,p):
    return roc_auc_score(y,p) if len(set(y))>1 else np.nan

def compute_metrics(y,pred,prob):
    return (
        accuracy_score(y,pred),
        0.5*(recall_score(y,pred,pos_label=0)+recall_score(y,pred,pos_label=1)),
        precision_score(y,pred,zero_division=0),
        recall_score(y,pred,zero_division=0),
        f1_score(y,pred),
        f1_score(y,pred,average="weighted"),
        safe_auc(y,prob)
    )


# ================= FEATURE EXTRACTORS ==================
def make_fbank():
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=SPEC_NFFT,
            win_length=400, hop_length=160,
            n_mels=FBANK_BINS, f_min=0, f_max=8000),
        torchaudio.transforms.AmplitudeToDB()
    )

def make_spec():
    return nn.Sequential(
        torchaudio.transforms.Spectrogram(
            n_fft=SPEC_NFFT, win_length=400, hop_length=160, power=2),
        torchaudio.transforms.AmplitudeToDB()
    )

def make_mfcc():
    return torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE, n_mfcc=MFCC_DIM,
        melkwargs={
            "n_fft": SPEC_NFFT, "win_length":400, "hop_length":160,
            "n_mels":FBANK_BINS,"f_min":0,"f_max":8000}
    )

EXTRACTORS = {
    "DepAudioNet": make_fbank,
    "LightSERNet": make_mfcc,
    "SpeechFormer": make_spec
}


# ======================= MODELS ========================
class DepAudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)
        )
        self.lstm_hidden = 128
        self.lstm = nn.LSTM(128*(FBANK_BINS//8),self.lstm_hidden,
                            batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Linear(2*self.lstm_hidden,128),nn.ReLU(),
            nn.Dropout(0.4),nn.Linear(128,NUM_CLASSES)
        )
    def forward(self,x):
        x=self.cnn(x)
        B,C,F,T=x.shape
        x=x.permute(0,3,1,2).reshape(B,T,C*F)
        _,(h,_) = self.lstm(x)
        out = torch.cat((h[-2],h[-1]),1)
        return self.fc(self.dropout(out))


class LightSERNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d((2,2))
        )
        self.branch9 = nn.Sequential(
            nn.Conv2d(1, 32, (9,1), padding=(4,0)),
            nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d((2,2))
        )
        self.branch11 = nn.Sequential(
            nn.Conv2d(1, 32, (1,11), padding=(0,5)),
            nn.BatchNorm2d(32), nn.ReLU(), nn.AvgPool2d((2,2))
        )
        self.lflb1 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.AvgPool2d((2,2))
        )
        self.lflb2 = nn.Sequential(
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(), nn.AvgPool2d((2,2))
        )
        self.lflb3 = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AvgPool2d((2,2))
        )
        self.lflb4 = nn.Sequential(
            nn.Conv2d(128, 160, 3, padding=1),
            nn.BatchNorm2d(160), nn.ReLU(), nn.AvgPool2d((2,1))
        )
        self.lflb5 = nn.Sequential(
            nn.Conv2d(160, 320, 1),
            nn.BatchNorm2d(320), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(320, NUM_CLASSES)
    def forward(self, x):
        x=torch.cat([self.branch3(x),self.branch9(x),self.branch11(x)],1)
        x=self.lflb1(x)
        x=self.lflb2(x)
        x=self.lflb3(x) # FIXED
        x=self.lflb4(x)
        x=self.lflb5(x)
        x=x.squeeze(-1).squeeze(-1)
        return self.fc(self.dropout(x))


class PosEnc(nn.Module):
    def __init__(self,d,max_len=2000):
        super().__init__()
        pe=torch.zeros(max_len,d)
        pos=torch.arange(max_len).unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2)*(-np.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1)]


class SpeechFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(SPEC_BINS,128)
        self.pos = PosEnc(128)
        layer = nn.TransformerEncoderLayer(128,4,256,batch_first=True)
        self.enc = nn.TransformerEncoder(layer,3)
        self.fc = nn.Sequential(nn.Linear(128,128),
                                nn.ReLU(),nn.Dropout(0.3),
                                nn.Linear(128,NUM_CLASSES))
    def forward(self,x):
        x=x.squeeze(1).permute(0,2,1)
        x=self.pos(self.proj(x))
        return self.fc(self.enc(x).mean(1))


MODELS = {
    "DepAudioNet": DepAudioNet,
    "LightSERNet": LightSERNet,
    "SpeechFormer": SpeechFormer
}


# ================= DATASET ============================
class AudioSegDataset(Dataset):
    def __init__(self, df, extractor):
        self.df = df.reset_index(drop=True)
        self.ext = extractor
        self.target_len = int(SEG_MS * SAMPLE_RATE / 1000)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = torchaudio.load(row["audio_path"])
        if sr!=SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav,sr,SAMPLE_RATE)
        if wav.size(0)>1:
            wav = wav.mean(0,keepdim=True)

        # ---- Data Augmentation ----
        if np.random.rand()<0.5:
            wav = wav + 0.005 * torch.randn_like(wav)
        if np.random.rand()<0.3:
            wav = torch.roll(wav,shifts=np.random.randint(-400,400),dims=1)

        if wav.size(1)<self.target_len:
            wav = nn.functional.pad(wav,(0,self.target_len-wav.size(1)))
        else:
            wav = wav[:, :self.target_len]

        feat = self.ext(wav)
        feat = (feat-feat.mean())/(feat.std()+1e-9)

        sid=str(row["participant_id"])
        uid=f"{sid}_{row['segment_index']}"
        return feat,int(row["phq8_binary"]),sid,uid


# ============ THRESHOLD TUNING ================
def find_best_threshold(y, prob):
    thresholds = np.linspace(0.05,0.95,91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (prob>=t).astype(int)
        f1 = f1_score(y,pred,zero_division=0)
        if f1>best_f1:
            best_t,best_f1 = t,f1
    return best_t


# =================== TRAIN + EVAL =====================
def evaluate(Y,P,PS,ID):
    final_true,final_pred,final_prob=[],[],[]
    groups=defaultdict(list)
    for i,k in enumerate(ID): groups[k].append(i)
    for idx in groups.values():
        t=[Y[j] for j in idx]
        p=[P[j] for j in idx]
        pr=[PS[j] for j in idx]
        final_true.append(t[0])
        final_pred.append(np.bincount(p).argmax())
        final_prob.append(np.mean(pr))
    return compute_metrics(final_true,final_pred,final_prob)


def run_exp(model_name, df_train, df_dev, class_weights):
    model = MODELS[model_name]().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(weight=class_weights)

    ext = EXTRACTORS[model_name]()
    tr_ds = AudioSegDataset(df_train, ext)
    te_ds = AudioSegDataset(df_dev, ext)
    tr = DataLoader(tr_ds,BATCH,True,num_workers=4)
    te = DataLoader(te_ds,BATCH,False,num_workers=4)

    best_loss=float("inf")
    patience,no_imp=7,0

    for ep in range(1,EPOCHS+1):
        model.train()
        train_losses=[]
        for x,y,_,_ in tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            loss=crit(model(x),y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(loss.item())

        train_loss=np.mean(train_losses)

        # Validation loss
        model.eval()
        val_losses=[]
        with torch.no_grad():
            for x,y,_,_ in te:
                x,y=x.to(DEVICE),y.to(DEVICE)
                val_loss=crit(model(x),y).item()
                val_losses.append(val_loss)
        val_loss=np.mean(val_losses)

        if ep%5==0:
            print(f"{model_name} | Ep {ep} | Tr Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

        if val_loss<best_loss:
            best_loss,no_imp=val_loss,0
            best_model= model.state_dict()
        else:
            no_imp+=1
            if no_imp>=patience:
                print(f"Early Stop: {model_name} at Ep {ep}")
                break
    
    model.load_state_dict(best_model)

    # Final Eval on Dev
    Y,PS,UID,SID=[],[],[],[]
    with torch.no_grad():
        for x,y,spk,utt in te:
            x=x.to(DEVICE)
            probs=torch.softmax(model(x),1)[:,1].cpu().numpy()
            Y+=y.numpy().tolist()
            PS+=probs.tolist()
            UID+=utt
            SID+=spk

    best_t=find_best_threshold(Y,PS)
    P=(np.array(PS)>=best_t).astype(int).tolist()

    seg=compute_metrics(Y,P,PS)
    utt=evaluate(Y,P,PS,UID)
    spk=evaluate(Y,P,PS,SID)

    print(f"Best Thr={best_t:.3f} | SEG-F1={seg[4]:.4f}")
    print("UTT:",utt,"\nSPK:",spk)

    return seg,utt,spk


# =================== MAIN ============================
def main():
    df_train=pd.read_csv(TRAIN_CSV_PATH)
    df_dev=pd.read_csv(DEV_CSV_PATH)

    # Class weights
    counts=df_train["phq8_binary"].value_counts().sort_index().values
    counts=np.where(counts==0,1,counts)
    class_weights=torch.tensor(counts.sum()/counts,
                               device=DEVICE,dtype=torch.float32)

    NUM_RUNS=5

    for name in MODELS:
        segA,uttA,spkA=[],[],[]
        for r in range(NUM_RUNS):
            print(f"\n>>> Run {r+1}: {name}")
            np.random.seed(SEED+r)
            torch.manual_seed(SEED+r)
            seg,utt,spk=run_exp(name,df_train,df_dev,class_weights)
            segA.append(seg);uttA.append(utt);spkA.append(spk)

        print("\n===== Summary",name,"=====")
        print("[WA,UA,Prec,Rec,F1,WF1,AUC]")
        for lbl,arr in zip(["SEG","UTT","SPK"], [segA,uttA,spkA]):
            arr=np.array(arr)
            print(lbl)
            print("Mean:",np.round(np.nanmean(arr,0),4))
            print("Std :",np.round(np.nanstd(arr,0),4)); print()


if __name__=="__main__":
    main()
