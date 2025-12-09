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
N_FOLDS = 5
BATCH = 64
EPOCHS = 50
LR = 2e-4
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000   # standard
SEG_MS = 4000

# Feature settings:
FBANK_BINS = 64      # DepAudioNet
SPEC_NFFT = 1024     # SpeechFormer
SPEC_BINS = SPEC_NFFT // 2 + 1
MFCC_DIM = 40        # Light-SERNet

CSV_PATH = "/home/bubai-maji/bubai/Itali/segment_metadata.csv"
NUM_CLASSES = 2

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
    "SpeechFormer": make_spec,
    "LightSERNet": make_mfcc
}


# ===================== DATASET =========================
class AudioSegDataset(Dataset):
    def __init__(self, df, extractor):
        self.df = df.reset_index(drop=True)
        self.ext = extractor
        self.target_len = int(SEG_MS * SAMPLE_RATE / 1000)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = torchaudio.load(row["seg_path"])
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        if wav.size(0) > 1:
            wav = wav.mean(0, keepdim=True)

        if wav.size(1) < self.target_len:
            wav = nn.functional.pad(wav, (0, self.target_len - wav.size(1)))
        else:
            wav = wav[:, :self.target_len]

        feat = self.ext(wav)
        feat = (feat - feat.mean()) / (feat.std() + 1e-9)

        speaker_id = str(row["speaker_id"])
        utt_id = speaker_id  # using speaker ID as utterance grouping

        return feat, int(row["label"]), speaker_id, utt_id


# ======================= MODELS ========================
### DepAudioNet
class DepAudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
        )
        self.lstm_hidden = 128
        self.lstm = nn.LSTM(128*(FBANK_BINS//8),self.lstm_hidden,
                            batch_first=True,bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(2*self.lstm_hidden,128),nn.ReLU(),
            nn.Dropout(0.3),nn.Linear(128,NUM_CLASSES)
        )

    def forward(self,x):
        x = self.cnn(x) # [B,128,F',T']
        B,C,Fp,Tp = x.shape
        x = x.permute(0,3,1,2).reshape(B,Tp,C*Fp)
        _,(h,_) = self.lstm(x)
        out = torch.cat((h[-2],h[-1]),1)
        return self.fc(out)


### Light-SERNet
class LightSERNet(nn.Module):
    def __init__(self, feat_bins=40, num_classes=NUM_CLASSES):
        super().__init__()

        # Body Part I: Parallel Paths
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((2,2))
        )
        self.branch9 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(9,1), padding=(4,0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((2,2))
        )
        self.branch11 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1,11), padding=(0,5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d((2,2))
        )

        # Body Part II: Feature Learning (FLFB blocks)
        self.lflb1 = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AvgPool2d((2,2))
        )
        self.lflb2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.AvgPool2d((2,2))
        )
        self.lflb3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AvgPool2d((2,2))
        )
        self.lflb4 = nn.Sequential(
            nn.Conv2d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160), nn.ReLU(),
            nn.AvgPool2d((2,1))
        )
        self.lflb5 = nn.Sequential(
            nn.Conv2d(160, 320, kernel_size=1),
            nn.BatchNorm2d(320), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # GAP
        )

        # Head
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        # x: [B,1,MFCC_DIM,T]
        b1 = self.branch3(x)
        b2 = self.branch9(x)
        b3 = self.branch11(x)
        x = torch.cat([b1, b2, b3], dim=1)  # [B,96,*,*]

        x = self.lflb1(x)
        x = self.lflb2(x)
        x = self.lflb3(x)
        x = self.lflb4(x)
        x = self.lflb5(x)

        x = x.squeeze(-1).squeeze(-1)  # [B,320]
        x = self.dropout(x)
        return self.fc(x)


### SpeechFormer
class PosEnc(nn.Module):
    def __init__(self,d, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len,d)
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
        x=self.proj(x)
        x=self.pos(x)
        x=self.enc(x).mean(1)
        return self.fc(x)


MODELS = {
    #"DepAudioNet":DepAudioNet,
    "LightSERNet":LightSERNet,
    "SpeechFormer":SpeechFormer
}


# =================== TRAIN + EVAL =====================
def evaluate(Y,P,PS,ID):
    true,pred,prob=[],[],[]
    for k,idx in defaultdict(list, {k:[] for k in set(ID)}).items():
        for i,x in enumerate(ID): 
            if x==k: idx.append(i)
        t=[Y[i] for i in idx]
        p=[P[i] for i in idx]
        pr=[PS[i] for i in idx]
        true.append(t[0])
        pred.append(np.bincount(p).argmax())
        prob.append(np.mean(pr))
    return compute_metrics(true,pred,prob)


def run_fold(model_name, df_train, df_test, fold):
    print(f"\nFold {fold} | {model_name}")
    model = MODELS[model_name]().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    
    ext = EXTRACTORS[model_name]()
    tr_ds = AudioSegDataset(df_train,ext)
    te_ds = AudioSegDataset(df_test,ext)
    tr = DataLoader(tr_ds,BATCH,True,num_workers=4)
    te = DataLoader(te_ds,BATCH,False,num_workers=4)

    for ep in range(1,EPOCHS+1):
        model.train()
        for x,y,_,_ in tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            loss=crit(model(x),y)
            loss.backward()
            opt.step()
        if ep%10==0: print("Epoch:",ep,"Loss:",loss.item())

    model.eval()
    Y,P,PS,UID,SID=[],[],[],[],[]
    with torch.no_grad():
        for x,y,spk,utt in te:
            x=x.to(DEVICE)
            logits=model(x)
            probs=torch.softmax(logits,1)[:,1].cpu().numpy()
            pred=logits.argmax(1).cpu().numpy()
            Y+=y.numpy().tolist()
            P+=pred.tolist(); PS+=probs.tolist()
            UID+=utt; SID+=spk

    seg=compute_metrics(Y,P,PS)
    utt=evaluate(Y,P,PS,UID)
    spk=evaluate(Y,P,PS,SID)

    print("SEG:",seg)
    print("UTT:",utt)
    print("SPK:",spk)

    return seg,utt,spk


def main():
    df=pd.read_csv(CSV_PATH)
    for name in MODELS.keys():
        segA,uttA,spkA=[],[],[]
        for f in range(1,N_FOLDS+1):
            seg,utt,spk = run_fold(name,
                                   df[df.fold!=f],
                                   df[df.fold==f],
                                   f)
            segA.append(seg); uttA.append(utt); spkA.append(spk)

        print("\n### Summary:",name)
        for lbl,arr in zip(["SEG","UTT","SPK"],[segA,uttA,spkA]):
            arr=np.array(arr); mean=np.nanmean(arr,0); std=np.nanstd(arr,0)
            print(lbl,"=>")
            print(["WA","UA","Prec","Rec","F1","WF1","AUC"])
            print(mean,std,"\n")

if __name__=="__main__":
    main()
