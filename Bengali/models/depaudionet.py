
import os
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ======================= CONFIG =======================
SEED        = 42
N_FOLDS     = 5
BATCH       = 64
EPOCHS      = 50
LR          = 2e-4
DEVICE      = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

SAMPLE_RATE = 16000       # change if your audio is different
SEG_MS      = 4000        # must match your segmentation script

# Feature dimensions (per paper style)
FBANK_BINS  = 64          # DepAudioNet (Mel filterbanks)
SPEC_NFFT   = 1024        # SpeechFormer (spectrogram)
SPEC_BINS   = SPEC_NFFT // 2 + 1
MFCC_DIM    = 40          # LIGHT-SERNET (MFCC)

META_CSV    = "bangla_5fold_metadata.csv"
NUM_CLASSES = 2

torch.manual_seed(SEED)
np.random.seed(SEED)

# Optional: silence torchaudio deprecation spam
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


# ======================= UTILS ========================

def safe_auc(y, p):
    y = np.array(y)
    p = np.array(p)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def compute_metrics(ys, yp, ps):
    ys = np.array(ys)
    yp = np.array(yp)
    ps = np.array(ps)
    WA   = accuracy_score(ys, yp)
    UA   = 0.5 * (recall_score(ys, yp, pos_label=0) +
                  recall_score(ys, yp, pos_label=1))
    Prec = precision_score(ys, yp, zero_division=0)
    Rec  = recall_score(ys, yp, zero_division=0)
    F1   = f1_score(ys, yp)
    WF1  = f1_score(ys, yp, average="weighted")
    AUC  = safe_auc(ys, ps)
    return WA, UA, Prec, Rec, F1, WF1, AUC


def print_metrics(prefix, m):
    WA, UA, Prec, Rec, F1, WF1, AUC = m
    print(f"{prefix} | WA={WA:.3f} UA={UA:.3f} P={Prec:.3f} R={Rec:.3f} "
          f"F1={F1:.3f} WF1={WF1:.3f} AUC={AUC:.3f}")


# ================== FEATURE EXTRACTORS =================

def make_fbank_extractor():
    # DepAudioNet: Mel filterbanks (FBANK)
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=SPEC_NFFT,
        win_length=400,
        hop_length=160,
        n_mels=FBANK_BINS,
        f_min=0,
        f_max=8000,
        power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    return torch.nn.Sequential(melspec, to_db)


def make_spec_extractor():
    # SpeechFormer: linear spectrogram
    spec = torchaudio.transforms.Spectrogram(
        n_fft=SPEC_NFFT,
        win_length=400,
        hop_length=160,
        power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
    return torch.nn.Sequential(spec, to_db)


def make_mfcc_extractor():
    # LIGHT-SERNET: MFCCs
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=MFCC_DIM,
        melkwargs={
            "n_fft": SPEC_NFFT,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": FBANK_BINS,
            "f_min": 0,
            "f_max": 8000,
            "power": 2.0
        }
    )
    return mfcc


def get_extractor_and_dim(model_name: str):
    if model_name == "DepAudioNetLike":
        return make_fbank_extractor(), FBANK_BINS
    elif model_name == "SpeechFormerLike":
        return make_spec_extractor(), SPEC_BINS
    elif model_name == "LightSERNetLike":
        return make_mfcc_extractor(), MFCC_DIM
    else:
        raise ValueError(f"Unknown model name {model_name}")


# ===================== DATASET ========================

class BanglaFeatureDataset(Dataset):
    def __init__(self, df, extractor, sample_rate=SAMPLE_RATE, seg_ms=SEG_MS):
        self.df = df.reset_index(drop=True)
        self.extractor = extractor
        self.sample_rate = sample_rate
        self.target_len = int(seg_ms * sample_rate / 1000)

    def __len__(self):
        return len(self.df)

    def _load_and_pad(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [C, T]
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        T = wav.shape[1]
        if T < self.target_len:
            pad_len = self.target_len - T
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        elif T > self.target_len:
            wav = wav[:, :self.target_len]

        return wav  # [1, target_len]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["seg_path"]
        label = int(row["label"])
        spk_id = str(row["speaker_id"])
        utt_id = str(row["utterance_id"])

        wav = self._load_and_pad(path)
        feat = self.extractor(wav)   # [C, F, T']

        # per-utterance normalization
        mean = feat.mean()
        std = feat.std() + 1e-9
        feat = (feat - mean) / std

        return feat, label, spk_id, utt_id


# ======================= MODELS =======================

# ----- 1. DepAudioNet-inspired: FBANK -> CNN + BiLSTM -----

class DepAudioNet(nn.Module):
    def __init__(self, feat_bins=FBANK_BINS, num_classes=NUM_CLASSES, lstm_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2))  # -> [B, 128, F', T']
        )

        self.freq_after_pool = feat_bins // 8  # 3x pool of 2
        self.lstm_input_dim = 128 * self.freq_after_pool
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, C=1, F=FBANK_BINS, T]
        x = self.cnn(x)  # [B, 128, F', T']
        B, C, Fp, Tp = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T', C, F']
        x = x.view(B, Tp, C * Fp)               # [B, T', Feat]

        _, (h_n, _) = self.lstm(x)             # h_n: [2, B, H]
        h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, 2H]
        out = self.fc(h_last)
        return out


# ----- 2. LIGHT-SERNET-like: MFCC -> multi-branch CNN -----
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

# ----- 3. SpeechFormer-like: Spectrogram -> Transformer -----

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return x


class SpeechFormer(nn.Module):
    def __init__(self, feat_bins=SPEC_BINS, num_classes=NUM_CLASSES,
                 d_model=128, nhead=4, num_layers=3, dim_ff=256):
        super().__init__()
        self.proj = nn.Linear(feat_bins, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, F=SPEC_BINS, T]
        x = x.squeeze(1)          # [B, F, T]
        x = x.permute(0, 2, 1)    # [B, T, F]
        x = self.proj(x)          # [B, T, d_model]
        x = self.pos_enc(x)
        x = self.encoder(x)       # [B, T, d_model]
        x = x.mean(dim=1)         # global average over time
        out = self.fc(x)
        return out


# ==================== TRAIN / EVAL ====================

def run_fold(model_cls, model_name, feat_bins, df_train, df_test, fold_id):
    print(f"\n====== {model_name} | Fold {fold_id} ======")

    extractor, _ = get_extractor_and_dim(model_name)  # extractor already chosen per model
    train_ds = BanglaFeatureDataset(df_train, extractor)
    test_ds  = BanglaFeatureDataset(df_test, extractor)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # pass feat_bins where needed
    model = model_cls(feat_bins=feat_bins).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- TRAIN ----------
    for ep in range(1, EPOCHS+1):
        model.train()
        ep_loss = 0.0
        for feats, labels, _, _ in train_loader:
            feats = feats.to(DEVICE)          # [B, 1, F, T]
            labels = labels.to(DEVICE)

            opt.zero_grad()
            logits = model(feats)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * feats.size(0)

        ep_loss /= len(train_ds)
        if ep == 1 or ep % 5 == 0:
            print(f"[{model_name}] Fold {fold_id} Epoch {ep}/{EPOCHS} "
                  f"Loss={ep_loss:.4f}")

    # ---------- TEST ----------
    model.eval()
    seg_y_true, seg_y_pred, seg_prob = [], [], []
    seg_utt_ids, seg_spk_ids = [], []

    with torch.no_grad():
        for feats, labels, spk_ids, utt_ids in test_loader:
            feats = feats.to(DEVICE)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            seg_y_true.extend(labels.tolist())
            seg_y_pred.extend(preds.tolist())
            seg_prob.extend(probs.tolist())
            seg_utt_ids.extend(list(utt_ids))
            seg_spk_ids.extend(list(spk_ids))

    # ----- segment level -----
    seg_metrics = compute_metrics(seg_y_true, seg_y_pred, seg_prob)
    print_metrics("Segment", seg_metrics)

    # ----- utterance level -----
    utt_true, utt_pred, utt_prob = [], [], []
    utt_to_idx = defaultdict(list)
    for i, u in enumerate(seg_utt_ids):
        utt_to_idx[u].append(i)

    for u, idxs in utt_to_idx.items():
        ys = [seg_y_true[i] for i in idxs]
        ps = [seg_prob[i] for i in idxs]
        preds = [seg_y_pred[i] for i in idxs]

        utt_true.append(ys[0])  # all same within utterance
        counts = np.bincount(preds, minlength=NUM_CLASSES)
        utt_pred.append(np.argmax(counts))
        utt_prob.append(np.mean(ps))

    utt_metrics = compute_metrics(utt_true, utt_pred, utt_prob)
    print_metrics("Utterance", utt_metrics)

    # ----- speaker level -----
    spk_true, spk_pred, spk_prob = [], [], []
    spk_to_idx = defaultdict(list)
    for i, s in enumerate(seg_spk_ids):
        spk_to_idx[s].append(i)

    for s, idxs in spk_to_idx.items():
        ys = [seg_y_true[i] for i in idxs]
        ps = [seg_prob[i] for i in idxs]
        preds = [seg_y_pred[i] for i in idxs]

        spk_true.append(ys[0])  # all same within speaker
        counts = np.bincount(preds, minlength=NUM_CLASSES)
        spk_pred.append(np.argmax(counts))
        spk_prob.append(np.mean(ps))

    spk_metrics = compute_metrics(spk_true, spk_pred, spk_prob)
    print_metrics("Speaker", spk_metrics)

    return seg_metrics, utt_metrics, spk_metrics


def summarize(level_name, arr):
    arr = np.array(arr, dtype=float)   # [folds, 7]
    names = ["WA", "UA", "Prec", "Rec", "F1", "WF1", "AUC"]
    print(f"\n---- {level_name} 5-Fold Avg ----")
    for i, n in enumerate(names):
        mean = np.nanmean(arr[:, i])
        std  = np.nanstd(arr[:, i])
        print(f"{n:4}: {mean:.3f} ± {std:.3f}")


# ======================= MAIN ========================

def main():
    meta = pd.read_csv(META_CSV)
    print("Loaded metadata:", meta.shape)

    models = {
       # "DepAudioNet" : (DepAudioNet, FBANK_BINS),
        #"SpeechFormer": (SpeechFormer, SPEC_BINS),
        "LightSERNet" : (LightSERNet, MFCC_DIM),
    }

    for model_name, (model_cls, feat_bins) in models.items():
        print("\n" + "="*60)
        print(f"          Running 5-fold CV for {model_name}")
        print("="*60)

        seg_all, utt_all, spk_all = [], [], []

        for fold_id in range(1, N_FOLDS+1):
            df_test  = meta[meta["fold"] == fold_id]
            df_train = meta[meta["fold"] != fold_id]

            seg_m, utt_m, spk_m = run_fold(
                model_cls=model_cls,
                model_name=model_name,
                feat_bins=feat_bins,
                df_train=df_train,
                df_test=df_test,
                fold_id=fold_id
            )
            seg_all.append(seg_m)
            utt_all.append(utt_m)
            spk_all.append(spk_m)

        print(f"\n########## {model_name} SUMMARY ##########")
        summarize("Segment", seg_all)
        summarize("Utterance", utt_all)
        summarize("Speaker", spk_all)
        print("##########################################\n")


if __name__ == "__main__":
    main()
