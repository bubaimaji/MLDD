import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ===================== CONFIG =====================
TRAIN_DIR = "/home/bubai-maji/bubai/English/layerwise_features/wavlm-large-layer/train"
DEV_DIR   = "/home/bubai-maji/bubai/English/layerwise_features/wavlm-large-layer/dev"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 4e-4
EPOCHS = 60
EMB_DIM = 256   # output of FC projection
C_SVM = 10

# ================== LOAD LABELS ==================
y_train = np.load(os.path.join(TRAIN_DIR, "speaker_y.npy"))
y_dev   = np.load(os.path.join(DEV_DIR,   "speaker_y.npy"))

# ================== GET LAYERS ==================
layer_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.startswith("layer")])
N_LAYERS = len(layer_files)

print(f"Detected {N_LAYERS} layers.")


# ================== FC PROJECTOR ==================
class FCProjector(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        return self.fc(x)


def train_fc(model, X, y):
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for ep in range(EPOCHS):
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total += loss.item()


def extract_fc_features(model, X):
    model.eval()
    with torch.no_grad():
        Z = model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return Z


# ================== RUN LAYER-WISE ==================
print("\n===== LAYER-WISE RESULTS =====\n")

for idx, fname in enumerate(layer_files):

    # load layer X
    X_train = np.load(os.path.join(TRAIN_DIR, fname))
    X_dev   = np.load(os.path.join(DEV_DIR,   fname))

    FEAT_DIM = X_train.shape[1]

    # Build projector
    model = FCProjector(FEAT_DIM, EMB_DIM).to(DEVICE)

    # Train FC
    train_fc(model, X_train, y_train)

    # Extract embeddings
    Z_train = extract_fc_features(model, X_train)
    Z_dev   = extract_fc_features(model, X_dev)

    # Normalize
    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_dev_s   = scaler.transform(Z_dev)

    # SVM
    svm = SVC(kernel="rbf", C=C_SVM)
    svm.fit(Z_train_s, y_train)
    pred = svm.predict(Z_dev_s)

    acc = accuracy_score(y_dev, pred)
    f1  = f1_score(y_dev, pred)

    print(f"Layer {idx:02d} | {fname} | ACC={acc:.3f} | F1={f1:.3f}")

print("\n======= DONE =======\n")
