import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# ====== CONFIG ======
SEED = 42
RUNS = 2
BASE = "/home/bubai-maji/bubai/English/edic_features_npy/IS10"


np.random.seed(SEED)

def load(split):
    X = np.load(f"{BASE}/{split}/segment_X.npy")
    y = np.load(f"{BASE}/{split}/segment_y.npy").astype(int)
    spk = np.load(f"{BASE}/{split}/segment_speaker_id.npy")
    y[y > 1] = 1
    return X, y, spk

X_tr_all, y_tr_all, spk_tr_all = load("train")
X_te_all, y_te_all, spk_te_all = load("test")

# Normalize
scaler = StandardScaler().fit(X_tr_all)
X_tr_all = scaler.transform(X_tr_all)
X_te_all = scaler.transform(X_te_all)

models = {
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced'),
    "RF": RandomForestClassifier(n_estimators=400, class_weight='balanced')
}

results = {
    "SVM": {"acc":[], "f1":[], "wf1":[], "auc":[], "ua":[]},
    "RF":  {"acc":[], "f1":[], "wf1":[], "auc":[], "ua":[]}
}

for model_name, model in models.items():
    print(f"\n=========== {model_name} Evaluations ===========")

    for r in range(RUNS):
        print(f"\nRun {r+1}/{RUNS}")

        # shuffle introduces randomness for STD
        X_tr, y_tr = shuffle(X_tr_all, y_tr_all, random_state=SEED+r)

        # Train
        model.fit(X_tr, y_tr)

        # Predict probabilities
        seg_prob = model.predict_proba(X_te_all)[:,1]
        seg_y = y_te_all
        seg_spk = spk_te_all

        # Speaker-level aggregation
        spk_prob = []
        spk_true = []
        for s in np.unique(seg_spk):
            idx = np.where(seg_spk == s)[0]
            spk_prob.append(seg_prob[idx].mean())
            spk_true.append(seg_y[idx][0])
        spk_prob = np.array(spk_prob)
        spk_true = np.array(spk_true)

        # Optimal threshold
        fpr, tpr, th = roc_curve(spk_true, spk_prob)
        best_th = th[np.argmax(tpr - fpr)]
        spk_pred = (spk_prob >= best_th).astype(int)

        acc  = accuracy_score(spk_true, spk_pred)
        f1   = f1_score(spk_true, spk_pred)
        wf1  = f1_score(spk_true, spk_pred, average="weighted")
        auc  = roc_auc_score(spk_true, spk_prob)
        tn, fp, fn, tp = confusion_matrix(spk_true, spk_pred).ravel()
        ua   = 0.5*((tp/(tp+fn+1e-6)) + (tn/(tn+fp+1e-6)))

        print(f"Acc={acc:.3f}, F1={f1:.3f}, WF1={wf1:.3f}, AUC={auc:.3f}, UA={ua:.3f}")

        results[model_name]["acc"].append(acc)
        results[model_name]["f1"].append(f1)
        results[model_name]["wf1"].append(wf1)
        results[model_name]["auc"].append(auc)
        results[model_name]["ua"].append(ua)

# ===== FINAL RESULTS =====
for name in results:
    print(f"\n===== {name} Final (Speaker-Level) =====")
    for m,v in results[name].items():
        print(f"{m.upper():5s}: {np.mean(v):.3f} ± {np.std(v):.3f}")
