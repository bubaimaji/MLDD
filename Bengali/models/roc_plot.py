# plot_roc_three.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import pathlib

UNIMOD_DIR = "/home/bubai-maji/bubai/Bangla/bn_results/unimodal_probs"
FUSION_DIR  = "/home/bubai-maji/bubai/Bangla/bn_results/fusion_probs"
OUT_DIR = "/home/bubai-maji/bubai/Bangla/bn_results"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

FOLDS = 5

ALL_IS10_PROBS = []
ALL_PRE_PROBS = []
ALL_FUSION_PROBS = []
ALL_LABELS = []

# load probs/labels
for f in range(1, FOLDS+1):
    is10_p = np.load(os.path.join(UNIMOD_DIR, f"is10_probs_fold{f}.npy"))
    pre_p  = np.load(os.path.join(UNIMOD_DIR, f"pre_probs_fold{f}.npy"))
    fus_p  = np.load(os.path.join(FUSION_DIR, f"fusion_probs_fold{f}.npy"))
    labels = np.load(os.path.join(FUSION_DIR, f"labels_fold{f}.npy"))

    ALL_IS10_PROBS.append(is10_p)
    ALL_PRE_PROBS.append(pre_p)
    ALL_FUSION_PROBS.append(fus_p)
    ALL_LABELS.append(labels)

    print(f"[Loaded fold {f}] speakers={len(labels)}")

# compute mean ROC
def compute_mean_roc(probs_folds, labels_folds):
    fold_fprs = []
    fold_tprs = []
    aucs = []
    for probs, labels in zip(probs_folds, labels_folds):
        fpr, tpr, _ = roc_curve(labels, probs)
        fold_fprs.append(fpr); fold_tprs.append(tpr)
        aucs.append(auc(fpr, tpr))
    mean_fpr = np.linspace(0, 1, 300)
    interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fold_fprs, fold_tprs)]
    interp_tprs = np.array(interp_tprs)
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    return mean_fpr, mean_tpr, np.mean(aucs), np.std(aucs)

mf_is10, mt_is10, auc_is10, std_is10 = compute_mean_roc(ALL_IS10_PROBS, ALL_LABELS)
mf_pre,  mt_pre,  auc_pre,  std_pre  = compute_mean_roc(ALL_PRE_PROBS,  ALL_LABELS)
mf_fus,  mt_fus,  auc_fus,  std_fus  = compute_mean_roc(ALL_FUSION_PROBS, ALL_LABELS)

# plot ROC
plt.figure(figsize=(7,6))
plt.plot(mf_is10, mt_is10, color='blue', linewidth=2.5, label=f"IS10 (AUC={auc_is10:.3f} ± {std_is10:.3f})")
plt.plot(mf_pre,  mt_pre,  color='red',  linewidth=2.5, label=f"Pretrained (AUC={auc_pre:.3f} ± {std_pre:.3f})")
plt.plot(mf_fus,  mt_fus,  color='gold', linewidth=2.5, label=f"Fusion (AUC={auc_fus:.3f} ± {std_fus:.3f})")
plt.plot([0,1],[0,1], 'k--', linewidth=1)
plt.xlim([0,1]); plt.ylim([0,1])
plt.xlabel("False Positive Rate", fontsize=14); plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curves: IS10 vs Pretrained vs Fusion (Mean 5-Fold)", fontsize=16)
plt.legend(loc='lower right', fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_three_models_mean.png"), dpi=400)
plt.close()
print("Saved ROC plot to:", os.path.join(OUT_DIR, "roc_three_models_mean.png"))

# optional: produce TSNEs if per-fold embeddings are available (from either unimodal or fusion outputs)
is10_emb_files = [os.path.join(FUSION_DIR, f"is10_emb_fold{f}.npy") for f in range(1, FOLDS+1)]
pre_emb_files  = [os.path.join(FUSION_DIR, f"pre_emb_fold{f}.npy")  for f in range(1, FOLDS+1)]
fus_emb_files  = [os.path.join(FUSION_DIR, f"fusion_emb_fold{f}.npy")  for f in range(1, FOLDS+1)]

if all(os.path.exists(p) for p in is10_emb_files + pre_emb_files + fus_emb_files):
    is10_all = np.vstack([np.load(p) for p in is10_emb_files])
    pre_all  = np.vstack([np.load(p) for p in pre_emb_files])
    fuse_all = np.vstack([np.load(p) for p in fus_emb_files])
    labels_all = np.hstack(ALL_LABELS)

    def tsne_save(emb, lab, title, outpath):
        z = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate='auto').fit_transform(emb)
        plt.figure(figsize=(4,3))
        for c, col, name in [(0, "#1f77b4", "Healthy"), (1, "#d62728", "Depression")]:
            mask = (lab == c)
            plt.scatter(z[mask,0], z[mask,1], s=12, alpha=0.7, color=col, label=name)
        plt.title(title); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(outpath, dpi=400); plt.close()

    tsne_save(is10_all, labels_all, "t-SNE IS10",  os.path.join(OUT_DIR, "tsne_is10.png"))
    tsne_save(pre_all,  labels_all, "t-SNE Pretrained", os.path.join(OUT_DIR, "tsne_pretrained.png"))
    tsne_save(fuse_all,  labels_all, "t-SNE Fusion", os.path.join(OUT_DIR, "tsne_fusion.png"))
    print("Saved TSNE plots to:", OUT_DIR)
else:
    print("Embeddings missing; skipping TSNE generation.")
