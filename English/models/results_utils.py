# ============================
# results_utils.py (FIXED)
# ============================
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.manifold import TSNE

# ---------------------------------------
# Speaker aggregation (same)
# ---------------------------------------
def speaker_aggregate(y_true_seg, y_prob_seg, spk_ids_seg, threshold=0.5):
    y_true_seg = np.asarray(y_true_seg)
    y_prob_seg = np.asarray(y_prob_seg)
    spk_ids_seg = np.asarray(spk_ids_seg)

    speakers = np.unique(spk_ids_seg)
    y_true_spk, y_prob_spk = [], []

    for s in speakers:
        idx = np.where(spk_ids_seg == s)[0]
        y_true_spk.append(int(y_true_seg[idx][0]))
        y_prob_spk.append(float(np.mean(y_prob_seg[idx])))

    y_true_spk = np.array(y_true_spk)
    y_prob_spk = np.array(y_prob_spk)

    y_pred_spk = (y_prob_spk >= threshold).astype(int)
    return y_true_spk, y_prob_spk, y_pred_spk, speakers

# ---------------------------------------
# Confusion Matrix (same)
# ---------------------------------------
def plot_confusion_speaker_counts(y_true_spk, y_pred_spk, savepath=None):
    cm = confusion_matrix(y_true_spk, y_pred_spk)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy","Depressed"])
    fig,ax = plt.subplots(figsize=(4.8,4.2))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (Speaker-level)")
    fig.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=400, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------
# ROC PLOT with mean ± std across runs
# ---------------------------------------
def plot_roc_avg(y_true_runs, y_prob_runs, save):
    """
    y_true_runs: list of arrays (speaker labels per run)
    y_prob_runs: list of arrays (speaker probabilities per run)
    """
    assert len(y_true_runs) == len(y_prob_runs)
    N = len(y_true_runs)

    fprs = []
    tprs = []
    aucs = []

    # compute ROC per run
    for i in range(N):
        fpr, tpr, _ = roc_curve(y_true_runs[i], y_prob_runs[i])
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc(fpr, tpr))

    # interpolate to average ROC
    mean_fpr = np.linspace(0,1,100)
    tpr_interp = []
    for i in range(N):
        tpr_interp.append(np.interp(mean_fpr, fprs[i], tprs[i]))
    mean_tpr = np.mean(tpr_interp, axis=0)
    std_tpr = np.std(tpr_interp, axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.figure(figsize=(5,4))

    # shade std band
    plt.fill_between(mean_fpr, 
                     mean_tpr-std_tpr, 
                     mean_tpr+std_tpr, 
                     alpha=0.2, color="blue")

    # mean ROC
    plt.plot(mean_fpr, mean_tpr, color="blue",
             label=f"AUC = {mean_auc:.3f} ± {std_auc:.3f}", linewidth=2)

    # diagonal
    plt.plot([0,1],[0,1],"--",color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC (Speaker‑level, Averaged)")
    plt.legend()

    os.makedirs(os.path.dirname(save), exist_ok=True)
    plt.savefig(save, dpi=400, bbox_inches="tight")
    plt.close()

# ---------------------------------------
# t‑SNE (safe)
# ---------------------------------------
def plot_tsne(X_raw, Z, y, out):
    y = np.asarray(y)
    colors = np.where(y==0,"teal","darkorange")

    # Avoid TSNE crash when single sample
    if X_raw.shape[0] < 3:
        print("Skipping t‑SNE — not enough samples.")
        return

    tsne = TSNE(n_components=2, perplexity=min(30, X_raw.shape[0]-1), random_state=42)
    Xr = tsne.fit_transform(X_raw)

    tsne2 = TSNE(n_components=2, perplexity=min(30, Z.shape[0]-1), random_state=42)
    Zr = tsne2.fit_transform(Z)

    fig,ax = plt.subplots(1,2,figsize=(9,4))
    ax[0].scatter(Xr[:,0],Xr[:,1],c=colors,s=10)
    ax[0].set_title("Input Space")
    ax[1].scatter(Zr[:,0],Zr[:,1],c=colors,s=10)
    ax[1].set_title("Fusion Latent Space")

    os.makedirs(out,exist_ok=True)
    plt.savefig(f"{out}/tsne.png",dpi=350,bbox_inches="tight")
    plt.close()


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_stacked_attention_bars(attention_weights, labels, save_path="stacked_attention_bars.png"):
    """
    Plot horizontal stacked color bars for average attention scores of Healthy and Depressed classes.
    """

    attention_weights = np.array(attention_weights)
    labels = np.array(labels)

    healthy_avg = attention_weights[labels == 0].mean(axis=0)
    depressed_avg = attention_weights[labels == 1].mean(axis=0)

    data = np.stack([healthy_avg, depressed_avg])
    class_names = ["Healthy", "Depressed"]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 1.5))

    norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = cm.get_cmap('Blues')

    for i, row in enumerate(data):
        x_start = 0
        for val in row:
            width = 1
            color = cmap(norm(val))
            ax.barh(i, width=width, left=x_start, height=0.8, color=color, edgecolor='none')
            x_start += width

    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xticks([])
    ax.set_xlim(0, data.shape[1])
    ax.set_frame_on(False)

    # Add colorbar using the correct axis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.3)
    cbar.set_label("Attention Score")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

