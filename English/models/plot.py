import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import seaborn as sns

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(3.5,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Depression"],
                yticklabels=["Healthy", "Depression"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_auc_curve_folds(fpr, tpr, auc_score, save_path):
    plt.figure(figsize=(6,5))
    mean_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.interp(mean_fpr, fpr, tpr)

    plt.plot(fpr, tpr, color="gray", alpha=0.5)
    plt.plot(mean_fpr, mean_tpr, color="blue", linewidth=3,
             label=f"AUC={auc_score:.2f}")
    plt.plot([0,1],[0,1],"--",color="black")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# put at top of plot.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_tsne(initial_emb, final_emb, labels, save_path,
              pca_dim=50, tsne_perplexity=30, random_state=42):
    """
    Robust t-SNE plotting when initial_emb and final_emb have different dims.

    initial_emb : (N, d1) numpy array
    final_emb   : (N, d2) numpy array
    labels      : (N,)   numpy array with integer class labels (0/1)
    save_path   : path to save plot

    pca_dim     : desired PCA dim before t-SNE (will be clipped to <= N-1)
    tsne_perplexity : perplexity for TSNE (will be clipped to be < N)
    """

    # -- Basic checks
    initial_emb = np.asarray(initial_emb)
    final_emb   = np.asarray(final_emb)
    labels = np.asarray(labels)

    if initial_emb.ndim != 2 or final_emb.ndim != 2:
        raise ValueError("initial_emb and final_emb must be 2D arrays (N, D)")

    if initial_emb.shape[0] != final_emb.shape[0] or initial_emb.shape[0] != labels.shape[0]:
        raise ValueError("initial_emb, final_emb and labels must have same number of rows (N)")

    N = initial_emb.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 samples for t-SNE")

    # -- Choose safe PCA dimension (must be <= N-1 and <= each feature dim)
    max_pca_allowed = max(2, min(N - 1, initial_emb.shape[1], final_emb.shape[1]))
    pca_dim = int(min(pca_dim, max_pca_allowed))
    if pca_dim < 2:
        pca_dim = 2

    # -- Scale embeddings separately (helps PCA/t-SNE)
    sc1 = StandardScaler()
    sc2 = StandardScaler()
    init_scaled = sc1.fit_transform(initial_emb)
    final_scaled = sc2.fit_transform(final_emb)

    # -- PCA reduce each to same dim
    pca1 = PCA(n_components=pca_dim, random_state=random_state)
    pca2 = PCA(n_components=pca_dim, random_state=random_state)

    try:
        Z1 = pca1.fit_transform(init_scaled)
    except Exception:
        # fallback: if PCA fails (rare), use the first pca_dim columns
        Z1 = init_scaled[:, :pca_dim]

    try:
        Z2 = pca2.fit_transform(final_scaled)
    except Exception:
        Z2 = final_scaled[:, :pca_dim]

    # -- Combine and run one TSNE so both share coordinates
    all_emb = np.vstack([Z1, Z2])   # shape = (2N, pca_dim)

    # safe perplexity: must be < N (perplexity rule of thumb)
    safe_perp = int(min(tsne_perplexity, max(2, (N // 3))))
    if safe_perp < 2:
        safe_perp = 2

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=safe_perp)
    all_Z = tsne.fit_transform(all_emb)

    Z_init = all_Z[:N]
    Z_final = all_Z[N:]

    # -- Plot
    plt.figure(figsize=(6, 3))
    class_names = {0: "Healthy", 1: "Depression"}
    colors = {0: "blue", 1: "red"}

    ax1 = plt.subplot(1, 2, 1)
    for c in np.unique(labels):
        m = (labels == c)
        ax1.scatter(Z_init[m, 0], Z_init[m, 1], s=10, c=colors.get(int(c), "gray"),
                    label=class_names.get(int(c), str(c)))
    ax1.set_title("Initial Representation")
    ax1.axis("off")
    ax1.legend(fontsize=6, loc="best")

    ax2 = plt.subplot(1, 2, 2)
    for c in np.unique(labels):
        m = (labels == c)
        ax2.scatter(Z_final[m, 0], Z_final[m, 1], s=10, c=colors.get(int(c), "gray"),
                    label=class_names.get(int(c), str(c)))
    ax2.set_title("Learned Representation")
    ax2.axis("off")
    ax2.legend(fontsize=6, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
