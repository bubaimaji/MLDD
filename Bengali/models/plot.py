import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
import seaborn as sns


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4,3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Depression"],
                yticklabels=["Healthy", "Depression"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    #plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc(y_true, prob_fusion, save_path):
    fpr_fus,  tpr_fus,  _ = roc_curve(y_true, prob_fusion)
    auc_fus  = auc(fpr_fus,  tpr_fus)

    # ---- Create plot ----
    plt.figure(figsize=(3.5,3))

    plt.plot(fpr_fus, tpr_fus,
             label=f"AUC={auc_fus:.2f}",
             color="gold", linewidth=2)

    # Random chance line
    plt.plot([0,1], [0,1], 'k--', linewidth=1)

    # Style
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    #plt.title("ROC Curves: IS10 vs Pretrained vs Fusion", fontsize=10)

    plt.legend(
        loc="lower right",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="gray"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
