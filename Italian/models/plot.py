import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
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
    #plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_three_model_roc(y_true,
                         prob_is10,
                         prob_pretrained,
                         prob_fusion,
                         save_path):

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # ---- Compute ROC curves ----
    fpr_is10, tpr_is10, _ = roc_curve(y_true, prob_is10)
    fpr_pre,  tpr_pre,  _ = roc_curve(y_true, prob_pretrained)
    fpr_fus,  tpr_fus,  _ = roc_curve(y_true, prob_fusion)

    auc_is10 = auc(fpr_is10, tpr_is10)
    auc_pre  = auc(fpr_pre,  tpr_pre)
    auc_fus  = auc(fpr_fus,  tpr_fus)

    # ---- Create plot ----
    plt.figure(figsize=(6,5))

    # Plot all models (matching your example)
    plt.plot(fpr_is10, tpr_is10,
             label=f"IS10 (AUC={auc_is10:.2f})",
             color="blue", linewidth=2.5)

    plt.plot(fpr_pre, tpr_pre,
             label=f"Pretrained (AUC={auc_pre:.2f})",
             color="red", linewidth=2.5)

    plt.plot(fpr_fus, tpr_fus,
             label=f"Fusion (AUC={auc_fus:.2f})",
             color="gold", linewidth=2.5)

    # Random chance line
    plt.plot([0,1], [0,1], 'k--', linewidth=1)

    # Style
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves: IS10 vs Pretrained vs Fusion", fontsize=14)

    plt.legend(
        loc="lower right",
        fontsize=10,
        frameon=True,
        facecolor="white",
        edgecolor="gray"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
