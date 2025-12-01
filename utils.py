import os
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_metrics_plots(history, out_dir, fold):
    """history is a dict with keys: train_loss, val_loss, train_bal, val_bal, train_roc, val_roc, train_ap, val_ap"""
    ensure_dir(out_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title(f"Loss - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"loss_fold_{fold}.png"))
    plt.close()

    # Balanced Accuracy
    plt.figure()
    plt.plot(epochs, history["train_bal"], label="Train Balanced Acc")
    plt.plot(epochs, history["val_bal"], label="Val Balanced Acc")
    plt.title(f"Balanced Accuracy - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"balacc_fold_{fold}.png"))
    plt.close()

    # ROC-AUC
    plt.figure()
    plt.plot(epochs, history["train_roc"], label="Train ROC-AUC")
    plt.plot(epochs, history["val_roc"], label="Val ROC-AUC")
    plt.title(f"ROC-AUC - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"rocauc_fold_{fold}.png"))
    plt.close()

    # Average Precision
    plt.figure()
    plt.plot(epochs, history["train_ap"], label="Train Average Precision")
    plt.plot(epochs, history["val_ap"], label="Val Average Precision")
    plt.title(f"Average Precision - Fold {fold}")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"avgprecision_fold_{fold}.png"))
    plt.close()
