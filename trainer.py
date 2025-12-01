import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import os


def train_and_validate(model, train_loader, val_loader, args, fold):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = model.to(device)

    # --- Define loss and optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    # --- Tracking ---
    num_epochs = args.epochs
    train_loss_list, val_loss_list = [], []
    train_bal_acc_list, val_bal_acc_list = [], []
    train_roc_auc_list, val_roc_auc_list = [], []
    train_ap_list, val_ap_list = [], []

    best_bal_acc = 0.0
    best_model_path = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        for batch in train_loader:
            imgs = batch["img"]
            labels = batch["label"].to(device)

            imgs = imgs.float().to(device)

            optimizer.zero_grad()
            with amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(imgs)
                loss = criterion(outputs, labels.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        train_loss_list.append(epoch_loss)

        # compute metrics
        bal_acc, roc_auc, avg_prec = calculate_metrics(all_labels, all_probs, all_preds)
        train_bal_acc_list.append(bal_acc)
        train_roc_auc_list.append(roc_auc)
        train_ap_list.append(avg_prec)

        # --- Validation Phase ---
        val_loss, val_bal_acc, val_roc_auc, val_ap = validate(model, val_loader, criterion, device)
        val_loss_list.append(val_loss)
        val_bal_acc_list.append(val_bal_acc)
        val_roc_auc_list.append(val_roc_auc)
        val_ap_list.append(val_ap)

        # Save best model based on Balanced Accuracy
        if val_bal_acc > best_bal_acc:
            best_bal_acc = val_bal_acc
            torch.save(model.state_dict(), best_model_path)

        print(f"Fold [{fold}] Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val BalAcc: {val_bal_acc:.4f} | ROC-AUC: {val_roc_auc:.4f} | AP: {val_ap:.4f}")

    # --- Plotting ---
    plot_metrics(
        fold, args.out_dir,
        train_loss_list, val_loss_list,
        train_bal_acc_list, val_bal_acc_list,
        train_roc_auc_list, val_roc_auc_list,
        train_ap_list, val_ap_list
    )


# ------------------------- Helper Functions ---------------------------- #

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["img"]
            labels = batch["label"].to(device)

            imgs = imgs.float().to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels.long())
            running_loss += loss.item() * imgs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.detach().cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    bal_acc, roc_auc, avg_prec = calculate_metrics(all_labels, all_probs, all_preds)
    return avg_loss, bal_acc, roc_auc, avg_prec


def calculate_metrics(labels, probs, preds):
    try:
        bal_acc = balanced_accuracy_score(labels, preds)
    except:
        bal_acc = np.nan

    try:
        roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    except:
        roc_auc = np.nan

    try:
        avg_prec = average_precision_score(labels, probs, average="macro")
    except:
        avg_prec = np.nan

    return bal_acc, roc_auc, avg_prec


def plot_metrics(fold, out_dir,
                 train_loss, val_loss,
                 train_bal_acc, val_bal_acc,
                 train_roc_auc, val_roc_auc,
                 train_ap, val_ap):
                 

    metrics = [
        ("Loss", train_loss, val_loss),
        ("Balanced Accuracy", train_bal_acc, val_bal_acc),
        ("ROC-AUC", train_roc_auc, val_roc_auc),
        ("Average Precision", train_ap, val_ap)
    ]

    for name, train_values, val_values in metrics:
        plt.figure()
        plt.plot(train_values, label=f"Train {name}")
        plt.plot(val_values, label=f"Val {name}")
        plt.title(f"{name} over Epochs - Fold {fold}")
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{name.replace(' ', '_').lower()}_fold_{fold}.png"))
        plt.close()
