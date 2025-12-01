import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from args import get_args
from model import Mymodel


@torch.no_grad()
def evaluate_recall():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧪 Using device: {device}")

    # Load the test CSV
    test_csv = os.path.join(args.csv_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"❌ test.csv not found in {args.csv_dir}")
    test_df = pd.read_csv(test_csv)

    # Image transformation (same as training input)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Model checkpoints directory
    session_dir = args.out_dir  # should be "session"
    model_paths = sorted([
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.startswith("best_model_fold_") and f.endswith(".pth")
    ])

    if not model_paths:
        raise FileNotFoundError(f"❌ No trained model found in {session_dir}")

    print(f"🔍 Found {len(model_paths)} trained models:\n{model_paths}\n")

    results = []

    # Iterate through each fold model
    for fold, model_path in enumerate(model_paths):
        print(f"📂 Evaluating {model_path} ...")

        # Load trained model
        model = Mymodel(args.backbone)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        # Loop through each test image
        for _, row in test_df.iterrows():
            img_path = row["Path"]
            label = int(row["KL"])

            if not os.path.exists(img_path):
                print(f"⚠️ Skipping missing file: {img_path}")
                continue

            # Load image
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            outputs = model(img)
            preds = torch.argmax(outputs, dim=1).item()

            all_preds.append(preds)
            all_labels.append(label)

        # Compute recall metrics
        macro = recall_score(all_labels, all_preds, average="macro")
        micro = recall_score(all_labels, all_preds, average="micro")
        weighted = recall_score(all_labels, all_preds, average="weighted")

        print(f"✅ Fold {fold}: Macro={macro:.4f}, Micro={micro:.4f}, Weighted={weighted:.4f}\n")

        results.append({
            "Fold": fold,
            "Macro Recall": macro,
            "Micro Recall": micro,
            "Weighted Recall": weighted
        })

    # Save results
    df_results = pd.DataFrame(results)
    avg_row = {
        "Fold": "Average",
        "Macro Recall": df_results["Macro Recall"].mean(),
        "Micro Recall": df_results["Micro Recall"].mean(),
        "Weighted Recall": df_results["Weighted Recall"].mean(),
    }
    df_results = pd.concat([df_results, pd.DataFrame([avg_row])])
    df_results.to_csv("recall_results.csv", index=False)
    print("💾 Saved recall metrics → recall_results.csv")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(df_results["Fold"][:-1], df_results["Macro Recall"][:-1], marker="o", label="Macro Recall")
    plt.plot(df_results["Fold"][:-1], df_results["Micro Recall"][:-1], marker="o", label="Micro Recall")
    plt.plot(df_results["Fold"][:-1], df_results["Weighted Recall"][:-1], marker="o", label="Weighted Recall")
    plt.title("Recall Metrics per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("recall_plot.png")
    print("📊 Saved recall plot → recall_plot.png")

    print("\n🎯 Evaluation complete for all folds.")


if __name__ == "__main__":
    evaluate_recall()
