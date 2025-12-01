import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


def build_splits(main_dir, csv_dir):
    metadata_path = os.path.join(main_dir, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    train_val, test = train_test_split(
        metadata,
        test_size=0.2,
        stratify=metadata["KL"],
        random_state=42
    )

    os.makedirs(csv_dir, exist_ok=True)

    test_path = os.path.join(csv_dir, "test.csv")
    test.to_csv(test_path, index=False)
    print(f"Saved test set: {test_path}")

    plt.figure()
    sns.countplot(x="KL", data=metadata)
    plt.title("KL Grade Distribution - Entire Dataset")
    plt.savefig(os.path.join(csv_dir, "kl_distribution_entire.png"))

    plt.figure()
    sns.countplot(x="KL", data=test)
    plt.title("KL Grade Distribution - Test Set")
    plt.savefig(os.path.join(csv_dir, "kl_distribution_test.png"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val, train_val["KL"])):
        train_fold = train_val.iloc[train_idx]
        val_fold = train_val.iloc[val_idx]

        train_filename = os.path.join(csv_dir, f"fold_{fold}_train.csv")
        val_filename = os.path.join(csv_dir, f"fold_{fold}_val.csv")

        train_fold.to_csv(train_filename, index=False)
        val_fold.to_csv(val_filename, index=False)

        print(f"Saved: {train_filename} and {val_filename}")

        plt.figure()
        sns.countplot(x="KL", data=train_fold)
        plt.title(f"KL Grade Distribution - Train Fold {fold}")
        plt.savefig(os.path.join(csv_dir, f"kl_distribution_train_fold_{fold}.png"))

        plt.figure()
        sns.countplot(x="KL", data=val_fold)
        plt.title(f"KL Grade Distribution - Validation Fold {fold}")
        plt.savefig(os.path.join(csv_dir, f"kl_distribution_val_fold_{fold}.png"))

    print("\nAll folds and test set have been generated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CSV splits for knee X-ray classification")
    parser.add_argument("--data_root", type=str, default=os.getcwd(),
                        help="Root directory containing metadata.csv and images")
    parser.add_argument("--csv_dir", type=str, default=None,
                        help="Output directory for CSV splits (default: <data_root>/CSVs)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_dir = os.path.abspath(args.data_root)
    csv_dir = args.csv_dir or os.path.join(main_dir, "CSVs")
    build_splits(main_dir, csv_dir)
