import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# for grade in range(5): # folder_path = os.path.join(main_dir, str(grade)) # for image in os.listdir(folder_path): # if image.endswith(".png"): # image_path = os.path.join(folder_path, image) # metadata = metadata._append({"Name": image, "Path": image_path, "KL": grade}, ignore_index=True) # metadata.to_csv("metadata.csv", index=False)
# === CONFIGURATION ===
main_dir = r"D:\ass_vscode\Knee Classification"
csv_dir = os.path.join(main_dir, "CSVs")
os.makedirs(csv_dir, exist_ok=True)

# === LOAD METADATA ===
metadata_path = os.path.join(main_dir, "metadata.csv")
metadata = pd.read_csv(metadata_path)

# === TRAIN / TEST SPLIT ===
train_val, test = train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata["KL"],
    random_state=42
)

# Save test set
test_path = os.path.join(csv_dir, "test.csv")
test.to_csv(test_path, index=False)
print(f"✅ Saved test set: {test_path}")

# === PLOT KL DISTRIBUTION ===
plt.figure()
sns.countplot(x="KL", data=metadata)
plt.title("KL Grade Distribution - Entire Dataset")
plt.savefig(os.path.join(csv_dir, "kl_distribution_entire.png"))

plt.figure()
sns.countplot(x="KL", data=test)
plt.title("KL Grade Distribution - Test Set")
plt.savefig(os.path.join(csv_dir, "kl_distribution_test.png"))

# === STRATIFIED 5-FOLD SPLIT ON TRAIN/VAL ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_val, train_val["KL"])):
    train_fold = train_val.iloc[train_idx]
    val_fold = train_val.iloc[val_idx]

    # Define file names
    train_filename = os.path.join(csv_dir, f"fold_{fold}_train.csv")
    val_filename = os.path.join(csv_dir, f"fold_{fold}_val.csv")

    # Save CSVs
    train_fold.to_csv(train_filename, index=False)
    val_fold.to_csv(val_filename, index=False)

    print(f"✅ Saved: {train_filename} and {val_filename}")

    # Plot distributions
    plt.figure()
    sns.countplot(x="KL", data=train_fold)
    plt.title(f"KL Grade Distribution - Train Fold {fold}")
    plt.savefig(os.path.join(csv_dir, f"kl_distribution_train_fold_{fold}.png"))

    plt.figure()
    sns.countplot(x="KL", data=val_fold)
    plt.title(f"KL Grade Distribution - Validation Fold {fold}")
    plt.savefig(os.path.join(csv_dir, f"kl_distribution_val_fold_{fold}.png"))

print("\n🎯 All folds and test set have been successfully generated and saved.")
