import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description="Knee X-ray Classification Training Arguments")

    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "small_cnn"])

    parser.add_argument("--csv_dir", type=str,
                        default=r"D:\ass_vscode\Knee Classification\CSVs",
                        help="Path to folder containing all CSV folds")

    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        choices=[8, 16, 32, 64, 128])

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)

    parser.add_argument('--out_dir', type=str,
                        default=r"D:\ass_vscode\Knee Classification\session")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for training')
    parser.add_argument('--save_best', action='store_true', help='Save best model based on validation accuracy')


    return parser.parse_args()


