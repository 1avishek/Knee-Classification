import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description="Knee X-ray Classification Training Arguments")

    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "small_cnn"])

    parser.add_argument("--csv_dir", type=str, default="CSVs",
                        help="Path to folder containing all CSV folds")
    parser.add_argument("--img_root", type=str, default=None,
                        help="Optional root to prepend to relative image paths in CSVs")

    parser.add_argument('--batch_size', '-bs', type=int, default=32,
                        choices=[8, 16, 32, 64, 128])

    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)

    parser.add_argument('--out_dir', type=str, default="session",
                        help="Directory to save checkpoints and plots")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for training: "auto", "cuda", or "cpu"')
    parser.add_argument('--pretrained', action='store_true', help='Use ImageNet-pretrained backbone')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--pin_memory', action='store_true', help='Pin dataloader memory (GPU training)')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision (CUDA only)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_best', action='store_true', help='Save best model based on validation accuracy')

    return parser.parse_args()
