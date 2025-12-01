from args import get_args
import os
import torch
import pandas as pd
from dataset import KneeXrayDataset
from torch.utils.data import DataLoader
from model import Mymodel
from trainer import train_and_validate  
import random
import numpy as np


def main():
    # get arguments
    args = get_args()

    # set seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # iterate among the folds
    for fold in range(5):
        print(f"training fold: {fold}\n")

        # load train and validation CSVs
        train_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_train.csv'))
        val_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_val.csv'))

        # prepare datasets
        train_dataset = KneeXrayDataset(train_set, img_root=args.img_root, train=True)
        val_dataset = KneeXrayDataset(val_set, img_root=args.img_root, train=False)

        # create dataloaders
        common_loader_kwargs = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)

        # initialize model
        model = Mymodel(args.backbone, pretrained=args.pretrained)

        # train and validate this fold
        train_and_validate(model, train_loader, val_loader, args, fold)


if __name__ == '__main__':
    main()
