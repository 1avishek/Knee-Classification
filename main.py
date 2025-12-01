from args import get_args
import os
import torch
import pandas as pd
from dataset import KneeXrayDataset
from torch.utils.data import DataLoader
from model import Mymodel
from trainer import train_and_validate  


def main():
    # get arguments
    args = get_args()

    # iterate among the folds
    for fold in range(5):
        print(f"training fold: {fold}\n")

        # load train and validation CSVs
        train_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_train.csv'))
        val_set = pd.read_csv(os.path.join(args.csv_dir, f'fold_{fold}_val.csv'))

        # prepare datasets
        train_dataset = KneeXrayDataset(train_set)
        val_dataset = KneeXrayDataset(val_set)

        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # initialize model
        model = Mymodel(args.backbone)

        # train and validate this fold
        train_and_validate(model, train_loader, val_loader, args, fold)


if __name__ == '__main__':
    main()
