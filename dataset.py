import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    xray = cv2.cvtColor(xray, cv2.COLOR_GRAY2RGB)  # convert grayscale → RGB
    return xray

class KneeXrayDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img = read_xray(self.dataset['Path'].iloc[index])
        label = self.dataset['KL'].iloc[index]

        img = self.transform(img)  # apply transform
        label = torch.tensor(label, dtype=torch.long)

        res = {'img': img, 'label': label}

        return res
