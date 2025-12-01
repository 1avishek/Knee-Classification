import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_path(path, img_root=None):
    if os.path.isabs(path):
        return path
    if img_root:
        return os.path.join(img_root, path)
    return path


def build_transforms(train=False):
    if train:
        t = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    else:
        t = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    return transforms.Compose(t)


class KneeXrayDataset(Dataset):
    def __init__(self, dataset, img_root=None, train=False):
        self.dataset = dataset
        self.img_root = img_root
        self.transform = build_transforms(train=train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        path = resolve_path(self.dataset['Path'].iloc[index], self.img_root)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        label = self.dataset['KL'].iloc[index]
        img = self.transform(img)
        label = torch.tensor(label, dtype=torch.long)

        return {'img': img, 'label': label}
