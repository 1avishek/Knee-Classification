import torch
import torch.nn as nn
import torchvision.models as models


class Mymodel(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=False):
        super(Mymodel, self).__init__()

        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet18(weights=weights)
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.resnet34(weights=weights)
        else:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.model = models.resnet50(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 5)

    def forward(self, x):
        return self.model(x)
