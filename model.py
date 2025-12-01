import torch
import torch.nn as nn
import torchvision.models as models

class Mymodel(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(Mymodel, self).__init__()

        if backbone == 'resnet18':
            self.model = models.resnet18(weights=None)
        elif backbone == 'resnet34':
            self.model = models.resnet34(weights=None)
        else:
            self.model = models.resnet50(weights=None)

        # Replace the final classification layer for 5 classes (KL grades)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 5)

    def forward(self, x):
        return self.model(x)
