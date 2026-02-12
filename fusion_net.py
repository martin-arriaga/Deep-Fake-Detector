import torch
import torch.nn as nn
from torchvision import models

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        self.spatial = models.efficientnet_b0(pretrained=True)
        self.spatial.classifier = nn.Identity()
        """
        Create our own CCN 
        """
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()

        )

        self.classifier = nn.Sequential(
            nn.Linear(1280 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        def forward(self, x, dct):
            features_x = self.spatial(x)
            features_xdct = self.frequency_branch(dct)
            concat = torch.cat((features_x, features_xdct), 1)
            return self.classifier(concat)


