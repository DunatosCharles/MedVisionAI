import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BreastCancerResNet(nn.Module):

    def __init__(self, pretrained=True, freeze_early_layers=True):

        super().__init__()

        # ImageNet weights are only needed when training; inference
        # loads a checkpoint, so skip the download there
        self.model = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Preserve pretrained knowledge by averaging RGB filters
        old_conv = self.model.conv1

        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(
                dim=1,
                keepdim=True
            )

        self.model.conv1 = new_conv

        # Freeze the stem and layer1, fine tune deeper layers
        if freeze_early_layers:

            for param in self.model.conv1.parameters():
                param.requires_grad = False

            for param in self.model.bn1.parameters():
                param.requires_grad = False

            for param in self.model.layer1.parameters():
                param.requires_grad = False


        # Replace classifier
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(
                self.model.fc.in_features,
                2
            )
        )

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)