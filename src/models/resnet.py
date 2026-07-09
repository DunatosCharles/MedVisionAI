import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BreastCancerResNet(nn.Module):

    def __init__(self):

        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)

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

        # Freeze pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False


        # Fine tune deeper layers
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True


        # Replace classifier
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

    def forward(self, x):
        return self.model(x)