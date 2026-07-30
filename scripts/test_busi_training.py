import torch

from src.models.resnet import BreastCancerResNet
from src.data.resnet_dataloader import get_resnet_dataloaders


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


device = get_device()

print("Device:", device)


train_loader, val_loader, test_loader = get_resnet_dataloaders(
    dataset="busi",
    batch_size=8
)


model = BreastCancerResNet().to(device)


images, labels = next(iter(train_loader))

images = images.to(device)
labels = labels.to(device)


outputs = model(images)


print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Labels:", labels)


print("Training pipeline ready!")