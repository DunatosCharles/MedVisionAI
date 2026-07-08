import torch
from torch.utils.data import DataLoader
from medmnist import BreastMNIST
from torchvision import transforms


def get_dataloaders(batch_size=32):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

    train_dataset = BreastMNIST(
        split="train",
        download=True,
        transform=transform
    )

    val_dataset = BreastMNIST(
        split="val",
        download=True,
        transform=transform
    )

    test_dataset = BreastMNIST(
        split="test",
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader