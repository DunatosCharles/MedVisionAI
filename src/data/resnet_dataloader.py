import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from medmnist import BreastMNIST

from src.data.busi_dataset import BUSIDataset


def get_resnet_dataloaders(
    dataset="breastmnist",
    batch_size=32
):

    train_transform = transforms.Compose([

        transforms.Resize(
            (224,224)
        ),

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05,0.05)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.4829],
            std=[0.229]
        )
    ])


    test_transform = transforms.Compose([

        transforms.Resize(
            (224,224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.4829],
            std=[0.229]
        )
    ])


    if dataset.lower() == "breastmnist":

        train_dataset = BreastMNIST(
            split="train",
            download=True,
            transform=train_transform
        )


        val_dataset = BreastMNIST(
            split="val",
            download=True,
            transform=test_transform
        )


        test_dataset = BreastMNIST(
            split="test",
            download=True,
            transform=test_transform
        )


    elif dataset.lower() == "busi":

        full_dataset = BUSIDataset(
            root_dir="datasets/busi/Dataset_BUSI_with_GT",
            transform=None
        )


        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size


        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            range(len(full_dataset)),
            [
                train_size,
                val_size,
                test_size
            ],
            generator=torch.Generator().manual_seed(42)
        )


        train_dataset = BUSIDataset(
            root_dir="datasets/busi/Dataset_BUSI_with_GT",
            transform=train_transform
        )

        val_dataset = BUSIDataset(
            root_dir="datasets/busi/Dataset_BUSI_with_GT",
            transform=test_transform
        )

        test_dataset = BUSIDataset(
            root_dir="datasets/busi/Dataset_BUSI_with_GT",
            transform=test_transform
        )


        train_dataset = torch.utils.data.Subset(
            train_dataset,
            train_indices.indices
        )

        val_dataset = torch.utils.data.Subset(
            val_dataset,
            val_indices.indices
        )

        test_dataset = torch.utils.data.Subset(
            test_dataset,
            test_indices.indices
        )


    else:

        raise ValueError(
            f"Unknown dataset: {dataset}"
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