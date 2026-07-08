from torch.utils.data import DataLoader
from medmnist import BreastMNIST
from torchvision import transforms


def get_dataloaders(batch_size=32):

    train_transform = transforms.Compose([

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])


    test_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])


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