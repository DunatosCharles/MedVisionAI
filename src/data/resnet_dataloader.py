from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.data.busi_dataset import BUSIDataset
from src.data.transforms import get_train_transform, get_eval_transform


BUSI_ROOT = "datasets/busi/Dataset_BUSI_with_GT"


def get_busi_split_indices(labels, seed=42):

    indices = list(range(len(labels)))

    # 70 / 15 / 15 split, stratified so every split keeps the
    # benign / malignant ratio
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=0.3,
        stratify=labels,
        random_state=seed
    )

    holdout_labels = [labels[i] for i in holdout_indices]

    val_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=0.5,
        stratify=holdout_labels,
        random_state=seed
    )

    return train_indices, val_indices, test_indices


def get_resnet_dataloaders(
    dataset="busi",
    batch_size=32,
    num_workers=0,
    seed=42
):

    train_transform = get_train_transform()
    test_transform = get_eval_transform()


    if dataset.lower() == "breastmnist":

        from medmnist import BreastMNIST

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

        train_dataset = BUSIDataset(
            root_dir=BUSI_ROOT,
            transform=train_transform
        )

        eval_dataset = BUSIDataset(
            root_dir=BUSI_ROOT,
            transform=test_transform
        )


        train_indices, val_indices, test_indices = get_busi_split_indices(
            train_dataset.labels,
            seed=seed
        )


        train_dataset = Subset(
            train_dataset,
            train_indices
        )

        val_dataset = Subset(
            eval_dataset,
            val_indices
        )

        test_dataset = Subset(
            eval_dataset,
            test_indices
        )


    else:

        raise ValueError(
            f"Unknown dataset: {dataset}"
        )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )


    return train_loader, val_loader, test_loader
