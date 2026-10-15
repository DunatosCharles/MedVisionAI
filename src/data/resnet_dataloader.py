import json
import os

import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold

from src.data.busi_dataset import BUSIDataset
from src.data.transforms import get_train_transform, get_eval_transform


BUSI_ROOT = "datasets/busi/Dataset_BUSI_with_GT"
BUSI_GROUPS_FILE = "data/busi_groups.json"


def load_busi_groups(image_paths):

    if not os.path.exists(BUSI_GROUPS_FILE):
        raise FileNotFoundError(
            f"{BUSI_GROUPS_FILE} not found. "
            "Run: python -m scripts.find_duplicates"
        )

    with open(BUSI_GROUPS_FILE) as f:
        data = json.load(f)

    names = [os.path.relpath(p, BUSI_ROOT) for p in image_paths]

    groups = np.array([data["groups"][name] for name in names])

    conflicting = set(data["conflicting_labels"])

    usable = np.array([name not in conflicting for name in names])

    return groups, usable


def group_split(indices, labels, groups, n_splits, fold, seed):

    # Stratified by label, but every image of a lesion stays on the same
    # side, so near-duplicate frames never leak across splits
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    splits = list(splitter.split(indices, labels[indices], groups[indices]))

    keep, held_out = splits[fold]

    return indices[keep], indices[held_out]


def get_busi_split_indices(image_paths, labels, fold=None, n_folds=5, seed=42):

    labels = np.asarray(labels)

    groups, usable = load_busi_groups(image_paths)

    indices = np.where(usable)[0]

    if fold is None:
        # Single hold-out split: about 70 / 15 / 15
        rest, test_indices = group_split(indices, labels, groups, 7, 0, seed)
    else:
        rest, test_indices = group_split(indices, labels, groups, n_folds, fold, seed)

    train_indices, val_indices = group_split(rest, labels, groups, 6, 0, seed)

    return (
        train_indices.tolist(),
        val_indices.tolist(),
        test_indices.tolist()
    )


def get_resnet_dataloaders(
    dataset="busi",
    batch_size=32,
    num_workers=0,
    seed=42,
    fold=None,
    n_folds=5
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
            train_dataset.images,
            train_dataset.labels,
            fold=fold,
            n_folds=n_folds,
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
