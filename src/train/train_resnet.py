import argparse
import json
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.models.resnet import BreastCancerResNet
from src.data.resnet_dataloader import get_resnet_dataloaders


CHECKPOINT_DIR = "models/checkpoints"
BEST_CHECKPOINT = f"{CHECKPOINT_DIR}/resnet18_busi_best_f1.pth"
FINAL_CHECKPOINT = f"{CHECKPOINT_DIR}/resnet18_busi_final.pth"
METRICS_FILE = f"{CHECKPOINT_DIR}/resnet18_busi_metrics.json"
HISTORY_FILE = "training_history.json"


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device, criterion=None):

    model.eval()

    running_loss = 0.0

    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            outputs = model(images)

            if criterion is not None:
                running_loss += criterion(outputs, labels).item() * labels.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return {
        "loss": running_loss / len(true_labels),
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, zero_division=0),
        "recall": recall_score(true_labels, predictions, zero_division=0),
        "f1": f1_score(true_labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(true_labels, probabilities),
        "confusion_matrix": confusion_matrix(true_labels, predictions).tolist()
    }


def get_class_weights(train_loader):

    # Inverse frequency weights so malignant (minority) errors cost more
    subset = train_loader.dataset
    labels = [subset.dataset.labels[i] for i in subset.indices]

    counts = np.bincount(labels, minlength=2)
    weights = len(labels) / (2 * counts)

    return torch.tensor(weights, dtype=torch.float32)


def train(args):

    set_seed(args.seed)

    device = get_device()

    print("Using device:", device)

    train_loader, val_loader, test_loader = get_resnet_dataloaders(
        dataset="busi",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    print(
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    model = BreastCancerResNet(
        pretrained=True,
        freeze_early_layers=not args.full_finetune
    ).to(device)

    class_weights = get_class_weights(train_loader).to(device)

    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # New classifier head learns faster than the pretrained backbone
    head_params = list(model.model.fc.parameters())
    head_ids = {id(p) for p in head_params}

    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_ids
    ]

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * 10}
        ],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }

    for epoch in range(args.epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.view(-1).long().to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_accuracy = correct / total

        val = evaluate(
            model,
            val_loader,
            device,
            criterion
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val['loss']:.4f} | "
            f"Val Acc: {val['accuracy']:.4f} | "
            f"Val F1: {val['f1']:.4f} | "
            f"LR: {lr:.2e}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_accuracy)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1"])
        history["lr"].append(lr)

        # Best by validation F1, ties broken by lower validation loss
        improved = (
            val["f1"] > best_val_f1
            or (val["f1"] == best_val_f1 and val["loss"] < best_val_loss)
        )

        if improved:

            best_val_f1 = val["f1"]
            best_val_loss = val["loss"]
            counter = 0

            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT
            )

            print("Saved best model!")

        else:

            counter += 1

        if counter >= args.patience:

            print("Early stopping!")
            break

    print(f"Best validation F1: {best_val_f1:.4f}")

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

    print("Training history saved!")


    # Final evaluation of the best checkpoint on the held-out test set
    model.load_state_dict(
        torch.load(BEST_CHECKPOINT, map_location=device, weights_only=True)
    )

    test = evaluate(model, test_loader, device)

    print("\nTest Results")

    for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{name}: {test[name]:.4f}")

    print("Confusion Matrix:", test["confusion_matrix"])

    with open(METRICS_FILE, "w") as f:
        json.dump(
            {
                "best_val_f1": best_val_f1,
                "test": test,
                "args": vars(args)
            },
            f,
            indent=4
        )

    # The web app serves this checkpoint
    shutil.copyfile(BEST_CHECKPOINT, FINAL_CHECKPOINT)

    print(f"Copied best model to {FINAL_CHECKPOINT}")


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train ResNet18 on the BUSI dataset"
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="Also train the stem and layer1"
    )

    return parser.parse_args()


if __name__ == "__main__":

    train(parse_args())
