import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score

from src.models.resnet import BreastCancerResNet
from src.data.resnet_dataloader import get_resnet_dataloaders


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


def validate(model, val_loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    predictions = []
    true_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = correct / total

    f1 = f1_score(
        true_labels,
        predictions
    )

    return running_loss, accuracy, f1


def train():

    device = get_device()

    print("Using device:", device)

    train_loader, val_loader, _ = get_resnet_dataloaders()

    model = BreastCancerResNet().to(device)

    class_weights = torch.tensor(
        [2.7, 1.0],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    epochs = 50

    best_val_f1 = 0

    patience = 10
    counter = 0

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_accuracy = correct / total

        val_loss, val_accuracy, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_f1)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {running_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']}"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            counter = 0

            torch.save(
                model.state_dict(),
                "models/checkpoints/resnet18_finetuned_v3.pth"
            )

            print("Saved best model!")

        else:

            counter += 1

        if counter >= patience:

            print("Early stopping!")
            break

    print(
        f"Best validation F1: {best_val_f1:.4f}"
    )


if __name__ == "__main__":

    train()