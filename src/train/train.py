import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.cnn import BreastCancerCNN
from src.data.dataloader import get_dataloaders


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train():

    device = get_device()

    print("Using device:", device)

    train_loader, val_loader, _ = get_dataloaders()

    model = BreastCancerCNN().to(device)

    class_weights = torch.tensor(
        [399/147, 1.0],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 10

    best_val_accuracy = 0

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

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (predictions == labels).sum().item()
            total += labels.size(0)


        train_accuracy = correct / total

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {running_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "models/checkpoints/best_model.pth"
        )

        print("Saved best model!")

if __name__ == "__main__":

    def validate(model, val_loader, criterion, device):

        model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.squeeze().long().to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                running_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        return running_loss, accuracy

    train()
