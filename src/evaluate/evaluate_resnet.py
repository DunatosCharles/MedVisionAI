import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.models.resnet import BreastCancerResNet
from src.data.resnet_dataloader import get_resnet_dataloaders

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


def evaluate():

    device = get_device()

    print("Using device:", device)

    _, _, test_loader = get_resnet_dataloaders()

    model = BreastCancerResNet().to(device)

    model.load_state_dict(
        torch.load(
            "models/checkpoints/resnet18_finetuned.pth",
            map_location=device
        )
    )

    print(model)

    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1
            ).cpu()

            predictions.extend(
                preds.numpy()
            )

            true_labels.extend(
                labels.squeeze().numpy()
            )


    print(
        "Accuracy:",
        accuracy_score(true_labels, predictions)
    )

    print(
        "Precision:",
        precision_score(true_labels, predictions)
    )

    print(
        "Recall:",
        recall_score(true_labels, predictions)
    )

    print(
        "F1:",
        f1_score(true_labels, predictions)
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            true_labels,
            predictions
        )
    )


if __name__ == "__main__":
    evaluate()