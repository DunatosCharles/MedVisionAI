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


device = get_device()

print("Device:", device)


_, _, test_loader = get_resnet_dataloaders(
    dataset="busi"
)


model = BreastCancerResNet().to(device)

model.load_state_dict(
    torch.load(
        "models/checkpoints/resnet18_busi_best_f1.pth",
        map_location=device
    )
)

model.eval()


predictions = []
labels = []


with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(device)

        targets = targets.view(-1).long().to(device)

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        )


        predictions.extend(
            preds.cpu().numpy()
        )

        labels.extend(
            targets.cpu().numpy()
        )


print("\nResults")

print(
    "Accuracy:",
    accuracy_score(labels, predictions)
)

print(
    "Precision:",
    precision_score(labels, predictions)
)

print(
    "Recall:",
    recall_score(labels, predictions)
)

print(
    "F1:",
    f1_score(labels, predictions)
)


print("\nConfusion Matrix")

print(
    confusion_matrix(
        labels,
        predictions
    )
)