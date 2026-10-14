import argparse

import torch

from src.models.resnet import BreastCancerResNet
from src.data.resnet_dataloader import get_resnet_dataloaders
from src.train.train_resnet import evaluate, get_device, FINAL_CHECKPOINT


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the BUSI test split"
    )

    parser.add_argument("--checkpoint", default=FINAL_CHECKPOINT)

    args = parser.parse_args()

    device = get_device()

    print("Device:", device)

    _, _, test_loader = get_resnet_dataloaders(
        dataset="busi"
    )

    model = BreastCancerResNet(pretrained=False).to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    results = evaluate(model, test_loader, device)

    print("\nResults")

    for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{name.capitalize()}: {results[name]:.4f}")

    print("\nConfusion Matrix")

    print(results["confusion_matrix"])


if __name__ == "__main__":

    main()
