import sys
import torch
from PIL import Image
from torchvision import transforms

from src.models.resnet import BreastCancerResNet


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


def predict(image_path):

    device = get_device()

    print("Using device:", device)


    transform = transforms.Compose([

        transforms.Resize(
            (224,224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.4829],
            std=[0.229]
        )
    ])


    image = Image.open(
        image_path
    ).convert("L")


    image = transform(
        image
    )

    image = image.unsqueeze(0).to(device)


    model = BreastCancerResNet().to(device)


    model.load_state_dict(
        torch.load(
            "models/checkpoints/resnet18_breastmnist_final_best_f1.pth",
            map_location=device
        )
    )


    model.eval()


    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()


        confidence = (
            probabilities[0][prediction]
            .item()
            * 100
        )


    classes = {
        0: "Benign",
        1: "Malignant"
    }


    print(
        f"\nPrediction: {classes[prediction]}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python scripts/predict.py <image_path>"
        )

        exit()


    predict(
        sys.argv[1]
    )