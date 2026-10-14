import argparse

import torch
from PIL import Image

from src.models.resnet import BreastCancerResNet
from src.data.transforms import get_eval_transform


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    else:
        return torch.device("cpu")


def load_model():

    device = get_device()

    model = BreastCancerResNet(pretrained=False).to(device)

    model.load_state_dict(
        torch.load(
            "models/checkpoints/resnet18_busi_final.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model, device



def preprocess_image(image_path):

    transform = get_eval_transform()


    image = Image.open(
        image_path
    ).convert("L")


    image = transform(image)

    image = image.unsqueeze(0)

    return image



def predict(image_path):

    model, device = load_model()

    image = preprocess_image(
        image_path
    ).to(device)


    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            output,
            dim=1
        ).item()


    classes = {
        0: "Benign",
        1: "Malignant"
    }


    print(
        "Prediction:",
        classes[prediction]
    )

    print(
        "Confidence:",
        round(
            probabilities[0][prediction].item()*100,
            2
        ),
        "%"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Classify a breast ultrasound image"
    )

    parser.add_argument("image_path")

    predict(parser.parse_args().image_path)