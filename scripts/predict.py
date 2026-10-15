import argparse

import torch
from PIL import Image

from src.models.resnet import BreastCancerResNet
from src.data.transforms import get_eval_transform
from src.evaluate.calibration import (
    apply_calibration,
    flag_threshold,
    load_calibration,
)


METRICS_PATH = "models/checkpoints/resnet18_busi_metrics.json"


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

    # Calibrated on the validation set by scripts/evaluate_busi.py
    calibration = load_calibration(METRICS_PATH)

    probabilities = apply_calibration(
        output.cpu(),
        calibration
    )

    # Same cautious decision rule as the web app
    prediction = int(
        probabilities[0][1].item() >= flag_threshold(calibration)
    )


    classes = {
        0: "Benign",
        1: "Malignant"
    }


    print(
        "Prediction:",
        classes[prediction]
    )

    print(
        "Probability of malignancy:",
        round(
            probabilities[0][1].item()*100,
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