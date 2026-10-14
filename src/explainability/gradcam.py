import argparse

import torch
import matplotlib.pyplot as plt

from PIL import Image

from src.models.resnet import BreastCancerResNet
from src.data.transforms import get_eval_transform
from src.explainability.gradcam_utils import generate_gradcam


CHECKPOINT = "models/checkpoints/resnet18_busi_final.pth"

CLASSES = {
    0: "Benign",
    1: "Malignant"
}


def main():

    parser = argparse.ArgumentParser(
        description="Save a Grad-CAM visualization for an ultrasound image"
    )

    parser.add_argument("image_path")

    parser.add_argument(
        "--output",
        default="screenshots/gradcam_example.png"
    )

    args = parser.parse_args()


    device = torch.device("cpu")

    model = BreastCancerResNet(pretrained=False).to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()


    image = Image.open(args.image_path).convert("L")

    input_tensor = get_eval_transform()(image).unsqueeze(0).to(device)


    with torch.no_grad():

        probabilities = torch.softmax(
            model(input_tensor),
            dim=1
        )

    prediction = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][prediction].item() * 100


    print(f"Prediction: {CLASSES[prediction]}")
    print(f"Confidence: {confidence:.2f}%")


    visualization = generate_gradcam(
        model,
        input_tensor,
        image,
        device
    )


    plt.figure(figsize=(6,6))

    plt.imshow(visualization)

    plt.axis("off")

    plt.title(
        f"Prediction: {CLASSES[prediction]}\n"
        f"Confidence: {confidence:.2f}%"
    )

    plt.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved Grad-CAM visualization to {args.output}")


if __name__ == "__main__":

    main()
