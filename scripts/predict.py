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


def load_model():

    device = get_device()

    model = BreastCancerResNet().to(device)

    model.load_state_dict(
        torch.load(
            "models/checkpoints/resnet18_busi_final.pth",
            map_location=device
        )
    )

    model.eval()

    return model, device



def preprocess_image(image_path):

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

    predict(
        "/Users/apple/Documents/MedVisionAI/datasets/busi/Dataset_BUSI_with_GT/benign/benign (1).png"
    )