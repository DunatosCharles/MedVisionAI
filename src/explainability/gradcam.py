import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from torchvision import transforms

from src.models.resnet import BreastCancerResNet


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


device = get_device()


# Load model

model = BreastCancerResNet().to(device)

model.load_state_dict(
    torch.load(
        "models/checkpoints/resnet18_breastmnist_final.pth",
        map_location=device
    )
)

model.eval()



# Image preprocessing

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



# Load image

from medmnist import BreastMNIST


dataset = BreastMNIST(
    split="test",
    download=True,
    transform=None
)


for index in range(len(dataset)):

    image, label = dataset[index]

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(image_tensor)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()


    if prediction == label[0]:

        print(
            "Using correctly classified image:",
            index
        )

        break

image = image.convert("L")

print(type(image))
print(image.size)
print("Label:", label)



input_tensor = transform(image)
input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)



# Convert grayscale for visualization

rgb_image = np.array(
    image.resize((224,224))
).astype(np.float32)

rgb_image = np.stack(
    [rgb_image]*3,
    axis=-1
)

rgb_image = rgb_image / 255.0



# Target final convolution layer

target_layers = [
    model.model.layer4[-1]
]


cam = GradCAM(
    model=model,
    target_layers=target_layers
)



output = model(
    input_tensor
)

probabilities = torch.softmax(
    output,
    dim=1
)

prediction = torch.argmax(
    probabilities,
    dim=1
).item()

confidence = probabilities[0][prediction].item() * 100


classes = {
    0: "Benign",
    1: "Malignant"
}


print(
    f"Prediction: {classes[prediction]}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print(
    f"True Label: {classes[label[0]]}"
)



targets = [
    ClassifierOutputTarget(prediction)
]


grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]


visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)



plt.figure(figsize=(6,6))

plt.imshow(
    visualization
)

plt.axis("off")

plt.title(
    f"Prediction: {classes[prediction]}\n"
    f"Confidence: {confidence:.2f}%"
)


plt.savefig(
    "screenshots/gradcam_example.png",
    dpi=300,
    bbox_inches="tight"
)


print(
    "Saved Grad-CAM visualization"
)