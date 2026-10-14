import torch
import numpy as np

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_gradcam(
    model,
    image_tensor,
    original_image,
    device
):

    model.eval()


    target_layers = [
        model.model.layer4[-1]
    ]


    with torch.no_grad():

        output = model(
            image_tensor
        )


    prediction = torch.argmax(
        output
    ).item()


    targets = [
        ClassifierOutputTarget(
            prediction
        )
    ]


    # Context manager removes the hooks GradCAM registers on the model,
    # which is cached and reused across Streamlit reruns
    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets
        )[0]


    img = np.array(
        original_image.resize(
            (224,224)
        )
    )


    # Convert grayscale to RGB
    if len(img.shape) == 2:
        img = np.stack(
            [img, img, img],
            axis=-1
        )


    img = img.astype(np.float32) / 255.0


    visualization = show_cam_on_image(
        img,
        grayscale_cam,
        use_rgb=True
    )


    return Image.fromarray(
        visualization
    )