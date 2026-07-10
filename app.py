import os

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import torch

torch.set_num_threads(1)

import streamlit as st
from PIL import Image
from torchvision import transforms

from src.models.resnet import BreastCancerResNet


# -----------------------
# Page Configuration
# -----------------------

st.set_page_config(
    page_title="MedVisionAI",
    page_icon="🩺",
    layout="centered"
)


# -----------------------
# Device
# -----------------------

device = torch.device("cpu")


# -----------------------
# Load Model
# -----------------------

@st.cache_resource(show_spinner=False)
def load_model():

    model = BreastCancerResNet()

    checkpoint = torch.load(
        "models/checkpoints/resnet18_breastmnist_final_best_f1.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model


with st.spinner("Loading MedVisionAI model..."):
    model = load_model()


# -----------------------
# Image Transform
# -----------------------

transform = transforms.Compose(
    [
        transforms.Resize((224,224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.4829],
            std=[0.229]
        )
    ]
)


# -----------------------
# UI
# -----------------------

st.title("🩺 MedVisionAI")

st.subheader(
    "Breast Ultrasound Classification Using Deep Learning"
)

st.write(
    """
    This application uses a fine-tuned ResNet18 model
    to classify breast ultrasound images as benign or malignant.
    """
)


uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("L")


    st.image(
        image,
        caption="Uploaded Ultrasound Image",
        width=300
    )


    input_tensor = transform(image).unsqueeze(0)


    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )


    classes = [
        "Benign",
        "Malignant"
    ]

    result = classes[prediction.item()]

    confidence = confidence.item() * 100


    st.divider()

    if result == "Malignant":
        st.error(
            f"Prediction: {result}"
        )
    else:
        st.success(
            f"Prediction: {result}"
        )


    st.info(
        f"Confidence: {confidence:.2f}%"
    )


    # -----------------------
    # Grad-CAM (lazy loading)
    # -----------------------

    if st.checkbox("Show Grad-CAM explanation"):

        with st.spinner("Generating explanation..."):

            from src.explainability.gradcam_utils import generate_gradcam

            cam_image = generate_gradcam(
                model,
                input_tensor,
                image,
                device
            )

            st.image(
                cam_image,
                caption="Grad-CAM Visualization",
                width=400
            )


# -----------------------
# Disclaimer
# -----------------------

st.divider()

st.warning(
    """
    ⚠️ Disclaimer:

    This project is for educational and research purposes only.
    It is not a medical diagnostic system and should not be used
    for clinical decisions.
    """
)


# -----------------------
# Sidebar
# -----------------------

st.sidebar.title("About MedVisionAI")

st.sidebar.write(
    """
    **Model:** ResNet18 Transfer Learning

    **Dataset:** BreastMNIST

    **Task:** Binary Breast Ultrasound Classification

    **Classes:**
    - Benign
    - Malignant

    **Test F1 Score:**
    92.17%

    **Frameworks:**
    - PyTorch
    - Streamlit
    - Grad-CAM
    """
)