import json
import os

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import torch

torch.set_num_threads(1)

import streamlit as st
from PIL import Image

from src.models.resnet import BreastCancerResNet
from src.data.transforms import get_eval_transform
from src.evaluate.calibration import (
    apply_calibration,
    flag_threshold,
    load_calibration,
)


MODEL_PATH = "models/checkpoints/resnet18_busi_final.pth"
METRICS_PATH = "models/checkpoints/resnet18_busi_metrics.json"
CV_METRICS_PATH = "results/busi_cv/cv_metrics.json"

SAMPLE_DIR = "datasets/busi/Dataset_BUSI_with_GT"

# Held-out test images the model never saw during training
SAMPLES = {
    "Benign": "benign/benign (19).png",
    "Malignant": "malignant/malignant (153).png",
    "Borderline": "malignant/malignant (108).png",
}


# -----------------------
# Page Configuration
# -----------------------

st.set_page_config(
    page_title="MedVisionAI · Breast Ultrasound Classifier",
    page_icon="🩺",
    layout="wide"
)


# Accent colours matching .streamlit/config.toml; custom CSS cannot read
# the theme's colours, so pick them from the active theme
dark_theme = st.context.theme.type == "dark"

accent = "#2DD4BF" if dark_theme else "#0F766E"
benign_color = "#4ADE80" if dark_theme else "#15803D"
malignant_color = "#F87171" if dark_theme else "#B91C1C"


st.markdown(
    f"""
    <style>
    :root {{
        --mv-accent: {accent};
        --mv-benign: {benign_color};
        --mv-malignant: {malignant_color};
    }}
    """
    """
    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
    }

    .mv-eyebrow {
        color: var(--mv-accent);
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .mv-title {
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0 0 0.5rem 0;
    }

    .mv-subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        max-width: 46rem;
        margin-bottom: 1.5rem;
    }

    .mv-step {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0.6;
        margin-bottom: 0.5rem;
    }

    .mv-result-label {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
        margin: 0.25rem 0 0.25rem 0;
    }

    .mv-benign { color: var(--mv-benign); }
    .mv-malignant { color: var(--mv-malignant); }

    .mv-bar {
        position: relative;
        height: 0.6rem;
        border-radius: 999px;
        background: rgba(127, 127, 127, 0.18);
        margin-top: 0.25rem;
    }

    .mv-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: var(--mv-accent);
    }

    .mv-bar-mark {
        position: absolute;
        top: -0.3rem;
        bottom: -0.3rem;
        width: 2px;
        margin-left: -1px;
        background: currentColor;
        opacity: 0.7;
    }

    .mv-bar-labels {
        position: relative;
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        opacity: 0.65;
        margin: 0.4rem 0 0.75rem 0;
        height: 1.1rem;
    }

    .mv-bar-mark-label {
        position: absolute;
        transform: translateX(-50%);
        white-space: nowrap;
    }

    .mv-empty {
        text-align: center;
        padding: 3rem 1rem;
        opacity: 0.6;
    }

    .mv-empty-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .mv-footer {
        font-size: 0.8rem;
        opacity: 0.6;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
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

    model = BreastCancerResNet(pretrained=False)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model


with st.spinner("Loading MedVisionAI model..."):
    model = load_model()


# Calibration fitted on the validation set by scripts/evaluate_busi.py,
# so the displayed probability is calibrated (identity if missing)
@st.cache_data(show_spinner=False)
def get_calibration():

    return load_calibration(METRICS_PATH)


calibration = get_calibration()


# Calibrated probabilities, but flagged at the uncalibrated model's more
# cautious decision boundary (see flag_threshold)
threshold = flag_threshold(calibration)


def logit(p):

    p = min(max(p, 1e-6), 1 - 1e-6)

    return float(torch.log(torch.tensor(p / (1 - p))))


# -----------------------
# Image Transform
# -----------------------

transform = get_eval_transform()


@st.cache_data(show_spinner=False)
def load_json(path):

    try:
        with open(path) as f:
            return json.load(f)

    except (OSError, ValueError):
        return None


def load_metrics():

    data = load_json(METRICS_PATH)

    if not data:
        return None

    # The app keeps the uncalibrated model's decisions (see
    # flag_threshold), so its raw test metrics describe what it flags
    metrics = data.get("test")

    if metrics and "specificity" not in metrics and "confusion_matrix" in metrics:

        (tn, fp), _ = metrics["confusion_matrix"]

        metrics = {**metrics, "specificity": tn / (tn + fp)}

    return metrics


def load_cv_summary():

    data = load_json(CV_METRICS_PATH)

    return data.get("summary") if data else None


# -----------------------
# Sidebar
# -----------------------

with st.sidebar:

    st.markdown("### About the model")

    st.markdown(
        "A **ResNet18** fine-tuned on the "
        "[BUSI breast ultrasound dataset]"
        "(https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) "
        "to classify lesions as **benign** or **malignant**."
    )

    metrics = load_metrics()

    if metrics:

        st.markdown("##### Held-out test set")

        col_a, col_b = st.columns(2)

        col_a.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        col_b.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

        col_a.metric(
            "Sensitivity",
            f"{metrics['recall'] * 100:.1f}%",
            help="Share of malignant lesions correctly flagged"
        )

        if "specificity" in metrics:
            col_b.metric(
                "Specificity",
                f"{metrics['specificity'] * 100:.1f}%",
                help="Share of benign lesions correctly cleared"
            )

    cv = load_cv_summary()

    if cv:

        st.markdown("##### 5-fold cross-validation")

        st.caption(
            f"Accuracy {cv['accuracy']['pooled'] * 100:.1f}% "
            f"(95% CI {cv['accuracy']['pooled_ci_95'][0] * 100:.0f}–"
            f"{cv['accuracy']['pooled_ci_95'][1] * 100:.0f}%) · "
            f"ROC AUC {cv['roc_auc']['pooled']:.3f}, "
            "with images of the same lesion never split between "
            "training and testing."
        )

    st.divider()

    st.markdown("##### Built with")

    st.caption("PyTorch · Grad-CAM · Streamlit")

    st.markdown(
        "[Source code on GitHub]"
        "(https://github.com/DunatosCharles/MedVisionAI)"
    )


# -----------------------
# Header
# -----------------------

st.markdown(
    """
    <div class="mv-eyebrow">Research demo · Breast ultrasound</div>
    <div class="mv-title">MedVisionAI</div>
    <div class="mv-subtitle">
        Upload a breast ultrasound image to get a benign / malignant
        prediction from a deep learning model, with a Grad-CAM heatmap
        showing which regions drove the decision.
    </div>
    """,
    unsafe_allow_html=True
)


# -----------------------
# Main Layout
# -----------------------

input_col, result_col = st.columns([1, 1], gap="large")


with input_col:

    with st.container(border=True):

        st.markdown(
            '<div class="mv-step">1 · Choose an image</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload ultrasound image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        sample = None

        if os.path.isdir(SAMPLE_DIR):

            # Pills wrap as whole chips in narrow columns, unlike buttons
            sample = st.pills(
                "No image at hand? Try a sample from the test set:",
                list(SAMPLES)
            )

        with st.expander("Image guidelines"):

            st.markdown(
                """
                Works best with **grayscale breast ultrasound images**
                (PNG or JPG) similar to the BUSI dataset.

                Results are unreliable for:
                - MRI, CT, X-ray or other imaging types
                - Color images or non-breast ultrasound
                - Images from very different machines or hospitals
                """
            )


image = None
image_name = None

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    image_name = uploaded_file.name

elif sample:

    image = Image.open(
        os.path.join(SAMPLE_DIR, SAMPLES[sample])
    ).convert("L")

    image_name = f"Sample: {sample}"


with result_col:

    with st.container(border=True):

        st.markdown(
            '<div class="mv-step">2 · Result</div>',
            unsafe_allow_html=True
        )

        if image is None:

            st.markdown(
                """
                <div class="mv-empty">
                    <div class="mv-empty-icon">🔬</div>
                    Upload an image or pick a sample to see the prediction.
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            input_tensor = transform(image).unsqueeze(0)

            with torch.inference_mode():

                outputs = model(input_tensor)

                probabilities = apply_calibration(
                    outputs,
                    calibration
                )

            malignant_probability = probabilities[0, 1].item()

            prediction = int(malignant_probability >= threshold)

            result = ["Benign", "Malignant"][prediction]

            # Distance from the decision boundary in logit space; 2.2 and
            # 0.85 correspond to 90% and 70% around a 50% boundary
            distance = abs(logit(malignant_probability) - logit(threshold))

            if distance >= 2.2:
                certainty = "High confidence"
            elif distance >= 0.85:
                certainty = "Moderate confidence"
            else:
                certainty = "Low confidence: close to the decision boundary"

            st.markdown(
                f'<div class="mv-result-label mv-{result.lower()}">'
                f"{result}</div>",
                unsafe_allow_html=True
            )

            st.caption(
                certainty
                + (" · calibrated" if calibration["method"] != "none" else "")
            )

            st.metric(
                "Probability of malignancy",
                f"{malignant_probability * 100:.1f}%"
            )

            fill = malignant_probability * 100
            mark = threshold * 100

            st.markdown(
                f"""
                <div class="mv-bar">
                    <div class="mv-bar-fill" style="width:{fill:.1f}%"></div>
                    <div class="mv-bar-mark" style="left:{mark:.1f}%"></div>
                </div>
                <div class="mv-bar-labels">
                    <span class="mv-bar-mark-label" style="left:{mark:.1f}%">
                        flag ≥ {mark:.0f}%
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            if threshold < 0.5:

                st.caption(
                    f"Lesions are flagged as malignant from {mark:.0f}% "
                    "calibrated probability, so that fewer cancers are "
                    "missed."
                )


# -----------------------
# Image and Grad-CAM
# -----------------------

if image is not None:

    st.write("")

    with st.container(border=True):

        header_col, toggle_col = st.columns(
            [3, 2],
            vertical_alignment="center"
        )

        header_col.markdown(
            '<div class="mv-step" style="margin-bottom:0">'
            "3 · Explanation</div>",
            unsafe_allow_html=True
        )

        show_cam = toggle_col.toggle(
            "Show Grad-CAM heatmap",
            value=True
        )

        image_col, cam_col = st.columns(2, gap="medium")

        image_col.image(
            image.resize((448, 448)),
            caption=image_name,
            width="stretch"
        )

        if show_cam:

            with cam_col:

                with st.spinner("Generating explanation..."):

                    from src.explainability.gradcam_utils import generate_gradcam

                    cam_image = generate_gradcam(
                        model,
                        input_tensor,
                        image,
                        device,
                        target_class=prediction
                    )

                st.image(
                    cam_image.resize((448, 448)),
                    caption=f"Grad-CAM: regions supporting “{result}”",
                    width="stretch"
                )

        else:

            cam_col.info(
                "Grad-CAM highlights the image regions that most "
                "influenced the prediction. Turn it on to compare."
            )


# -----------------------
# Disclaimer
# -----------------------

st.markdown(
    """
    <div class="mv-footer">
    ⚠️ <b>Research and educational use only.</b>
    MedVisionAI is not a medical device and must not be used for
    diagnosis or treatment decisions. It was trained on the public BUSI
    dataset and may not generalize to other equipment, hospitals or
    patient populations.
    </div>
    """,
    unsafe_allow_html=True
)
