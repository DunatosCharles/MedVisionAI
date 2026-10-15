import json

import torch
import torch.nn.functional as F


# Both calibrators map the logit margin (malignant - benign) to
# P(malignant) = sigmoid(scale * margin + bias)
#   temperature scaling: scale = 1 / T, bias = 0 (never changes the prediction)
#   Platt scaling:       scale and bias both fitted (the bias moves the
#                        0.5 decision boundary, so predictions can change)
IDENTITY = {"method": "none", "scale": 1.0, "bias": 0.0}


def _margin(logits):

    logits = torch.as_tensor(logits, dtype=torch.float32)

    return logits[:, 1] - logits[:, 0]


def _fit(logits, labels, fit_bias, max_iter=200):

    margin = _margin(logits)
    labels = torch.as_tensor(labels, dtype=torch.float32)

    # Optimise log scale so the scale stays positive
    log_scale = torch.zeros(1, requires_grad=True)
    bias = torch.zeros(1, requires_grad=fit_bias)

    params = [log_scale, bias] if fit_bias else [log_scale]

    optimizer = torch.optim.LBFGS(params, lr=0.1, max_iter=max_iter)

    def closure():

        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            log_scale.exp() * margin + bias, labels
        )
        loss.backward()

        return loss

    optimizer.step(closure)

    return float(log_scale.exp().item()), float(bias.item())


def fit_temperature(logits, labels):

    # Temperature scaling (Guo et al., 2017): one scalar T minimising NLL
    scale, _ = _fit(logits, labels, fit_bias=False)

    return {"method": "temperature", "scale": scale, "bias": 0.0,
            "temperature": 1.0 / scale}


def fit_platt(logits, labels):

    # Platt scaling: a temperature plus a bias, also minimising NLL. The
    # bias corrects a systematic shift toward one class, e.g. from the
    # class-weighted training loss
    scale, bias = _fit(logits, labels, fit_bias=True)

    return {"method": "platt", "scale": scale, "bias": bias}


def apply_calibration(logits, calibration=IDENTITY):

    # Returns [P(benign), P(malignant)] per row, like softmax over logits
    malignant = torch.sigmoid(
        calibration["scale"] * _margin(logits) + calibration["bias"]
    )

    return torch.stack([1 - malignant, malignant], dim=-1)


def flag_threshold(calibration=IDENTITY):

    # Probability at which a lesion is flagged malignant. Platt's bias
    # lowers every malignant probability, so a 0.5 cut-off would miss more
    # cancers than the uncalibrated model. Flagging at the probability of
    # the uncalibrated decision boundary (logit margin 0) keeps its more
    # cautious decisions while showing calibrated probabilities:
    # sigmoid(scale * 0 + bias) = sigmoid(bias), which is 0.5 for
    # temperature scaling or no calibration
    return float(torch.sigmoid(torch.tensor(calibration["bias"])))


def load_calibration(metrics_path):

    # Fall back to the identity (uncalibrated) if the model was never calibrated
    try:
        with open(metrics_path) as f:
            calibration = json.load(f)["calibration"]

        return {
            "method": calibration["method"],
            "scale": float(calibration["scale"]),
            "bias": float(calibration["bias"]),
        }

    except (OSError, KeyError, TypeError, ValueError):
        return dict(IDENTITY)


def save_calibration(metrics_path, calibration, details=None):

    # Merge into the training metrics file rather than overwriting it
    try:
        with open(metrics_path) as f:
            data = json.load(f)

    except (OSError, ValueError):
        data = {}

    data["calibration"] = {**calibration, **(details or {})}

    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=4)


def load_temperature(metrics_path):

    # Deprecated: kept so callers written for temperature scaling still run.
    # Use load_calibration + apply_calibration, which also apply the bias.
    # A Platt calibration has no pure temperature, so this returns 1
    # (uncalibrated) rather than a scale without its bias
    calibration = load_calibration(metrics_path)

    if calibration["method"] != "temperature":
        return 1.0

    return 1.0 / calibration["scale"]


def apply_temperature(logits, temperature=1.0):

    # Deprecated: see load_temperature
    return torch.softmax(torch.as_tensor(logits) / temperature, dim=-1)
