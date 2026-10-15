import json

import torch
import torch.nn.functional as F


def fit_temperature(logits, labels, max_iter=200):

    # Temperature scaling (Guo et al., 2017): divide the logits by a single
    # scalar T fitted on held-out data. It changes confidence, never the
    # predicted class, so accuracy, recall and ROC AUC stay the same
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)

    # Optimise log T so the temperature stays positive
    log_t = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():

        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()

        return loss

    optimizer.step(closure)

    return float(log_t.exp().item())


def apply_temperature(logits, temperature=1.0):

    return torch.softmax(torch.as_tensor(logits) / temperature, dim=-1)


def load_temperature(metrics_path):

    # Fall back to T = 1 (uncalibrated) if the model was never calibrated
    try:
        with open(metrics_path) as f:
            return float(json.load(f)["calibration"]["temperature"])

    except (OSError, KeyError, TypeError, ValueError):
        return 1.0


def save_temperature(metrics_path, temperature, details=None):

    # Merge into the training metrics file rather than overwriting it
    try:
        with open(metrics_path) as f:
            data = json.load(f)

    except (OSError, ValueError):
        data = {}

    data["calibration"] = {"temperature": temperature, **(details or {})}

    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=4)
