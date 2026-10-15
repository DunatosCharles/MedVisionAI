import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(labels, probabilities, threshold=0.5):

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        # Specificity matters in screening: how many benign cases are cleared
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
        "brier": brier_score_loss(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
    }


def expected_calibration_error(labels, probabilities, n_bins=10):

    # Confidence is the probability of the predicted class, accuracy is
    # whether that prediction was right; ECE is their weighted gap per bin
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    predictions = (probabilities >= 0.5).astype(int)
    confidences = np.where(predictions == 1, probabilities, 1 - probabilities)
    correct = (predictions == labels).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0

    for low, high in zip(edges[:-1], edges[1:]):

        in_bin = (confidences > low) & (confidences <= high)

        if low == 0.5:
            in_bin |= confidences == 0.5

        if in_bin.any():
            gap = abs(correct[in_bin].mean() - confidences[in_bin].mean())
            ece += in_bin.mean() * gap

    return float(ece)


def bootstrap_confidence_intervals(
    labels,
    probabilities,
    n_resamples=2000,
    alpha=0.05,
    seed=42
):

    # Resample the test set with replacement to estimate how much each
    # metric would move with a different draw of ~100 test images
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    rng = np.random.default_rng(seed)
    samples = {}

    for _ in range(n_resamples):

        idx = rng.integers(0, len(labels), len(labels))

        # AUC is undefined when a resample contains only one class
        if len(np.unique(labels[idx])) < 2:
            continue

        for name, value in classification_metrics(
            labels[idx], probabilities[idx]
        ).items():
            samples.setdefault(name, []).append(value)

    return {
        name: (
            float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2)))
        )
        for name, values in samples.items()
    }
