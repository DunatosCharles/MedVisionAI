import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.models.resnet import BreastCancerResNet
from src.data.resnet_dataloader import get_resnet_dataloaders
from src.evaluate.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
)
from src.evaluate.calibration import (
    apply_temperature,
    fit_temperature,
    save_temperature,
)
from src.train.train_resnet import get_device, FINAL_CHECKPOINT, METRICS_FILE


CLASS_NAMES = ["Benign", "Malignant"]


def collect_logits(model, loader, device):

    model.eval()

    logits = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in loader:

            logits.append(model(images.to(device)).cpu())
            labels.extend(batch_labels.view(-1).numpy())

    return np.array(labels), torch.cat(logits)


def malignant_probability(logits, temperature=1.0):

    return apply_temperature(logits, temperature)[:, 1].numpy()


def plot_confusion_matrix(labels, probabilities, path):

    predictions = (probabilities >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(4.5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(labels, predictions),
        display_labels=CLASS_NAMES
    ).plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Test Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(labels, probabilities, metrics, path):

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)

    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10, 4.5))

    roc_ax.plot(fpr, tpr, label=f"ResNet18 (AUC = {metrics['roc_auc']:.3f})")
    roc_ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curve")
    roc_ax.legend(loc="lower right")

    prevalence = labels.mean()

    pr_ax.plot(recall, precision, label=f"ResNet18 (AP = {metrics['pr_auc']:.3f})")
    pr_ax.axhline(prevalence, linestyle="--", color="gray",
                  label=f"Chance (prevalence = {prevalence:.2f})")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-Recall Curve")
    pr_ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(labels, curves, path):

    # curves: list of (label, probabilities, metrics), drawn on one diagram
    fig, (curve_ax, hist_ax) = plt.subplots(
        2, 1, figsize=(5, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    curve_ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    for name, probabilities, metrics in curves:

        # Few test images, so use 5 bins to keep each point meaningful
        frac_positive, mean_predicted = calibration_curve(
            labels, probabilities, n_bins=5, strategy="quantile"
        )

        curve_ax.plot(mean_predicted, frac_positive, "o-",
                      label=f"{name} (ECE = {metrics['ece']:.3f})")

        hist_ax.hist(probabilities, bins=20, range=(0, 1),
                     histtype="step", linewidth=1.5)

    curve_ax.set_ylabel("Observed fraction malignant")
    curve_ax.set_title("Reliability Diagram")
    curve_ax.legend(loc="upper left")

    hist_ax.set_xlabel("Predicted probability of malignant")
    hist_ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the BUSI test split"
    )

    parser.add_argument("--checkpoint", default=FINAL_CHECKPOINT)
    parser.add_argument(
        "--metrics-file", default=METRICS_FILE,
        help="JSON next to the checkpoint where the fitted temperature is saved"
    )
    parser.add_argument("--output-dir", default="results/busi")
    parser.add_argument("--bootstrap", type=int, default=2000)

    args = parser.parse_args()

    device = get_device()

    print("Device:", device)

    _, val_loader, test_loader = get_resnet_dataloaders(
        dataset="busi"
    )

    model = BreastCancerResNet(pretrained=False).to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    # Fit the temperature on validation only, so the test set stays unseen
    val_labels, val_logits = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)

    val_ece_before = classification_metrics(
        val_labels, malignant_probability(val_logits))["ece"]
    val_ece_after = classification_metrics(
        val_labels, malignant_probability(val_logits, temperature))["ece"]

    print(f"\nTemperature fitted on {len(val_labels)} validation images: "
          f"T = {temperature:.3f}")
    print(f"Validation ECE {val_ece_before:.4f} -> {val_ece_after:.4f}")

    save_temperature(
        args.metrics_file, temperature,
        {"fitted_on": "validation", "n_val": int(len(val_labels))}
    )

    labels, logits = collect_logits(model, test_loader, device)

    raw_probabilities = malignant_probability(logits)
    probabilities = malignant_probability(logits, temperature)

    raw_metrics = classification_metrics(labels, raw_probabilities)
    raw_intervals = bootstrap_confidence_intervals(
        labels, raw_probabilities, n_resamples=args.bootstrap
    )

    metrics = classification_metrics(labels, probabilities)
    intervals = bootstrap_confidence_intervals(
        labels, probabilities, n_resamples=args.bootstrap
    )

    print(f"\nResults on {len(labels)} test images (95% bootstrap CI)\n")
    print(f"{'':<12} {'Uncalibrated':<26} {f'Calibrated (T = {temperature:.2f})'}")

    for name, value in metrics.items():
        raw_low, raw_high = raw_intervals[name]
        low, high = intervals[name]
        print(f"{name:<12} {raw_metrics[name]:.4f} [{raw_low:.4f}, {raw_high:.4f}]"
              f"  {value:.4f} [{low:.4f}, {high:.4f}]")

    print("\nConfusion Matrix")

    print(confusion_matrix(labels, (probabilities >= 0.5).astype(int)))

    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        labels, probabilities,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )
    plot_roc_pr(
        labels, probabilities, metrics,
        os.path.join(args.output_dir, "roc_pr_curves.png")
    )
    plot_calibration(
        labels,
        [
            ("Uncalibrated", raw_probabilities, raw_metrics),
            (f"Temperature scaled (T = {temperature:.2f})", probabilities, metrics),
        ],
        os.path.join(args.output_dir, "calibration.png")
    )

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "n_test": int(len(labels)),
                "temperature": temperature,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "ci_95": intervals,
                "metrics_uncalibrated": {
                    k: float(v) for k, v in raw_metrics.items()
                },
                "ci_95_uncalibrated": raw_intervals,
            },
            f,
            indent=4
        )

    print(f"\nSaved plots and metrics to {args.output_dir}/")


if __name__ == "__main__":

    main()
