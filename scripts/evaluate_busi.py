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
    IDENTITY,
    apply_calibration,
    flag_threshold,
    fit_platt,
    fit_temperature,
    save_calibration,
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


def malignant_probability(logits, calibration=IDENTITY):

    return apply_calibration(logits, calibration)[:, 1].numpy()


def log_loss(labels, probabilities):

    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)

    return float(-np.mean(
        labels * np.log(probabilities)
        + (1 - labels) * np.log(1 - probabilities)
    ))


def plot_confusion_matrix(labels, probabilities, path, threshold=0.5):

    predictions = (probabilities >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(4.5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(labels, predictions),
        display_labels=CLASS_NAMES
    ).plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title(f"Test Set Confusion Matrix (flag ≥ {threshold:.0%})")
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
    curve_ax.legend(loc="upper left", fontsize=8)

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
        help="JSON next to the checkpoint where the fitted calibration is saved"
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

    # Fit both calibrators on validation only, so the test set stays unseen
    val_labels, val_logits = collect_logits(model, val_loader, device)
    labels, logits = collect_logits(model, test_loader, device)

    temperature = fit_temperature(val_logits, val_labels)
    platt = fit_platt(val_logits, val_labels)

    # Platt scaling is the adopted calibrator: on validation it improved
    # NLL and Brier over temperature scaling (see README, Calibration)
    calibrators = {
        "uncalibrated": IDENTITY,
        "temperature": temperature,
        "platt": platt,
    }

    print(f"\nFitted on {len(val_labels)} validation images: "
          f"temperature T = {temperature['temperature']:.3f}, "
          f"Platt scale = {platt['scale']:.3f}, bias = {platt['bias']:.3f}")

    print("\nValidation (in-sample)")

    for name, calibration in calibrators.items():
        p = malignant_probability(val_logits, calibration)
        m = classification_metrics(val_labels, p)
        print(f"{name:<14} NLL {log_loss(val_labels, p):.4f}  "
              f"Brier {m['brier']:.4f}  ECE {m['ece']:.4f}")

    test_probabilities = {
        name: malignant_probability(logits, calibration)
        for name, calibration in calibrators.items()
    }
    test_metrics = {
        name: classification_metrics(labels, p)
        for name, p in test_probabilities.items()
    }
    test_intervals = {
        name: bootstrap_confidence_intervals(
            labels, p, n_resamples=args.bootstrap
        )
        for name, p in test_probabilities.items()
    }

    probabilities = test_probabilities["platt"]
    metrics = test_metrics["platt"]

    print(f"\nResults on {len(labels)} test images (95% bootstrap CI)\n")
    print(f"{'':<12} " + "".join(f"{name:<27}" for name in calibrators))

    for metric in metrics:
        print(f"{metric:<12} " + "".join(
            f"{test_metrics[name][metric]:.4f} "
            f"[{test_intervals[name][metric][0]:.4f}, "
            f"{test_intervals[name][metric][1]:.4f}]  "
            for name in calibrators
        ))

    print(f"{'nll':<12} " + "".join(
        f"{log_loss(labels, p):<27.4f}" for p in test_probabilities.values()
    ))

    print("\nConfusion Matrix (Platt calibrated, threshold 0.5)")

    print(confusion_matrix(labels, (probabilities >= 0.5).astype(int)))

    # The app shows Platt probabilities but flags at flag_threshold, which
    # reproduces the raw model's decisions
    threshold = flag_threshold(platt)

    print(
        f"\nConfusion Matrix (deployed: Platt >= {threshold:.3f}, "
        "same decisions as the raw model)"
    )

    print(confusion_matrix(labels, (probabilities >= threshold).astype(int)))

    # Platt metrics at 0.5, kept for comparison with the deployed rule
    save_calibration(
        args.metrics_file, platt,
        {
            "fitted_on": "validation",
            "n_val": int(len(val_labels)),
            "test": {k: float(v) for k, v in metrics.items()},
        }
    )

    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        labels, probabilities,
        os.path.join(args.output_dir, "confusion_matrix.png"),
        threshold=threshold
    )
    plot_roc_pr(
        labels, probabilities, metrics,
        os.path.join(args.output_dir, "roc_pr_curves.png")
    )
    plot_calibration(
        labels,
        [
            ("Uncalibrated", test_probabilities["uncalibrated"],
             test_metrics["uncalibrated"]),
            (f"Temperature (T = {temperature['temperature']:.2f})",
             test_probabilities["temperature"], test_metrics["temperature"]),
            ("Platt", probabilities, metrics),
        ],
        os.path.join(args.output_dir, "calibration.png")
    )

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "n_test": int(len(labels)),
                "calibration": platt,
                "metrics": {k: float(v) for k, v in metrics.items()},
                "ci_95": test_intervals["platt"],
                "comparison": {
                    name: {
                        "calibration": calibrators[name],
                        "metrics": {
                            k: float(v) for k, v in test_metrics[name].items()
                        },
                        "nll": log_loss(labels, test_probabilities[name]),
                        "ci_95": test_intervals[name],
                    }
                    for name in calibrators
                },
            },
            f,
            indent=4
        )

    print(f"\nSaved plots and metrics to {args.output_dir}/")


if __name__ == "__main__":

    main()
