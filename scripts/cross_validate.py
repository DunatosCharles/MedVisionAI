import argparse
import json
import os

import numpy as np

from src.evaluate.metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
)
from src.train.train_resnet import train


METRICS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
]


def main():

    parser = argparse.ArgumentParser(
        description="Lesion-grouped k-fold cross-validation on BUSI"
    )

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--output-dir", default="results/busi_cv")

    args = parser.parse_args()


    fold_metrics = []
    all_labels = []
    all_probabilities = []

    for fold in range(args.n_folds):

        print(f"\n===== Fold {fold + 1}/{args.n_folds} =====")

        fold_args = argparse.Namespace(**vars(args), fold=fold)

        test = train(fold_args)

        metrics = classification_metrics(
            test["labels"],
            test["probabilities"]
        )

        fold_metrics.append(metrics)

        all_labels.extend(test["labels"])
        all_probabilities.extend(test["probabilities"])


    # Every usable image is in exactly one test fold, so the pooled
    # out-of-fold predictions cover the whole dataset once
    pooled = classification_metrics(all_labels, all_probabilities)

    intervals = bootstrap_confidence_intervals(
        all_labels,
        all_probabilities
    )

    summary = {
        name: {
            "mean": float(np.mean([m[name] for m in fold_metrics])),
            "std": float(np.std([m[name] for m in fold_metrics])),
            "pooled": float(pooled[name]),
            "pooled_ci_95": intervals[name],
        }
        for name in METRICS
    }


    print(f"\nCross-validation results ({args.n_folds} folds, "
          f"{len(all_labels)} images)\n")

    print(f"{'metric':<12} {'mean ± std':>16}   {'pooled [95% CI]':>26}")

    for name, row in summary.items():

        low, high = row["pooled_ci_95"]

        print(
            f"{name:<12} {row['mean']:.3f} ± {row['std']:.3f}   "
            f"{row['pooled']:.3f} [{low:.3f}, {high:.3f}]"
        )


    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "cv_metrics.json"), "w") as f:
        json.dump(
            {
                "n_folds": args.n_folds,
                "n_images": len(all_labels),
                "summary": summary,
                "folds": [
                    {k: float(v) for k, v in m.items()}
                    for m in fold_metrics
                ],
                "args": vars(args),
            },
            f,
            indent=4
        )

    print(f"\nSaved {args.output_dir}/cv_metrics.json")


if __name__ == "__main__":

    main()
