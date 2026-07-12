"""Evaluation metrics and reporting for IDS benchmark."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

from src.utils import ensure_dir

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    average: str = "weighted",
) -> dict:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probability estimates (for ROC-AUC).
        average: Averaging method for multi-class metrics.

    Returns:
        Dictionary of metric name -> value.
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["f1_score"] = f1_score(
        y_true, y_pred, average=average, zero_division=0
    )

    # ROC-AUC (multi-class via one-vs-rest)
    if y_proba is not None:
        try:
            lb = LabelBinarizer()
            y_true_bin = lb.fit_transform(y_true)
            # Handle binary case
            if y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
            # Ensure probability columns match label classes
            if y_proba.shape[1] == y_true_bin.shape[1]:
                metrics["roc_auc"] = roc_auc_score(
                    y_true_bin, y_proba, average=average, multi_class="ovr"
                )
            else:
                metrics["roc_auc"] = None
                logger.warning("ROC-AUC: class mismatch between y_true and y_proba")
        except Exception as e:
            metrics["roc_auc"] = None
            logger.warning(f"ROC-AUC computation failed: {e}")
    else:
        metrics["roc_auc"] = None

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["confusion_matrix_labels"] = labels

    # Per-class report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics["classification_report"] = report

    # Detection rate and false positive rate (binary: ATTACK vs BENIGN)
    metrics.update(_compute_ids_rates(y_true, y_pred))

    return metrics


def _compute_ids_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute IDS-specific metrics: detection rate and false positive rate.

    Detection rate = proportion of actual attacks correctly identified.
    False positive rate = proportion of benign samples misclassified as attacks.

    Labels are binarized: ``"BENIGN"`` vs everything else (attack).
    """
    # Binarize: 1 = attack, 0 = benign
    y_true_bin = np.array([0 if str(y) == "BENIGN" else 1 for y in y_true])
    y_pred_bin = np.array([0 if str(y) == "BENIGN" else 1 for y in y_pred])

    # True positives / False positives / etc.
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
    }


def format_results(
    model_name: str,
    split_name: str,
    metrics: dict,
    train_time: float,
    inference_time: float,
    n_train: int,
    n_test: int,
    dataset_variant: str = "original",
) -> dict:
    """Format experiment results into a structured dictionary.

    Args:
        model_name: Name of the model.
        split_name: Name of the splitting strategy.
        metrics: Computed metrics dictionary.
        train_time: Training time in seconds.
        inference_time: Inference time in seconds.
        n_train: Number of training samples.
        n_test: Number of test samples.
        dataset_variant: Dataset variant name (e.g. ``"original"``,
            ``"jitter_augmented"``).

    Returns:
        Formatted results dictionary.
    """
    return {
        "model": model_name,
        "split_strategy": split_name,
        "dataset_variant": dataset_variant,
        "n_train": n_train,
        "n_test": n_test,
        "train_time_seconds": round(train_time, 3),
        "inference_time_seconds": round(inference_time, 3),
        "per_sample_latency_ms": round(inference_time / n_test * 1000, 4),
        "accuracy": round(metrics["accuracy"], 6),
        "precision": round(metrics["precision"], 6),
        "recall": round(metrics["recall"], 6),
        "f1_score": round(metrics["f1_score"], 6),
        "roc_auc": round(metrics["roc_auc"], 6) if metrics["roc_auc"] else None,
        "detection_rate": round(metrics.get("detection_rate", 0), 6),
        "false_positive_rate": round(metrics.get("false_positive_rate", 0), 6),
        "confusion_matrix": metrics["confusion_matrix"],
        "confusion_matrix_labels": metrics["confusion_matrix_labels"],
    }


def save_results(
    results: list[dict],
    output_dir: str | Path,
    filename: str = "experiment_results.json",
) -> Path:
    """Save experiment results to JSON.

    Args:
        results: List of result dictionaries.
        output_dir: Output directory path.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = ensure_dir(output_dir)
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return filepath


def print_summary_table(results: list[dict]) -> None:
    """Print a summary comparison table to console.

    Args:
        results: List of result dictionaries.
    """
    if not results:
        logger.warning("No results to display")
        return

    header = (
        f"{'Model':<18} {'Split':<25} {'Acc':>8} {'Prec':>8} "
        f"{'Rec':>8} {'F1':>8} {'AUC':>8} {'DR':>8} {'FPR':>8} {'Lat(ms)':>10}"
    )
    separator = "-" * len(header)

    print(f"\n{separator}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(separator)
    print(header)
    print(separator)

    for r in results:
        auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "  N/A  "
        print(
            f"{r['model']:<18} {r['split_strategy']:<25} "
            f"{r['accuracy']:>8.4f} {r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} {r['f1_score']:>8.4f} "
            f"{auc_str:>8} {r.get('detection_rate', 0):>8.4f} "
            f"{r.get('false_positive_rate', 0):>8.4f} "
            f"{r['per_sample_latency_ms']:>10.4f}"
        )

    print(separator)
