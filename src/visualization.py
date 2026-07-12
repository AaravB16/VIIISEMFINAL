"""Visualization utilities for IDS experiment results.

Generates confusion matrix heatmaps, per-class ROC curves, model
comparison bar charts, and training-time comparisons.  All plots are
saved to the configured results directory.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

from src.utils import ensure_dir

logger = logging.getLogger(__name__)

# Consistent styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12
})


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    labels: list[str],
    model_name: str,
    split_name: str,
    output_dir: str | Path,
) -> Path:
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2-D array or nested list).
        labels: Class labels in matrix order.
        model_name: Model name (used in title and filename).
        split_name: Split strategy name.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = ensure_dir(Path(output_dir) / "figures")
    cm = np.array(cm)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name} ({split_name})")

    filename = f"cm_{model_name}_{split_name}.png"
    filepath = output_dir / filename
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Confusion matrix saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
    model_name: str,
    split_name: str,
    output_dir: str | Path,
) -> Path:
    """Plot per-class ROC curves with micro/macro averages.

    Args:
        y_true: Ground truth labels (string or integer).
        y_proba: Predicted probability matrix (n_samples × n_classes).
        labels: Ordered class labels matching columns of *y_proba*.
        model_name: Model name.
        split_name: Split strategy name.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = ensure_dir(Path(output_dir) / "figures")

    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y_true)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    n_classes = y_bin.shape[1]
    fig, ax = plt.subplots(figsize=(11, 6))

    # Per-class curves
    for i in range(min(n_classes, y_proba.shape[1])):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        label_name = labels[i] if i < len(labels) else f"Class {i}"
        ax.plot(fpr, tpr, lw=1.5, label=f"{label_name} (AUC={roc_auc:.3f})")

    # Macro-average ROC
    all_fpr = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(min(n_classes, y_proba.shape[1])):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(
        all_fpr, mean_tpr, "k--", lw=2,
        label=f"Macro-avg (AUC={macro_auc:.3f})",
    )

    ax.plot([0, 1], [0, 1], "gray", linestyle=":", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name} ({split_name})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    filename = f"roc_{model_name}_{split_name}.png"
    filepath = output_dir / filename
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"ROC curves saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Model comparison bar charts
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: list[dict],
    metric: str = "f1_score",
    output_dir: str | Path = "results",
) -> Path:
    """Bar chart comparing models across split strategies for a given metric.

    Args:
        results: List of experiment result dicts.
        metric: Metric to compare (e.g. ``"f1_score"``, ``"accuracy"``).
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = ensure_dir(Path(output_dir) / "figures")

    models = sorted(set(r["model"] for r in results))
    splits = sorted(set(r["split_strategy"] for r in results))

    fig, ax = plt.subplots(figsize=(max(11, len(models) * 2.5), 5))
    width = 0.8 / max(len(splits), 1)

    for i, split in enumerate(splits):
        values = []
        for model in models:
            match = [r for r in results if r["model"] == model and r["split_strategy"] == split]
            values.append(match[0][metric] if match else 0)
        x = np.arange(len(models)) + i * width
        ax.bar(x, values, width, label=split)

    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")
    ax.set_xticks(np.arange(len(models)) + width * (len(splits) - 1) / 2)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    filepath = output_dir / f"comparison_{metric}.png"
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Comparison chart saved: {filepath}")
    return filepath


def plot_training_time_comparison(
    results: list[dict],
    output_dir: str | Path = "results",
) -> Path:
    """Bar chart comparing training times across models.

    Args:
        results: List of experiment result dicts.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    output_dir = ensure_dir(Path(output_dir) / "figures")

    # Average training time per model across splits
    model_times: dict[str, list[float]] = {}
    for r in results:
        model_times.setdefault(r["model"], []).append(r["train_time_seconds"])

    models = sorted(model_times.keys())
    avg_times = [np.mean(model_times[m]) for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), 5))
    bars = ax.bar(models, avg_times, color=sns.color_palette("muted", len(models)))
    ax.set_xlabel("Model")
    ax.set_ylabel("Avg Training Time (s)")
    ax.set_title("Training Time Comparison")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    for bar, t in zip(bars, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.1f}s", ha="center", va="bottom", fontsize=8)

    filepath = output_dir / "training_time_comparison.png"
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Training time chart saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Robustness visualizations
# ---------------------------------------------------------------------------

def plot_robustness_heatmap(
    deltas: list[dict],
    output_dir: str | Path,
) -> Path | None:
    """Heatmap of F1 deltas: models × augmented variants.

    Args:
        deltas: List of robustness delta dicts.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure, or *None* if not enough data.
    """
    if not deltas:
        return None

    output_dir = ensure_dir(Path(output_dir) / "figures")

    # Filter to F1 deltas and average across splits
    f1_deltas = [d for d in deltas if d["metric"] == "f1_score"]
    if not f1_deltas:
        return None

    models = sorted(set(d["model"] for d in f1_deltas))
    variants = sorted(set(d["dataset_variant"] for d in f1_deltas))

    matrix = np.zeros((len(models), len(variants)))
    for d in f1_deltas:
        mi = models.index(d["model"])
        vi = variants.index(d["dataset_variant"])
        matrix[mi, vi] += d["delta"]

    # Average over splits that contributed to each cell
    counts = np.zeros_like(matrix)
    for d in f1_deltas:
        mi = models.index(d["model"])
        vi = variants.index(d["dataset_variant"])
        counts[mi, vi] += 1
    counts[counts == 0] = 1
    matrix /= counts

    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 2.5), max(4, len(models))))
    sns.heatmap(
        matrix, annot=True, fmt="+.4f", cmap="RdYlGn", center=0,
        xticklabels=variants, yticklabels=models, ax=ax,
    )
    ax.set_xlabel("Augmentation Variant")
    ax.set_ylabel("Model")
    ax.set_title("Robustness Heatmap — Avg F1 Score Delta")

    filepath = output_dir / "robustness_heatmap_f1.png"
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Robustness heatmap saved: {filepath}")
    return filepath


def plot_performance_drop_chart(
    deltas: list[dict],
    output_dir: str | Path,
) -> Path | None:
    """Grouped bar chart showing per-metric avg delta per model.

    Args:
        deltas: List of robustness delta dicts.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure, or *None* if not enough data.
    """
    if not deltas:
        return None

    output_dir = ensure_dir(Path(output_dir) / "figures")

    quality = ["accuracy", "precision", "recall", "f1_score"]
    models = sorted(set(d["model"] for d in deltas))
    if not models:
        return None

    # Average delta per model per metric (across all variants and splits)
    metric_avgs: dict[str, list[float]] = {m: [] for m in quality}
    for model in models:
        for metric in quality:
            vals = [
                d["delta"] for d in deltas
                if d["model"] == model and d["metric"] == metric
            ]
            metric_avgs[metric].append(float(np.mean(vals)) if vals else 0.0)

    x = np.arange(len(models))
    width = 0.8 / len(quality)
    fig, ax = plt.subplots(figsize=(max(11, len(models) * 2.5), 5))

    for i, metric in enumerate(quality):
        ax.bar(x + i * width, metric_avgs[metric], width, label=metric)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Model")
    ax.set_ylabel("Avg Δ (augmented − original)")
    ax.set_title("Performance Drop Under Augmentation")
    ax.set_xticks(x + width * (len(quality) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    filepath = output_dir / "performance_drop_chart.png"
    fig.savefig(filepath)
    plt.close(fig)

    logger.info(f"Performance drop chart saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Generate all visualizations
# ---------------------------------------------------------------------------

def generate_all_visualizations(
    results: list[dict],
    output_dir: str | Path,
    y_true_map: dict | None = None,
    y_proba_map: dict | None = None,
    robustness_deltas: list[dict] | None = None,
) -> None:
    """Generate all available visualizations from experiment results.

    Args:
        results: List of experiment result dicts.
        output_dir: Base output directory.
        y_true_map: Optional mapping ``{(model, split): y_true}`` for ROC.
        y_proba_map: Optional mapping ``{(model, split): y_proba}`` for ROC.
        robustness_deltas: Optional list of robustness delta dicts.
    """
    output_dir = Path(output_dir)

    # Confusion matrices
    for r in results:
        if "confusion_matrix" in r and r["confusion_matrix"]:
            plot_confusion_matrix(
                cm=r["confusion_matrix"],
                labels=r.get("confusion_matrix_labels", []),
                model_name=r["model"],
                split_name=r["split_strategy"],
                output_dir=output_dir,
            )

    # ROC curves (if probability data available)
    if y_true_map and y_proba_map:
        for key in y_proba_map:
            model_name, split_name = key
            y_true = y_true_map[key]
            y_proba = y_proba_map[key]
            labels = sorted(set(y_true))
            plot_roc_curves(
                y_true=y_true,
                y_proba=y_proba,
                labels=labels,
                model_name=model_name,
                split_name=split_name,
                output_dir=output_dir,
            )

    # Comparison charts
    for metric in ["accuracy", "f1_score", "precision", "recall"]:
        if any(metric in r for r in results):
            plot_model_comparison(results, metric=metric, output_dir=output_dir)

    plot_training_time_comparison(results, output_dir=output_dir)

    # Robustness visualizations (when augmentation data is available)
    if robustness_deltas:
        plot_robustness_heatmap(robustness_deltas, output_dir=output_dir)
        plot_performance_drop_chart(robustness_deltas, output_dir=output_dir)

    logger.info(f"All visualizations saved to {output_dir / 'figures'}")
