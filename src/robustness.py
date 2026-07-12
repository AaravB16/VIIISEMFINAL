"""Robustness analysis for augmented-dataset evaluation.

Computes performance deltas between the original dataset and each
augmented variant, ranks models by stability, and persists a structured
robustness report.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import ensure_dir

logger = logging.getLogger(__name__)

# Metrics to compare for robustness (must exist in result dicts).
_ROBUSTNESS_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "detection_rate",
    "false_positive_rate",
    "train_time_seconds",
    "per_sample_latency_ms",
]


def compute_robustness_deltas(
    all_results: list[dict],
) -> list[dict]:
    """Compute performance deltas between original and augmented results.

    For every ``(model, split, metric)`` triple, the delta is:

        Δ = value_augmented − value_original

    A negative Δ for a quality metric (e.g. F1) means degradation.

    Args:
        all_results: List of result dicts, each containing a
            ``"dataset_variant"`` field.

    Returns:
        List of delta dicts, each with keys:
        ``model``, ``split_strategy``, ``dataset_variant``,
        ``metric``, ``original_value``, ``augmented_value``, ``delta``.
    """
    # Index original results by (model, split)
    original_lookup: dict[tuple[str, str], dict] = {}
    for r in all_results:
        if r.get("dataset_variant", "original") == "original":
            key = (r["model"], r["split_strategy"])
            original_lookup[key] = r

    deltas: list[dict] = []
    for r in all_results:
        variant = r.get("dataset_variant", "original")
        if variant == "original":
            continue

        key = (r["model"], r["split_strategy"])
        orig = original_lookup.get(key)
        if orig is None:
            continue

        for metric in _ROBUSTNESS_METRICS:
            orig_val = orig.get(metric)
            aug_val = r.get(metric)
            if orig_val is None or aug_val is None:
                continue
            delta = float(aug_val) - float(orig_val)
            deltas.append({
                "model": r["model"],
                "split_strategy": r["split_strategy"],
                "dataset_variant": variant,
                "metric": metric,
                "original_value": round(float(orig_val), 6),
                "augmented_value": round(float(aug_val), 6),
                "delta": round(delta, 6),
            })

    logger.info("Computed %d robustness deltas", len(deltas))
    return deltas


def find_most_robust_model(
    deltas: list[dict],
) -> dict:
    """Identify the most robust model across all augmentations.

    Robustness score = mean |Δ| for the core quality metrics
    (accuracy, precision, recall, f1_score, roc_auc) across all
    augmented variants and splits.  Lower is better.

    Args:
        deltas: Output of :func:`compute_robustness_deltas`.

    Returns:
        Dict with ``model``, ``robustness_score``, and per-metric
        average absolute deltas.
    """
    quality_metrics = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    # model -> list of |delta| values
    model_abs_deltas: dict[str, list[float]] = defaultdict(list)
    model_metric_deltas: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for d in deltas:
        if d["metric"] not in quality_metrics:
            continue
        model_abs_deltas[d["model"]].append(abs(d["delta"]))
        model_metric_deltas[d["model"]][d["metric"]].append(abs(d["delta"]))

    if not model_abs_deltas:
        return {"model": None, "robustness_score": None, "details": {}}

    # Rank models by average absolute delta (lower = more robust)
    rankings: list[dict] = []
    for model, abs_vals in model_abs_deltas.items():
        per_metric = {
            m: round(float(np.mean(vals)), 6)
            for m, vals in model_metric_deltas[model].items()
        }
        rankings.append({
            "model": model,
            "robustness_score": round(float(np.mean(abs_vals)), 6),
            "per_metric_avg_abs_delta": per_metric,
        })

    rankings.sort(key=lambda x: x["robustness_score"])
    best = rankings[0]

    logger.info(
        "Most robust model: %s (robustness_score=%.6f)",
        best["model"], best["robustness_score"],
    )
    return {
        "most_robust": best,
        "rankings": rankings,
    }


def save_robustness_report(
    deltas: list[dict],
    robustness_summary: dict,
    output_dir: str | Path,
    filename: str = "robustness_analysis.json",
) -> Path:
    """Persist the full robustness analysis to JSON.

    Args:
        deltas: List of delta dicts.
        robustness_summary: Output of :func:`find_most_robust_model`.
        output_dir: Directory for the report file.
        filename: Report filename.

    Returns:
        Path to the saved file.
    """
    output_dir = ensure_dir(output_dir)
    filepath = output_dir / filename

    report = {
        "robustness_deltas": deltas,
        "summary": robustness_summary,
    }
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Robustness report saved to %s", filepath)
    return filepath


def print_robustness_summary(
    deltas: list[dict],
    robustness_summary: dict,
) -> None:
    """Print a concise robustness summary to stdout."""
    if not deltas:
        print("\nNo robustness data available (augmentation disabled?).\n")
        return

    variants = sorted(set(d["dataset_variant"] for d in deltas))
    models = sorted(set(d["model"] for d in deltas))

    print("\n" + "=" * 80)
    print("ROBUSTNESS EVALUATION")
    print("=" * 80)

    # Per-variant summary
    quality = {"accuracy", "precision", "recall", "f1_score"}
    for variant in variants:
        print(f"\n{'─' * 60}")
        print(f"Variant: {variant}")
        print(f"{'─' * 60}")
        header = f"  {'Model':<18} {'ΔAcc':>10} {'ΔPrec':>10} {'ΔRec':>10} {'ΔF1':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for model in models:
            row_deltas = {
                d["metric"]: d["delta"]
                for d in deltas
                if d["model"] == model
                and d["dataset_variant"] == variant
                and d["metric"] in quality
            }
            # Average across splits for each metric
            metric_means: dict[str, float] = {}
            for m in quality:
                vals = [
                    d["delta"] for d in deltas
                    if d["model"] == model
                    and d["dataset_variant"] == variant
                    and d["metric"] == m
                ]
                metric_means[m] = float(np.mean(vals)) if vals else 0.0

            print(
                f"  {model:<18} "
                f"{metric_means.get('accuracy', 0):>+10.4f} "
                f"{metric_means.get('precision', 0):>+10.4f} "
                f"{metric_means.get('recall', 0):>+10.4f} "
                f"{metric_means.get('f1_score', 0):>+10.4f}"
            )

    # Overall best
    rankings = robustness_summary.get("rankings", [])
    if rankings:
        print(f"\n{'=' * 80}")
        print("ROBUSTNESS RANKING (lower score = more stable)")
        print("=" * 80)
        for i, r in enumerate(rankings, 1):
            marker = " ← MOST ROBUST" if i == 1 else ""
            print(f"  {i}. {r['model']:<18} score={r['robustness_score']:.6f}{marker}")
    print()
