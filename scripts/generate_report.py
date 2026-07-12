#!/usr/bin/env python3
"""
Generate a comparison report from saved experiment results.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --results results/experiment_results.json
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_project_root


def load_results(filepath: str | Path) -> list[dict]:
    """Load experiment results from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def _format_variant_table(
    results: list[dict],
    variant: str,
) -> list[str]:
    """Build a performance table for a single dataset variant."""
    lines: list[str] = []
    subset = [r for r in results if r.get("dataset_variant", "original") == variant]
    if not subset:
        return lines

    # Group by split strategy
    splits: dict[str, list[dict]] = {}
    for r in subset:
        splits.setdefault(r["split_strategy"], []).append(r)

    for split_name, split_results in splits.items():
        lines.append(f"\n{'─' * 60}")
        lines.append(f"Split Strategy: {split_name}")
        lines.append(f"{'─' * 60}")

        header = (
            f"  {'Model':<18} {'Acc':>8} {'Prec':>8} "
            f"{'Rec':>8} {'F1':>8} {'AUC':>8} {'Train(s)':>10} {'Lat(ms)':>10}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for r in split_results:
            auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "  N/A  "
            lines.append(
                f"  {r['model']:<18} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
                f"{r['recall']:>8.4f} {r['f1_score']:>8.4f} "
                f"{auc_str:>8} {r['train_time_seconds']:>10.2f} "
                f"{r['per_sample_latency_ms']:>10.4f}"
            )

        lines.append("")
        best = max(split_results, key=lambda x: x["f1_score"])
        lines.append(f"  Best (F1): {best['model']} (F1={best['f1_score']:.4f})")

    return lines


def generate_text_report(
    results: list[dict],
    robustness_data: dict | None = None,
) -> str:
    """Generate a formatted text report.

    Args:
        results: List of experiment result dicts.
        robustness_data: Optional dict with keys ``"deltas"`` and
            ``"summary"`` from the robustness analysis JSON.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("CICIoV2024 EVALUATION FRAMEWORK - EXPERIMENT REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Determine variants present
    variants = sorted(set(r.get("dataset_variant", "original") for r in results))
    has_augmentation = len(variants) > 1

    # ── Section 1: Baseline Performance (original dataset) ──
    lines.append("\nSECTION 1: BASELINE PERFORMANCE (original dataset)")
    lines.append("=" * 60)
    lines.extend(_format_variant_table(results, "original"))

    # Overall best (baseline)
    original_results = [r for r in results if r.get("dataset_variant", "original") == "original"]
    if original_results:
        lines.append(f"\n{'=' * 80}")
        lines.append("OVERALL BEST MODELS (baseline)")
        lines.append("=" * 80)

        best_overall = max(original_results, key=lambda x: x["f1_score"])
        lines.append(
            f"  Best F1: {best_overall['model']} on {best_overall['split_strategy']} "
            f"(F1={best_overall['f1_score']:.4f})"
        )

        fastest = min(original_results, key=lambda x: x["per_sample_latency_ms"])
        lines.append(
            f"  Fastest: {fastest['model']} on {fastest['split_strategy']} "
            f"(Latency={fastest['per_sample_latency_ms']:.4f}ms)"
        )

    # ── Section 2: Augmentation Performance Tables ──
    if has_augmentation:
        aug_variants = [v for v in variants if v != "original"]
        for idx, variant in enumerate(aug_variants, 1):
            lines.append(f"\n\nSECTION 2.{idx}: AUGMENTATION PERFORMANCE — {variant}")
            lines.append("=" * 60)
            lines.extend(_format_variant_table(results, variant))

    # ── Section 3: Robustness Comparison Summary ──
    if robustness_data and robustness_data.get("deltas"):
        deltas = robustness_data["deltas"]
        summary = robustness_data.get("summary", {})

        lines.append(f"\n\n{'=' * 80}")
        lines.append("SECTION 3: ROBUSTNESS COMPARISON SUMMARY")
        lines.append("=" * 80)

        quality_metrics = ["accuracy", "precision", "recall", "f1_score"]
        aug_variants = sorted(set(d["dataset_variant"] for d in deltas))
        models = sorted(set(d["model"] for d in deltas))

        for variant in aug_variants:
            lines.append(f"\n  Variant: {variant}")
            lines.append(f"  {'Model':<18} {'ΔAcc':>10} {'ΔPrec':>10} {'ΔRec':>10} {'ΔF1':>10}")
            lines.append("  " + "-" * 60)

            for model in models:
                metric_means: dict[str, float] = {}
                for m in quality_metrics:
                    vals = [
                        d["delta"] for d in deltas
                        if d["model"] == model
                        and d["dataset_variant"] == variant
                        and d["metric"] == m
                    ]
                    metric_means[m] = sum(vals) / len(vals) if vals else 0.0

                lines.append(
                    f"  {model:<18} "
                    f"{metric_means.get('accuracy', 0):>+10.4f} "
                    f"{metric_means.get('precision', 0):>+10.4f} "
                    f"{metric_means.get('recall', 0):>+10.4f} "
                    f"{metric_means.get('f1_score', 0):>+10.4f}"
                )

        # Most robust model
        rankings = summary.get("rankings", [])
        if rankings:
            lines.append(f"\n{'─' * 60}")
            lines.append("MOST ROBUST MODEL")
            lines.append(f"{'─' * 60}")
            for i, r in enumerate(rankings, 1):
                marker = " ← MOST ROBUST" if i == 1 else ""
                lines.append(
                    f"  {i}. {r['model']:<18} "
                    f"robustness_score={r['robustness_score']:.6f}{marker}"
                )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output report file path (prints to stdout if not specified)",
    )
    args = parser.parse_args()

    results_path = args.results
    if results_path is None:
        results_path = get_project_root() / "results" / "experiment_results.json"

    results = load_results(results_path)

    # Load robustness analysis if available
    robustness_data = None
    robustness_path = Path(results_path).parent / "robustness_analysis.json"
    if robustness_path.exists():
        robustness_data = load_results(robustness_path)

    report = generate_text_report(results, robustness_data=robustness_data)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"Report saved to {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
