#!/usr/bin/env python3
"""Create timestamped visual summaries from experiment artifacts.

This script reads saved result artifacts and generates a curated set of
plots for quick project review.  All plots are saved into:

    results/runs/<timestamp>[/<tag>]/figures/

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --tag phase2_update
    python scripts/visualize_results.py --results results/experiment_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, get_project_root
from src.visualization import (
    plot_model_comparison,
    plot_performance_drop_chart,
    plot_robustness_heatmap,
    plot_training_time_comparison,
)

logger = logging.getLogger("visualize_results")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate timestamped visualizations from Project Aarav results",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to experiment results JSON (default: results/experiment_results.json)",
    )
    parser.add_argument(
        "--robustness",
        type=str,
        default=None,
        help="Path to robustness JSON (default: results/robustness_analysis.json)",
    )
    parser.add_argument(
        "--feature-benchmark",
        type=str,
        default=None,
        help=(
            "Path to feature-selection benchmark JSON "
            "(default: auto-detect _ii, standard, or _smoke file)"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output root directory (default: results/runs)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional suffix tag appended to timestamp directory name",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _read_json(path: Path, required: bool = False):
    if not path.exists():
        msg = f"File not found: {path}"
        if required:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return None
    with open(path, "r") as f:
        return json.load(f)


def _resolve_feature_benchmark_path(base_results: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)

    candidates = [
        base_results / "feature_selection_benchmark_ii.json",
        base_results / "feature_selection_benchmark.json",
        base_results / "feature_selection_benchmark_smoke.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _build_run_dir(output_root: Path, tag: str | None) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")
    dirname = f"{ts}_{tag}" if tag else ts
    run_dir = ensure_dir(output_root / dirname)
    ensure_dir(run_dir / "figures")
    return run_dir


def _save_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str,
    run_dir: Path,
) -> Path | None:
    if metric not in df.columns:
        return None

    pivot = (
        df.pivot_table(
            index="model",
            columns="split_strategy",
            values=metric,
            aggfunc="mean",
        )
        .sort_index()
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.6), max(4, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Split Strategy")
    ax.set_ylabel("Model")

    out = run_dir / "figures" / filename
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved heatmap: %s", out)
    return out


def _plot_variant_delta_f1(df_all: pd.DataFrame, run_dir: Path) -> Path | None:
    if "dataset_variant" not in df_all.columns:
        return None
    if df_all["dataset_variant"].nunique() <= 1:
        return None

    base = (
        df_all[df_all["dataset_variant"] == "original"][
            ["model", "split_strategy", "f1_score"]
        ]
        .rename(columns={"f1_score": "f1_original"})
    )
    aug = df_all[df_all["dataset_variant"] != "original"][
        ["model", "split_strategy", "dataset_variant", "f1_score"]
    ]
    if base.empty or aug.empty:
        return None

    merged = aug.merge(base, on=["model", "split_strategy"], how="left")
    merged = merged.dropna(subset=["f1_original"])
    if merged.empty:
        return None
    merged["delta_f1"] = merged["f1_score"] - merged["f1_original"]

    grouped = (
        merged.groupby(["dataset_variant", "model"], as_index=False)["delta_f1"]
        .mean()
        .sort_values(["dataset_variant", "model"])
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(
        data=grouped,
        x="dataset_variant",
        y="delta_f1",
        hue="model",
        ax=ax,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Average F1 Delta vs Original by Augmentation Variant")
    ax.set_xlabel("Dataset Variant")
    ax.set_ylabel("Avg ΔF1 (variant - original)")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')

    out = run_dir / "figures" / "variant_delta_f1.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved augmentation delta chart: %s", out)
    return out


def _plot_robustness_rankings(robustness_data: dict, run_dir: Path) -> Path | None:
    summary = robustness_data.get("summary", {}) if robustness_data else {}
    rankings = summary.get("rankings", [])
    if not rankings:
        return None

    df = pd.DataFrame(rankings)
    if "model" not in df.columns or "robustness_score" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df.sort_values("robustness_score"),
        x="robustness_score",
        y="model",
        orient="h",
        ax=ax,
    )
    ax.set_title("Model Robustness Ranking (Lower is Better)")
    ax.set_xlabel("Robustness Score")
    ax.set_ylabel("Model")

    out = run_dir / "figures" / "robustness_ranking.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved robustness ranking chart: %s", out)
    return out


def _plot_feature_selection_tradeoffs(
    fs_data: dict,
    run_dir: Path,
) -> list[Path]:
    generated: list[Path] = []
    if not fs_data:
        return generated

    rows = fs_data.get("results", [])
    if rows:
        df = pd.DataFrame(rows)
        if not df.empty and {"feature_count", "f1_score", "model"} <= set(df.columns):
            fig, ax = plt.subplots(figsize=(12, 5))
            for model, sub in df.groupby("model"):
                agg = (
                    sub.groupby("feature_count", as_index=False)["f1_score"]
                    .max()
                    .sort_values("feature_count")
                )
                ax.plot(
                    agg["feature_count"],
                    agg["f1_score"],
                    marker="o",
                    linewidth=1.8,
                    label=model,
                )
            ax.set_title("Feature Count vs Best F1 (by Model)")
            ax.set_xlabel("Selected Feature Count")
            ax.set_ylabel("F1 Score")
            ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
            out = run_dir / "figures" / "feature_selection_tradeoff_f1.png"
            fig.savefig(out)
            plt.close(fig)
            generated.append(out)
            logger.info("Saved feature-selection tradeoff chart: %s", out)

    recs = fs_data.get("recommendations", {})
    if recs:
        rec_rows: list[dict] = []
        for model, rec in recs.items():
            setting = rec.get("recommended_setting", {}) or {}
            rec_rows.append({
                "model": model,
                "feature_count": setting.get("feature_count"),
                "f1_score": setting.get("f1_score"),
                "setting_name": setting.get("setting_name"),
            })

        rec_df = pd.DataFrame(rec_rows).dropna(subset=["feature_count"])
        if not rec_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.barplot(data=rec_df, x="model", y="feature_count", ax=ax)
            ax.set_title("Recommended Feature Count by Model")
            ax.set_xlabel("Model")
            ax.set_ylabel("Feature Count")
            ax.set_ylim(0, max(rec_df["feature_count"]) + 2)

            for i, row in rec_df.reset_index(drop=True).iterrows():
                text = f"{row['setting_name']}\\nF1={row['f1_score']:.4f}"
                ax.text(i, row["feature_count"] + 0.05, text, ha="center", va="bottom", fontsize=8)

            out = run_dir / "figures" / "feature_selection_recommendations.png"
            fig.savefig(out)
            plt.close(fig)
            generated.append(out)
            logger.info("Saved feature-selection recommendations chart: %s", out)

    return generated


def _write_run_manifest(
    run_dir: Path,
    source_paths: dict[str, str],
    generated_files: list[Path],
) -> Path:
    payload = {
        "run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifacts": source_paths,
        "generated_files": [str(p) for p in generated_files],
        "n_generated_files": len(generated_files),
    }
    out = run_dir / "run_summary.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    results_root = get_project_root() / "results"
    results_path = Path(args.results) if args.results else results_root / "experiment_results.json"
    robustness_path = Path(args.robustness) if args.robustness else results_root / "robustness_analysis.json"
    feature_benchmark_path = _resolve_feature_benchmark_path(results_root, args.feature_benchmark)
    output_root = Path(args.output_root) if args.output_root else results_root / "runs"

    run_dir = _build_run_dir(output_root=output_root, tag=args.tag)
    logger.info("Output run directory: %s", run_dir)

    results = _read_json(results_path, required=True)
    robustness_data = _read_json(robustness_path, required=False)
    fs_data = _read_json(feature_benchmark_path, required=False)

    if not isinstance(results, list) or not results:
        raise ValueError(f"Results file has no usable rows: {results_path}")

    df_all = pd.DataFrame(results)
    if "dataset_variant" not in df_all.columns:
        df_all["dataset_variant"] = "original"
    df_original = df_all[df_all["dataset_variant"].fillna("original") == "original"].copy()
    if df_original.empty:
        df_original = df_all.copy()

    generated: list[Path] = []
    original_records = df_original.to_dict(orient="records")

    # Standard model comparison plots on original-variant results.
    for metric in ["accuracy", "f1_score", "precision", "recall", "detection_rate"]:
        if metric in df_original.columns:
            p = plot_model_comparison(original_records, metric=metric, output_dir=run_dir)
            generated.append(p)
    generated.append(plot_training_time_comparison(original_records, output_dir=run_dir))

    # Matrix-style overview plots.
    p = _save_metric_heatmap(
        df=df_original,
        metric="f1_score",
        title="F1 Score Heatmap (Original Variant)",
        filename="heatmap_f1_original.png",
        run_dir=run_dir,
    )
    if p:
        generated.append(p)

    p = _save_metric_heatmap(
        df=df_original,
        metric="detection_rate",
        title="Detection Rate Heatmap (Original Variant)",
        filename="heatmap_detection_rate_original.png",
        run_dir=run_dir,
    )
    if p:
        generated.append(p)

    p = _plot_variant_delta_f1(df_all=df_all, run_dir=run_dir)
    if p:
        generated.append(p)

    # Robustness charts from robustness artifact.
    if robustness_data:
        deltas = robustness_data.get("robustness_deltas", [])
        p = plot_robustness_heatmap(deltas, output_dir=run_dir)
        if p:
            generated.append(p)
        p = plot_performance_drop_chart(deltas, output_dir=run_dir)
        if p:
            generated.append(p)
        p = _plot_robustness_rankings(robustness_data, run_dir=run_dir)
        if p:
            generated.append(p)

    # Feature-selection charts (if artifact is available).
    generated.extend(_plot_feature_selection_tradeoffs(fs_data, run_dir=run_dir))

    manifest = _write_run_manifest(
        run_dir=run_dir,
        source_paths={
            "results": str(results_path),
            "robustness": str(robustness_path),
            "feature_benchmark": str(feature_benchmark_path),
        },
        generated_files=generated,
    )

    print(f"Visualization run complete: {run_dir}")
    print(f"Generated figures: {len(generated)}")
    print(f"Run summary: {manifest}")


if __name__ == "__main__":
    main()
