#!/usr/bin/env python3
"""Benchmark feature-selection strategies and recommend minimal feature sets."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.deep_learning import is_deep_learning_model
from src.evaluation import compute_metrics
from src.feature_engineering import prepare_features
from src.feature_selection import apply_feature_selection
from src.models import get_enabled_models, get_model, predict_model, train_model
from src.preprocessing import preprocess_postsplit, preprocess_presplit
from src.splitting import random_split, scenario_split
from src.utils import ensure_dir, get_project_root, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature-selection benchmark for Project Aarav",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["random", "scenario"],
        help="Data split strategy for benchmarking",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run benchmark only for one model",
    )
    parser.add_argument(
        "--target-f1",
        type=float,
        default=None,
        help="Absolute F1 target for recommendation (default: baseline - drop_tolerance)",
    )
    parser.add_argument(
        "--drop-tolerance",
        type=float,
        default=0.002,
        help="Allowed F1 drop from baseline for minimal-feature recommendation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/feature_selection_benchmark.json)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    unique = np.unique(y)
    forward = {int(old): int(new) for new, old in enumerate(unique)}
    reverse = {int(new): int(old) for new, old in enumerate(unique)}
    return np.array([forward[int(v)] for v in y], dtype=np.intp), reverse


def _build_candidate_fs_settings(max_features: int) -> list[dict]:
    """Create benchmark settings across MI/correlation/PCA combinations."""
    k_values = sorted(set([
        3, 4, 5, 6, 8, min(10, max_features), min(12, max_features),
    ]))
    k_values = [k for k in k_values if 1 <= k <= max_features]

    settings: list[dict] = [
        {
            "name": "baseline_no_fs",
            "config": {"enabled": False},
        },
        {
            "name": "corr_only_0.95",
            "config": {
                "enabled": True,
                "method": None,
                "k": "all",
                "correlation_filter": {"enabled": True, "threshold": 0.95},
                "pca": {"enabled": False, "n_components": 0.95},
            },
        },
        {
            "name": "corr_only_0.90",
            "config": {
                "enabled": True,
                "method": None,
                "k": "all",
                "correlation_filter": {"enabled": True, "threshold": 0.90},
                "pca": {"enabled": False, "n_components": 0.95},
            },
        },
        {
            "name": "pca_only_0.95",
            "config": {
                "enabled": True,
                "method": None,
                "k": "all",
                "correlation_filter": {"enabled": False, "threshold": 0.95},
                "pca": {"enabled": True, "n_components": 0.95},
            },
        },
        {
            "name": "corr_pca_0.95",
            "config": {
                "enabled": True,
                "method": None,
                "k": "all",
                "correlation_filter": {"enabled": True, "threshold": 0.95},
                "pca": {"enabled": True, "n_components": 0.95},
            },
        },
    ]

    for k in k_values:
        settings.append(
            {
                "name": f"mi_k_{k}",
                "config": {
                    "enabled": True,
                    "method": "mutual_information",
                    "k": k,
                    "correlation_filter": {"enabled": False, "threshold": 0.95},
                    "pca": {"enabled": False, "n_components": 0.95},
                },
            }
        )
        settings.append(
            {
                "name": f"corr_mi_k_{k}",
                "config": {
                    "enabled": True,
                    "method": "mutual_information",
                    "k": k,
                    "correlation_filter": {"enabled": True, "threshold": 0.95},
                    "pca": {"enabled": False, "n_components": 0.95},
                },
            }
        )

    # De-duplicate by name while preserving order.
    unique = []
    seen = set()
    for s in settings:
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        unique.append(s)
    return unique


def _get_split(df, config: dict, split_name: str) -> dict:
    if split_name == "random":
        return random_split(df, config)
    if split_name == "scenario":
        return scenario_split(df, config)
    raise ValueError(f"Unsupported split: {split_name}")


def _evaluate_single_setting(
    train_df,
    test_df,
    config: dict,
    model_name: str,
    model_params: dict,
    fs_setting: dict,
):
    cfg = copy.deepcopy(config)
    cfg["feature_selection"] = fs_setting

    train_norm, test_norm, _ = preprocess_postsplit(train_df.copy(), test_df.copy(), cfg)
    X_train, y_train_raw = prepare_features(train_norm, cfg, for_sequence_model=False)
    X_test, y_test_raw = prepare_features(test_norm, cfg, for_sequence_model=False)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    y_train_contiguous, reverse_label_map = _remap_labels(y_train)

    X_tr, X_te = apply_feature_selection(X_train, y_train_contiguous, X_test, cfg)
    feature_count = int(X_tr.shape[1])

    model = get_model(
        model_name,
        model_params,
        seed=cfg["seed"],
        n_jobs=cfg.get("n_jobs", -1),
    )
    model, train_time = train_model(model, X_tr, y_train_contiguous)
    y_pred_contiguous, y_proba, inference_time = predict_model(model, X_te)
    y_pred = np.array(
        [reverse_label_map.get(int(v), int(v)) for v in y_pred_contiguous],
        dtype=np.intp,
    )

    y_true_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    metrics = compute_metrics(
        y_true_decoded,
        y_pred_decoded,
        y_proba=y_proba,
        average=cfg["evaluation"]["average"],
    )

    return {
        "model": model_name,
        "setting_name": fs_setting.get("name", "unknown"),
        "feature_count": feature_count,
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "roc_auc": float(metrics["roc_auc"]) if metrics.get("roc_auc") else None,
        "detection_rate": float(metrics.get("detection_rate", 0.0)),
        "false_positive_rate": float(metrics.get("false_positive_rate", 0.0)),
        "train_time_seconds": float(train_time),
        "per_sample_latency_ms": float(inference_time / max(len(X_te), 1) * 1000.0),
    }


def run_benchmark(config: dict, args: argparse.Namespace) -> dict:
    logger = setup_logging(args.log_level)
    set_seed(config["seed"])

    logger.info("Loading CICIoV2024 for feature-selection benchmarking")
    df = preprocess_presplit(load_dataset(config), config)
    split = _get_split(df, config, args.split)
    train_df, test_df = split["train"], split["test"]

    # Determine baseline feature dimension (without feature selection).
    tmp_cfg = copy.deepcopy(config)
    tmp_cfg["feature_selection"] = {"enabled": False}
    train_tmp, test_tmp, _ = preprocess_postsplit(train_df.copy(), test_df.copy(), tmp_cfg)
    X_tmp, _ = prepare_features(train_tmp, tmp_cfg, for_sequence_model=False)
    max_features = int(X_tmp.shape[1])
    fs_settings = _build_candidate_fs_settings(max_features=max_features)

    models = get_enabled_models(config)
    models = [(n, p) for n, p in models if not is_deep_learning_model(n)]
    if args.model:
        models = [(n, p) for n, p in models if n == args.model]
    if not models:
        raise ValueError("No non-deep-learning models selected for benchmarking")

    rows: list[dict] = []
    for model_name, model_params in models:
        logger.info("Benchmarking model: %s", model_name)
        for setting in fs_settings:
            logger.info("  Setting: %s", setting["name"])
            row = _evaluate_single_setting(
                train_df=train_df,
                test_df=test_df,
                config=config,
                model_name=model_name,
                model_params=model_params,
                fs_setting={"name": setting["name"], **setting["config"]},
            )
            rows.append(row)

    recommendations = {}
    for model_name, _ in models:
        model_rows = [r for r in rows if r["model"] == model_name]
        baseline = next((r for r in model_rows if r["setting_name"] == "baseline_no_fs"), None)
        if baseline is None:
            continue

        dynamic_target = baseline["f1_score"] - float(args.drop_tolerance)
        effective_target = max(dynamic_target, args.target_f1) if args.target_f1 is not None else dynamic_target
        eligible = [r for r in model_rows if r["f1_score"] >= effective_target]

        if eligible:
            best = sorted(
                eligible,
                key=lambda r: (r["feature_count"], -r["f1_score"]),
            )[0]
        else:
            best = sorted(
                model_rows,
                key=lambda r: (-r["f1_score"], r["feature_count"]),
            )[0]

        recommendations[model_name] = {
            "baseline_f1_score": baseline["f1_score"],
            "effective_target_f1": effective_target,
            "recommended_setting": best,
        }

    return {
        "split_strategy": args.split,
        "drop_tolerance": float(args.drop_tolerance),
        "target_f1": float(args.target_f1) if args.target_f1 is not None else None,
        "results": rows,
        "recommendations": recommendations,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    report = run_benchmark(config, args)

    output_path = Path(args.output) if args.output else (
        get_project_root() / "results" / "feature_selection_benchmark.json"
    )
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Feature-selection benchmark saved to {output_path}")


if __name__ == "__main__":
    main()
