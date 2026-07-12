#!/usr/bin/env python3
"""Run cross-dataset validation (train on CICIoV2024, test externally)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_dataset import load_external_dataset
from src.data_loader import load_dataset
from src.deep_learning import is_deep_learning_model
from src.evaluation import compute_metrics, format_results, save_results
from src.feature_engineering import prepare_features
from src.feature_selection import apply_feature_selection
from src.models import get_enabled_models, get_model, predict_model, train_model
from src.preprocessing import preprocess_postsplit, preprocess_presplit
from src.utils import ensure_dir, get_project_root, load_config, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-dataset validation for Project Aarav",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="car_hacking,otids,road",
        help="Comma-separated dataset keys to run from cross_dataset_validation.datasets",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip datasets that are missing on disk instead of failing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results/cross_dataset_results.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only one model name",
    )
    parser.add_argument(
        "--include-dl",
        action="store_true",
        help="Include deep-learning models if enabled in config",
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


def _load_enabled_external_datasets(
    config: dict,
    dataset_keys: list[str],
    allow_missing: bool,
    logger: logging.Logger,
) -> dict[str, object]:
    cfg = config.get("cross_dataset_validation", {}).get("datasets", {})
    loaded: dict[str, object] = {}

    for key in dataset_keys:
        key = key.strip()
        if not key:
            continue
        ds_cfg = cfg.get(key)
        if ds_cfg is None:
            logger.warning("Dataset key '%s' not found in cross_dataset_validation.datasets", key)
            continue
        if not ds_cfg.get("enabled", False):
            logger.info("Skipping disabled dataset '%s'", key)
            continue

        path = ds_cfg.get("path")
        try:
            loaded[key] = load_external_dataset(
                path=path,
                source_name=key,
                id_column=ds_cfg.get("id_column"),
                byte_columns=ds_cfg.get("byte_columns"),
                label_column=ds_cfg.get("label_column"),
                label_mapping=ds_cfg.get("label_mapping", {}),
                read_csv_kwargs=ds_cfg.get("read_csv_kwargs"),
            )
            logger.info("Loaded external dataset '%s' from %s", key, path)
        except FileNotFoundError as exc:
            if allow_missing:
                logger.warning("%s (skipped)", exc)
                continue
            raise

    return loaded


def run_cross_dataset_validation(config: dict, args: argparse.Namespace) -> list[dict]:
    logger = setup_logging(args.log_level)
    set_seed(config["seed"])

    logger.info("=" * 60)
    logger.info("Loading and preprocessing CICIoV2024 training dataset")
    logger.info("=" * 60)
    train_df = preprocess_presplit(load_dataset(config), config)

    dataset_keys = [x.strip() for x in args.datasets.split(",") if x.strip()]
    external_datasets = _load_enabled_external_datasets(
        config,
        dataset_keys=dataset_keys,
        allow_missing=args.allow_missing or config.get("cross_dataset_validation", {}).get("allow_missing", True),
        logger=logger,
    )
    if not external_datasets:
        logger.warning("No external datasets loaded. Nothing to evaluate.")
        return []

    models = get_enabled_models(config)
    if not args.include_dl:
        models = [(n, p) for n, p in models if not is_deep_learning_model(n)]
    if args.model:
        models = [(n, p) for n, p in models if n == args.model]

    if not models:
        logger.warning("No models selected for cross-dataset validation.")
        return []

    results: list[dict] = []
    for model_name, model_params in models:
        is_sequential_dl = is_deep_learning_model(model_name) and model_name in ("lstm", "cnn1d")
        if is_sequential_dl and not config["features"].get("use_sliding_window", False):
            logger.warning(
                "Skipping %s: sequence model requires use_sliding_window=true",
                model_name,
            )
            continue

        logger.info("-" * 60)
        logger.info("Model: %s", model_name)
        logger.info("-" * 60)

        for ds_name, ext_df in external_datasets.items():
            logger.info("Evaluating on external dataset: %s", ds_name)

            train_norm, ext_norm, _ = preprocess_postsplit(
                train_df.copy(),
                ext_df.copy(),
                config,
            )

            X_train, y_train_raw = prepare_features(
                train_norm,
                config,
                for_sequence_model=is_sequential_dl,
            )
            X_test, y_test_raw = prepare_features(
                ext_norm,
                config,
                for_sequence_model=is_sequential_dl,
            )

            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train_raw)
            y_train_contiguous, reverse_label_map = _remap_labels(y_train)

            if is_sequential_dl:
                X_tr, X_te = X_train, X_test
                if config.get("feature_selection", {}).get("enabled", False):
                    logger.warning(
                        "Feature selection enabled but skipped for sequence model %s",
                        model_name,
                    )
            else:
                X_tr, X_te = apply_feature_selection(
                    X_train, y_train_contiguous, X_test, config,
                )

            model = get_model(
                model_name,
                model_params,
                seed=config["seed"],
                n_jobs=config.get("n_jobs", -1),
            )
            model, train_time = train_model(model, X_tr, y_train_contiguous)
            y_pred_contiguous, y_proba, inference_time = predict_model(model, X_te)

            y_pred_global = np.array(
                [reverse_label_map.get(int(v), int(v)) for v in y_pred_contiguous],
                dtype=np.intp,
            )
            y_pred = label_encoder.inverse_transform(y_pred_global)
            y_true = np.array(y_test_raw, dtype=object)

            metrics = compute_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                average=config["evaluation"]["average"],
            )
            result = format_results(
                model_name=model_name,
                split_name=f"cross_dataset_{ds_name}",
                metrics=metrics,
                train_time=train_time,
                inference_time=inference_time,
                n_train=len(X_tr),
                n_test=len(X_te),
                dataset_variant="external_validation",
            )
            result["target_dataset"] = ds_name
            results.append(result)

    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    config["log_level"] = args.log_level

    output_file = Path(args.output) if args.output else (
        get_project_root() / "results" / "cross_dataset_results.json"
    )
    ensure_dir(output_file.parent)

    results = run_cross_dataset_validation(config, args)
    if results:
        save_results(results, output_file.parent, filename=output_file.name)
        print(f"Saved {len(results)} cross-dataset result rows to {output_file}")
    else:
        print("No cross-dataset results generated.")


if __name__ == "__main__":
    main()
