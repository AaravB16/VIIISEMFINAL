#!/usr/bin/env python3
"""
Main experiment runner for the CICIoV2024 Evaluation Framework.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --config config/custom.yaml
    python scripts/run_experiment.py --split random --model random_forest
    python scripts/run_experiment.py --no-viz   # skip visualizations
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.augmentation_engine import generate_augmented_datasets, save_augmented_dataset
from src.data_loader import load_dataset
from src.deep_learning import is_deep_learning_model
from src.evaluation import (
    compute_metrics,
    format_results,
    print_summary_table,
    save_results,
)
from src.feature_engineering import prepare_features
from src.feature_selection import apply_feature_selection
from src.models import get_enabled_models, get_model, predict_model, train_model
from src.preprocessing import preprocess_presplit, preprocess_postsplit
from src.reproducibility import build_experiment_metadata, save_experiment_metadata
from src.robustness import (
    compute_robustness_deltas,
    find_most_robust_model,
    print_robustness_summary,
    save_robustness_report,
)
from src.splitting import generate_splits
from src.utils import ensure_dir, get_project_root, load_config, set_seed, setup_logging
from src.visualization import generate_all_visualizations


# ---------------------------------------------------------------------------
# tqdm-compatible logging handler — log messages render above the progress bar
# ---------------------------------------------------------------------------

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes through tqdm.write() to avoid bar corruption."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CICIoV2024 IDS Evaluation Framework",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Run only a specific split strategy",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run only a specific model",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--enable-dl", action="store_true",
        help="Force-enable deep learning models defined under deep_learning.models",
    )
    return parser.parse_args()


def _remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Remap integer labels to contiguous 0…N-1.

    XGBoost (and some other learners) require labels to form a contiguous
    range starting at 0.  In attack-holdout splits, the held-out class
    leaves a gap in the encoded label space which causes a crash.

    Returns:
        (remapped_y, reverse_map)  where ``reverse_map[new] == old``.
    """
    unique = np.unique(y)
    forward = {int(old): int(new) for new, old in enumerate(unique)}
    reverse = {int(new): int(old) for new, old in enumerate(unique)}
    return np.array([forward[int(v)] for v in y], dtype=np.intp), reverse


def _count_experiments(config: dict, n_variants: int = 1) -> int:
    """Estimate the total number of (variant, split, model) triples."""
    strategies = config["splitting"]["strategies"]
    n_splits = 0
    for s in strategies:
        if s == "random":
            n_splits += 1
        elif s == "scenario":
            n_splits += 1
        elif s == "attack_holdout":
            n_splits += len(config["splitting"].get("holdout_attacks", []))
        elif s == "cross_validation":
            n_splits += config.get("cross_validation", {}).get("n_folds", 5)
        else:
            n_splits += 1

    n_models = len(get_enabled_models(config))
    return n_splits * n_models * n_variants


def run_experiment(
    config: dict,
    split_filter: str | None,
    model_filter: str | None,
    generate_viz: bool = True,
):
    """Execute the full experiment pipeline."""
    logger = setup_logging(config.get("log_level", "INFO"))

    # Replace default handlers with tqdm-aware handler so log lines
    # render cleanly above the progress bar.
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    root_logger.addHandler(tqdm_handler)
    root_logger.setLevel(getattr(logging, config.get("log_level", "INFO").upper()))

    seed = config["seed"]
    set_seed(seed)

    output_dir = get_project_root() / config["output"]["results_dir"]
    ensure_dir(output_dir)

    # ---- 1. Load data ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading CICIoV2024 dataset")
    logger.info("=" * 60)
    df = load_dataset(config)

    # ---- 2. Pre-split preprocessing (no normalization) ----
    logger.info("=" * 60)
    logger.info("STEP 2: Pre-split preprocessing")
    logger.info("=" * 60)
    df = preprocess_presplit(df, config)

    # ---- 3. Encode labels ----
    label_encoder = LabelEncoder()
    label_encoder.fit(df["specific_class"])
    n_global_classes = len(label_encoder.classes_)
    logger.info(f"Label classes ({n_global_classes}): {list(label_encoder.classes_)}")

    # ---- 4. Get models to run ----
    models_to_run = get_enabled_models(config)
    if model_filter:
        models_to_run = [(n, p) for n, p in models_to_run if n == model_filter]
        if not models_to_run:
            logger.error(f"Model '{model_filter}' not found or not enabled")
            return

    # ---- 5. Filter splits if requested ----
    if split_filter:
        config = {**config, "splitting": {**config["splitting"], "strategies": [split_filter]}}

    # ---- 6. Build dataset variants (original + augmented) ----
    dataset_variants: list[tuple[str, pd.DataFrame]] = [("original", df)]

    aug_cfg = config.get("augmentation", {})
    if aug_cfg.get("enabled", False):
        logger.info("=" * 60)
        logger.info("STEP 3a: Generating augmented datasets")
        logger.info("=" * 60)
        augmented = generate_augmented_datasets(df, config, seed=seed)
        dataset_variants.extend(augmented)

        # Optionally persist augmented DataFrames
        if aug_cfg.get("save_augmented_datasets", False):
            for name, aug_df in augmented:
                save_augmented_dataset(name, aug_df)

    n_variants = len(dataset_variants)
    variant_names = [v[0] for v in dataset_variants]
    logger.info(f"Dataset variants ({n_variants}): {variant_names}")

    # ---- 7. Run experiments ----
    all_results = []
    y_true_map = {}   # for ROC visualization
    y_proba_map = {}
    n_total = _count_experiments(config, n_variants=n_variants)
    n_done = 0
    n_fail = 0

    logger.info("=" * 60)
    logger.info("STEP 4: Running experiments")
    logger.info("=" * 60)

    pbar = tqdm(
        total=n_total,
        desc="Experiments",
        unit="run",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        dynamic_ncols=True,
    )

    for variant_name, variant_df in dataset_variants:

        for split_name, split_data in generate_splits(variant_df, config):
            train_df = split_data["train"]
            test_df = split_data["test"]

            # Post-split normalization (fit on train only)
            train_df, test_df, scaler = preprocess_postsplit(train_df, test_df, config)


            for model_name, model_params in models_to_run:
                tag = f"{model_name} | {split_name} | {variant_name}"
                pbar.set_postfix_str(tag, refresh=True)
                logger.info(f"\n--- {tag} ---")

                try:
                    is_sequential_dl = (
                        is_deep_learning_model(model_name)
                        and model_name in ("lstm", "cnn1d")
                    )

                    if is_sequential_dl and not config["features"].get(
                        "use_sliding_window", False
                    ):
                        logger.warning(
                            "%s requires sliding-window features. Skipping this run.",
                            model_name,
                        )
                        continue

                    # Feature extraction is done per model so that sequence
                    # models can consume 3D tensors while classical models
                    # use frame/flat/aggregated 2D features.
                    X_train, y_train_raw = prepare_features(
                        train_df,
                        config,
                        for_sequence_model=is_sequential_dl,
                    )
                    X_test, y_test_raw = prepare_features(
                        test_df,
                        config,
                        for_sequence_model=is_sequential_dl,
                    )

                    # Encode labels using the global encoder
                    y_train = label_encoder.transform(y_train_raw)
                    y_test = label_encoder.transform(y_test_raw)

                    # Remap training labels to contiguous 0…N-1 so that models
                    # expecting dense indices (e.g., XGBoost) do not fail on
                    # attack-holdout gaps.
                    y_train_contiguous, reverse_label_map = _remap_labels(y_train)

                    # Apply feature selection only to 2D feature matrices.
                    if is_sequential_dl:
                        X_tr, X_te = X_train, X_test
                        if config.get("feature_selection", {}).get("enabled", False):
                            logger.warning(
                                "Feature selection is enabled but skipped for %s "
                                "(sequence tensor input).",
                                model_name,
                            )
                    else:
                        X_tr, X_te = apply_feature_selection(
                            X_train, y_train_contiguous, X_test, config,
                        )

                    # Initialize model
                    model = get_model(
                        model_name, model_params, seed=seed,
                        n_jobs=config.get("n_jobs", -1),
                    )

                    # Train with contiguous labels
                    model, train_time = train_model(model, X_tr, y_train_contiguous)

                    # Predict (returns contiguous-space labels)
                    y_pred_contiguous, y_proba, inference_time = predict_model(model, X_te)

                    # Map predictions back to original global label space
                    y_pred = np.array(
                        [reverse_label_map.get(int(p), int(p)) for p in y_pred_contiguous],
                        dtype=np.intp,
                    )

                    # Decode to string labels for evaluation
                    y_test_decoded = label_encoder.inverse_transform(y_test)
                    y_pred_decoded = label_encoder.inverse_transform(y_pred)

                    # If the model was trained on fewer classes than the global
                    # set, expand y_proba columns to the global label space so
                    # that downstream ROC/metric code sees consistent shapes.
                    if y_proba is not None and y_proba.shape[1] < n_global_classes:
                        y_proba_full = np.zeros(
                            (y_proba.shape[0], n_global_classes), dtype=y_proba.dtype,
                        )
                        for contiguous_idx, global_idx in reverse_label_map.items():
                            if contiguous_idx < y_proba.shape[1]:
                                y_proba_full[:, global_idx] = y_proba[:, contiguous_idx]
                        y_proba = y_proba_full

                    # Compute metrics
                    metrics = compute_metrics(
                        y_test_decoded, y_pred_decoded,
                        y_proba=y_proba,
                        average=config["evaluation"]["average"],
                    )

                    # Format and store results
                    result = format_results(
                        model_name=model_name,
                        split_name=split_name,
                        metrics=metrics,
                        train_time=train_time,
                        inference_time=inference_time,
                        n_train=len(X_tr),
                        n_test=len(X_te),
                        dataset_variant=variant_name,
                    )
                    all_results.append(result)
                    n_done += 1

                    # Store data for ROC visualization (original variant only)
                    if y_proba is not None and variant_name == "original":
                        y_true_map[(model_name, split_name)] = y_test_decoded
                        y_proba_map[(model_name, split_name)] = y_proba

                except Exception as e:
                    logger.error(f"FAILED: {tag}: {e}")
                    import traceback
                    traceback.print_exc()
                    n_fail += 1
                    continue
                finally:
                    pbar.update(1)

    pbar.close()
    logger.info(f"Experiments finished: {n_done} succeeded, {n_fail} failed")

    # ---- 8. Save results ----
    if all_results:
        save_results(all_results, output_dir)
        print_summary_table(all_results)

        # ---- 8a. Robustness analysis ----
        robustness_deltas = []
        robustness_summary = {}
        if n_variants > 1:
            logger.info("=" * 60)
            logger.info("STEP 5: Robustness analysis")
            logger.info("=" * 60)
            robustness_deltas = compute_robustness_deltas(all_results)
            robustness_summary = find_most_robust_model(robustness_deltas)
            save_robustness_report(robustness_deltas, robustness_summary, output_dir)
            print_robustness_summary(robustness_deltas, robustness_summary)

        # ---- 9. Visualizations ----
        if generate_viz:
            logger.info("=" * 60)
            logger.info("STEP 6: Generating visualizations")
            logger.info("=" * 60)
            try:
                generate_all_visualizations(
                    all_results, output_dir,
                    y_true_map=y_true_map,
                    y_proba_map=y_proba_map,
                    robustness_deltas=robustness_deltas if robustness_deltas else None,
                )
            except Exception as e:
                logger.error(f"Visualization generation failed: {e}")

        # ---- 10. Reproducibility metadata ----
        logger.info("=" * 60)
        logger.info("STEP 7: Saving reproducibility metadata")
        logger.info("=" * 60)
        metadata = build_experiment_metadata(config, all_results)
        save_experiment_metadata(metadata, output_dir)
    else:
        logger.warning("No results were generated")

    logger.info("Experiment complete.")


def main():
    args = parse_args()
    config = load_config(args.config)
    config["log_level"] = args.log_level
    if args.enable_dl:
        config.setdefault("deep_learning", {})
        config["deep_learning"]["enabled"] = True
    run_experiment(
        config,
        split_filter=args.split,
        model_filter=args.model,
        generate_viz=not args.no_viz,
    )


if __name__ == "__main__":
    main()
