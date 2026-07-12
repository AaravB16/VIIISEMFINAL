"""Data splitting strategies for evaluation."""

import logging
from typing import Generator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


def random_split(
    df: pd.DataFrame,
    config: dict,
) -> dict:
    """Split data randomly with a fixed seed.

    Args:
        df: Full preprocessed DataFrame.
        config: Configuration dictionary.

    Returns:
        Dictionary with 'train' and 'test' DataFrames.
    """
    seed = config["seed"]
    test_size = config["splitting"]["test_size"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["specific_class"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        f"Random split: train={len(train_df)}, test={len(test_df)} "
        f"(seed={seed}, test_size={test_size})"
    )
    _log_split_distribution(train_df, test_df)

    return {"train": train_df, "test": test_df}


def scenario_split(
    df: pd.DataFrame,
    config: dict,
) -> dict:
    """Split data by scenario (driving session / file source).

    Ensures that complete scenarios stay together in either train or test.
    For each attack scenario, a portion of its frames goes to test.
    Benign data is split proportionally.

    Args:
        df: Full preprocessed DataFrame with 'scenario' column.
        config: Configuration dictionary.

    Returns:
        Dictionary with 'train' and 'test' DataFrames.
    """
    seed = config["seed"]
    test_size = config["splitting"]["test_size"]
    rng = np.random.RandomState(seed)

    scenarios = df["scenario"].unique()
    train_frames = []
    test_frames = []

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario].copy()
        n = len(scenario_df)

        # Split each scenario's data preserving contiguous blocks
        # Take the last `test_size` fraction as test (temporal consistency)
        split_idx = int(n * (1 - test_size))
        train_frames.append(scenario_df.iloc[:split_idx])
        test_frames.append(scenario_df.iloc[split_idx:])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    logger.info(
        f"Scenario split: train={len(train_df)}, test={len(test_df)} "
        f"({len(scenarios)} scenarios)"
    )
    _log_split_distribution(train_df, test_df)

    return {"train": train_df, "test": test_df}


def attack_holdout_split(
    df: pd.DataFrame,
    config: dict,
    holdout_attack: str,
) -> dict:
    """Split data by holding out an entire attack type from training.

    All frames of the held-out attack type go to the test set.
    The remaining attack types and benign data are split normally.
    This evaluates the model's ability to detect unseen attack types.

    Args:
        df: Full preprocessed DataFrame.
        config: Configuration dictionary.
        holdout_attack: The specific_class value to hold out (e.g., "DoS").

    Returns:
        Dictionary with 'train' and 'test' DataFrames, and 'holdout_attack'.
    """
    seed = config["seed"]
    test_size = config["splitting"]["test_size"]

    # Separate holdout attack data
    holdout_mask = df["specific_class"] == holdout_attack
    holdout_df = df[holdout_mask].copy()
    remaining_df = df[~holdout_mask].copy()

    if len(holdout_df) == 0:
        logger.warning(f"No frames found for holdout attack '{holdout_attack}'")
        # Fall back to random split
        return random_split(df, config)

    # Split the remaining data
    train_df, test_other = train_test_split(
        remaining_df,
        test_size=test_size,
        random_state=seed,
        stratify=remaining_df["specific_class"],
    )

    # Test set = held-out attack + portion of other classes
    test_df = pd.concat([test_other, holdout_df], ignore_index=True)
    train_df = train_df.reset_index(drop=True)

    logger.info(
        f"Attack-holdout split (holdout={holdout_attack}): "
        f"train={len(train_df)}, test={len(test_df)} "
        f"(holdout frames={len(holdout_df)})"
    )
    _log_split_distribution(train_df, test_df)

    return {
        "train": train_df,
        "test": test_df,
        "holdout_attack": holdout_attack,
    }


def cross_validation_splits(
    df: pd.DataFrame,
    config: dict,
) -> Generator[tuple[str, dict], None, None]:
    """Generate stratified k-fold cross-validation splits.

    Args:
        df: Full preprocessed DataFrame.
        config: Configuration dictionary.

    Yields:
        Tuples of (fold_name, split_dict).
    """
    cv_config = config.get("cross_validation", {})
    n_folds = cv_config.get("n_folds", 5)
    seed = config["seed"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(df, df["specific_class"])
    ):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        fold_name = f"cv_fold_{fold_idx + 1}"
        logger.info(
            f"{fold_name}: train={len(train_df)}, test={len(test_df)}"
        )
        yield (fold_name, {"train": train_df, "test": test_df})


def generate_splits(
    df: pd.DataFrame,
    config: dict,
) -> Generator[tuple[str, dict], None, None]:
    """Generate all configured data splits.

    Yields (split_name, split_dict) tuples for each splitting strategy.

    Args:
        df: Full preprocessed DataFrame.
        config: Configuration dictionary.

    Yields:
        Tuples of (split_name, split_data_dict).
    """
    strategies = config["splitting"]["strategies"]

    for strategy in strategies:
        if strategy == "random":
            yield ("random", random_split(df, config))

        elif strategy == "scenario":
            yield ("scenario", scenario_split(df, config))

        elif strategy == "attack_holdout":
            holdout_attacks = config["splitting"].get(
                "holdout_attacks",
                ["DoS", "GAS", "RPM", "SPEED", "STEERING_WHEEL"],
            )
            for attack in holdout_attacks:
                split = attack_holdout_split(df, config, holdout_attack=attack)
                yield (f"attack_holdout_{attack}", split)

        elif strategy == "cross_validation":
            yield from cross_validation_splits(df, config)

        else:
            logger.warning(f"Unknown splitting strategy: {strategy}")


def _log_split_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log class distribution for train and test sets."""
    for name, split_df in [("train", train_df), ("test", test_df)]:
        counts = split_df["specific_class"].value_counts()
        dist = ", ".join(f"{cls}={cnt}" for cls, cnt in counts.items())
        logger.info(f"  {name}: {dist}")
