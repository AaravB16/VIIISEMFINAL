"""Preprocessing pipeline for CICIoV2024 CAN bus data."""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Remove duplicate CAN frames based on specified columns.

    Preserves chronological ordering by keeping the first occurrence.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for identifying duplicates.
                If None, uses all payload columns.

    Returns:
        DataFrame with duplicates removed.
    """
    n_before = len(df)

    if subset is None:
        subset = ["ID"] + [f"DATA_{i}" for i in range(8)]

    # Keep label/category columns in duplicate detection to avoid
    # removing frames that have identical payloads but different labels
    # (e.g., a benign frame and an injected attack frame with same content)
    subset_with_label = subset + ["label"]
    available = [c for c in subset_with_label if c in df.columns]

    df = df.drop_duplicates(subset=available, keep="first").reset_index(drop=True)
    n_removed = n_before - len(df)

    logger.info(
        f"Duplicate removal: {n_before} -> {len(df)} frames "
        f"({n_removed} removed, {100 * n_removed / n_before:.1f}%)"
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and handle missing or corrupted entries.

    Strategy:
    - Drop rows with any NaN in payload columns (these are corrupted frames).
    - Preserve chronological ordering.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values handled.
    """
    payload_cols = [f"DATA_{i}" for i in range(8)]
    essential_cols = ["ID"] + payload_cols + ["label"]
    available = [c for c in essential_cols if c in df.columns]

    n_before = len(df)
    n_missing = df[available].isna().any(axis=1).sum()

    if n_missing > 0:
        df = df.dropna(subset=available).reset_index(drop=True)
        logger.warning(
            f"Dropped {n_missing} rows with missing values in essential columns"
        )
    else:
        logger.info("No missing values detected in essential columns")

    return df


def add_inter_arrival_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add a positional index as a proxy for inter-arrival time.

    Since the CICIoV2024 decimal format does not include timestamps,
    the frame index serves as a monotonic proxy for temporal ordering.
    Within each CAN ID group, we compute the positional difference
    between consecutive frames of the same ID.

    Args:
        df: Input DataFrame (must be in chronological order).

    Returns:
        DataFrame with 'frame_index' and 'inter_arrival_idx' columns.
    """
    df = df.copy()
    df["frame_index"] = np.arange(len(df))

    # Inter-arrival: difference in frame_index for same CAN ID
    df["inter_arrival_idx"] = (
        df.groupby("ID")["frame_index"]
        .diff()
        .fillna(0)
        .astype(np.float32)
    )

    logger.info("Added frame_index and inter_arrival_idx features")
    return df


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    method: str = "minmax",
    scaler: MinMaxScaler | StandardScaler | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler | StandardScaler]:
    """Normalize numeric features.

    Args:
        df: Input DataFrame.
        feature_cols: Columns to normalize.
        method: "minmax" for [0, 1] scaling or "standard" for z-score.
        scaler: Pre-fitted scaler (for applying to test data). If None,
                a new scaler is fitted on the provided data.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler).
    """
    df = df.copy()
    available_cols = [c for c in feature_cols if c in df.columns]

    if scaler is None:
        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        df[available_cols] = scaler.fit_transform(df[available_cols].values)
        logger.info(f"Fitted {method} scaler on {len(available_cols)} features")
    else:
        df[available_cols] = scaler.transform(df[available_cols].values)
        logger.info(f"Applied pre-fitted scaler on {len(available_cols)} features")

    return df, scaler


def preprocess_presplit(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Pre-split preprocessing: operations safe to run on the full dataset.

    Includes missing-value handling, duplicate removal, and inter-arrival
    index construction.  Normalization is deliberately deferred to
    ``preprocess_postsplit`` to avoid data leakage.

    Args:
        df: Raw DataFrame.
        config: Configuration dictionary.

    Returns:
        Cleaned DataFrame.
    """
    pp_config = config["preprocessing"]

    # Step 1: Handle missing values
    df = handle_missing_values(df)

    # Step 2: Remove duplicates
    if pp_config.get("remove_duplicates", True):
        subset = pp_config.get("duplicate_subset")
        df = remove_duplicates(df, subset=subset)

    # Step 3: Add inter-arrival index
    if pp_config.get("add_inter_arrival_index", True):
        df = add_inter_arrival_index(df)

    return df


def preprocess_postsplit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler | StandardScaler | None]:
    """Post-split preprocessing: normalization fitted on training data only.

    The scaler is fitted on *train_df* and applied to both splits,
    preventing information leakage from the test set.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (normalized train_df, normalized test_df, fitted scaler).
    """
    pp_config = config["preprocessing"]
    scaler = None

    if pp_config.get("normalize", True):
        feature_cols = list(config["features"]["frame_level"])
        if "inter_arrival_idx" in train_df.columns:
            feature_cols = feature_cols + ["inter_arrival_idx"]

        method = pp_config.get("normalization_method", "minmax")

        # Fit on train, transform both
        train_df, scaler = normalize_features(
            train_df, feature_cols, method=method, scaler=None,
        )
        test_df, _ = normalize_features(
            test_df, feature_cols, method=method, scaler=scaler,
        )

    return train_df, test_df, scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    config: dict,
    scaler: MinMaxScaler | StandardScaler | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler | StandardScaler | None]:
    """Legacy convenience wrapper that runs the full pipeline on one DataFrame.

    .. note:: For proper evaluation, prefer ``preprocess_presplit`` followed
       by splitting and ``preprocess_postsplit``.

    Args:
        df: Raw DataFrame.
        config: Configuration dictionary.
        scaler: Pre-fitted scaler for test data. None for training data.

    Returns:
        Tuple of (preprocessed DataFrame, fitted scaler).
    """
    df = preprocess_presplit(df, config)

    pp_config = config["preprocessing"]
    if pp_config.get("normalize", True):
        feature_cols = list(config["features"]["frame_level"])
        if "inter_arrival_idx" in df.columns:
            feature_cols = feature_cols + ["inter_arrival_idx"]
        method = pp_config.get("normalization_method", "minmax")
        df, scaler = normalize_features(df, feature_cols, method=method, scaler=scaler)

    return df, scaler
