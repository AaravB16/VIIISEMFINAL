"""Feature selection and dimensionality reduction for IDS evaluation.

Provides statistical feature selection (mutual information, chi-squared),
correlation-based filtering, and PCA dimensionality reduction.  All
selectors follow the fit/transform pattern so they can be fitted on
training data and applied identically to test data.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    mutual_info_classif,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statistical feature selection
# ---------------------------------------------------------------------------

def select_by_mutual_information(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int | str = "all",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Select top-k features ranked by mutual information with the target.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_test: Test feature matrix.
        k: Number of features to keep, or ``"all"``.
        seed: Random seed for MI estimation.

    Returns:
        Tuple of (X_train_selected, X_test_selected, fitted selector).
    """
    def mi_scorer(X, y):
        return mutual_info_classif(X, y, random_state=seed)

    selector = SelectKBest(score_func=mi_scorer, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    n_selected = X_train_sel.shape[1]
    logger.info(
        f"Mutual information selection: {X_train.shape[1]} -> {n_selected} features"
    )
    return X_train_sel, X_test_sel, selector


def select_by_chi_squared(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int | str = "all",
) -> tuple[np.ndarray, np.ndarray, SelectKBest]:
    """Select top-k features using the chi-squared statistic.

    Note: chi-squared requires non-negative feature values,
    so data should be min-max scaled beforehand.

    Args:
        X_train: Training feature matrix (non-negative).
        y_train: Training labels.
        X_test: Test feature matrix (non-negative).
        k: Number of features to keep, or ``"all"``.

    Returns:
        Tuple of (X_train_selected, X_test_selected, fitted selector).
    """
    # Ensure non-negative for chi2
    X_train_nn = np.clip(X_train, 0, None)
    X_test_nn = np.clip(X_test, 0, None)

    selector = SelectKBest(score_func=chi2, k=k)
    X_train_sel = selector.fit_transform(X_train_nn, y_train)
    X_test_sel = selector.transform(X_test_nn)

    n_selected = X_train_sel.shape[1]
    logger.info(
        f"Chi-squared selection: {X_train.shape[1]} -> {n_selected} features"
    )
    return X_train_sel, X_test_sel, selector


# ---------------------------------------------------------------------------
# Correlation-based filtering
# ---------------------------------------------------------------------------

def filter_correlated_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Remove highly correlated features to reduce multicollinearity.

    For each pair of features whose Pearson correlation exceeds the
    threshold, the second feature (by column index) is dropped.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        feature_names: Optional list of feature names for logging.
        threshold: Absolute correlation threshold (default 0.95).

    Returns:
        Tuple of (X_train_filtered, X_test_filtered, kept_indices).
    """
    corr_matrix = np.corrcoef(X_train, rowvar=False)
    n_features = corr_matrix.shape[0]
    drop_indices = set()

    for i in range(n_features):
        if i in drop_indices:
            continue
        for j in range(i + 1, n_features):
            if j in drop_indices:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                drop_indices.add(j)

    keep_indices = sorted(set(range(n_features)) - drop_indices)
    X_train_filtered = X_train[:, keep_indices]
    X_test_filtered = X_test[:, keep_indices]

    n_dropped = len(drop_indices)
    if n_dropped > 0:
        if feature_names:
            dropped_names = [feature_names[i] for i in sorted(drop_indices)]
            logger.info(
                f"Correlation filtering: dropped {n_dropped} features "
                f"(threshold={threshold}): {dropped_names}"
            )
        else:
            logger.info(
                f"Correlation filtering: {n_features} -> {len(keep_indices)} features "
                f"(dropped {n_dropped}, threshold={threshold})"
            )
    else:
        logger.info(f"Correlation filtering: no features dropped (threshold={threshold})")

    return X_train_filtered, X_test_filtered, keep_indices


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int | float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Apply PCA for dimensionality reduction.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        n_components: Number of components to keep.  If a float in (0, 1),
            selects the number of components to explain that fraction of
            total variance.
        seed: Random seed.

    Returns:
        Tuple of (X_train_pca, X_test_pca, fitted PCA object).
    """
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained = sum(pca.explained_variance_ratio_) * 100
    logger.info(
        f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} components "
        f"({explained:.1f}% variance explained)"
    )
    return X_train_pca, X_test_pca, pca


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def apply_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the feature selection pipeline configured in ``config``.

    Multiple methods can be chained.  The order is:
    1. Correlation filtering
    2. Statistical selection (mutual information **or** chi-squared)
    3. PCA (if enabled)

    Args:
        X_train: Training features.
        y_train: Training labels (encoded).
        X_test: Test features.
        config: Full configuration dictionary.

    Returns:
        Tuple of (X_train_processed, X_test_processed).
    """
    fs_config = config.get("feature_selection", {})
    if not fs_config.get("enabled", False):
        return X_train, X_test

    seed = config.get("seed", 42)

    # 1. Correlation filtering
    if fs_config.get("correlation_filter", {}).get("enabled", False):
        threshold = fs_config["correlation_filter"].get("threshold", 0.95)
        X_train, X_test, _ = filter_correlated_features(
            X_train, X_test, threshold=threshold,
        )

    # 2. Statistical selection
    stat_method = fs_config.get("method", None)
    k = fs_config.get("k", "all")

    if stat_method == "mutual_information":
        X_train, X_test, _ = select_by_mutual_information(
            X_train, y_train, X_test, k=k, seed=seed,
        )
    elif stat_method == "chi_squared":
        X_train, X_test, _ = select_by_chi_squared(
            X_train, y_train, X_test, k=k,
        )

    # 3. PCA
    if fs_config.get("pca", {}).get("enabled", False):
        n_components = fs_config["pca"].get("n_components", 0.95)
        X_train, X_test, _ = apply_pca(
            X_train, X_test, n_components=n_components, seed=seed,
        )

    return X_train, X_test
