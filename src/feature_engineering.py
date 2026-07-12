"""Feature engineering for CAN bus intrusion detection."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_feature_columns(df: pd.DataFrame, config: dict) -> list[str]:
    """Resolve available feature columns from config and DataFrame."""
    feature_cols = list(config["features"]["frame_level"])
    if "inter_arrival_idx" in df.columns:
        feature_cols.append("inter_arrival_idx")
    return [c for c in feature_cols if c in df.columns]


def extract_frame_features(
    df: pd.DataFrame,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract frame-level features and labels.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (feature array X, label array y).
    """
    available = _resolve_feature_columns(df, config)
    X = df[available].values.astype(np.float32)
    y = df["specific_class"].values

    logger.info(f"Extracted frame features: X={X.shape}, y={y.shape}")
    return X, y


def _aggregate_window(window: np.ndarray, stats: list[str]) -> np.ndarray:
    """Aggregate one sliding window using configured summary statistics."""
    chunks: list[np.ndarray] = []

    for stat in stats:
        if stat == "mean":
            chunks.append(window.mean(axis=0))
        elif stat == "std":
            chunks.append(window.std(axis=0))
        elif stat == "min":
            chunks.append(window.min(axis=0))
        elif stat == "max":
            chunks.append(window.max(axis=0))
        elif stat == "first":
            chunks.append(window[0])
        elif stat == "last":
            chunks.append(window[-1])
        elif stat == "delta":
            chunks.append(window[-1] - window[0])
        elif stat == "median":
            chunks.append(np.median(window, axis=0))
        else:
            logger.warning("Unknown aggregate statistic '%s' - skipped", stat)

    if not chunks:
        # Safe fallback to mean if user config is invalid.
        chunks = [window.mean(axis=0)]

    return np.concatenate(chunks, axis=0).astype(np.float32)


def build_sliding_windows(
    df: pd.DataFrame,
    config: dict,
    for_sequence_model: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window features for temporal modeling.

    Supports three classical representations:
    - ``flatten``: flatten each window into one long vector.
    - ``aggregate``: summary-statistic vectors per window.
    - ``hybrid``: concatenate flattened + aggregated vectors.

    For sequence models (LSTM/CNN), set ``for_sequence_model=True`` to
    return 3D tensors of shape ``(n_windows, window_size, n_features)``.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary with window settings.
        for_sequence_model: Whether to return sequence tensors (3D).

    Returns:
        Tuple of (windowed feature array X, label array y).
    """
    feat_config = config["features"]
    window_size = int(feat_config.get("window_size", 10))
    stride = int(feat_config.get("window_stride", 1))
    representation = feat_config.get("window_representation", "flatten")
    aggregate_stats = list(
        feat_config.get(
            "aggregate_statistics",
            ["mean", "std", "min", "max", "last", "delta"],
        )
    )

    available = _resolve_feature_columns(df, config)
    values = df[available].values.astype(np.float32)
    labels = df["specific_class"].values

    n_frames = len(values)
    n_features = values.shape[1]

    if n_frames < window_size:
        logger.warning(
            "Dataset has %d frames, less than window_size=%d. "
            "Falling back to frame-level features.",
            n_frames, window_size,
        )
        if for_sequence_model:
            # Shape to (samples, 1, features) for sequence-model compatibility.
            return values[:, None, :], labels
        return values, labels

    window_tensors: list[np.ndarray] = []
    window_vectors: list[np.ndarray] = []
    window_labels: list[str] = []

    for i in range(0, n_frames - window_size + 1, stride):
        seq_window = values[i : i + window_size]
        window_tensors.append(seq_window)
        window_labels.append(labels[i + window_size - 1])

        if not for_sequence_model:
            flat = seq_window.flatten()
            if representation == "flatten":
                window_vectors.append(flat.astype(np.float32))
            elif representation == "aggregate":
                window_vectors.append(_aggregate_window(seq_window, aggregate_stats))
            elif representation == "hybrid":
                agg = _aggregate_window(seq_window, aggregate_stats)
                window_vectors.append(np.concatenate([flat, agg]).astype(np.float32))
            else:
                logger.warning(
                    "Unknown window_representation='%s', using 'flatten'",
                    representation,
                )
                window_vectors.append(flat.astype(np.float32))

    y = np.array(window_labels)

    if for_sequence_model:
        X = np.array(window_tensors, dtype=np.float32)
        logger.info(
            "Built sliding-window sequence tensors: X=%s, y=%s, window_size=%d, "
            "stride=%d, features_per_timestep=%d",
            X.shape, y.shape, window_size, stride, n_features,
        )
        return X, y

    X = np.array(window_vectors, dtype=np.float32)
    logger.info(
        "Built sliding-window features: X=%s, y=%s, window_size=%d, stride=%d, "
        "representation=%s",
        X.shape, y.shape, window_size, stride, representation,
    )
    return X, y


def prepare_features(
    df: pd.DataFrame,
    config: dict,
    for_sequence_model: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare features based on configuration.

    Dispatches to either frame-level or sliding-window feature extraction
    depending on config settings.

    Args:
        df: Preprocessed DataFrame.
        config: Configuration dictionary.
        for_sequence_model: Whether caller is a sequence model requiring
            3D sliding-window tensors.

    Returns:
        Tuple of (feature array X, label array y).
    """
    use_window = config["features"].get("use_sliding_window", False)

    if use_window:
        return build_sliding_windows(df, config, for_sequence_model=for_sequence_model)
    return extract_frame_features(df, config)
