"""IDS model implementations for benchmark evaluation."""

import logging
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def get_model(model_name: str, params: dict, seed: int, n_jobs: int = -1):
    """Instantiate a model by name with given parameters.

    Args:
        model_name: One of 'random_forest', 'svm', 'xgboost', 'lightgbm'.
        params: Model hyperparameters dictionary.
        seed: Random seed.
        n_jobs: Number of parallel jobs.

    Returns:
        Instantiated sklearn-compatible model.
    """
    if model_name == "random_forest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )

    elif model_name == "svm":
        # SVM with probability estimates for ROC-AUC
        return SVC(
            random_state=seed,
            probability=True,
            **params,
        )

    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            use_label_encoder=False,
            **params,
        )

    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=seed,
            n_jobs=n_jobs,
            **params,
        )

    else:
        # Try deep learning models
        from src.deep_learning import is_deep_learning_model, get_deep_learning_model

        if is_deep_learning_model(model_name):
            return get_deep_learning_model(model_name, params, seed=seed)

        raise ValueError(f"Unknown model: {model_name}")


def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Train a model and return training time.

    Args:
        model: Sklearn-compatible model instance.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Tuple of (trained model, training_time_seconds).
    """
    model_name = type(model).__name__
    logger.info(f"Training {model_name} on {X_train.shape[0]} samples...")

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    logger.info(f"  Training completed in {train_time:.2f}s")
    return model, train_time


def predict_model(
    model,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Generate predictions and measure inference latency.

    Args:
        model: Trained model.
        X_test: Test features.

    Returns:
        Tuple of (predictions, probability_estimates, inference_time_seconds).
    """
    model_name = type(model).__name__

    # Predictions
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start

    # Probability estimates (for ROC-AUC)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception as e:
            logger.warning(f"Could not get probability estimates: {e}")

    per_sample_latency = inference_time / len(X_test) * 1000  # ms
    logger.info(
        f"  {model_name} inference: {inference_time:.2f}s total, "
        f"{per_sample_latency:.4f}ms/sample"
    )

    return y_pred, y_proba, inference_time


def get_enabled_models(config: dict) -> list[tuple[str, dict]]:
    """Get list of enabled models and their parameters from config.

    Combines traditional ML models from ``config["models"]`` with deep
    learning models from ``config["deep_learning"]["models"]``.

    Args:
        config: Configuration dictionary.

    Returns:
        List of (model_name, params) tuples.
    """
    models = []

    # Traditional ML models
    for model_name, model_config in config.get("models", {}).items():
        if model_config.get("enabled", True):
            models.append((model_name, model_config.get("params", {})))

    # Deep learning models
    dl_config = config.get("deep_learning", {})
    if dl_config.get("enabled", False):
        for model_name, model_config in dl_config.get("models", {}).items():
            if model_config.get("enabled", True):
                models.append((model_name, model_config.get("params", {})))

    return models
