"""Utility functions for configuration, logging, and reproducibility."""

import logging
import os
import random
from pathlib import Path

import numpy as np
import yaml


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file. Defaults to config/config.yaml
                     relative to project root.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seeds for ML libraries if available
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pass

    logging.getLogger(__name__).info(f"Random seed set to {seed}")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging for the framework.

    Args:
        level: Logging level string.

    Returns:
        Root logger instance.
    """
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("cicIoV2024")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
