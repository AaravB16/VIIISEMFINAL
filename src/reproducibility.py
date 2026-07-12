"""Reproducibility utilities for experiment transparency.

Records the execution environment (Python version, installed packages,
hardware info), computes dataset file checksums, and archives full
experiment metadata alongside results.
"""

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.utils import ensure_dir, get_project_root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------

def capture_environment() -> dict:
    """Capture a snapshot of the current execution environment.

    Returns:
        Dictionary with Python version, platform info, and installed packages.
    """
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    # Installed packages
    try:
        import importlib.metadata as meta
        packages = {d.metadata["Name"]: d.version for d in meta.distributions()}
        env["installed_packages"] = dict(sorted(packages.items()))
    except Exception:
        env["installed_packages"] = "unavailable"

    # Key library versions
    key_libs = [
        "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm",
        "tensorflow", "matplotlib", "seaborn", "pyyaml",
    ]
    versions = {}
    for lib in key_libs:
        try:
            mod = __import__(lib.replace("-", "_"))
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    env["key_library_versions"] = versions

    return env


# ---------------------------------------------------------------------------
# Dataset checksums
# ---------------------------------------------------------------------------

def compute_file_checksum(filepath: str | Path, algorithm: str = "sha256") -> str:
    """Compute the hash digest of a file.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm (default ``"sha256"``).

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_checksums(config: dict) -> dict:
    """Compute checksums for all dataset files listed in the config.

    Args:
        config: Configuration dictionary.

    Returns:
        Mapping from filename to SHA-256 digest.
    """
    ds_config = config["dataset"]
    fmt = ds_config["format"]
    base_dir = get_project_root() / ds_config["base_dir"] / fmt

    checksums = {}
    for file_key, filename in ds_config["files"].items():
        filepath = base_dir / filename
        if filepath.exists():
            digest = compute_file_checksum(filepath)
            checksums[filename] = digest
            logger.debug(f"Checksum {filename}: {digest[:16]}…")
        else:
            checksums[filename] = "FILE_NOT_FOUND"
            logger.warning(f"Dataset file not found for checksum: {filepath}")

    logger.info(f"Computed SHA-256 checksums for {len(checksums)} dataset files")
    return checksums


# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------

def build_experiment_metadata(
    config: dict,
    results: list[dict] | None = None,
) -> dict:
    """Build a comprehensive metadata record for the experiment run.

    Args:
        config: Full experiment configuration.
        results: Optional list of result dictionaries.

    Returns:
        Metadata dictionary suitable for JSON serialization.
    """
    metadata = {
        "framework": "CICIoV2024 Evaluation Framework",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": _make_serializable(config),
        "environment": capture_environment(),
        "dataset_checksums": compute_dataset_checksums(config),
    }

    if results:
        metadata["n_experiments"] = len(results)
        metadata["models_evaluated"] = sorted(set(r["model"] for r in results))
        metadata["splits_used"] = sorted(set(r["split_strategy"] for r in results))
        # Record dataset variants (augmentation)
        variants = sorted(set(
            r.get("dataset_variant", "original") for r in results
        ))
        metadata["dataset_variants"] = variants

    # Augmentation configuration
    aug_cfg = config.get("augmentation", {})
    if aug_cfg.get("enabled", False):
        metadata["augmentation"] = _make_serializable(aug_cfg)

    return metadata


def save_experiment_metadata(
    metadata: dict,
    output_dir: str | Path,
    filename: str = "experiment_metadata.json",
) -> Path:
    """Save experiment metadata to JSON.

    Args:
        metadata: Metadata dictionary.
        output_dir: Output directory.
        filename: Output filename.

    Returns:
        Path to the saved file.
    """
    output_dir = ensure_dir(output_dir)
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Experiment metadata saved to {filepath}")
    return filepath


def _make_serializable(obj):
    """Convert non-serializable types for JSON output."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Path):
        return str(obj)
    return obj
