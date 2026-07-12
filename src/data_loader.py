"""Data loading utilities for the CICIoV2024 dataset."""

import logging
from pathlib import Path

import pandas as pd

from src.utils import get_project_root

logger = logging.getLogger(__name__)

# Column definitions by format
DECIMAL_COLS = [
    "ID", "DATA_0", "DATA_1", "DATA_2", "DATA_3",
    "DATA_4", "DATA_5", "DATA_6", "DATA_7",
    "label", "category", "specific_class",
]

HEXADECIMAL_COLS = [
    "Interface", "ID", "DLC", "DATA_0", "DATA_1", "DATA_2", "DATA_3",
    "DATA_4", "DATA_5", "DATA_6", "DATA_7",
    "label", "category", "specific_class",
]

# Mapping from file keys to attack scenario names
SCENARIO_MAP = {
    "benign": "BENIGN",
    "DoS": "DoS",
    "spoofing_GAS": "GAS",
    "spoofing_RPM": "RPM",
    "spoofing_SPEED": "SPEED",
    "spoofing_STEERING_WHEEL": "STEERING_WHEEL",
}


def load_single_file(filepath: str | Path, fmt: str = "decimal") -> pd.DataFrame:
    """Load a single CICIoV2024 CSV file.

    Args:
        filepath: Path to the CSV file.
        fmt: Data format, either "decimal" or "hexadecimal".

    Returns:
        DataFrame with CAN frame data.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} frames from {filepath.name}")
    return df


def load_dataset(config: dict) -> pd.DataFrame:
    """Load the full CICIoV2024 dataset from all scenario files.

    Each file is tagged with a 'scenario' column indicating its source
    (e.g., BENIGN, DoS, GAS, RPM, SPEED, STEERING_WHEEL). This is used
    for scenario-based splitting.

    Args:
        config: Configuration dictionary with dataset settings.

    Returns:
        Combined DataFrame with all scenarios and a 'scenario' column.
    """
    ds_config = config["dataset"]
    fmt = ds_config["format"]
    base_dir = get_project_root() / ds_config["base_dir"] / fmt

    frames = []
    for file_key, filename in ds_config["files"].items():
        filepath = base_dir / filename
        df = load_single_file(filepath, fmt=fmt)

        # Tag with scenario source
        scenario_name = SCENARIO_MAP.get(file_key, file_key)
        df["scenario"] = scenario_name

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Combined dataset: {len(combined)} frames, "
        f"{combined['scenario'].nunique()} scenarios"
    )

    # Log class distribution
    class_counts = combined["specific_class"].value_counts()
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count} frames ({100 * count / len(combined):.1f}%)")

    return combined


def get_hex_to_decimal_mapping() -> dict:
    """Return a mapping for converting hex payload bytes to decimal.

    Useful when loading hexadecimal format data and converting to numeric.

    Returns:
        Dictionary with conversion info.
    """
    return {
        "base": 16,
        "payload_cols": [f"DATA_{i}" for i in range(8)],
    }
