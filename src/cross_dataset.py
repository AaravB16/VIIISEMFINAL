"""Cross-dataset utilities for evaluating CIC-trained IDS models externally."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.preprocessing import add_inter_arrival_index, handle_missing_values

logger = logging.getLogger(__name__)

_ID_CANDIDATES = [
    "ID", "id", "Id", "can_id", "CAN_ID", "arbitration_id", "Arbitration_ID",
]
_LABEL_CANDIDATES = [
    "specific_class", "label", "Label", "class", "Class", "attack", "Attack",
    "target", "Target",
]


def _parse_numeric(value):
    """Parse mixed-format numeric tokens (decimal, hex, binary) into ints."""
    if pd.isna(value):
        return value

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip()
    if text == "":
        return value

    # Remove wrappers often found in CAN logs.
    text = text.replace("[", "").replace("]", "")
    text = text.replace("0X", "0x")

    # Common hexadecimal indicators.
    if text.startswith("0x"):
        return int(text, 16)
    if any(ch in text for ch in "ABCDEFabcdef"):
        return int(text, 16)

    # Binary-looking payloads.
    if set(text) <= {"0", "1"} and len(text) > 1:
        return int(text, 2)

    # Decimal fallback (supports "12.0").
    return int(float(text))


def _resolve_column(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not infer {role} column; checked {candidates}")


def _resolve_byte_columns(df: pd.DataFrame) -> list[str]:
    """Resolve DATA_0..DATA_7 columns across common naming conventions."""
    resolved: list[str] = []
    for i in range(8):
        candidates = [
            f"DATA_{i}", f"data_{i}", f"Data_{i}",
            f"BYTE_{i}", f"byte_{i}", f"Byte_{i}",
            f"BYTE{i}", f"byte{i}", f"Byte{i}",
        ]
        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise KeyError(
                f"Could not infer payload byte column for index {i}; "
                f"checked {candidates}"
            )
        resolved.append(found)
    return resolved


def _normalize_specific_class(raw_label: str, label_mapping: dict[str, str]) -> str:
    """Map external labels into canonical Project Aarav labels."""
    text = str(raw_label).strip()
    key = text.lower()
    if key in label_mapping:
        return label_mapping[key]

    # Heuristic fallback mappings.
    low = key.replace("-", "_").replace(" ", "_")
    if any(tok in low for tok in ["benign", "normal"]):
        return "BENIGN"
    if "dos" in low:
        return "DoS"
    if "rpm" in low:
        return "RPM"
    if "speed" in low:
        return "SPEED"
    if "steer" in low:
        return "STEERING_WHEEL"
    if any(tok in low for tok in ["gas", "throttle"]):
        return "GAS"
    if "fuzzy" in low:
        return "FUZZY"
    if "gear" in low:
        return "GEAR"
    return text.upper().replace(" ", "_")


def canonicalize_external_dataset(
    df: pd.DataFrame,
    source_name: str,
    id_column: str | None = None,
    byte_columns: list[str] | None = None,
    label_column: str | None = None,
    label_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Canonicalize external CAN IDS dataset to Project Aarav schema."""
    if label_mapping is None:
        label_mapping = {}

    # Case-insensitive mapping keys for robust lookups.
    label_mapping = {str(k).lower(): str(v) for k, v in label_mapping.items()}

    id_col = id_column or _resolve_column(df, _ID_CANDIDATES, role="CAN ID")
    payload_cols = byte_columns or _resolve_byte_columns(df)
    lbl_col = label_column or _resolve_column(df, _LABEL_CANDIDATES, role="label")

    out = pd.DataFrame()
    out["ID"] = df[id_col].map(_parse_numeric)
    for i, col in enumerate(payload_cols):
        out[f"DATA_{i}"] = df[col].map(_parse_numeric)

    out["specific_class"] = df[lbl_col].map(
        lambda x: _normalize_specific_class(x, label_mapping)
    )
    out["label"] = out["specific_class"].map(
        lambda x: "BENIGN" if x == "BENIGN" else "ATTACK"
    )
    out["category"] = out["specific_class"].map(
        lambda x: "BENIGN" if x == "BENIGN" else "EXTERNAL"
    )
    out["scenario"] = source_name.upper()

    logger.info(
        "Canonicalized external dataset '%s': rows=%d, classes=%s",
        source_name,
        len(out),
        sorted(out["specific_class"].unique()),
    )
    return out


def preprocess_external_dataset(
    df: pd.DataFrame,
    add_inter_arrival: bool = True,
) -> pd.DataFrame:
    """Apply non-leaky preprocessing suitable for external inference sets."""
    out = handle_missing_values(df.copy())
    if add_inter_arrival:
        out = add_inter_arrival_index(out)
    return out


def load_external_dataset(
    path: str | Path,
    source_name: str,
    *,
    id_column: str | None = None,
    byte_columns: list[str] | None = None,
    label_column: str | None = None,
    label_mapping: dict[str, str] | None = None,
    read_csv_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Load + canonicalize an external benchmark dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"External dataset not found: {path}")

    kwargs = read_csv_kwargs or {}
    raw = pd.read_csv(path, **kwargs)
    canonical = canonicalize_external_dataset(
        raw,
        source_name=source_name,
        id_column=id_column,
        byte_columns=byte_columns,
        label_column=label_column,
        label_mapping=label_mapping,
    )
    return preprocess_external_dataset(canonical, add_inter_arrival=True)
