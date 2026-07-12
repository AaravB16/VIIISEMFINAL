"""Controlled dataset augmentation for IoV robustness evaluation.

Generates augmented copies of the CICIoV2024 dataset that simulate
realistic variations in CAN bus traffic.  Four strategies are provided:

1. **Timing Jitter** – small perturbation of inter-arrival features.
2. **Feature Noise** – Gaussian noise on numeric payload columns.
3. **Attack Intensity Scaling** – scale attack-traffic features up/down.
4. **Adversarial Evasion** – constrained perturbations that push attack
   samples toward benign-looking feature patterns.

All strategies preserve labels, IDs, and chronological ordering.  Each
uses a deterministic ``numpy.random.Generator`` derived from the global
seed so that results are fully reproducible.
"""

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that must NEVER be modified by any augmentation.
_PROTECTED_COLS = frozenset({
    "ID", "label", "category", "specific_class", "scenario", "frame_index",
})

# Numeric payload columns that are eligible for perturbation.
_PAYLOAD_COLS = [f"DATA_{i}" for i in range(8)]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _timing_jitter(
    df: pd.DataFrame,
    jitter_pct: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate ECU scheduling differences and bus jitter.

    Perturbs ``inter_arrival_idx`` by a small Gaussian noise proportional
    to its per-CAN-ID standard deviation.  Values are clamped ≥ 0 so that
    chronological ordering is preserved.

    Args:
        df: Input DataFrame (copied internally).
        jitter_pct: Noise magnitude as a fraction (e.g. 0.02 = 2%).
        rng: Seeded random generator.

    Returns:
        Augmented DataFrame.
    """
    col = "inter_arrival_idx"
    if col not in df.columns:
        logger.warning("Timing jitter: '%s' column not present – skipped", col)
        return df.copy()

    out = df.copy()
    per_id_std = out.groupby("ID")[col].transform("std").fillna(0).values
    noise = rng.normal(loc=0.0, scale=jitter_pct * per_id_std, size=len(out))
    out[col] = np.maximum(out[col].values + noise, 0.0).astype(np.float32)

    logger.info(
        "Timing jitter applied: jitter_pct=%.3f, affected %d rows",
        jitter_pct, int((per_id_std > 0).sum()),
    )
    return out


def _feature_noise(
    df: pd.DataFrame,
    noise_level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Inject small Gaussian noise into numeric features.

    Noise σ is ``noise_level × feature_std``.  Values are clipped to the
    original ``[min, max]`` range of each feature so that the augmented
    data stays within realistic vehicle signal bounds.

    Args:
        df: Input DataFrame (copied internally).
        noise_level: Multiplier for per-feature standard deviation.
        rng: Seeded random generator.

    Returns:
        Augmented DataFrame.
    """
    out = df.copy()

    # Augment payload columns + inter_arrival_idx if present
    target_cols = [c for c in _PAYLOAD_COLS if c in out.columns]
    if "inter_arrival_idx" in out.columns:
        target_cols.append("inter_arrival_idx")

    for col in target_cols:
        vals = out[col].values.astype(np.float64)
        col_std = np.std(vals)
        if col_std == 0:
            continue
        col_min, col_max = np.min(vals), np.max(vals)
        noise = rng.normal(loc=0.0, scale=noise_level * col_std, size=len(vals))
        out[col] = np.clip(vals + noise, col_min, col_max).astype(np.float32)

    logger.info(
        "Feature noise applied: noise_level=%.4f, columns=%d",
        noise_level, len(target_cols),
    )
    return out


def _attack_intensity_scaling(
    df: pd.DataFrame,
    scale_range: Sequence[float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Scale attack-traffic features to simulate weaker/stronger attacks.

    For every attack row, a random scale factor is drawn from
    ``Uniform(scale_min, scale_max)`` and applied to the numeric payload
    columns.  Benign rows are left untouched.  Values are clipped to
    original feature ranges.

    Args:
        df: Input DataFrame (copied internally).
        scale_range: ``(scale_min, scale_max)`` e.g. ``(0.85, 1.15)``.
        rng: Seeded random generator.

    Returns:
        Augmented DataFrame.
    """
    out = df.copy()
    scale_min, scale_max = float(scale_range[0]), float(scale_range[1])

    attack_mask = out["specific_class"] != "BENIGN"
    n_attack = int(attack_mask.sum())
    if n_attack == 0:
        logger.warning("Attack intensity scaling: no attack rows found – skipped")
        return out

    target_cols = [c for c in _PAYLOAD_COLS if c in out.columns]
    if "inter_arrival_idx" in out.columns:
        target_cols.append("inter_arrival_idx")

    # Upcast integer columns to float so scaled values can be stored
    for col in target_cols:
        if out[col].dtype.kind == "i":  # integer
            out[col] = out[col].astype(np.float32)

    # Per-row random scale factor for attack frames
    scale_factors = rng.uniform(scale_min, scale_max, size=n_attack)

    for col in target_cols:
        vals = out.loc[attack_mask, col].values.astype(np.float64)
        col_min = np.min(out[col].values.astype(np.float64))
        col_max = np.max(out[col].values.astype(np.float64))
        scaled = vals * scale_factors
        out.loc[attack_mask, col] = np.clip(scaled, col_min, col_max).astype(
            np.float32
        )

    logger.info(
        "Attack intensity scaling applied: range=[%.2f, %.2f], "
        "attack_rows=%d, columns=%d",
        scale_min, scale_max, n_attack, len(target_cols),
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_augmented_datasets(
    df: pd.DataFrame,
    config: dict,
    seed: int,
) -> list[tuple[str, pd.DataFrame]]:
    """Generate augmented dataset variants according to config.

    Each enabled strategy produces one augmented copy of *df*.  The
    original DataFrame is **not** included in the returned list (the
    caller prepends it as ``("original", df)``).

    Args:
        df: Pre-processed DataFrame (after ``preprocess_presplit``).
        config: Full experiment configuration.
        seed: Global random seed for deterministic augmentation.

    Returns:
        List of ``(variant_name, augmented_df)`` pairs.
    """
    aug_cfg = config.get("augmentation", {})
    if not aug_cfg.get("enabled", False):
        return []

    strategies = aug_cfg.get("strategies", {})
    variants: list[tuple[str, pd.DataFrame]] = []

    # Each strategy gets its own deterministic Generator.
    base_seed = seed

    # --- Timing Jitter ---
    jitter_cfg = strategies.get("timing_jitter", {})
    if jitter_cfg.get("enabled", False):
        rng = np.random.default_rng(base_seed + 1)
        jitter_pct = float(jitter_cfg.get("jitter_pct", 0.02))
        aug_df = _timing_jitter(df, jitter_pct=jitter_pct, rng=rng)
        variants.append(("jitter_augmented", aug_df))

    # --- Feature Noise ---
    noise_cfg = strategies.get("feature_noise", {})
    if noise_cfg.get("enabled", False):
        rng = np.random.default_rng(base_seed + 2)
        noise_level = float(noise_cfg.get("noise_level", 0.01))
        aug_df = _feature_noise(df, noise_level=noise_level, rng=rng)
        variants.append(("noise_augmented", aug_df))

    # --- Attack Intensity Scaling ---
    attack_cfg = strategies.get("attack_intensity", {})
    if attack_cfg.get("enabled", False):
        rng = np.random.default_rng(base_seed + 3)
        scale_range = attack_cfg.get("scale_range", [0.85, 1.15])
        aug_df = _attack_intensity_scaling(df, scale_range=scale_range, rng=rng)
        variants.append(("scaled_attack", aug_df))

    # --- Adversarial Evasion ---
    adversarial_cfg = strategies.get("adversarial_evasion", {})
    if adversarial_cfg.get("enabled", False):
        from src.adversarial import generate_evasion_variant

        rng = np.random.default_rng(base_seed + 4)
        epsilon = float(adversarial_cfg.get("epsilon", 0.03))
        attack_fraction = float(adversarial_cfg.get("attack_fraction", 1.0))
        benign_pull = float(adversarial_cfg.get("benign_pull", 0.7))
        noise_scale = float(adversarial_cfg.get("noise_scale", 0.01))
        aug_df = generate_evasion_variant(
            df,
            epsilon=epsilon,
            attack_fraction=attack_fraction,
            benign_pull=benign_pull,
            noise_scale=noise_scale,
            rng=rng,
        )
        variants.append(("adversarial_evasion", aug_df))

    logger.info("Generated %d augmented dataset variant(s)", len(variants))
    return variants


def save_augmented_dataset(
    name: str,
    df: pd.DataFrame,
    output_dir: str | None = None,
) -> None:
    """Persist an augmented DataFrame to CSV.

    Args:
        name: Variant name (used as filename stem).
        df: Augmented DataFrame.
        output_dir: Directory to write into.  Defaults to
            ``results/augmented_datasets/``.
    """
    from src.utils import ensure_dir, get_project_root

    if output_dir is None:
        output_dir = get_project_root() / "results" / "augmented_datasets"
    out_path = ensure_dir(output_dir) / f"{name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved augmented dataset: %s (%d rows)", out_path, len(df))
