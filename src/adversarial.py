"""Adversarial sample generation for CAN intrusion-detection robustness tests.

The routines in this module create *evasion-oriented* perturbations of
attack frames. Perturbations are constrained by per-feature ranges so
generated samples remain plausible while being shifted toward benign traffic
statistics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PAYLOAD_COLS = [f"DATA_{i}" for i in range(8)]


def _target_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns eligible for adversarial perturbation."""
    cols = [c for c in _PAYLOAD_COLS if c in df.columns]
    if "inter_arrival_idx" in df.columns:
        cols.append("inter_arrival_idx")
    return cols


def generate_evasion_variant(
    df: pd.DataFrame,
    epsilon: float,
    attack_fraction: float,
    benign_pull: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate an adversarial-evasion variant of a CAN dataframe.

    Strategy:
    1. Select attack rows (``specific_class != BENIGN``).
    2. Compute benign centroid for each perturbable feature.
    3. Move selected attack samples toward benign centroid plus noise.
    4. Project perturbation into an ``epsilon`` L-infinity ball around the
       original sample and clip to global feature ranges.

    Args:
        df: Input DataFrame.
        epsilon: Max perturbation as fraction of per-feature global range.
        attack_fraction: Fraction of attack rows to perturb.
        benign_pull: Strength of deterministic pull toward benign centroid.
        noise_scale: Random noise scale as fraction of feature range.
        rng: Deterministic random generator.

    Returns:
        Perturbed DataFrame copy.
    """
    out = df.copy()
    cols = _target_columns(out)
    if not cols:
        logger.warning("Adversarial evasion skipped: no perturbable feature columns")
        return out

    if "specific_class" not in out.columns:
        logger.warning("Adversarial evasion skipped: missing 'specific_class' column")
        return out

    attack_mask = out["specific_class"] != "BENIGN"
    benign_mask = out["specific_class"] == "BENIGN"
    n_attack_total = int(attack_mask.sum())
    n_benign = int(benign_mask.sum())

    if n_attack_total == 0 or n_benign == 0:
        logger.warning(
            "Adversarial evasion skipped: attack_rows=%d, benign_rows=%d",
            n_attack_total, n_benign,
        )
        return out

    # Select a subset of attack rows for perturbation.
    attack_indices = np.flatnonzero(attack_mask.values)
    n_selected = max(1, int(round(n_attack_total * float(attack_fraction))))
    n_selected = min(n_selected, n_attack_total)
    selected_idx = rng.choice(attack_indices, size=n_selected, replace=False)

    values = out[cols].astype(np.float64).to_numpy()
    benign_centroid = values[benign_mask.values].mean(axis=0)
    feat_min = values.min(axis=0)
    feat_max = values.max(axis=0)
    feat_range = np.maximum(feat_max - feat_min, 1e-12)

    # Ensure columns can store float perturbations.
    for col in cols:
        if out[col].dtype.kind == "i":
            out[col] = out[col].astype(np.float32)

    alpha = float(benign_pull)
    eps = float(epsilon)
    ns = float(noise_scale)

    original = out.loc[selected_idx, cols].to_numpy(dtype=np.float64)
    direction = benign_centroid[None, :] - original
    stochastic = rng.normal(
        loc=0.0,
        scale=ns * feat_range[None, :],
        size=direction.shape,
    )
    candidate = original + alpha * direction + stochastic

    # Project into epsilon-box around original and clip to valid feature bounds.
    delta = np.clip(candidate - original, -eps * feat_range, eps * feat_range)
    adversarial = np.clip(original + delta, feat_min, feat_max).astype(np.float32)
    out.loc[selected_idx, cols] = adversarial

    logger.info(
        "Adversarial evasion applied: attack_rows=%d, perturbed=%d, epsilon=%.4f, "
        "benign_pull=%.3f, noise_scale=%.4f, cols=%d",
        n_attack_total, n_selected, eps, alpha, ns, len(cols),
    )
    return out
