"""Authoritative SDSS photometry validation and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

MAGNITUDE_COLUMNS = [f"modelMag_{band}" for band in "ugriz"]
COLOR_COLUMNS = ["color_u_g", "color_g_r", "color_r_i", "color_i_z"]
MODEL_FEATURE_COLUMNS = MAGNITUDE_COLUMNS + COLOR_COLUMNS
SENTINEL = -9999


def _numeric_magnitudes(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(MAGNITUDE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required SDSS magnitude columns: {missing}")
    return frame[MAGNITUDE_COLUMNS].apply(pd.to_numeric, errors="coerce")


def validate_magnitudes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return numeric values, a valid-row mask, and human-readable reasons."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input photometry is empty")
    numeric = _numeric_magnitudes(frame)
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    sentinel = numeric.eq(SENTINEL).any(axis=1).to_numpy()
    valid = pd.Series(finite & ~sentinel, index=frame.index)
    reasons = pd.Series("", index=frame.index, dtype="object")
    for index in frame.index[~valid]:
        row = numeric.loc[index]
        parts = []
        if not np.isfinite(row.to_numpy(dtype=float)).all():
            parts.append("missing/non-numeric/non-finite magnitude")
        if (row == SENTINEL).any():
            parts.append("SDSS sentinel -9999")
        reasons.loc[index] = "; ".join(parts)
    return numeric, valid, reasons


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate model features from five valid SDSS magnitudes."""
    numeric, valid, reasons = validate_magnitudes(frame)
    if not valid.all():
        bad = reasons.loc[~valid].unique().tolist()
        raise ValueError(f"Cannot engineer features from invalid photometry: {bad}")
    features = numeric.copy()
    for blue, red in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        features[f"color_{blue}_{red}"] = features[f"modelMag_{blue}"] - features[f"modelMag_{red}"]
    return features[MODEL_FEATURE_COLUMNS]


def prepare_inference(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare valid rows for prediction while retaining invalid-row reasons."""
    numeric, valid, reasons = validate_magnitudes(frame)
    features = pd.DataFrame(index=frame.index, columns=MODEL_FEATURE_COLUMNS, dtype=float)
    if valid.any():
        features.loc[valid, :] = engineer_features(numeric.loc[valid])
    return features, valid, reasons
