"""Clean the raw SDSS extract and add interpretable adjacent-band colors."""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.features import COLOR_COLUMNS, MAGNITUDE_COLUMNS, SENTINEL, engineer_features
except ModuleNotFoundError:  # direct invocation: python src/preprocessing.py
    from features import COLOR_COLUMNS, MAGNITUDE_COLUMNS, SENTINEL, engineer_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sdss_dr17_galaxies_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "sdss_dr17_galaxies_processed.csv"
REPORT_PATH = PROCESSED_DIR / "cleaning_report.csv"

COLOR_PAIRS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]
REQUIRED_COLUMNS = ["objID", *MAGNITUDE_COLUMNS, "specz"]


def clean_and_engineer(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return modeling rows and a compact, stage-by-stage row-count report.

    Rows with an invalid identifier or redshift are excluded first. Rows with
    missing, non-finite, or SDSS sentinel magnitudes are then excluded because
    all five bands are required for the four adjacent colors in this phase.
    The raw input is never modified.
    """
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Raw dataset is missing columns: {missing_columns}")

    data = raw[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    report_rows = [{"stage": "raw input", "rows_remaining": int(len(data)), "rows_removed": 0}]

    valid_target = (
        np.isfinite(data["objID"])
        & np.isfinite(data["specz"])
        & data["specz"].between(0.01, 0.5)
    )
    before = len(data)
    data = data.loc[valid_target].copy()
    report_rows.append({
        "stage": "valid identifier and specz in [0.01, 0.5]",
        "rows_remaining": int(len(data)),
        "rows_removed": int(before - len(data)),
    })

    finite_magnitudes = np.isfinite(data[MAGNITUDE_COLUMNS]).all(axis=1)
    measured_magnitudes = data[MAGNITUDE_COLUMNS].ne(SENTINEL).all(axis=1)
    valid_photometry = finite_magnitudes & measured_magnitudes
    before = len(data)
    data = data.loc[valid_photometry].copy()
    report_rows.append({
        "stage": "finite, non-sentinel measurements in all five bands",
        "rows_remaining": int(len(data)),
        "rows_removed": int(before - len(data)),
    })

    before = len(data)
    data = data.drop_duplicates(subset="objID", keep="first").copy()
    report_rows.append({
        "stage": "unique objID",
        "rows_remaining": int(len(data)),
        "rows_removed": int(before - len(data)),
    })

    engineered = engineer_features(data[MAGNITUDE_COLUMNS])
    for column in COLOR_COLUMNS:
        data[column] = engineered[column]

    return data.reset_index(drop=True), pd.DataFrame(report_rows)


def main() -> None:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH)
    processed, report = clean_and_engineer(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)
    report.to_csv(REPORT_PATH, index=False)

    print(f"Raw rows: {len(raw):,}")
    print(report.to_string(index=False))
    print(f"Processed rows: {len(processed):,}")
    print(f"Processed data: {PROCESSED_PATH}")
    print(f"Cleaning report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
