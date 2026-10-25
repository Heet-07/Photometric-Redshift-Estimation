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

# New 50 K batch — separate files, originals never touched
NEW_BATCH_RAW_PATH       = PROJECT_ROOT / "data" / "raw"       / "sdss_dr17_new_batch_raw.csv"
NEW_BATCH_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sdss_dr17_new_batch_processed.csv"
NEW_BATCH_REPORT_PATH    = PROJECT_ROOT / "data" / "processed" / "new_batch_cleaning_report.csv"

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


def process_new_batch(
    raw_path: Path = NEW_BATCH_RAW_PATH,
    processed_path: Path = NEW_BATCH_PROCESSED_PATH,
    report_path: Path = NEW_BATCH_REPORT_PATH,
) -> pd.DataFrame:
    """
    Apply the same cleaning and feature-engineering pipeline to the new
    50 K raw batch and save the result to a separate processed CSV.

    Parameters
    ----------
    raw_path       : path to the new batch raw CSV
    processed_path : destination for the cleaned CSV
    report_path    : destination for the cleaning-stage report

    Returns
    -------
    pd.DataFrame  — the cleaned, feature-engineered batch
    """
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"New batch raw CSV not found: {raw_path}\n"
            "Run fetch_new_batch() in data_loading.py first."
        )

    print(f"Reading raw batch: {raw_path}")
    raw = pd.read_csv(raw_path)
    print(f"  Rows in raw batch: {len(raw):,}")

    processed, report = clean_and_engineer(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    report.to_csv(report_path, index=False)

    print(report.to_string(index=False))
    print(f"Processed rows  : {len(processed):,}")
    print(f"Processed data  : {processed_path}")
    print(f"Cleaning report : {report_path}")
    return processed


if __name__ == "__main__":
    main()

