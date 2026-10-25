"""
Download galaxy data from SDSS DR17 SkyServer and split into train / test.

Downloads 100,000 galaxies in a single query, then deterministically splits:
  - First 50,000 rows → data/raw/sdss_dr17_galaxies_raw.csv       (train)
  - Last  50,000 rows → data/raw/sdss_dr17_new_batch_raw.csv      (test)

Because both files come from the same ordered API response, the split is
guaranteed to be non-overlapping by construction — no duplicate objIDs.

Usage:
    python src/data_loading.py

Data source:
    Sloan Digital Sky Survey, Data Release 17
    https://skyserver.sdss.org/dr17/
"""

import argparse
import os
import time
import requests
import numpy as np
import pandas as pd
from io import StringIO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SKYSERVER_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "sdss_dr17_galaxies_raw.csv")    # train  (50 K)
NEW_BATCH_FILE = os.path.join(OUTPUT_DIR, "sdss_dr17_new_batch_raw.csv")   # test   (50 K)

TOTAL_ROWS = 100_000
TRAIN_ROWS = 50_000   # first 50 K  → train
TEST_ROWS  = 50_000   # last  50 K  → test

EXPECTED_COLUMNS = [
    "objID",
    "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
    "specz",
]
MAGNITUDE_COLUMNS = [
    "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
]

# Single SQL query — fetches 100 K rows with quality filters.
#
# Quality filters:
#   class = 'GALAXY'   → only galaxies, not stars or quasars
#   zWarning = 0       → only reliable redshift measurements
#   z BETWEEN 0.01-0.5 → main galaxy survey redshift range
#
# Column names verified against DR17 schema:
#   https://skyserver.sdss.org/dr17/en/help/browser/browser.aspx
SQL_QUERY_TEMPLATE = """
SELECT TOP {limit}
    p.objID,
    p.modelMag_u,
    p.modelMag_g,
    p.modelMag_r,
    p.modelMag_i,
    p.modelMag_z,
    s.z AS specz
FROM SpecPhoto AS sp
    JOIN PhotoObj AS p ON sp.objID = p.objID
    JOIN SpecObj  AS s ON sp.specObjID = s.specObjID
WHERE
    s.class = 'GALAXY'
    AND s.zWarning = 0
    AND s.z BETWEEN 0.01 AND 0.5
""".strip()


def build_query(limit: int = TOTAL_ROWS) -> str:
    """Build the SQL query with a bounded result size."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    return SQL_QUERY_TEMPLATE.format(limit=int(limit))


def download_from_skyserver(query: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Send a SQL query to SDSS SkyServer and return the result as a DataFrame.

    Parameters
    ----------
    query : str
        A valid SDSS CAS SQL query.
    max_retries : int
        Number of times to retry on failure.

    Returns
    -------
    pd.DataFrame
        Query results as a pandas DataFrame.
    """
    params = {"cmd": query, "format": "csv"}

    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}... ", end="", flush=True)
        try:
            response = requests.get(SKYSERVER_URL, params=params, timeout=180)
            response.raise_for_status()

            text = response.text.strip()
            if text.startswith("<"):
                raise ValueError(
                    f"SkyServer returned an error page. First 300 chars: {text[:300]}"
                )

            lines = text.split("\n")
            clean_lines = [line for line in lines if not line.startswith("#")]
            clean_text = "\n".join(clean_lines).strip()

            if not clean_text:
                raise ValueError("Response was empty after removing comment lines")

            df = pd.read_csv(StringIO(clean_text))
            print(f"OK — received {len(df):,} rows")
            return df

        except (requests.RequestException, ValueError) as e:
            print(f"FAILED — {e}")
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print("  All retries exhausted.")
                raise


def validate_dataframe(df: pd.DataFrame, minimum_rows: int = 1) -> dict:
    """Run basic sanity checks. Raises AssertionError if anything looks wrong."""
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    assert not missing_columns, f"Missing expected columns: {missing_columns}"
    assert len(df) >= minimum_rows, f"Too few rows: {len(df)}"
    assert pd.api.types.is_numeric_dtype(df["objID"]), "objID must be numeric"
    assert pd.api.types.is_numeric_dtype(df["specz"]), "specz must be numeric"
    assert df["objID"].is_unique, "Duplicate objID values found"

    numeric_frame = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    non_finite_counts = {
        col: int((~np.isfinite(numeric_frame[col].to_numpy(dtype=float))).sum())
        for col in EXPECTED_COLUMNS
    }
    assert sum(non_finite_counts.values()) == 0, f"Non-finite values: {non_finite_counts}"
    assert df["specz"].min() >= 0.0, f"Negative redshift: {df['specz'].min()}"
    assert df["specz"].max() <= 1.0, f"Unexpectedly high redshift: {df['specz'].max()}"

    sentinel_counts = {col: int((df[col] == -9999).sum()) for col in MAGNITUDE_COLUMNS}
    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_objID_count": int(df["objID"].duplicated().sum()),
        "null_counts": {col: int(df[col].isna().sum()) for col in EXPECTED_COLUMNS},
        "sentinel_counts": sentinel_counts,
        "redshift_min": float(df["specz"].min()),
        "redshift_max": float(df["specz"].max()),
        "redshift_median": float(df["specz"].median()),
    }
    print("  Validation passed")
    print(f"  Rows: {report['rows']:,};  sentinel magnitudes: {sentinel_counts}")
    return report


def load_and_validate_existing(path: str) -> dict:
    """Validate an existing raw CSV without changing it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    print(f"Validating existing raw CSV: {path}")
    return validate_dataframe(pd.read_csv(path), minimum_rows=1)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    """Atomically write a DataFrame to CSV using a .part temp file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.part"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def download_and_split(
    total: int = TOTAL_ROWS,
    train_size: int = TRAIN_ROWS,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download `total` galaxies from SDSS DR17 in one API call, then split:
      - First `train_size` rows → OUTPUT_FILE    (train raw)
      - Remaining rows          → NEW_BATCH_FILE  (test  raw)

    Because the split is a positional slice of a single ordered response,
    train and test are guaranteed to be non-overlapping by construction.

    Parameters
    ----------
    total      : total rows to download (default 100 000)
    train_size : rows assigned to train  (default 50 000)
    force      : re-download even if output files already exist

    Returns
    -------
    (train_df, test_df)
    """
    test_size = total - train_size

    print("=" * 60)
    print("SDSS DR17 — Download & Split")
    print(f"  Total download : {total:,} rows")
    print(f"  Train split    : first {train_size:,} rows → {os.path.basename(OUTPUT_FILE)}")
    print(f"  Test  split    : last  {test_size:,} rows → {os.path.basename(NEW_BATCH_FILE)}")
    print("=" * 60)

    # Use cached files if available and force=False
    if not force and os.path.isfile(OUTPUT_FILE) and os.path.isfile(NEW_BATCH_FILE):
        print("\nBoth split files already exist — loading from cache.")
        print("  (Pass force=True to re-download.)")
        train_df = pd.read_csv(OUTPUT_FILE)
        test_df  = pd.read_csv(NEW_BATCH_FILE)
        print(f"  Train : {len(train_df):,} rows")
        print(f"  Test  : {len(test_df):,} rows")
        return train_df, test_df

    # Download
    query = build_query(total)
    print("\nSQL Query:")
    print("-" * 40)
    print(query)
    print("-" * 40)
    print(f"\nDownloading {total:,} rows from SkyServer...")
    full_df = download_from_skyserver(query)

    # Validate full dataset
    print("\nValidating full dataset...")
    validate_dataframe(full_df, minimum_rows=total)

    # Split by position
    train_df = full_df.iloc[:train_size].reset_index(drop=True)
    test_df  = full_df.iloc[train_size:].reset_index(drop=True)

    # Confirm zero overlap
    overlap = set(train_df["objID"]) & set(test_df["objID"])
    assert len(overlap) == 0, f"Unexpected overlap of {len(overlap)} objIDs after split!"
    print(f"  Overlap check passed — 0 shared objIDs between train and test splits.")

    # Save
    print(f"\nSaving train split ({len(train_df):,} rows)...")
    _save_csv(train_df, OUTPUT_FILE)
    print(f"  → {OUTPUT_FILE}  ({os.path.getsize(OUTPUT_FILE)/1e6:.1f} MB)")

    print(f"\nSaving test split ({len(test_df):,} rows)...")
    _save_csv(test_df, NEW_BATCH_FILE)
    print(f"  → {NEW_BATCH_FILE}  ({os.path.getsize(NEW_BATCH_FILE)/1e6:.1f} MB)")

    print("\n✓ Download and split complete.")
    print(f"  Train : {len(train_df):,} rows  |  Test : {len(test_df):,} rows")
    return train_df, test_df


def fetch_new_batch(force: bool = False) -> pd.DataFrame:
    """
    Return the test split (50 K rows).

    If the file already exists, loads from cache.
    If not, calls download_and_split() to create both files first.

    Parameters
    ----------
    force : re-download from API even if file exists

    Returns
    -------
    pd.DataFrame — the 50 K test raw data
    """
    if not force and os.path.isfile(NEW_BATCH_FILE):
        print(f"Loading cached test split: {NEW_BATCH_FILE}")
        df = pd.read_csv(NEW_BATCH_FILE)
        print(f"  Loaded {len(df):,} rows.")
        return df

    print("Test split not found — running download_and_split() first...")
    _, test_df = download_and_split(force=force)
    return test_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total",      type=int, default=TOTAL_ROWS,
                        help=f"total rows to download (default: {TOTAL_ROWS})")
    parser.add_argument("--train-size", type=int, default=TRAIN_ROWS,
                        help=f"rows assigned to train split (default: {TRAIN_ROWS})")
    parser.add_argument("--force",      action="store_true",
                        help="re-download even if output files already exist")
    parser.add_argument("--validate-existing", action="store_true",
                        help="validate existing CSVs and skip downloading")
    args = parser.parse_args()

    if args.validate_existing:
        print("Validating existing files...")
        r_train = load_and_validate_existing(OUTPUT_FILE)
        r_test  = load_and_validate_existing(NEW_BATCH_FILE)
        print(f"Train : {r_train['rows']:,} rows")
        print(f"Test  : {r_test['rows']:,} rows")
        return

    download_and_split(
        total=args.total,
        train_size=args.train_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
