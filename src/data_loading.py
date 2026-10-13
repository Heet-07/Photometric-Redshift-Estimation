"""
Download galaxy data from SDSS DR17 SkyServer.

This script sends a SQL query to the SDSS SkyServer REST API and saves
the result as a CSV file. No account or authentication is required.

Usage:
    python src/data_loading.py

Output:
    data/raw/sdss_dr17_galaxies_raw.csv  (~50,000 rows)

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

# SDSS SkyServer SQL Search endpoint (DR17)
SKYSERVER_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

# Where to save the downloaded data
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sdss_dr17_galaxies_raw.csv")

EXPECTED_COLUMNS = [
    "objID",
    "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
    "specz",
]
MAGNITUDE_COLUMNS = [
    "modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
]

# The SQL query that defines our dataset.
#
# What this query does:
#   - Pulls 50,000 galaxies from SDSS DR17
#   - Gets brightness measurements in 5 filters (u, g, r, i, z)
#   - Gets the spectroscopic redshift (our prediction target)
#
# Quality filters applied:
#   - class = 'GALAXY'   → only galaxies, not stars or quasars
#   - zWarning = 0        → only reliable redshift measurements
#   - z BETWEEN 0.01-0.5  → main galaxy survey redshift range
#
# Column names verified against DR17 schema:
#   https://skyserver.sdss.org/dr17/en/help/browser/browser.aspx
#
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


def build_query(limit: int = 50000) -> str:
    """Build the documented query with a bounded result size."""
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
    params = {
        "cmd": query,
        "format": "csv",
    }

    for attempt in range(1, max_retries + 1):
        print(f"  Attempt {attempt}/{max_retries}... ", end="", flush=True)
        try:
            response = requests.get(
                SKYSERVER_URL,
                params=params,
                timeout=120,  # 2-minute timeout
            )
            response.raise_for_status()

            # SkyServer returns CSV text; check for error messages
            text = response.text.strip()
            if text.startswith("<"):
                # HTML error page returned instead of CSV
                raise ValueError(
                    f"SkyServer returned an error page. "
                    f"First 300 chars: {text[:300]}"
                )

            # SkyServer CSV often includes comment lines starting with '#'
            # (e.g. '#Table1'). Strip those before parsing.
            lines = text.split("\n")
            clean_lines = [line for line in lines if not line.startswith("#")]
            clean_text = "\n".join(clean_lines).strip()

            if not clean_text:
                raise ValueError("Response was empty after removing comment lines")

            # Parse CSV
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
    """
    Run basic sanity checks on the downloaded data.
    Raises AssertionError if anything looks wrong.
    """
    # Check columns exist
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    assert not missing_columns, f"Missing expected columns: {missing_columns}"

    # Check we got a reasonable number of rows
    assert len(df) >= minimum_rows, f"Too few rows downloaded: {len(df)}"

    # Ensure identifiers and target are numeric and usable.
    assert pd.api.types.is_numeric_dtype(df["objID"]), "objID must be numeric"
    assert pd.api.types.is_numeric_dtype(df["specz"]), "specz must be numeric"
    assert df["objID"].is_unique, "Duplicate objID values found"

    numeric_frame = df[EXPECTED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    non_finite_counts = {
        col: int((~np.isfinite(numeric_frame[col].to_numpy(dtype=float))).sum())
        for col in EXPECTED_COLUMNS
    }
    assert sum(non_finite_counts.values()) == 0, f"Non-finite values found: {non_finite_counts}"

    # Check redshift range
    assert df["specz"].min() >= 0.0, f"Negative redshift found: {df['specz'].min()}"
    assert df["specz"].max() <= 1.0, f"Unexpectedly high redshift: {df['specz'].max()}"

    null_counts = {col: int(df[col].isna().sum()) for col in EXPECTED_COLUMNS}
    sentinel_counts = {
        col: int((df[col] == -9999).sum()) for col in MAGNITUDE_COLUMNS
    }

    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "duplicate_objID_count": int(df["objID"].duplicated().sum()),
        "null_counts": null_counts,
        "sentinel_counts": sentinel_counts,
        "redshift_min": float(df["specz"].min()),
        "redshift_max": float(df["specz"].max()),
        "redshift_median": float(df["specz"].median()),
    }
    print("  Validation passed")
    print(f"  Rows: {report['rows']:,}; sentinel magnitudes: {sentinel_counts}")
    return report


def load_and_validate_existing(path: str) -> dict:
    """Validate an existing raw CSV without changing it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    print(f"Validating existing raw CSV: {path}")
    return validate_dataframe(pd.read_csv(path), minimum_rows=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50000,
                        help="maximum rows to request (default: 50000)")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help="raw CSV output path")
    parser.add_argument("--force", action="store_true",
                        help="allow replacing an existing output CSV")
    parser.add_argument("--validate-existing", action="store_true",
                        help="validate an existing CSV and skip downloading")
    args = parser.parse_args()

    print("=" * 60)
    print("SDSS DR17 Galaxy Data Download")
    print("=" * 60)

    if args.validate_existing:
        report = load_and_validate_existing(args.output)
        print(report)
        return report

    query = build_query(args.limit)
    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing raw CSV: {args.output}. "
            "Use --validate-existing or choose another --output path."
        )

    # Show the query for transparency
    print("\nSQL Query:")
    print("-" * 40)
    print(query)
    print("-" * 40)

    # Download
    print("\nDownloading from SkyServer...")
    df = download_from_skyserver(query)

    # Validate
    print("\nValidating data...")
    validate_dataframe(df, minimum_rows=1)

    # Summary
    print(f"\nDataset summary:")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Redshift range: {df['specz'].min():.4f} — {df['specz'].max():.4f}")
    print(f"  Redshift median: {df['specz'].median():.4f}")

    # Save
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    temporary_output = f"{args.output}.part"
    try:
        df.to_csv(temporary_output, index=False)
        os.replace(temporary_output, args.output)
    finally:
        if os.path.exists(temporary_output):
            os.remove(temporary_output)
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved to: {args.output}")
    print(f"File size: {file_size_mb:.1f} MB")

    print("\n✓ Done! Next step: run the EDA notebook.")
    return df


if __name__ == "__main__":
    main()
