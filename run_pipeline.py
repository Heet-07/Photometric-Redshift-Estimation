"""Run the validated raw-data, preprocessing, and EDA pipeline."""

from pathlib import Path

import pandas as pd

from src.data_loading import OUTPUT_FILE, load_and_validate_existing
from src.eda import PROCESSED_PATH, run_eda
from src.preprocessing import (
    PROCESSED_DIR,
    REPORT_PATH,
    clean_and_engineer,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 72)
    print("SDSS PHOTOMETRIC REDSHIFT PIPELINE")
    print("=" * 72)

    print("\n[1/3] Validating existing raw CSV")
    validation = load_and_validate_existing(OUTPUT_FILE)
    print(f"Validated {validation['rows']:,} raw rows")

    print("\n[2/3] Cleaning data and engineering colors")
    raw = pd.read_csv(OUTPUT_FILE)
    processed, cleaning_report = clean_and_engineer(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_PATH, index=False)
    cleaning_report.to_csv(REPORT_PATH, index=False)
    print(cleaning_report.to_string(index=False))
    print(f"Saved {len(processed):,} processed rows to {PROCESSED_PATH}")

    print("\n[3/3] Creating exploratory analysis outputs")
    outputs = run_eda(processed)
    for label, path in outputs.items():
        print(f"{label}: {path}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
