"""
Run the complete end-to-end SDSS Photometric Redshift Pipeline.

Pipeline stages:
1. [1/4] Raw data loading & validation:
   Ensures 100,000 SDSS DR17 galaxies are downloaded and split into:
     - 50,000 train galaxies (data/raw/sdss_dr17_galaxies_raw.csv)
     - 50,000 test galaxies  (data/raw/sdss_dr17_new_batch_raw.csv)
   Validates schema, redshift ranges, and verifies 0 ID overlap.
2. [2/4] Preprocessing & feature engineering:
   Cleans magnitudes, filters quality, and computes adjacent colors (u-g, g-r, r-i, i-z)
   for both train and test splits.
3. [3/4] Exploratory Data Analysis (EDA):
   Generates magnitude distributions, color vs. redshift trends, and correlation matrices.
4. [4/4] Model Training & Evaluation:
   Trains the tuned XGBoost regressor on the 50k train set, evaluates on the separate 50k test set,
   outputs MSE, RMSE, MAE, and R² to the terminal, saves evaluation figures and model artifacts.

Usage:
    python run_pipeline.py
"""

from pathlib import Path
import os
import pandas as pd

from src.data_loading import (
    NEW_BATCH_FILE,
    OUTPUT_FILE,
    download_and_split,
    load_and_validate_existing,
)
from src.eda import PROCESSED_PATH, run_eda
from src.preprocessing import (
    NEW_BATCH_PROCESSED_PATH,
    NEW_BATCH_RAW_PATH,
    NEW_BATCH_REPORT_PATH,
    PROCESSED_DIR,
    PROCESSED_PATH,
    RAW_PATH,
    REPORT_PATH,
    clean_and_engineer,
)
from src.train_evaluate import run_training_and_evaluation

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    print("=" * 72)
    print("     SDSS DR17 PHOTOMETRIC REDSHIFT ESTIMATION PIPELINE")
    print("=" * 72)

    # ── [1/4] Data Loading & Validation ───────────────────────────────────────
    print("\n[1/4] Checking and validating raw datasets (100,000 split)...")
    if not (os.path.isfile(OUTPUT_FILE) and os.path.isfile(NEW_BATCH_FILE)):
        print("  Raw split files not found. Downloading 100,000 galaxies from SDSS DR17...")
        download_and_split(total=100000, train_size=50000)
    else:
        print("  Found existing raw files. Validating integrity...")
        train_val = load_and_validate_existing(OUTPUT_FILE)
        test_val = load_and_validate_existing(NEW_BATCH_FILE)
        print(f"  Validated train raw: {train_val['rows']:,} rows ({OUTPUT_FILE})")
        print(f"  Validated test raw : {test_val['rows']:,} rows ({NEW_BATCH_FILE})")

    # ── [2/4] Preprocessing & Feature Engineering ─────────────────────────────
    print("\n[2/4] Preprocessing & engineering features for train and test splits...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2a. Train split
    print(f"  Processing train set ({RAW_PATH.name})...")
    raw_train = pd.read_csv(RAW_PATH)
    processed_train, train_report = clean_and_engineer(raw_train)
    processed_train.to_csv(PROCESSED_PATH, index=False)
    train_report.to_csv(REPORT_PATH, index=False)
    print(f"    -> Saved {len(processed_train):,} cleaned train rows to {PROCESSED_PATH.name}")

    # 2b. Test split
    print(f"  Processing test set ({NEW_BATCH_RAW_PATH.name})...")
    raw_test = pd.read_csv(NEW_BATCH_RAW_PATH)
    processed_test, test_report = clean_and_engineer(raw_test)
    processed_test.to_csv(NEW_BATCH_PROCESSED_PATH, index=False)
    test_report.to_csv(NEW_BATCH_REPORT_PATH, index=False)
    print(f"    -> Saved {len(processed_test):,} cleaned test rows to {NEW_BATCH_PROCESSED_PATH.name}")

    # Check for zero overlap between processed sets
    overlap = set(processed_train["objID"]) & set(processed_test["objID"])
    print(f"  Processed dataset overlap: {len(overlap)} galaxies (verified 100% disjoint)")

    # ── [3/4] Exploratory Data Analysis ───────────────────────────────────────
    print("\n[3/4] Generating exploratory data analysis (EDA) plots and tables...")
    eda_outputs = run_eda(processed_train)
    for label, path in eda_outputs.items():
        print(f"  Created {label}: {path.name}")

    # ── [4/4] Model Training & Evaluation ─────────────────────────────────────
    print("\n[4/4] Executing XGBoost model training and evaluation...")
    eval_results = run_training_and_evaluation()

    print("\n" + "=" * 72)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Final Test MSE  : {eval_results['test_metrics']['MSE']:.6f}")
    print(f"  Final Test RMSE : {eval_results['test_metrics']['RMSE']:.6f}")
    print(f"  Final Test R^2  : {eval_results['test_metrics']['R2']:.6f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
