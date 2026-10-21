"""Predict photometric redshifts from a CSV containing the model features.

Example: ``python src/predict.py --input new_galaxies.csv --output predictions.csv``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "xgboost_photometric_redshift.json"
METADATA = ROOT / "models" / "xgboost_photometric_redshift_metadata.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    data = pd.read_csv(args.input)
    features = metadata["feature_columns"]
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required feature columns: {missing}")
    model = xgb.XGBRegressor(); model.load_model(MODEL)
    output = data.copy(); output["predicted_specz"] = model.predict(data[features])
    args.output.parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output, index=False)
    print(f"Predicted {len(output):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
