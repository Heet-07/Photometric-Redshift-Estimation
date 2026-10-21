"""Predict photometric redshifts from a CSV containing five SDSS magnitudes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from src.features import MODEL_FEATURE_COLUMNS, prepare_inference
except ModuleNotFoundError:  # direct invocation: python src/predict.py
    from features import MODEL_FEATURE_COLUMNS, prepare_inference

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_VERSION = "photometric-redshift-xgb-v1"
ARTIFACTS = {
    "evaluation": (ROOT / "models" / "xgboost_photometric_redshift.json", ROOT / "models" / "xgboost_photometric_redshift_metadata.json"),
    "deployment": (ROOT / "models" / "xgboost_photometric_redshift_deployment.json", ROOT / "models" / "xgboost_photometric_redshift_deployment_metadata.json"),
}


def load_artifacts(kind: str = "deployment", model_path: Path | None = None, metadata_path: Path | None = None) -> tuple[xgb.XGBRegressor, dict]:
    """Load one explicitly selected model and verify its feature contract."""
    if kind not in ARTIFACTS:
        raise ValueError(f"Unknown model kind {kind!r}; choose evaluation or deployment")
    default_model, default_metadata = ARTIFACTS[kind]
    model_path = Path(model_path or default_model)
    metadata_path = Path(metadata_path or default_metadata)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_kind") not in (None, kind):
        raise ValueError(f"Selected {kind} artifact does not match metadata kind {metadata.get('artifact_kind')!r}")
    if metadata.get("artifact_version") not in (None, ARTIFACT_VERSION):
        raise ValueError("Unsupported model artifact version")
    if metadata.get("model_type") not in (None, "xgboost.XGBRegressor"):
        raise ValueError("Unsupported model type in metadata")
    if metadata.get("feature_columns") != MODEL_FEATURE_COLUMNS:
        raise ValueError("Model metadata feature schema does not match the current feature engineering contract")
    if metadata.get("target_column") != "specz":
        raise ValueError("Model metadata target column must be 'specz'")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    if model.get_booster().num_features() != len(MODEL_FEATURE_COLUMNS):
        raise ValueError("Model feature count does not match model metadata")
    return model, metadata


def predict_dataframe(data: pd.DataFrame, model: xgb.XGBRegressor) -> pd.DataFrame:
    """Return original columns plus predictions and invalid-row status."""
    if data.empty:
        raise ValueError("Input CSV contains no rows")
    features, valid, reasons = prepare_inference(data)
    output = data.copy()
    output["predicted_specz"] = np.nan
    output["prediction_status"] = "invalid"
    output["invalid_reason"] = reasons
    if valid.any():
        prediction = model.predict(features.loc[valid, MODEL_FEATURE_COLUMNS])
        if not np.isfinite(prediction).all():
            raise ValueError("Model produced a non-finite prediction")
        output.loc[valid, "predicted_specz"] = prediction
        output.loc[valid, "prediction_status"] = "predicted"
        output.loc[valid, "invalid_reason"] = ""
    return output


def write_predictions(output: pd.DataFrame, path: Path) -> None:
    """Write predictions atomically."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        output.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=sorted(ARTIFACTS), default="deployment", help="artifact to use (default: deployment)")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different")
    model, _ = load_artifacts(args.model)
    output = predict_dataframe(pd.read_csv(args.input), model)
    write_predictions(output, args.output)
    print(f"Predicted {int((output['prediction_status'] == 'predicted').sum()):,} rows; rejected {int((output['prediction_status'] == 'invalid').sum()):,} -> {args.output}")


if __name__ == "__main__":
    main()
