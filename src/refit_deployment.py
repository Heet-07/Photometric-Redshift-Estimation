"""Fit the selected configuration on all cleaned rows for deployment."""

from pathlib import Path
import json
import pandas as pd
import xgboost as xgb

try:
    from src.features import MODEL_FEATURE_COLUMNS
except ModuleNotFoundError:  # direct invocation: python src/refit_deployment.py
    from features import MODEL_FEATURE_COLUMNS

try:
    from src.train_evaluate import TUNED_PARAMS, choose_device
except ModuleNotFoundError:  # direct invocation: python src/refit_deployment.py
    from train_evaluate import TUNED_PARAMS, choose_device

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sdss_dr17_galaxies_processed.csv"
MODEL_DIR = ROOT / "models"
FEATURES = MODEL_FEATURE_COLUMNS
ARTIFACT_VERSION = "photometric-redshift-xgb-v1"


def main() -> None:
    if not DATA.is_file():
        raise FileNotFoundError(f"Processed training data not found: {DATA}")

    data = pd.read_csv(DATA)
    MODEL_DIR.mkdir(exist_ok=True)
    device, device_description = choose_device()
    params = dict(TUNED_PARAMS)
    model = xgb.XGBRegressor(
        **params,
        device=device,
        random_state=42,
        n_jobs=1 if device == "cuda" else -1,
        verbosity=0,
    )
    model.fit(data[FEATURES], data["specz"], verbose=False)
    path = MODEL_DIR / "xgboost_photometric_redshift_deployment.json"; model.save_model(path)
    metadata = {"artifact_version": ARTIFACT_VERSION, "artifact_kind": "deployment", "model_type": "xgboost.XGBRegressor", "xgboost_version": xgb.__version__, "device": device, "device_description": device_description, "feature_columns": FEATURES, "target_column": "specz", "training_rows": len(data), "purpose": "deployment refit on all cleaned training rows after independent test evaluation", "parameters": params}
    (MODEL_DIR / "xgboost_photometric_redshift_deployment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved deployment model: {path} ({len(data):,} rows)")


if __name__ == "__main__": main()
