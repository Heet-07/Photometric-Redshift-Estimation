"""Fit the selected configuration on all cleaned rows for deployment."""

from pathlib import Path
import json
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sdss_dr17_galaxies_processed.csv"
MODEL_DIR = ROOT / "models"
FEATURES = ["modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z", "color_u_g", "color_g_r", "color_r_i", "color_i_z"]
PARAMS = {"colsample_bytree": .8, "learning_rate": .05, "max_depth": 8, "min_child_weight": 3, "n_estimators": 200, "subsample": .8}


def main() -> None:
    data = pd.read_csv(DATA); MODEL_DIR.mkdir(exist_ok=True)
    model = xgb.XGBRegressor(**PARAMS, objective="reg:squarederror", tree_method="hist", device="cpu", random_state=42, n_jobs=-1)
    model.fit(data[FEATURES], data["specz"], verbose=False)
    path = MODEL_DIR / "xgboost_photometric_redshift_deployment.json"; model.save_model(path)
    metadata = {"feature_columns": FEATURES, "target_column": "specz", "training_rows": len(data), "purpose": "deployment refit on all cleaned rows", "parameters": PARAMS}
    (MODEL_DIR / "xgboost_photometric_redshift_deployment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved deployment model: {path} ({len(data):,} rows)")


if __name__ == "__main__": main()
