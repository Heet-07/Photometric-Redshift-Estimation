"""Train, evaluate, and persist the selected photometric-redshift model.

Run from the project root with ``python src/train_evaluate.py``. The script
uses the tuned XGBoost configuration recorded in the tuning notebook, creates
deterministic 80/10/10 partitions, writes diagnostics, and saves the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sdss_dr17_galaxies_processed.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR, METRICS_DIR, MODELS_DIR = RESULTS_DIR / "figures", RESULTS_DIR / "metrics", PROJECT_ROOT / "models"
FEATURE_COLUMNS = ["modelMag_u", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z", "color_u_g", "color_g_r", "color_r_i", "color_i_z"]
TARGET_COLUMN, SEED = "specz", 42
ARTIFACT_VERSION = "photometric-redshift-xgb-v1"
TUNED_PARAMS = {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 3, "n_estimators": 200, "subsample": 0.8}


def stratification_bins(y: pd.Series) -> pd.Series:
    return pd.qcut(y, q=20, duplicates="drop")


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    normalized = residual / (1.0 + y_true)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(residual)),
        "Normalized_MAD": float(1.4826 * np.median(np.abs(normalized - np.median(normalized)))),
        "Catastrophic_outlier_fraction": float(np.mean(np.abs(normalized) > 0.15)),
    }


def choose_device() -> str:
    """Use CPU by default; this avoids XGBoost silently downgrading CUDA runs."""
    return "cpu"


def save_tuning_artifacts() -> None:
    """Restore the compact results reported by the completed tuning notebook."""
    records = pd.DataFrame([
        {"Model": "XGBoost", "Baseline RMSE": .024869, "Tuned RMSE": .024223, "Baseline R2": .933818, "Tuned R2": .937211, "Search time (min)": 36.1},
        {"Model": "Extra Trees", "Baseline RMSE": .025023, "Tuned RMSE": .024518, "Baseline R2": .932995, "Tuned R2": .935672, "Search time (min)": 31.7},
        {"Model": "Random Forest", "Baseline RMSE": .025171, "Tuned RMSE": .024456, "Baseline R2": .932201, "Tuned R2": .936000, "Search time (min)": 85.3},
    ])
    records["RMSE improvement"] = records["Baseline RMSE"] - records["Tuned RMSE"]
    records["R2 improvement"] = records["Tuned R2"] - records["Baseline R2"]
    records.to_csv(METRICS_DIR / "tuning_results.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5)); x, width = np.arange(len(records)), .36
    for axis, baseline, tuned, title, ylabel in [
        (axes[0], "Baseline RMSE", "Tuned RMSE", "Tuning improvement", "RMSE (lower is better)"),
        (axes[1], "Baseline R2", "Tuned R2", "Tuning improvement", "R2 (higher is better)"),
    ]:
        axis.bar(x - width / 2, records[baseline], width, label="Baseline")
        axis.bar(x + width / 2, records[tuned], width, label="Tuned")
        axis.set(xticks=x, xticklabels=records["Model"], title=title, ylabel=ylabel)
        axis.legend(); axis.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "tuning_comparison.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")
    for directory in (FIGURE_DIR, METRICS_DIR, MODELS_DIR): directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    missing = sorted(set([*FEATURE_COLUMNS, TARGET_COLUMN]) - set(data.columns))
    if missing: raise ValueError(f"Processed dataset is missing columns: {missing}")
    X, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=.10, random_state=SEED, stratify=stratification_bins(y))
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 / 9, random_state=SEED, stratify=stratification_bins(y_train_val))
    device = choose_device()
    model = xgb.XGBRegressor(**TUNED_PARAMS, objective="reg:squarederror", tree_method="hist", device=device, random_state=SEED, n_jobs=1 if device == "cuda" else -1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    predictions = {"train": model.predict(X_train), "validation": model.predict(X_val), "test": model.predict(X_test)}
    metrics = pd.DataFrame([
        {"split": "train", **metrics_for(y_train.to_numpy(), predictions["train"])},
        {"split": "validation", **metrics_for(y_val.to_numpy(), predictions["validation"])},
        {"split": "test", **metrics_for(y_test.to_numpy(), predictions["test"])},
    ])
    metrics.to_csv(METRICS_DIR / "final_model_metrics.csv", index=False)
    residuals = pd.DataFrame({"objID": data.loc[X_test.index, "objID"].to_numpy(), "specz": y_test.to_numpy(), "predicted_specz": predictions["test"], "residual": predictions["test"] - y_test.to_numpy(), "normalized_residual": (predictions["test"] - y_test.to_numpy()) / (1 + y_test.to_numpy()), "modelMag_r": X_test["modelMag_r"].to_numpy()})
    residuals.to_csv(METRICS_DIR / "test_residuals.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7)); prediction = predictions["test"]
    limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
    axes[0].scatter(y_test, prediction, s=5, alpha=.2); axes[0].plot(limits, limits, "r--", lw=1); axes[0].set(xlabel="Spectroscopic redshift", ylabel="Predicted redshift", title="Predicted vs actual")
    axes[1].scatter(y_test, residuals["residual"], s=5, alpha=.2); axes[1].axhline(0, color="red", ls="--", lw=1); axes[1].set(xlabel="Spectroscopic redshift", ylabel="Prediction residual", title="Residuals by redshift")
    axes[2].scatter(residuals["modelMag_r"], residuals["residual"], s=5, alpha=.2); axes[2].axhline(0, color="red", ls="--", lw=1); axes[2].set(xlabel="r-band magnitude", ylabel="Prediction residual", title="Residuals by brightness")
    fig.suptitle(f"Tuned XGBoost diagnostics - test RMSE={metrics.iloc[2]['RMSE']:.5f}"); fig.tight_layout(); fig.savefig(FIGURE_DIR / "final_model_diagnostics.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    model_path = MODELS_DIR / "xgboost_photometric_redshift.json"; model.save_model(model_path)
    metadata = {"artifact_version": ARTIFACT_VERSION, "artifact_kind": "evaluation", "model_type": "xgboost.XGBRegressor", "xgboost_version": xgb.__version__, "device": device, "random_seed": SEED, "feature_columns": FEATURE_COLUMNS, "target_column": TARGET_COLUMN, "split": "stratified 80/10/10 train/validation/test", "parameters": TUNED_PARAMS, "test_metrics": metrics.iloc[2].drop("split").to_dict()}
    (MODELS_DIR / "xgboost_photometric_redshift_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_tuning_artifacts()
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}")); print(f"Saved model: {model_path}")


if __name__ == "__main__": main()
