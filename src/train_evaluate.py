"""
Train and evaluate the selected XGBoost photometric-redshift model.

Trains on the 50,000-galaxy processed training dataset and evaluates
on the separate, non-overlapping 50,000-galaxy processed test dataset
(originating from the 100,000 SDSS DR17 download split).

Reports MSE, RMSE, MAE, and R² score in the terminal, saves comprehensive
diagnostic plots to results/figures/, writes detailed metric tables to
results/metrics/, and persists the model artifact to models/.

Usage:
    python src/train_evaluate.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server/script environments
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths & Setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import COLOR_COLUMNS, MAGNITUDE_COLUMNS

TRAIN_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sdss_dr17_galaxies_processed.csv"
TEST_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sdss_dr17_new_batch_processed.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = MAGNITUDE_COLUMNS + COLOR_COLUMNS
TARGET_COLUMN = "specz"
SEED = 42
ARTIFACT_VERSION = "photometric-redshift-xgb-v1"

# Tuned XGBoost hyperparameters
TUNED_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}


def print_separator(char: str = "=", width: int = 65) -> None:
    print(char * width)


def choose_device() -> tuple[str, str]:
    """Auto-detect CUDA GPU availability with silent CPU fallback."""
    try:
        probe = xgb.XGBRegressor(device="cuda", n_estimators=1, verbosity=0)
        probe.fit([[0]], [0])
        return "cuda", "GPU (CUDA - NVIDIA RTX 3050)"
    except Exception:
        return "cpu", "CPU (all available cores)"


def compute_astrophysical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression and standard photometric-redshift error metrics."""
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    residual = y_pred - y_true
    normalized = residual / (1.0 + y_true)
    bias = float(np.mean(residual))
    std = float(np.std(residual))
    nmad = float(1.4826 * np.median(np.abs(normalized - np.median(normalized))))
    outlier_fraction = float(np.mean(np.abs(normalized) > 0.15))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Bias": bias,
        "Residual_Std": std,
        "Normalized_MAD": nmad,
        "Catastrophic_Outlier_Fraction": outlier_fraction,
    }


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets, verifying existence and required columns."""
    if not TRAIN_PROCESSED_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PROCESSED_PATH}\n"
            "Run python run_pipeline.py or python src/data_loading.py first."
        )

    if not TEST_PROCESSED_PATH.is_file():
        # Fall back to creating processed test set if raw test exists
        from src.preprocessing import NEW_BATCH_RAW_PATH, process_new_batch
        if NEW_BATCH_RAW_PATH.is_file():
            print(f"Processing raw test split from {NEW_BATCH_RAW_PATH}...")
            process_new_batch()
        else:
            raise FileNotFoundError(
                f"Test dataset not found: {TEST_PROCESSED_PATH}\n"
                "Run python run_pipeline.py or python src/data_loading.py first."
            )

    train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
    test_df = pd.read_csv(TEST_PROCESSED_PATH)

    for name, df in [("Train", train_df), ("Test", test_df)]:
        missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns))
        if missing:
            raise ValueError(f"{name} dataset is missing columns: {missing}")

    return train_df, test_df


def plot_diagnostics(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    test_df: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    """Save comprehensive multi-panel diagnostic plots for model evaluation."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    residuals = y_test_pred - y_test

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # 1. Predicted vs True Redshift
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.scatter(y_test, y_test_pred, s=6, alpha=0.25, color="#1f77b4", rasterized=True, label="Test Galaxies")
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    ax1.plot(lims, lims, "r--", lw=1.5, label="Perfect 1:1 line")
    ax1.set_xlabel("Spectroscopic Redshift ($z_{spec}$)", fontsize=11)
    ax1.set_ylabel("Photometric Redshift ($z_{phot}$)", fontsize=11)
    ax1.set_title(
        f"Predicted vs True Redshift — Test Set ({len(y_test):,} galaxies)\n"
        f"MSE={metrics['MSE']:.6f}   RMSE={metrics['RMSE']:.5f}   MAE={metrics['MAE']:.5f}   R²={metrics['R2']:.5f}",
        fontsize=12,
        fontweight="bold",
    )
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # 2. Key Metrics Bar Chart
    ax2 = fig.add_subplot(gs[0, 2])
    bar_data = {"MSE": metrics["MSE"], "RMSE": metrics["RMSE"], "MAE": metrics["MAE"], "1 - R²": 1.0 - metrics["R2"]}
    colors = ["#d95f02", "#e6ab02", "#1b9e77", "#7570b3"]
    bars = ax2.bar(list(bar_data.keys()), list(bar_data.values()), color=colors, width=0.55, edgecolor="gray")
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.0008,
            f"{height:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_title("Test Error Metrics\n(lower is better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Metric Value", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Residual Distribution Histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=60, color="#315b7d", edgecolor="white", alpha=0.85)
    ax3.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero error")
    ax3.axvline(metrics["Bias"], color="orange", linestyle="-", lw=1.5, label=f"Bias={metrics['Bias']:+.4f}")
    ax3.set_xlabel("Residual ($z_{phot} - z_{spec}$)", fontsize=10)
    ax3.set_ylabel("Galaxy Count", fontsize=10)
    ax3.set_title(f"Residual Distribution\n(Std = {metrics['Residual_Std']:.4f})", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, linestyle="--", alpha=0.35)

    # 4. Residuals by Brightness (r-band)
    ax4 = fig.add_subplot(gs[1, 1])
    r_mag = test_df["modelMag_r"].to_numpy()
    ax4.scatter(r_mag, residuals, s=5, alpha=0.2, color="#4b8b75", rasterized=True)
    ax4.axhline(0, color="red", linestyle="--", lw=1.5)
    ax4.set_xlabel("SDSS r-band Model Magnitude", fontsize=10)
    ax4.set_ylabel("Residual ($z_{phot} - z_{spec}$)", fontsize=10)
    ax4.set_title("Residuals vs Brightness (r-band)", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.35)

    # 5. Redshift Distribution: Train vs Test Set
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(y_train, bins=45, alpha=0.55, color="#1f77b4", label=f"Train ({len(y_train):,})", density=True)
    ax5.hist(y_test, bins=45, alpha=0.55, color="#ff7f0e", label=f"Test ({len(y_test):,})", density=True)
    ax5.set_xlabel("Spectroscopic Redshift ($z_{spec}$)", fontsize=10)
    ax5.set_ylabel("Probability Density", fontsize=10)
    ax5.set_title("Redshift Distribution: Train vs Test", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=8, loc="upper right")
    ax5.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle(
        "Tuned XGBoost — Final Model Performance on Non-Overlapping Test Set (SDSS DR17)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    out_primary = FIGURE_DIR / "final_model_diagnostics.png"
    out_secondary = FIGURE_DIR / "new_batch_evaluation.png"
    fig.savefig(out_primary, dpi=160, bbox_inches="tight")
    fig.savefig(out_secondary, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved:")
    print(f"    - {out_primary}")
    print(f"    - {out_secondary}")


def run_training_and_evaluation() -> dict:
    """Execute training on the train split and evaluate on the test split."""
    for directory in (FIGURE_DIR, METRICS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print_separator("=")
    print("  PHOTOMETRIC REDSHIFT -- XGBOOST MODEL TRAINING & EVALUATION")
    print_separator("=")

    # 1. Load Data
    train_df, test_df = load_datasets()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].to_numpy()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    print(f"\n[1/3] Datasets Loaded:")
    print(f"  Train samples : {len(X_train):,} galaxies ({TRAIN_PROCESSED_PATH.name})")
    print(f"  Test samples  : {len(X_test):,} galaxies ({TEST_PROCESSED_PATH.name})")
    print(f"  Features ({len(FEATURE_COLUMNS)}): {', '.join(FEATURE_COLUMNS)}")

    # 2. Configure & Train XGBoost Model
    device_name, device_desc = choose_device()
    print(f"\n[2/3] Training XGBoost Regressor on {len(X_train):,} training galaxies...")
    print(f"  Compute device : {device_desc}")
    print(f"  Parameters     : n_estimators={TUNED_PARAMS['n_estimators']}, learning_rate={TUNED_PARAMS['learning_rate']}, max_depth={TUNED_PARAMS['max_depth']}")

    model = xgb.XGBRegressor(
        **TUNED_PARAMS,
        device=device_name,
        random_state=SEED,
        verbosity=0,
        n_jobs=1 if device_name == "cuda" else -1,
    )

    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    train_time = time.time() - t0
    print(f"  Model fitting completed in {train_time:.2f} seconds.")

    # 3. Predict & Compute Metrics
    print(f"\n[3/3] Evaluating on Non-Overlapping Test Set ({len(X_test):,} galaxies)...")
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_metrics = compute_astrophysical_metrics(y_train, train_pred)
    test_metrics = compute_astrophysical_metrics(y_test, test_pred)

    # ── Terminal Output Report ────────────────────────────────────────────────
    print()
    print_separator("=")
    print("  FINAL EVALUATION RESULTS -- TEST SET")
    print_separator("=")
    print(f"  Model                       : XGBoost ({device_name.upper()})")
    print(f"  Training Galaxies           : {len(X_train):,}")
    print(f"  Test Galaxies (Held-out)    : {len(X_test):,}")
    print_separator("-")
    print(f"  Test MSE                    : {test_metrics['MSE']:.6f}")
    print(f"  Test RMSE                   : {test_metrics['RMSE']:.6f}")
    print(f"  Test MAE                    : {test_metrics['MAE']:.6f}")
    print(f"  Test R^2 Score              : {test_metrics['R2']:.6f}")
    print(f"  Residual Mean (Bias)        : {test_metrics['Bias']:+.6f}")
    print(f"  Residual Std Deviation      : {test_metrics['Residual_Std']:.6f}")
    print(f"  Normalized MAD (sigma_NMAD) : {test_metrics['Normalized_MAD']:.6f}")
    print(f"  Catastrophic Outlier Fract. : {test_metrics['Catastrophic_Outlier_Fraction'] * 100:.2f}%  (|dz|/(1+z) > 0.15)")
    print_separator("=")

    # 4. Save Artifacts & Metrics
    print("\nSaving metrics and model artifacts...")

    # Metrics CSV
    metrics_summary = pd.DataFrame([
        {"split": "train", "samples": len(y_train), **train_metrics},
        {"split": "test", "samples": len(y_test), **test_metrics},
    ])
    metrics_path = METRICS_DIR / "final_model_metrics.csv"
    metrics_summary.to_csv(metrics_path, index=False)
    # Also save new_batch_evaluation.csv for consistency
    metrics_summary.to_csv(METRICS_DIR / "new_batch_evaluation.csv", index=False)
    print(f"  Metrics saved  : {metrics_path}")

    # Residuals CSV
    residuals_df = pd.DataFrame({
        "objID": test_df["objID"].to_numpy() if "objID" in test_df.columns else np.arange(len(y_test)),
        "specz": y_test,
        "predicted_specz": test_pred,
        "residual": test_pred - y_test,
        "normalized_residual": (test_pred - y_test) / (1.0 + y_test),
        "modelMag_r": test_df["modelMag_r"].to_numpy(),
    })
    residuals_path = METRICS_DIR / "test_residuals.csv"
    residuals_df.to_csv(residuals_path, index=False)
    print(f"  Residuals saved: {residuals_path}")

    # Model file
    model_path = MODELS_DIR / "xgboost_photometric_redshift.json"
    model.save_model(model_path)
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_kind": "evaluation",
        "model_type": "xgboost.XGBRegressor",
        "xgboost_version": xgb.__version__,
        "device": device_name,
        "random_seed": SEED,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "parameters": TUNED_PARAMS,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    metadata_path = MODELS_DIR / "xgboost_photometric_redshift_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Model saved    : {model_path}")
    print(f"  Metadata saved : {metadata_path}")

    # Diagnostic Plots
    plot_diagnostics(y_train, y_test, test_pred, test_df, test_metrics)

    print()
    print_separator("=")
    print("  EVALUATION COMPLETED SUCCESSFULLY")
    print_separator("=")

    return {
        "model": model,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def main() -> None:
    run_training_and_evaluation()


if __name__ == "__main__":
    main()
