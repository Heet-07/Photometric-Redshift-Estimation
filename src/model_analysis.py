"""Run feature-ablation and slice-level diagnostics for the selected model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "sdss_dr17_galaxies_processed.csv"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
SEED = 42
MAGNITUDES = [f"modelMag_{band}" for band in "ugriz"]
COLORS = ["color_u_g", "color_g_r", "color_r_i", "color_i_z"]
ALL_FEATURES = MAGNITUDES + COLORS
PARAMS = {"colsample_bytree": .8, "learning_rate": .05, "max_depth": 8, "min_child_weight": 3, "n_estimators": 200, "subsample": .8}


def bins(values: pd.Series, count: int = 10) -> pd.Series:
    return pd.qcut(values, q=count, duplicates="drop")


def score(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** .5,
        "R2": r2_score(y_true, prediction),
        "Count": len(y_true),
    }


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    train_val, test = train_test_split(data, test_size=.10, random_state=SEED, stratify=bins(data["specz"], 20))
    train, _ = train_test_split(train_val, test_size=1 / 9, random_state=SEED, stratify=bins(train_val["specz"], 20))

    feature_sets = {"Magnitudes only": MAGNITUDES, "Colors only": COLORS, "Magnitudes + colors": ALL_FEATURES}
    ablation_rows = []
    predictions = None
    for name, features in feature_sets.items():
        model = xgb.XGBRegressor(**PARAMS, objective="reg:squarederror", tree_method="hist", device="cpu", random_state=SEED, n_jobs=-1)
        model.fit(train[features], train["specz"], verbose=False)
        prediction = model.predict(test[features])
        ablation_rows.append({"Feature set": name, **score(test["specz"], prediction)})
        if name == "Magnitudes + colors": predictions = prediction
    ablation = pd.DataFrame(ablation_rows).sort_values("RMSE")
    ablation.to_csv(METRICS / "feature_ablation.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(ablation["Feature set"], ablation["RMSE"]); axes[0].set_ylabel("RMSE (lower is better)"); axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(ablation["Feature set"], ablation["R2"]); axes[1].set_ylabel("R2 (higher is better)"); axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Feature ablation on the fixed evaluation split"); fig.tight_layout(); fig.savefig(FIGURES / "feature_ablation.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    evaluated = test[["specz", "modelMag_r"]].copy()
    evaluated["prediction"] = predictions
    evaluated["residual"] = evaluated["prediction"] - evaluated["specz"]
    evaluated["normalized_abs_error"] = evaluated["residual"].abs() / (1 + evaluated["specz"])
    rows = []
    for variable, label in (("specz", "redshift"), ("modelMag_r", "r-band magnitude")):
        evaluated["bin"] = bins(evaluated[variable], 10)
        for interval, group in evaluated.groupby("bin", observed=True):
            rows.append({"slice": label, "bin": str(interval), "lower": interval.left, "upper": interval.right, "count": len(group), "MAE": group["residual"].abs().mean(), "RMSE": np.sqrt(np.mean(group["residual"] ** 2)), "mean_normalized_abs_error": group["normalized_abs_error"].mean()})
    slices = pd.DataFrame(rows)
    slices.to_csv(METRICS / "slice_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, label in zip(axes, ["redshift", "r-band magnitude"]):
        subset = slices[slices["slice"] == label]
        axis.plot(subset["lower"], subset["RMSE"], marker="o", label="RMSE")
        axis.plot(subset["lower"], subset["mean_normalized_abs_error"], marker="o", label="Normalized absolute error")
        axis.set(title=f"Error by {label}", xlabel=f"{label} bin lower edge"); axis.legend(); axis.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGURES / "slice_metrics.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(ablation.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"Saved {METRICS / 'feature_ablation.csv'} and {METRICS / 'slice_metrics.csv'}")


if __name__ == "__main__":
    main()
