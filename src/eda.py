"""Create descriptive plots and tables for the cleaned SDSS sample."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sdss_dr17_galaxies_processed.csv"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
SUMMARY_DIR = PROJECT_ROOT / "results" / "metrics"
MAGNITUDE_COLUMNS = [f"modelMag_{band}" for band in "ugriz"]
COLOR_COLUMNS = ["color_u_g", "color_g_r", "color_r_i", "color_i_z"]

sns.set_theme(style="whitegrid", context="notebook")


def run_eda(data: pd.DataFrame) -> dict[str, Path]:
    """Save labeled EDA figures and summary tables and return their paths."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Log-count axes retain rare extreme measurements without flattening the core.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    axes[0].hist(data["specz"], bins=45, color="#315b7d", edgecolor="white", log=True)
    axes[0].set(title="Spectroscopic redshift", xlabel="$z_{spec}$", ylabel="Galaxy count")
    for axis, band in zip(axes[1:], "ugriz"):
        column = f"modelMag_{band}"
        axis.hist(data[column], bins=45, color="#4b8b75", edgecolor="white", log=True)
        axis.set(title=f"SDSS {band}-band model magnitude", xlabel=f"{band} (mag)", ylabel="Galaxy count")
    fig.suptitle("Redshift and photometric measurement distributions")
    fig.tight_layout()
    outputs["distributions"] = FIGURE_DIR / "distributions.png"
    fig.savefig(outputs["distributions"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Color versus redshift reveals broad photometric trends without implying causality.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, column in zip(axes.ravel(), COLOR_COLUMNS):
        axis.scatter(data["specz"], data[column], s=5, alpha=0.18, color="#315b7d", rasterized=True)
        axis.set(title=f"{column.replace('color_', '').replace('_', '−')} color", xlabel="$z_{spec}$", ylabel="Color (mag)")
        lower, upper = data[column].quantile([0.005, 0.995])
        axis.set_ylim(lower, upper)
        axis.text(
            0.02, 0.98,
            "display: 0.5th–99.5th percentiles",
            transform=axis.transAxes, va="top", fontsize=8,
        )
    fig.suptitle("Adjacent-band colors versus spectroscopic redshift")
    fig.tight_layout()
    outputs["colors_vs_redshift"] = FIGURE_DIR / "colors_vs_redshift.png"
    fig.savefig(outputs["colors_vs_redshift"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Correlation matrix excludes objID because it is an identifier, not a measurement.
    correlation_columns = [*MAGNITUDE_COLUMNS, *COLOR_COLUMNS, "specz"]
    correlation = data[correlation_columns].corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(correlation, cmap="vlag", center=0, annot=True, fmt=".2f", square=True, ax=axis)
    axis.set_title("Correlation among photometry, colors, and spectroscopic redshift")
    fig.tight_layout()
    outputs["correlation"] = FIGURE_DIR / "correlation_matrix.png"
    fig.savefig(outputs["correlation"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    summary = data[[*MAGNITUDE_COLUMNS, *COLOR_COLUMNS, "specz"]].describe().T
    summary.to_csv(SUMMARY_DIR / "eda_descriptive_statistics.csv")
    correlation.to_csv(SUMMARY_DIR / "eda_correlation_matrix.csv")
    outputs["descriptive_statistics"] = SUMMARY_DIR / "eda_descriptive_statistics.csv"
    outputs["correlation_table"] = SUMMARY_DIR / "eda_correlation_matrix.csv"
    return outputs


def main() -> None:
    if not PROCESSED_PATH.is_file():
        raise FileNotFoundError(
            f"Processed CSV not found: {PROCESSED_PATH}. Run `python src/preprocessing.py` first."
        )
    data = pd.read_csv(PROCESSED_PATH)
    outputs = run_eda(data)
    print(f"EDA completed for {len(data):,} galaxies.")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
