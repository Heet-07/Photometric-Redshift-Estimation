"""Render dark diagnostic figures from saved held-out predictions."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


BACKGROUND = "#202124"
TEXT = "#e8e4db"
GRID = "#49494d"
ACCENT = "#e8a184"
SECONDARY = "#90c7c1"
DENSITY = LinearSegmentedColormap.from_list(
    "galaxy_density", ["#665979", "#a45480", "#e58d72", "#ffe1a8"],
)


def _prediction_agreement(fig, axis, rows):
    measured = rows["specz"]
    predicted = rows["predicted_specz"]
    lower = min(measured.min(), predicted.min())
    upper = max(measured.max(), predicted.max())
    density = axis.hexbin(measured, predicted, gridsize=45, mincnt=1, bins="log", cmap=DENSITY)
    axis.plot([lower, upper], [lower, upper], color=SECONDARY, linestyle="--", label="Perfect agreement")
    axis.set(xlabel="Spectroscopic redshift", ylabel="Predicted redshift", aspect="equal")
    axis.legend(loc="upper left", frameon=False, labelcolor=TEXT)
    _add_count_scale(fig, axis, density)


def _residual_distribution(fig, axis, rows):
    residuals = rows["residual"]
    axis.hist(residuals, bins=60, color=ACCENT)
    axis.axvline(0, color=TEXT, linestyle="--", label="Zero error")
    axis.axvline(residuals.mean(), color=SECONDARY, label=f"Mean {residuals.mean():+.4f}")
    axis.set(xlabel="Predicted − spectroscopic redshift", ylabel="Galaxies")
    axis.legend(frameon=False, labelcolor=TEXT)


def _residuals_by_brightness(fig, axis, rows):
    density = axis.hexbin(
        rows["modelMag_r"], rows["residual"], gridsize=45, mincnt=1, bins="log", cmap=DENSITY,
    )
    axis.axhline(0, color=SECONDARY, linestyle="--")
    axis.set(xlabel="SDSS r-band model magnitude", ylabel="Redshift residual")
    _add_count_scale(fig, axis, density)


def _error_by_redshift(fig, axis, rows):
    redshift = rows["specz"].to_numpy()
    residual = rows["residual"].to_numpy()
    edges = np.linspace(redshift.min(), redshift.max(), 11)
    counts, _ = np.histogram(redshift, bins=edges)
    absolute_error, _ = np.histogram(redshift, bins=edges, weights=np.abs(residual))
    squared_error, _ = np.histogram(redshift, bins=edges, weights=np.square(residual))
    nonempty = counts > 0
    centers = (edges[:-1] + edges[1:]) / 2
    mae = np.full(len(counts), np.nan)
    rmse = np.full(len(counts), np.nan)
    mae[nonempty] = absolute_error[nonempty] / counts[nonempty]
    rmse[nonempty] = np.sqrt(squared_error[nonempty] / counts[nonempty])
    axis.plot(centers, mae, color=ACCENT, marker="o", label="MAE")
    axis.plot(centers, rmse, color=SECONDARY, marker="s", label="RMSE")
    axis.set(xlabel="Spectroscopic redshift (bin midpoint)", ylabel="Redshift error", ylim=(0, None))
    axis.legend(frameon=False, labelcolor=TEXT)


def _add_count_scale(fig, axis, density):
    scale = fig.colorbar(density, ax=axis, pad=0.035)
    scale.set_label("Galaxies per bin (log scale)", color=TEXT)
    scale.ax.tick_params(colors=TEXT)
    scale.outline.set_edgecolor(GRID)


DIAGNOSTICS = {
    "prediction-agreement": (
        "Predicted vs. measured redshift",
        "Denser bins contain more galaxies. The dashed line marks perfect agreement.",
        ("specz", "predicted_specz"),
        _prediction_agreement,
    ),
    "residual-distribution": (
        "Residual distribution",
        "Each residual is the prediction minus the spectroscopic redshift.",
        ("residual",),
        _residual_distribution,
    ),
    "residuals-by-brightness": (
        "Residuals by brightness",
        "Higher r-band magnitudes correspond to fainter galaxies.",
        ("modelMag_r", "residual"),
        _residuals_by_brightness,
    ),
    "error-by-redshift": (
        "Error by redshift",
        "MAE and RMSE for ten equal-width intervals in measured redshift.",
        ("specz", "residual"),
        _error_by_redshift,
    ),
}


def render_diagnostic(name: str, residual_path: Path) -> bytes | None:
    """Return a PNG or None when its required evaluation columns are unavailable."""
    _, _, columns, draw = DIAGNOSTICS[name]
    try:
        rows = pd.read_csv(residual_path, usecols=lambda column: column in columns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None
    if not set(columns).issubset(rows.columns):
        return None
    rows = rows.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if rows.empty or (name == "error-by-redshift" and rows["specz"].nunique() < 2):
        return None

    figure = Figure(figsize=(6.4, 4), dpi=130, facecolor=BACKGROUND, layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.set_facecolor(BACKGROUND)
    axis.set_axisbelow(True)
    axis.grid(color=GRID, linewidth=0.6)
    axis.tick_params(colors=TEXT)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    draw(figure, axis, rows)

    output = BytesIO()
    figure.savefig(output, format="png")
    return output.getvalue()
