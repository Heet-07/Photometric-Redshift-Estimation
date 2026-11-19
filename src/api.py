"""Local photometric-redshift dashboard and prediction API."""

from __future__ import annotations

from contextlib import asynccontextmanager
import csv
from io import StringIO
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from src.dashboard_charts import DIAGNOSTICS, render_diagnostic
    from src.predict import ARTIFACTS, load_artifacts, predict_dataframe
except ModuleNotFoundError:  # direct invocation from the project root
    from dashboard_charts import DIAGNOSTICS, render_diagnostic
    from predict import ARTIFACTS, load_artifacts, predict_dataframe

_model = None
ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"
RESULTS = ROOT / "results"
REPORT_FIGURES = {
    "final_model_diagnostics": ("Full XGBoost diagnostics", "Includes the training and test redshift distributions."),
    "model_comparison": ("Baseline model comparison", "Error scores and training time across the baseline regressors."),
    "tuning_comparison": ("Hyperparameter tuning", "Baseline and tuned scores on the tuning split."),
    "best_model_pred_vs_actual": ("Best baseline predictions", "Predicted and measured redshifts for the best baseline model."),
    "feature_ablation": ("Feature ablation", "Performance comparison for magnitudes, colours, and their combination."),
    "slice_metrics": ("Error by data slice", "Error across redshift and brightness groups on the analysis split."),
    "distributions": ("Photometry distributions", "Redshift and magnitude distributions in the exploratory dataset."),
    "colors_vs_redshift": ("Galaxy colour vs. redshift", "Four adjacent-band colour indices against spectroscopic redshift."),
    "correlation_matrix": ("Photometry correlations", "Pairwise correlations among magnitudes, colours, and redshift."),
}
MAX_BATCH_ROWS = 5000
MAX_CSV_BYTES = 2 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """A batch of rows containing SDSS five-band magnitudes."""

    rows: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH_ROWS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model
    try:
        _model, _ = load_artifacts("deployment")
    except (OSError, ValueError) as exc:
        LOGGER.warning("Deployment model unavailable: %s", exc)
    try:
        yield
    finally:
        _model = None


app = FastAPI(
    title="SDSS Photometric Redshift API",
    description="Predict galaxy redshifts from SDSS u, g, r, i, z model magnitudes.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/dashboard-assets", StaticFiles(directory=DASHBOARD / "assets"), name="dashboard-assets")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD / "index.html")


@app.get("/dashboard-data")
def dashboard_data() -> dict[str, Any]:
    """Summarize saved held-out evaluation results for the dashboard."""
    _, metadata_path = ARTIFACTS["evaluation"]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        metadata = {}

    distribution = []
    residual_path = RESULTS / "metrics" / "test_residuals.csv"
    try:
        residuals = pd.read_csv(residual_path, usecols=["specz"], encoding="utf-8")
        redshifts = pd.to_numeric(residuals["specz"], errors="coerce").dropna()
    except (FileNotFoundError, ValueError):
        redshifts = pd.Series(dtype=float)
    if not redshifts.empty:
        counts, edges = np.histogram(redshifts, bins=20, range=(0, 0.5))
        distribution = [
            {"lower": round(float(lower), 3), "upper": round(float(upper), 3), "count": int(count)}
            for lower, upper, count in zip(edges[:-1], edges[1:], counts)
        ]

    return {
        "model_ready": _model is not None,
        "training_rows": metadata.get("train_samples"),
        "test_rows": metadata.get("test_samples"),
        "test_metrics": metadata.get("test_metrics", {}),
        "redshift_distribution": distribution,
        "diagnostic_charts": [
            {"name": name, "title": title, "description": description}
            for name, (title, description, _, _) in DIAGNOSTICS.items()
            if residual_path.is_file()
        ],
        "report_figures": [
            {"name": name, "title": title, "description": description}
            for name, (title, description) in REPORT_FIGURES.items()
            if (RESULTS / "figures" / f"{name}.png").is_file()
        ],
    }


@app.get("/dashboard-charts/{name}", include_in_schema=False)
def dashboard_chart(name: str) -> Response:
    """Serve known evaluation charts and saved project reports."""
    if name in DIAGNOSTICS:
        png = render_diagnostic(name, RESULTS / "metrics" / "test_residuals.csv")
        if png is not None:
            return Response(png, media_type="image/png")
    elif name in REPORT_FIGURES:
        path = RESULTS / "figures" / f"{name}.png"
        if path.is_file():
            return FileResponse(path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Chart unavailable")


@app.get("/health")
def health() -> dict[str, str]:
    """Report whether the deployment model has loaded."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Deployment model is not loaded")
    return {"status": "ok", "model": "deployment"}


def score_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if _model is None:
        raise HTTPException(status_code=503, detail="Deployment model is not loaded")

    try:
        return predict_dataframe(rows, _model)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    """Predict a batch, retaining invalid inputs with an explanatory status."""
    output = score_rows(pd.DataFrame(request.rows))

    records: list[dict[str, Any]] = []
    for row in output.to_dict(orient="records"):
        clean_row: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
                clean_row[key] = None
            elif isinstance(value, np.generic):
                clean_row[key] = value.item()
            else:
                clean_row[key] = value
        records.append(clean_row)

    return {
        "model": "deployment",
        "rows": len(records),
        "predictions": records,
    }


def score_csv(content: bytearray) -> Response:
    try:
        reader = csv.reader(StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        headers = next(reader, [])
        if not headers or any(not name for name in headers) or len(set(headers)) != len(headers):
            raise ValueError("The CSV needs a header with unique, non-empty column names.")

        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise ValueError(f"CSV line {reader.line_num} has a different number of fields than the header.")
            if len(rows) == MAX_BATCH_ROWS:
                raise ValueError(f"The CSV can contain up to {MAX_BATCH_ROWS:,} rows.")
            rows.append(row)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="Save the CSV with UTF-8 encoding and try again.") from exc
    except (csv.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    output = score_rows(pd.DataFrame(rows, columns=headers))
    return Response(
        output.to_csv(index=False),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="redshift-predictions.csv"',
            "X-Predicted-Rows": str(int(output["prediction_status"].eq("predicted").sum())),
            "X-Input-Rows": str(len(output)),
        },
    )


@app.post("/predict-csv")
async def predict_csv(request: Request) -> Response:
    """Score a UTF-8 CSV sent as the raw request body."""
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_CSV_BYTES:
            raise HTTPException(status_code=413, detail="The CSV must be 2 MiB or smaller.")
        content.extend(chunk)
    return await run_in_threadpool(score_csv, content)
