# Photometric Redshift Estimation of Galaxies Using SDSS Data

A machine-learning project that uses photometric observations from the
**Sloan Digital Sky Survey (SDSS DR17)** to estimate galaxy redshift.

For a step-by-step explanation of the pipeline, files, outputs, and terms,
read [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## The Idea

Galaxies emit light across different wavelengths. SDSS measures each galaxy's
brightness in 5 optical filters (`u`, `g`, `r`, `i`, `z`). The differences
between these measurements ("colors") change systematically with a galaxy's
redshift. This project trains ML models to learn that relationship.

```
Galaxy photometry (u, g, r, i, z)  →  ML Model  →  Predicted redshift
```

## Project Structure

```
├── data/
│   ├── raw/                 # Original SDSS download (not modified)
│   └── processed/           # Cleaned data ready for modeling
├── notebooks/               # Jupyter notebooks for exploration & modeling
├── dashboard/               # HTML, CSS, JavaScript, and an example CSV
├── models/                  # Generated XGBoost models and metadata
├── src/
│   ├── data_loading.py       # Download and validate SDSS splits
│   ├── preprocessing.py      # Clean data and add color features
│   ├── features.py           # Shared photometry validation and color calculations
│   ├── eda.py                # Exploratory plots and summaries
│   ├── train_evaluate.py     # Fit XGBoost and score the held-out test split
│   ├── model_analysis.py     # Compare feature sets and inspect data slices
│   ├── refit_deployment.py   # Save the prediction model
│   ├── predict.py            # Batch prediction CLI
│   ├── api.py                # FastAPI dashboard and prediction endpoints
│   └── dashboard_charts.py   # Render dark charts from held-out predictions
├── results/
│   ├── figures/             # Saved plots
│   └── metrics/             # Saved metric tables
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Run the dashboard

From the project root, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

On a fresh checkout, create the local data, evaluation, and deployment artifacts:

```powershell
python run_pipeline.py
python src/refit_deployment.py
```

If `models/` and `results/` already contain the generated outputs, start here:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open [Redshift Observatory](http://127.0.0.1:8000/). The dark dashboard has three main workflows:

- **Single galaxy** accepts the five SDSS model magnitudes. Select **Load example** and then **Estimate redshift** for a quick demo.
- **Batch CSV** accepts a UTF-8 CSV of up to 5,000 data rows and 2 MiB. Select **Download example CSV** to get a template. **Predict and download** returns the original input fields and prediction results, including reasons for invalid rows.
- **Model evaluation** shows the held-out scores, predicted versus measured redshift, the residual distribution, residuals by brightness, MAE and RMSE by redshift, and the measured redshift histogram.

Open **Training and data exploration → View saved project figures** for the
original figures available under `results/figures/`. Select a chart image to
open it full size. The pipeline generates the EDA and final evaluation figures.
Run `python src/model_analysis.py` for the feature-ablation and slice figures.
The baseline and tuning notebooks produce their own comparison figures.

If predictions report that the deployment model is unavailable, run the refit
and restart Uvicorn. If the evaluation charts are empty, check that the
evaluation metadata and `results/metrics/test_residuals.csv` exist.

The pipeline validates and reuses the two raw CSVs when both are present. If
either file is missing, it downloads 100,000 galaxies and writes both
50,000-row splits as a new pair. It then preprocesses both splits, runs EDA on
the training data, and evaluates XGBoost on the held-out test data.

## Data Source

- **Survey**: Sloan Digital Sky Survey
- **Release**: DR17
- **API**: SkyServer SQL Search (no account required)
- **Current split**: 50,000 training rows plus a separate 50,000-row held-out test batch
- **Sample**: 100,000 galaxies split into 50,000 training and 50,000 test rows with spectroscopic redshift in range 0.01-0.5

## Data and Reproducible Workflow

The raw CSVs are stored in `data/raw/`. Preprocessing reads those files and
writes cleaned CSVs and cleaning reports to `data/processed/`. The raw
training and test files use the same schema:

| Column | Meaning |
|---|---|
| `objID` | SDSS object identifier; excluded from model features |
| `modelMag_u` through `modelMag_z` | Five-band SDSS model magnitudes |
| `specz` | Spectroscopic redshift target |

Validate both existing raw CSVs with:

```bash
python src/data_loading.py --validate-existing
```

The downloader checks the schema and reports sentinel values such as `-9999`.
When both splits exist, `--force` explicitly replaces the pair with a new
download. Keep the saved raw pair to reproduce the recorded evaluation because
the SkyServer query uses `TOP` without `ORDER BY`.

To preprocess the training CSV by itself, run:

```bash
python src/preprocessing.py
```

This removes rows with invalid targets or unusable measurements in any band,
removes duplicate object IDs, and computes the adjacent colors `u-g`, `g-r`,
`r-i`, and `i-z`. Row counts for each filtering stage are saved in
`data/processed/cleaning_report.csv`.

Create the initial EDA outputs with:

```bash
python src/eda.py
```

This saves photometry and redshift distributions, colors versus spectroscopic
redshift, a correlation matrix, descriptive statistics, and correlation data
under `results/`.

## Model training and evaluation

Run the selected tuned XGBoost model with:

```bash
python src/train_evaluate.py
```

The script trains on the processed training CSV and evaluates on the separate
processed test CSV. It writes MSE, MAE, RMSE, R², bias, normalized
MAD, outlier rate, per-galaxy test residuals, and diagnostic figures under
`results/`. The evaluated model and metadata are saved under `models/` (which
is intentionally not committed). The saved run used **49,964 training
galaxies** and **50,000 held-out galaxies**.

| Held-out metric | Saved result |
|---|---:|
| RMSE | 0.026434 |
| MAE | 0.017679 |
| R² | 0.930428 |
| Catastrophic outlier fraction | 0.17% |

Additional analysis and deployment commands:

```bash
python src/model_analysis.py
python src/refit_deployment.py
python src/predict.py --input new_galaxies.csv --output predictions.csv
```

`model_analysis.py` compares magnitudes-only, colors-only, and combined
features, and reports errors in redshift and brightness bins. This exploratory
script uses an internal split of the training catalog and its own XGBoost
configuration. The deployment refit uses the selected settings and all cleaned
training rows. The dashboard scores come from the separate held-out evaluation
of the evaluation artifact.

`predict.py` accepts `modelMag_u`, `modelMag_g`, `modelMag_r`, `modelMag_i`,
and `modelMag_z`; it calculates colors internally. It defaults to the
deployment model and preserves all input columns. Each output row receives a
`predicted_specz`, `prediction_status`, and `invalid_reason` column. Rows with
missing, non-numeric, non-finite, or `-9999` magnitudes are retained as
`invalid` rather than being silently predicted.

## Local prediction API

The [interactive API docs](http://127.0.0.1:8000/docs) are available while
Uvicorn is running. The command above binds the service to localhost.

| Route | Purpose |
|---|---|
| `GET /health` | Returns 200 when the deployment model is loaded and 503 when it is unavailable. |
| `POST /predict` | Accepts JSON with a `rows` array of up to 5,000 groups of five SDSS magnitudes. |
| `POST /predict-csv` | Accepts a raw UTF-8 CSV body of up to 5,000 data rows and 2 MiB. Returns a CSV download. |
| `GET /dashboard-data` | Lists the saved held-out metrics, histogram bins, and available figures. |
| `GET /dashboard-charts/{name}` | Serves a dark diagnostic chart or one of the saved project figures. |

Both prediction routes include `predicted_specz`, `prediction_status`, and
`invalid_reason` in their output. Invalid measurements get an empty prediction
and a reason.
