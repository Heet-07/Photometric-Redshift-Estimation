# SDSS Photometric Redshift Project Guide

## What this project does

| Question | Answer |
|---|---|
| What are we trying to predict? | A galaxy's spectroscopic redshift (`specz`). |
| What information do we use? | Brightness measurements in five SDSS filters: `u`, `g`, `r`, `i`, and `z`. |
| Why predict redshift? | It tells us how much a galaxy's light has shifted to longer wavelengths while travelling to us. |
| What will the model eventually produce? | A photometric redshift prediction from brightness measurements only. |
| What are we doing now? | Reproducing and documenting the selected tuned model. |

## The full flow

```text
SDSS download
    ↓
Raw CSV: original measurements
    ↓
Validation: check columns, values, IDs, and redshift range
    ↓
Cleaning: remove unusable measurements
    ↓
Feature engineering: calculate colors
    ↓
EDA: make plots and summary tables
    ↓
Next: split data, train models, evaluate predictions
```

## Current project status

| Step | Status | Result |
|---|---|---|
| Download raw SDSS data | Done | 50,000 galaxies from SDSS DR17. |
| Validate raw data | Done | Expected columns, valid redshift values, and unique object IDs checked. |
| Clean the data | Done | 37 rows removed because at least one brightness measurement was unusable. |
| Add color features | Done | `u-g`, `g-r`, `r-i`, and `i-z` added. |
| Explore the data | Done | Distributions, color-redshift plots, and correlations saved. |
| Compare machine-learning models | Done | 15 regression models evaluated. |
| Tune leading models | Done | XGBoost, Extra Trees, and Random Forest tuned with 5-fold CV. |
| Save reproducible final model | Done | Script saves model, metadata, metrics, and diagnostics. |

## The dataset we use

| Item | Current value |
|---|---|
| Survey | Sloan Digital Sky Survey (SDSS) |
| Release | DR17 |
| Download method | SkyServer SQL query |
| Raw rows | 50,000 |
| Processed rows | 49,963 |
| Redshift selection | `0.01 <= specz <= 0.5` |
| Object selection | Spectroscopically classified galaxies with `zWarning = 0` |

### Raw-data columns

| Column | Plain meaning | Used later by the model? |
|---|---|---|
| `objID` | Unique number for an SDSS object | No; it only identifies a row. |
| `modelMag_u` | Brightness through the `u` filter | Yes. |
| `modelMag_g` | Brightness through the `g` filter | Yes. |
| `modelMag_r` | Brightness through the `r` filter | Yes. |
| `modelMag_i` | Brightness through the `i` filter | Yes. |
| `modelMag_z` | Brightness through the `z` filter | Yes. |
| `specz` | Redshift measured from a spectrum | Yes, as the answer the model learns to predict. |

## Why we clean the data

Some SDSS magnitudes are stored as `-9999`.

| Value type | Meaning | What we do |
|---|---|---|
| Normal number | A usable brightness measurement | Keep it. |
| Empty or non-finite value | The measurement cannot be used | Remove that row from the processed dataset. |
| `-9999` | SDSS placeholder for an unusable measurement | Remove that row from the processed dataset. |

We need all five brightness measurements because every processed row must be
able to calculate all four colors.

### Cleaning result

| Stage | Rows left | Rows removed at this stage |
|---|---:|---:|
| Raw SDSS file | 50,000 | 0 |
| Valid object ID and redshift | 50,000 | 0 |
| Usable values in all five filters | 49,963 | 37 |
| Unique object ID | 49,963 | 0 |

The original file is kept unchanged in `data/raw/`.

## What colors mean

A color is the difference between two brightness measurements.

| Feature | Calculation | Simple interpretation |
|---|---|---|
| `color_u_g` | `modelMag_u - modelMag_g` | Difference between `u` and `g` brightness. |
| `color_g_r` | `modelMag_g - modelMag_r` | Difference between `g` and `r` brightness. |
| `color_r_i` | `modelMag_r - modelMag_i` | Difference between `r` and `i` brightness. |
| `color_i_z` | `modelMag_i - modelMag_z` | Difference between `i` and `z` brightness. |

Galaxy colors change with redshift because the wavelengths we observe shift as
the galaxy is farther away. The plots help us see whether those changes are
useful for predicting redshift.

## Files and what each one does

| File | Purpose |
|---|---|
| `run_pipeline.py` | Main script. Runs the current pipeline in order and shows progress. |
| `src/data_loading.py` | Downloads SDSS data or validates the existing raw CSV. |
| `src/preprocessing.py` | Cleans the raw data and calculates colors. |
| `src/features.py` | Validates five-band photometry and calculates model colors. |
| `src/eda.py` | Creates plots and summary tables. |
| `src/predict.py` | Predicts from five-band CSV input using a selected model artifact. |
| `src/refit_deployment.py` | Fits the frozen configuration on all cleaned rows for deployment. |
| `data/raw/sdss_dr17_galaxies_raw.csv` | Original SDSS download. Do not edit it. |
| `data/processed/sdss_dr17_galaxies_processed.csv` | Cleaned data with color columns. |
| `data/processed/cleaning_report.csv` | Row counts after each cleaning rule. |
| `results/figures/` | Image files created during EDA. |
| `results/metrics/` | Numerical summary tables created during EDA. |

## How to run the current pipeline

Run this command from the project folder:

```bash
python run_pipeline.py
```

### What you will see

| Pipeline step | What it checks or creates |
|---|---|
| `1/3` Validate raw CSV | Checks the dataset shape, columns, IDs, redshift range, and unusable values. |
| `2/3` Clean and add colors | Creates the processed CSV and cleaning report. |
| `3/3` EDA | Creates figures and numerical summary tables. |

### Outputs after running

| Output | What to look for |
|---|---|
| `results/figures/distributions.png` | How redshift and each filter's brightness values are spread. |
| `results/figures/colors_vs_redshift.png` | How each color changes across known redshifts. |
| `results/figures/correlation_matrix.png` | Which measurements and colors change together. |
| `results/metrics/eda_descriptive_statistics.csv` | Minimum, average, median, and maximum values. |
| `results/metrics/eda_correlation_matrix.csv` | Numerical version of the correlation plot. |

### Batch prediction

The prediction interface requires only the five raw magnitude columns; colors
are calculated by `src/features.py`:

```bash
python src/predict.py --input new_galaxies.csv --output predictions.csv
```

Deployment is the default artifact. Use `--model evaluation` when you need the
held-out evaluation artifact instead. The input columns and row order are
preserved. Invalid rows receive `prediction_status=invalid` and an explanatory
`invalid_reason`; they are never silently passed to the model.

## What EDA means

EDA means **exploratory data analysis**.

| EDA output | Question it helps answer |
|---|---|
| Redshift distribution | Which redshift values are common in our sample? |
| Magnitude distributions | Are most galaxies bright, faint, or spread across both? |
| Color versus redshift plots | Do galaxy colors show patterns as redshift changes? |
| Correlation matrix | Which input values carry similar information? |

The color plots show the middle 99% of displayed values so the main pattern is
easy to see. No rows are removed for this display choice. The CSV files still
contain every processed row.

## Tools and libraries

| Tool | Why we use it |
|---|---|
| Python | Main programming language. |
| pandas | Reads CSV files and works with table-shaped data. |
| NumPy | Checks numeric values and supports calculations. |
| Matplotlib | Draws plots. |
| Seaborn | Makes statistical plots easier to read. |
| scikit-learn | Will be used for the machine-learning models next. |
| SDSS SkyServer | Provides the real astronomy data. |

## Important terms

| Term | Meaning |
|---|---|
| Galaxy | A large group of stars, gas, dust, and dark matter. |
| Magnitude | A logarithmic brightness scale used in astronomy. Smaller magnitudes mean brighter objects. |
| Filter or band | A range of light wavelengths measured by the telescope. |
| Spectrum | Light split into many wavelengths; it gives a more precise redshift measurement. |
| Spectroscopic redshift | Redshift measured from a spectrum; our trusted answer column. |
| Photometric redshift | Redshift estimated from brightness values in filters. |
| Feature | An input value given to a model, such as a magnitude or color. |
| Target | The value the model tries to predict: `specz`. |
| EDA | Looking at data with summaries and plots before modeling. |

## Astronomy theory used in this project

### Redshift

Redshift describes how much light from a distant object has moved toward longer
wavelengths before reaching us.

| Symbol | Meaning |
|---|---|
| `z` | Redshift; it has no unit. |
| `λ_rest` | Wavelength measured in the galaxy's own reference frame. |
| `λ_observed` | Wavelength measured by our telescope. |

The basic relationship is:

```text
z = (λ_observed - λ_rest) / λ_rest
```

| Redshift value | Simple meaning |
|---:|---|
| `z = 0` | No wavelength shift. |
| Positive `z` | Light is observed at longer wavelengths. |
| Larger positive `z` | Usually means the galaxy is farther away in this survey sample. |

Redshift alone is not a distance. Turning redshift into distance requires a
chosen model of the expanding universe and its parameters. This project does
not convert redshift to distance.

### Why distant galaxies look different in fixed filters

| Step | What happens |
|---|---|
| 1 | A galaxy emits light across many wavelengths. |
| 2 | As the universe expands, the galaxy's light is shifted toward longer wavelengths. |
| 3 | SDSS measures that shifted light through five fixed filters. |
| 4 | The amount of measured light in each filter changes with redshift. |
| 5 | A model can learn statistical links between those five measurements and known redshifts. |

This is why colors can contain redshift information.

### SDSS filters

| Filter | Relative part of visible light | Used here as |
|---|---|---|
| `u` | Ultraviolet end | `modelMag_u` |
| `g` | Blue-green | `modelMag_g` |
| `r` | Red | `modelMag_r` |
| `i` | Near infrared | `modelMag_i` |
| `z` | Further into near infrared | `modelMag_z` |

The filter letters are names of wavelength ranges. They are not redshift bins.

### Magnitudes and brightness

Astronomy uses magnitudes instead of ordinary brightness numbers.

| Magnitude change | Brightness meaning |
|---:|---|
| Smaller magnitude | Brighter object |
| Larger magnitude | Fainter object |
| Difference of 1 magnitude | About 2.5 times change in brightness |
| Difference of 5 magnitudes | Exactly 100 times change in brightness |

Magnitudes use a logarithmic scale. That is why we subtract magnitudes to make
colors instead of dividing them.

### Colors

| Example | Calculation | Interpretation |
|---|---|---|
| `u-g` | `u magnitude - g magnitude` | Compares the ultraviolet-side measurement with the blue-green measurement. |
| Positive color | First-band magnitude is larger | The object is fainter in the first band than the second. |
| Negative color | First-band magnitude is smaller | The object is brighter in the first band than the second. |

Color does not measure redshift directly. Different galaxy types, dust, and
measurement noise can produce similar colors at different redshifts.

### Spectroscopic and photometric redshift

| Method | Data needed | Strength | Limitation |
|---|---|---|---|
| Spectroscopic redshift | Detailed spectrum | More precise reference measurement | Requires more observing time. |
| Photometric redshift | A few broad-band brightness measurements | Faster for very large galaxy samples | Less precise and can confuse similar-looking galaxies. |

In this project:

```text
Known spectroscopic redshift = training answer
Photometric measurements and colors = model inputs
Predicted photometric redshift = model output
```

### What `modelMag` means

`modelMag` is an SDSS brightness estimate based on the better-fitting of two
simple galaxy light profiles. It is useful for consistent colors because the
same profile choice is applied across the bands. We treat it as the selected
photometric measurement for this first project stage.

### Why `zWarning = 0` matters

| Field | Meaning in our selection |
|---|---|
| `class = 'GALAXY'` | Keeps objects spectroscopically classified as galaxies. |
| `zWarning = 0` | Keeps spectra with no warning flag from the redshift pipeline. |
| `0.01 <= z <= 0.5` | Keeps the initial low-to-moderate-redshift study range. |

These choices make the training answer more reliable, but they also limit the
population represented by the project.

## Machine-learning theory used in this project

### Regression problem

| Part | In this project |
|---|---|
| Input features `X` | Five magnitudes, four colors, or a tested subset. |
| Target `y` | `specz`. |
| Model | A mathematical method that learns from examples. |
| Prediction `ŷ` | Estimated photometric redshift. |

The goal is:

```text
model(photometry) → predicted redshift
```

This is regression because the answer is a continuous number such as `0.137`.
It is not a classification task such as “low redshift” or “high redshift.”

### Training, validation, and test data

| Split | Purpose |
|---|---|
| Training set | Lets the model learn patterns. |
| Validation set | Helps choose model settings and feature combinations. |
| Test set | Gives the final unbiased performance estimate. |

The test set must remain unseen while choosing models. Otherwise the reported
score may be unrealistically good.

### Data leakage

Data leakage happens when the model gets information it would not have for a
new galaxy.

| Safe input | Unsafe input |
|---|---|
| `modelMag_u` through `modelMag_z` | `specz` itself |
| Colors calculated from those magnitudes | A quantity calculated using known redshift |
| Documented photometric quality information | Object ID used as a shortcut |

### Error measures planned for later

| Name | What it measures | Lower or higher is better? |
|---|---|---|
| MAE | Average absolute difference between prediction and true redshift | Lower |
| RMSE | Error measure that penalizes large mistakes more strongly | Lower |
| R² | How much of the target variation the model explains | Higher |

We will also examine the residual:

```text
residual = predicted redshift - spectroscopic redshift
```

Large residuals are useful to study because they show where the method is less
reliable.

## Limits of this project

| Limitation | Why it matters |
|---|---|
| Five broad filters contain less detail than a spectrum | Some redshift information is missing. |
| Similar colors can occur at different redshifts | A model can confuse galaxies. |
| Dust can change observed colors | Dust can imitate some redshift effects. |
| Faint objects have noisier measurements | Predictions may worsen for faint galaxies. |
| Training sample selection is restricted | Results may not apply to every SDSS galaxy. |
| Current range ends at `z = 0.5` | We cannot claim performance for higher-redshift galaxies. |

## What happens next

1. Make a reproducible training, validation, and test split.
2. Train simple reference models first: median predictor and linear regression.
3. Train tree-based models such as Random Forest and Gradient Boosting.
4. Compare the feature sets: magnitudes only, colors only, and both together.
5. Measure error with MAE, RMSE, and R².
6. Study prediction errors against redshift and brightness.

## Rules we will keep following

- Never use `specz` as an input feature.
- Never use `objID` as an input feature.
- Keep the raw SDSS CSV unchanged.
- Record every cleaning rule and removed-row count.
- Fit future scaling and model choices using training data only.
- Keep final test data separate until the final evaluation.

## Current model result and reproducible run

The recorded hyperparameter search selected XGBoost. Its original 10% holdout
result was RMSE 0.024223 and R2 0.937211. To reproduce the current evaluation
from the processed CSV, run:

```bash
python src/train_evaluate.py
```

It uses an 80/10/10 redshift-stratified split and writes the fitted model and
feature metadata to `models/`, numeric metrics and per-galaxy residuals to
`results/metrics/`, and diagnostic plots to `results/figures/`. It reports MAE,
RMSE, R2, bias, normalized MAD, and the fraction of normalized residuals above
0.15. Since the earlier notebook's holdout informed model selection, these new
outputs should be treated as reproducible diagnostics, not an untouched final
test claim.

Feature-ablation results are written by `python src/model_analysis.py` to
`results/metrics/feature_ablation.csv`, with slice-level results in
`results/metrics/slice_metrics.csv`. For deployment after evaluation is frozen,
run `python src/refit_deployment.py`; use `python src/predict.py --input
<csv> --output <csv>` to score new rows.
