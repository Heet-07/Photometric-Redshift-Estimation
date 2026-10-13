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
├── src/
│   ├── data_loading.py      # Download data from SDSS SkyServer
│   ├── preprocessing.py     # Clean and filter data
│   └── eda.py               # Exploratory plots and summaries
├── results/
│   ├── figures/             # Saved plots
│   └── metrics/             # Saved metric tables
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ~50,000 galaxies from SDSS DR17
python src/data_loading.py

# 3. Clean the data and add adjacent-band colors
python src/preprocessing.py

# 4. Create EDA plots and summary tables
python src/eda.py
```

To run validation, preprocessing, color construction, and EDA together with
progress output, use the primary pipeline script:

```bash
python run_pipeline.py
```

The pipeline reads the existing raw CSV and does not download or overwrite it.

## Data Source

- **Survey**: Sloan Digital Sky Survey
- **Release**: DR17
- **API**: SkyServer SQL Search (no account required)
- **Sample**: ~50,000 galaxies with spectroscopic redshift in range 0.01–0.5

## Data and Reproducible Workflow

The raw CSV is stored in `data/raw/` and is never modified. The processed CSV
and cleaning report are written to `data/processed/`. The raw file contains:

| Column | Meaning |
|---|---|
| `objID` | SDSS object identifier; excluded from model features |
| `modelMag_u` through `modelMag_z` | Five-band SDSS model magnitudes |
| `specz` | Spectroscopic redshift target |

Validate the existing raw file with:

```bash
python src/data_loading.py --validate-existing
```

The downloader rejects invalid responses, checks the schema, detects sentinel
values such as `-9999`, and refuses to overwrite an existing raw CSV unless
`--force` is supplied. Raw measurement sentinels are reported and handled by
the preprocessing step rather than changed in place.

Create the modeling dataset with:

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
