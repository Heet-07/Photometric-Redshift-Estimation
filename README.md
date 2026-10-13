# Photometric Redshift Estimation of Galaxies Using SDSS Data

A machine-learning project that uses photometric observations from the
**Sloan Digital Sky Survey (SDSS DR17)** to estimate galaxy redshift.

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
│   ├── features.py          # Compute colors and feature sets
│   ├── models.py            # ML model wrappers
│   └── evaluation.py        # Metrics and plotting
├── results/
│   ├── figures/             # Saved plots
│   └── metrics/             # Saved metric tables
├── docs/                    # Design decisions and SDSS schema notes
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ~50,000 galaxies from SDSS DR17
python src/data_loading.py

# 3. Open the exploration notebook
jupyter lab notebooks/
```

## Data Source

- **Survey**: Sloan Digital Sky Survey
- **Release**: DR17
- **API**: SkyServer SQL Search (no account required)
- **Sample**: ~50,000 galaxies with spectroscopic redshift in range 0.01–0.5
