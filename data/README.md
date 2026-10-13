# Data Directory

## Structure

- **`raw/`** — Original data downloaded from SDSS. Never modify these files.
- **`processed/`** — Cleaned and feature-engineered data ready for modeling.

## Data Source

- **Survey**: Sloan Digital Sky Survey (SDSS)
- **Data Release**: DR17
- **Access Method**: SkyServer SQL Search REST API
- **Query**: See `src/data_loading.py` and the acquisition notes below for the exact SQL query used.

## How to Download

```bash
python src/data_loading.py
```

This will save the raw CSV to `data/raw/sdss_dr17_galaxies_raw.csv`.

To validate an existing raw CSV without downloading or modifying it:

```bash
python src/data_loading.py --validate-existing
```

The downloader refuses to overwrite an existing raw CSV unless `--force` is
provided. Downloads are first written to a temporary `.part` file and moved
into place only after validation succeeds.

## Columns (Phase 1)

| Column | Description |
|---|---|
| `objID` | Unique SDSS object identifier (not a model feature) |
| `modelMag_u` | Model magnitude in u band |
| `modelMag_g` | Model magnitude in g band |
| `modelMag_r` | Model magnitude in r band |
| `modelMag_i` | Model magnitude in i band |
| `modelMag_z` | Model magnitude in z band |
| `specz` | Spectroscopic redshift — **this is the prediction target** |

The raw file is preserved exactly as downloaded. SDSS measurement sentinels
such as `-9999` are reported by validation and are not converted or removed
until a separate, reproducible preprocessing step is implemented. A null
value and a sentinel value are therefore tracked separately.

## Acquisition Notes

This checkpoint covers only obtaining and validating the raw CSV. It does not
perform cleaning, feature engineering, exploratory analysis, or modeling.

### Source and query

- Survey: Sloan Digital Sky Survey (SDSS)
- Data release: DR17
- Endpoint: `https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch`
- Response format: CSV
- Default limit: 50,000 rows

The query joins `SpecPhoto`, `PhotoObj`, and `SpecObj`, then selects galaxies
with `zWarning = 0` and `0.01 <= z <= 0.5`. It returns the object identifier,
five `modelMag_*` measurements, and spectroscopic redshift as `specz`.

The authoritative query source is `SQL_QUERY_TEMPLATE` in
`src/data_loading.py`. SkyServer's DR17 documentation describes SQL search,
CSV output, and programmatic access; the schema browser remains the source of
truth for field definitions.

### Raw-file policy

The raw CSV is immutable project input. The loader:

1. Requests the CSV with bounded retries.
2. Rejects HTML/error responses and empty results.
3. Validates schema, numeric types, unique `objID` values, finite values, and
   redshift bounds.
4. Writes a validated download atomically, avoiding partial CSVs.

Existing output is not overwritten by default. Use `--validate-existing` to
check it without a network request, or provide a new `--output` path for a
separate sample. `--force` is reserved for intentional replacement.

### Reproduction commands

```bash
# Validate the existing 50,000-row file without changing it.
python src/data_loading.py --validate-existing

# Download and validate a small sample to a separate file.
python src/data_loading.py --limit 1000 \
  --output data/raw/sdss_dr17_galaxies_1000_raw.csv
```
