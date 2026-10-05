# Real meter data and transformations

All source download and processing steps are implemented in `household_electricity_prediction.ipynb`.

- Dataset: [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)
- Download URL: `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip`
- Attribution: UCI Machine Learning Repository, DOI `10.24432/C58C86`
- License: CC BY 4.0
- Source size: 261,335,609-byte ZIP; 710,998,915-byte extracted text file
- Archive SHA-256: `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`
- Raw TXT SHA-256: `d51565f2cb5a6b768d06ba1bbd3c084c6e2f3aab07f00c6f2dcb80e90175124b`

The archive is too large for GitHub's file limit and is therefore downloaded into ignored `data/raw/uci_load_diagrams/`. `data/source/source_metadata.json` is the reproducibility manifest. The notebook verifies the archive and extracted file hashes before reading them. No synthetic source table, target, household, or fallback generator is used.

## Processed tables

`processed/daily_consumption.csv` contains 540,200 rows: 1,460 complete dates × 370 clients. Each row contains daily kWh, client ID, year, month, season, and the 96 observed quarter-hour readings flag. Source zeros before a client's activation are retained as published.

`processed/forecast_records.csv` contains 71,040 rows from 192 weekly origins × 370 clients. Each target is the measured energy from `forecast_date` through `forecast_date + 29 days`. Targets overlap across weekly origins and therefore are not independent household examples.

`processed/dataset_metadata.json` records source hashes, row/client/date counts, units, and date bounds.

## Units and missing readings

The source values are average kW for each 15-minute interval. Daily energy is `sum(96 readings) / 4`, yielding kWh. The source table has no missing cells, so no readings are imputed. Partial endpoint dates are excluded from daily aggregation; complete dates have exactly 96 readings.

## Model inputs and target

| Column | Definition |
| --- | --- |
| `forecast_date` | First day of the predicted 30-day period |
| `previous_30_day_kwh` | Total energy over the preceding 30 days |
| `previous_7_day_mean_kwh` | Mean daily energy over the preceding seven days |
| `previous_day_kwh` | Energy on the immediately preceding day |
| `previous_90_day_mean_kwh` | Mean daily energy over the preceding 90 days |
| `month_sin`, `month_cos` | Cyclic month encoding from `forecast_date` |
| `season` | Winter, Spring, Summer, or Autumn |
| `next_30_day_kwh` | Actual target energy over the 30-day horizon |

Every history feature ends before `forecast_date`; only calendar fields describe the forecast origin. Client ID is retained for evaluation and app context but is not used as a model feature.

## Evaluation boundary

The latest 20% of weekly origins form the holdout. Earlier origins whose target would reach the holdout are excluded. Five expanding time-series folds use a five-origin gap, which is at least 35 days. A separate deterministic client-holdout check evaluates 74 clients not used during fitting, while still supplying their preceding history. This does not measure a completely new client without any history.

Reproduce the processed data, plots, models, and artifacts by running the notebook from a fresh kernel, then run `python validate_project.py`.
