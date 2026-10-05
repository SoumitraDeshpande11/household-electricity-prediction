# Household Electricity Consumption Prediction

Case Study 27 · B.Tech CSE · Semester V Machine Learning

This repository contains a complete regression study and a dark Streamlit dashboard for forecasting the **next 30 days of electricity consumption**. The executed notebook is the source of truth for downloading, cleaning, aggregating, training, evaluating, and exporting the model artifacts.

## Dataset

The project uses the real [UCI ElectricityLoadDiagrams20112014 dataset](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), also known as `ElectricityLoadDiagrams20112014`. It contains **140,256 fifteen-minute readings for 370 client meters** from 2011–2014. The published table has one timestamp column and one average-power column per client, with no missing cells. The source is licensed CC BY 4.0 (DOI [10.24432/C58C86](https://doi.org/10.24432/C58C86)).

The source archive is about 261 MB, so it is intentionally downloaded into ignored `data/raw/` rather than committed to GitHub. `data/source/source_metadata.json` pins the official UCI URL, archive hash, raw-file hash, size, and attribution. The notebook downloads the archive, verifies both hashes, extracts `LD2011_2014.txt`, and then performs every transformation. There is no synthetic-data fallback or generator.

This source calls its meters **clients**; it does not prove that every client is a household and does not contain household size, room count, appliance count, AC hours, weather, or occupancy. We therefore use real previous consumption and calendar features instead of inventing unavailable attributes.

## Forecast definition and processing

Each 15-minute reading is average power in kW. For each complete day, the notebook sums the 96 readings and divides by four to obtain daily kWh. It keeps one row per client and date, producing 540,200 client-days across 1,460 complete dates.

For each client, forecast origins are seven days apart. The model uses only readings before the origin: previous 30-day total, previous 7-day mean, previous-day total, previous 90-day mean, cyclic month sine/cosine, and season one-hot encoding. The target is the measured energy from the origin through the following 29 days. This is a fixed 30-day planning horizon, not an exact calendar-month billing total. The 71,040 labelled rows have overlapping targets, so they are not independent samples.

## Run the notebook and app

Use Python 3.11 or newer. The committed model was created with scikit-learn 1.8.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook household_electricity_prediction.ipynb
```

Choose **Restart Kernel and Run All Cells**. The notebook downloads and verifies the UCI source, displays EDA plots, compares all five required regressors, performs chronological cross-validation and residual analysis, then exports the processed tables and deployment artifacts.

For a headless run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace household_electricity_prediction.ipynb --ExecutePreprocessor.timeout=600
python validate_project.py
streamlit run app.py
```

The notebook is the only training/data-preparation implementation. Downloaded raw files remain ignored; processed tables and model artifacts are committed for Streamlit startup.

## Evaluation and results

The latest 20% of weekly origins form the chronological holdout. Training origins whose 30-day target would reach the holdout are removed. Five expanding `TimeSeriesSplit` folds use a five-origin (at least 35-day) gap. Preprocessing is fitted separately inside each fold.

| Model | Mean CV RMSE (kWh) | Holdout RMSE (kWh) | Holdout MAE (kWh) | Holdout R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 345,382.48 | 279,331.80 | 33,297.54 | 0.986 |
| Polynomial Regression (degree 2) | 1,295,913.59 | 196,428.58 | 31,230.65 | 0.993 |
| Decision Tree Regressor | 719,871.22 | 268,187.29 | 57,886.77 | 0.987 |
| Random Forest Regressor | 714,118.39 | 215,319.65 | 25,657.95 | 0.992 |
| **Gradient Boosting Regressor** | **654,293.09** | **174,497.18** | 26,869.06 | **0.995** |

**Linear Regression is selected by the lowest training CV RMSE** in this run. Gradient Boosting has the lowest final holdout RMSE. The holdout is reported separately and is not used to choose the deployment model.

The selected model improves on the persistence baseline that repeats the previous 30-day total. A deterministic client-holdout check excludes 74 clients from fitting and evaluates their later forecasts; this tests clients with prior history, not a brand-new meter with no history. Residual diagnostics, seasonal performance, feature permutation importance, and all fold metrics are saved in `artifacts/`.

## Streamlit dashboard

- **Predict consumption:** choose a measured client example or enter four history values and a forecast date; receive a 30-day kWh forecast and an RMSE-based planning range.
- **Data explorer:** inspect daily client demand, season/year filters, client scale, coverage, and downloadable processed data.
- **Model insights:** compare all five regressors, CV spread, persistence, holdout predictions, residuals, seasonal errors, and feature importance.
- **Project guide:** source provenance, units, processing choices, limitations, and reproduction commands.

Run locally with `streamlit run app.py`. For Streamlit Community Cloud, select this repository, branch `main`, and entry point `app.py`; the app uses committed processed tables and artifacts and does not retrain on startup.

## Files

| File | Purpose |
| --- | --- |
| `household_electricity_prediction.ipynb` | Executed download, cleaning, aggregation, EDA, training, evaluation, and export workflow |
| `app.py` | Interactive Streamlit dashboard |
| `data/source/source_metadata.json` | Official UCI URLs, hashes, sizes, DOI, and license |
| `data/processed/daily_consumption.csv` | 540,200 real client-day rows |
| `data/processed/forecast_records.csv` | 71,040 causal client-date forecast rows and targets |
| `artifacts/best_model.joblib` | CV-selected preprocessing and regression pipeline refit on labelled data |
| `artifacts/metrics.json` | Five-model metrics, splits, baseline, client-holdout and residual diagnostics |
| `artifacts/test_predictions.csv` | Predictions from the untouched chronological holdout |
| `validate_project.py` | Source lineage, timing, split, metrics, notebook, and artifact checks |

## Limitations

The dataset represents 370 client meters rather than a labelled population of households. It has no demographics, appliance inventory, weather, occupancy, tariff, or instantaneous peak-demand target. Weekly origins produce overlapping 30-day targets, so the effective sample size is smaller than the row count. Client-holdout evaluation still gives each held-out client its own preceding history. The displayed ±RMSE interval is a planning aid, not a calibrated probability interval. Operational demand forecasting needs recent multi-household data, weather, peak-load targets, uncertainty calibration, and prospective validation.

Source attribution: UCI Machine Learning Repository, *ElectricityLoadDiagrams20112014*, DOI 10.24432/C58C86, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Daily aggregation, feature engineering, modeling, and visualizations are this project's transformations.
