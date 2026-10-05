# Household Electricity Consumption Prediction

This project implements **Case Study 27** for the B.Tech CSE Machine Learning course. It predicts a household's monthly electricity consumption from household characteristics, air-conditioner usage, season, and the previous month's consumption. The project is intentionally notebook-led so that the data assumptions, analysis, model comparison, and conclusions can be followed during an examination or viva.

## Data source and why the modelling table is synthetic

The reference consumption readings come from the [Individual Household Electric Power Consumption dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set) on Kaggle, originally published through the UCI Machine Learning Repository. It contains more than two million minute-level readings with active power, voltage, intensity, and sub-metering columns. The source data is useful for learning realistic seasonal and monthly consumption patterns, but it does not contain household size, rooms, appliance count, or a directly labelled AC-usage field for every record.

The repository therefore builds a reproducible synthetic household-month table. When a local source file is supplied, its monthly summaries anchor the scale; otherwise the builder uses its documented fallback scale. Household attributes and AC usage are generated with documented distributions and a fixed random seed, then combined with previous-month consumption and noise. This is the approach requested by the case study, and it avoids presenting invented household attributes as if they were measured Kaggle columns. The large raw download is deliberately not committed to Git. The builder works offline or can use a locally downloaded UCI/Kaggle file for calibration.

## Repository layout

```text
.
├── household_electricity_prediction.ipynb  # complete analysis and modelling record
├── app.py                                  # Streamlit prediction application
├── scripts/                                # optional reproducible data-building helpers
├── data/                                   # source notes and generated data (if saved)
├── artifacts/                              # winning model, metrics, and feature metadata
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer. From the project directory:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

From the repository root, reproduce the committed processed panel with:

```bash
python3 scripts/build_dataset.py
```

The default build creates 500 households observed for 36 months (18,000 rows) at `data/processed/household_monthly_panel.csv`. To calibrate against a local source download, pass `--reference-path data/raw/household_power_consumption.txt`; no network download is required by the project. Open `household_electricity_prediction.ipynb` in Jupyter or VS Code and run the cells from top to bottom. The notebook reports missing readings and their treatment, encodes the season, trains all five required regressors, and saves the selected model and metrics under `artifacts/` for the application. Keep any raw Kaggle file under a local data directory and do not add it to Git.

## What the notebook covers

The analysis answers the case-study questions with computed evidence rather than fixed claims:

- exploratory summaries and plots for household attributes, monthly consumption, and seasons;
- missing-meter simulation and imputation, with the number of affected readings reported;
- previous-month consumption as a predictor and the effect of seasonal encoding;
- Linear Regression, Polynomial Regression, Decision Tree, Random Forest, and Gradient Boosting regressors;
- hold-out evaluation and five-fold cross-validation using R², MSE, RMSE, and MAE;
- actual-versus-predicted and residual plots for the selected model;
- feature influence, seasonal error checks, constant-variance discussion, and limitations.

RMSE is the primary planning measure because it is expressed in the same units as monthly consumption. The model is selected from the measured validation results, with the lowest suitable cross-validated RMSE preferred. The notebook records the exact split, preprocessing, random seed, and winning metrics so the result is reproducible.

## Run the Streamlit application

After running the notebook and creating the artifacts:

```bash
streamlit run app.py
```

The application follows the layout of the Wine Quality Predictor: a styled hero, example profiles, a prominent forecast card, and four tabs:

- **Predict consumption** accepts a household profile, shows the expected error range, and compares controlled season and AC-usage scenarios. Submitted predictions remain visible when exploring the other tabs.
- **Data explorer** filters records by season and year, plots household attributes and monthly histories, and downloads the filtered dataset.
- **Model insights** compares all five regressors and shows held-out actual-versus-predicted, residual, feature-influence, and seasonal-performance charts.
- **Project guide** explains the inputs, generation assumptions, evaluation, and limitations.

The visual theme is defined in `.streamlit/config.toml` and `assets/app.css`. Interactive charts use Altair. Input bounds follow observed ranges in the processed dataset.

Enter household size, number of rooms, appliance count, AC usage, season, and previous-month consumption. The app displays the predicted monthly consumption, the selected model, and an expected error range. The range is based on the validation RMSE (approximately `prediction ± RMSE`, clipped at zero where appropriate); it is a practical planning interval, not a formal prediction interval or a guarantee for an individual bill.

If the app reports missing artifacts, run the notebook once more from the first cell so that the model, metrics, feature configuration, and any preprocessing objects are saved into `artifacts/`.

The notebook also exports `artifacts/test_predictions.csv` and `artifacts/feature_importance.csv`. Diagnostic charts use the held-out model's predictions, while the prediction form uses the deployment pipeline refitted on the full dataset. Feature influence is the increase in test RMSE after shuffling an original input, averaged over five shuffles; correlated inputs can share predictive information.

## Limitations and real-world use

This model demonstrates the requested workflow and supports an academic demand-forecasting prototype. It is not a utility billing system. The generated household attributes are assumptions, the source data represents one household rather than a diverse utility population, and weather, tariff, occupancy schedules, holidays, solar generation, and appliance-level behaviour are absent. These omissions can cause systematic error, especially during unusual heat or cold. A production forecast should retrain on interval readings from the target service area, join reliable weather and calendar data, monitor drift, and report uncertainty alongside the point prediction.

## Reproducibility notes

All random generation and model settings are kept in the notebook or accompanying scripts. Do not commit Kaggle credentials, downloaded raw files, virtual environments, notebook checkpoints, caches, or generated logs. The committed notebook, source notes, and saved lightweight artifacts are sufficient for review; a fresh raw download may be needed to rebuild the reference monthly summaries.
