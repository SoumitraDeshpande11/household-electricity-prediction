# Household Electricity Consumption Prediction

Case Study 27 · B.Tech CSE · Semester V Machine Learning

A complete **real-data** regression project, with all analysis and training in [household_electricity_prediction.ipynb](household_electricity_prediction.ipynb) and a dark Streamlit dashboard with prediction, exploration, and model diagnostics.

## Actual dataset

We downloaded [Kaggle's Individual Household Electric Power Consumption dataset](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set). It contains **2,075,259 minute readings from one household** in Sceaux, France, from December 2006 to November 2010. [Original UCI documentation](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

**No synthetic households, demographic attributes, or targets are generated.** The previous synthetic table and generator have been removed. The exact 19.4 MiB source ZIP is committed in data/source; the notebook extracts it and verifies its SHA-256. If absent, the notebook downloads the same real dataset. Extracted raw files are ignored by Git.

The source does not contain household size, room count, appliance count, or separate AC hours. We adapt the case study to measured energy history and calendar inputs, rather than inventing those fields.

## Forecast definition and missing readings

The output is **next-30-day energy consumption in kWh**, a standardised monthly horizon. It is not an exact calendar-month billing total. Daily forecast origins provide 957 labelled examples; these overlap and are not 957 independent households.

Global active power is kW averaged during each minute, so summing 1,440 values and dividing by 60 gives daily kWh. Submeter readings are Wh, converted to kWh by dividing summed readings by 1,000. The third circuit combines water heating and AC.

There are 25,979 missing power minutes. Forward filling for at most five minutes repairs 156 readings using only past measurements. Longer gaps stay missing; incomplete energy days and their target windows are excluded. This yields 1,410 usable days. Remaining historical input gaps are median-imputed within each training fold.

The six history inputs are the previous 30-day total, previous seven-day daily average, previous-day total, and seven-day daily averages of kitchen, laundry, and combined water-heater/AC energy. Month sine/cosine and one-hot season add calendar information. Every historical input ends before its forecast starts.

## Run the notebook and app

Use Python 3.11 or newer. Model loading uses scikit-learn 1.8.0, the version used for the committed artifact.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook household_electricity_prediction.ipynb
~~~

Choose **Restart Kernel and Run All Cells**. The executed notebook already includes the EDA figures, model comparisons, residual analysis, and answers to the case-study questions.

Alternatively, execute it without opening Jupyter:

~~~bash
python -m jupyter nbconvert --to notebook --execute --inplace household_electricity_prediction.ipynb --ExecutePreprocessor.timeout=600
python validate_project.py
streamlit run app.py
~~~

The source archive is bundled, so rebuilding does not require Kaggle credentials. All ML logic is in the notebook; there is no separate training or data-generation script.

## Evaluation and results

We hold out the latest 192 forecast origins, starting 9 December 2009. The earlier training partition contains 736 origins; 29 boundary origins are discarded so training targets cannot overlap holdout targets.

Model selection uses **five expanding time-series folds with a 30-row gap**, at least 30 calendar days. Preprocessing is fitted separately in each fold. All five required regressors are compared:

| Model | Mean CV RMSE (kWh) | CV standard deviation | Holdout RMSE (kWh) | Holdout MAE (kWh) | Holdout R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | 165.54 | 134.12 | 80.04 | 66.84 | 0.735 |
| Polynomial Regression (degree 2) | 565.72 | 855.10 | 96.08 | 76.99 | 0.618 |
| Decision Tree Regressor | 143.50 | 57.08 | 59.68 | 41.42 | 0.853 |
| **Random Forest Regressor** | **119.43** | 45.22 | 56.16 | 43.11 | 0.870 |
| Gradient Boosting Regressor | 121.30 | 43.36 | **45.62** | **34.97** | **0.914** |

**Random Forest is selected by training CV.** Gradient Boosting has the lowest final holdout RMSE; we report that distinction and do not select retrospectively using the holdout. The high fold variability also limits confidence in the small CV difference.

On the same 99 holdout origins with complete prior 30-day history, Random Forest RMSE is **50.14 kWh**, compared with **133.19 kWh** for repeating the previous 30-day total. The full Random Forest holdout MSE is 3,153.90 kWh².

Season and month cosine have the greatest permutation importance for this fitted model. This is predictive reliance, not a causal household effect. Residual standard deviations differ by **4.59×** across prediction bands, and lag-one residual correlation is **0.811**. Constant variance and independent errors are not established.

Only 16 holdout targets contain no repaired power minutes; their RMSE is 50.00 kWh. This small sensitivity subset cannot establish broad robustness.

## Streamlit dashboard

- **Predict consumption:** actual recorded history presets, six meter-history inputs, forecast date, next-30-day prediction, ±RMSE range, and calendar/history sensitivity charts.
- **Data explorer:** daily demand, seasonal demand, circuit consumption, meter coverage, season/year filters, and CSV download.
- **Model insights:** all five models, CV variability, persistence comparison, actual-versus-predicted and residual plots, permutation importance, seasonal performance, and holdout time plots.
- **Project guide:** real-source provenance, preprocessing, metrics, and limitations.

The deployment pipeline is refitted on all labelled examples after evaluation. Saved holdout predictions remain from the training-only fit. Historical app examples are demonstrations, not new test results.

For Streamlit Community Cloud, select this repository, branch main, and entry point app.py. The committed model and processed tables allow the app to start without rerunning training. [Open Streamlit deployment](https://share.streamlit.io/deploy).

## Files

| File | Purpose |
| --- | --- |
| household_electricity_prediction.ipynb | Complete executed download, cleaning, EDA, training, evaluation, and export workflow |
| app.py | Interactive dashboard loading notebook artifacts |
| data/source/household_power_consumption.zip | Exact real Kaggle source download |
| data/source/source_metadata.json | URLs, hashes, size, DOI, and license |
| data/processed/daily_consumption.csv | Daily measured demand and coverage/repair flags |
| data/processed/forecast_records.csv | Causal historical inputs and actual future 30-day targets |
| artifacts/best_model.joblib | Selected preprocessing + regression pipeline, refitted for deployment |
| artifacts/metrics.json | All five model scores, baseline, splits, and residual diagnostics |
| artifacts/test_predictions.csv | Predictions from the untouched chronological holdout |
| validate_project.py | Source, aggregation, feature timing, evaluation, and artifact checks |

## Limitations

This dataset is one French home measured in 2006–2010. Holdout performance describes later periods of that home, **not new households**. Demographic factors and weather are unavailable. Adjacent targets overlap, reducing the effective sample size despite leakage-prevention gaps. Short-gap repair introduces an assumption; exclusions may bias coverage. Season is an incomplete substitute for temperature and occupancy.

The ±RMSE range is a rough planning aid with no guaranteed probability coverage. Operational deployment requires recent multi-household data, weather, time-aware uncertainty calibration, and prospective validation. A 30-day energy forecast cannot determine instantaneous peak grid capacity.

Source attribution: Hebrail, G. & Berard, A. (2006), Individual Household Electric Power Consumption, UCI Machine Learning Repository, DOI 10.24432/C58K54, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Daily aggregation, documented gap repair, feature engineering, and modeling are this project's transformations.
