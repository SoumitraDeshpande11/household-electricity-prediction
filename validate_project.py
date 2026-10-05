#!/usr/bin/env python3
"""Verify real-source lineage, causal features, splits, and saved evaluation."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import joblib
import nbformat
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data/processed"


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS:", message)


def score(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {"R2": r2_score(actual, predicted), "MSE": mse, "RMSE": np.sqrt(mse), "MAE": mean_absolute_error(actual, predicted)}


def check_scores(actual, predicted, saved, description):
    for name, value in score(actual, predicted).items():
        check(np.isclose(value, saved[name], rtol=1e-9, atol=1e-8), f"{description} {name} matches exported predictions")


def main():
    metadata = json.loads((PROCESSED / "dataset_metadata.json").read_text())
    config = json.loads((ARTIFACTS / "feature_config.json").read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    archive_path = ROOT / "data/source/household_power_consumption.zip"
    check(metadata["synthetic"] is False and metadata["raw_rows"] == 2_075_259 and metadata["households"] == 1, "source is the real one-home meter dataset")
    check(hashlib.sha256(archive_path.read_bytes()).hexdigest() == metadata["archive_sha256"], "bundled archive hash matches Kaggle provenance")
    with zipfile.ZipFile(archive_path) as archive:
        member = next(n for n in archive.namelist() if n.endswith("household_power_consumption.txt"))
        source = archive.read(member)
    check(hashlib.sha256(source).hexdigest() == metadata["raw_sha256"], "extracted measurements match the recorded raw hash")
    check(len(source.splitlines()) - 1 == metadata["raw_rows"], "archive contains the reported number of measured rows")

    daily = pd.read_csv(PROCESSED / "daily_consumption.csv", parse_dates=["date"]).set_index("date")
    records = pd.read_csv(PROCESSED / "forecast_records.csv", parse_dates=["forecast_date", "forecast_end"])
    tests = pd.read_csv(ARTIFACTS / "test_predictions.csv", parse_dates=["forecast_date", "forecast_end"])
    check(len(daily) == metadata["daily_rows"] and len(records) == metadata["forecast_examples"], "processed counts match source metadata")
    check(int(daily.repaired_minutes.sum()) == metadata["repaired_power_minutes"] == 156, "repairs remain explicit and limited")
    check(int(daily.usable_energy.sum()) == metadata["usable_energy_days"], "usable-day count matches metadata")
    check(daily.loc[~daily.usable_energy, "energy_kwh"].isna().all(), "unresolved and incomplete energy days are excluded")
    check(records.forecast_date.is_unique and records.forecast_date.is_monotonic_increasing, "forecast origins are unique and chronological")
    check(((records.forecast_end - records.forecast_date).dt.days == 29).all(), "every target spans exactly 30 days")

    expected_target = daily.energy_kwh.rolling(30, min_periods=30).sum().shift(-29)
    np.testing.assert_allclose(records.next_30_day_kwh, expected_target.loc[records.forecast_date], rtol=1e-10)
    print("PASS: all targets reconstruct from measured daily energy")
    histories = {
        "previous_30_day_kwh": daily.energy_kwh.rolling(30, min_periods=30).sum().shift(1),
        "previous_7_day_mean_kwh": daily.energy_kwh.rolling(7, min_periods=7).mean().shift(1),
        "previous_day_kwh": daily.energy_kwh.shift(1),
    }
    for circuit in ["kitchen", "laundry", "heating_ac"]:
        histories[f"{circuit}_7_day_mean_kwh"] = daily[f"{circuit}_kwh"].rolling(7, min_periods=7).mean().shift(1)
    for name, values in histories.items():
        np.testing.assert_allclose(records[name], values.loc[records.forecast_date], equal_nan=True, rtol=1e-9, atol=1e-8)
    print("PASS: all six history inputs use preceding days only")
    np.testing.assert_allclose(records.month_sin, np.sin(2 * np.pi * records.forecast_date.dt.month / 12), atol=1e-10)
    np.testing.assert_allclose(records.month_cos, np.cos(2 * np.pi * records.forecast_date.dt.month / 12), atol=1e-10)
    check(set(records.season) == set(config["season_options"]), "calendar configuration includes the observed French seasons")

    test_start = records.forecast_date.iloc[int(len(records) * .8)]
    training = records[records.forecast_end < test_start]
    expected_test = records[records.forecast_date >= test_start]
    check(len(training) == metrics["train_rows"] and len(tests) == metrics["test_rows"] == len(expected_test), "saved holdout matches chronological split")
    check(training.forecast_end.max() < tests.forecast_date.min(), "training targets end before holdout begins")
    check(tests.forecast_date.reset_index(drop=True).equals(expected_test.forecast_date.reset_index(drop=True)), "saved test origins are exactly the final holdout")
    np.testing.assert_allclose(tests.actual, expected_test.next_30_day_kwh)
    np.testing.assert_allclose(tests.residual, tests.actual - tests.predicted)
    check_scores(tests.actual, tests.predicted, metrics["test_metrics"], "Holdout")
    baseline = tests.previous_30_day_kwh.notna()
    check(int(baseline.sum()) == metrics["baseline_comparable_rows"], "baseline and model use identical eligible origins")
    check_scores(tests.loc[baseline, "actual"], tests.loc[baseline, "previous_30_day_kwh"], metrics["baseline_metrics"], "Persistence")
    check_scores(tests.loc[baseline, "actual"], tests.loc[baseline, "predicted"], metrics["model_on_baseline_rows"], "Comparable model")
    for train_idx, validation_idx in TimeSeriesSplit(n_splits=5, gap=30).split(training):
        check(training.iloc[train_idx].forecast_end.max() < training.iloc[validation_idx].forecast_date.min(), "CV training target ends before validation starts")

    required_models = {"Linear Regression", "Polynomial Regression", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"}
    check(set(metrics["all_model_metrics"]) == required_models, "all five required regressors are evaluated")
    folds = pd.read_csv(ARTIFACTS / "cv_fold_metrics.csv")
    check(len(folds) == 25 and folds.groupby("Model").size().eq(5).all(), "all five models have five validation folds")
    for name, group in folds.groupby("Model"):
        check(np.isclose(group.RMSE.mean(), metrics["all_model_metrics"][name]["CV_RMSE"]), f"{name} CV score matches fold results")
    winner = min(metrics["all_model_metrics"], key=lambda n: metrics["all_model_metrics"][n]["CV_RMSE"])
    check(winner == metrics["selected_model"], "model selection uses lowest training CV RMSE")

    notebook = nbformat.read(ROOT / "household_electricity_prediction.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    check(all(cell.execution_count is not None for cell in code_cells), "all notebook code cells have executed")
    check(not any(output.output_type == "error" for cell in code_cells for output in cell.outputs), "notebook completed without cell errors")
    check(not (ROOT / "scripts/build_dataset.py").exists(), "synthetic generator has been removed")
    check(not (PROCESSED / "household_monthly_panel.csv").exists(), "synthetic training table has been removed")
    ast.parse((ROOT / "app.py").read_text())
    check(sklearn.__version__ == metrics["sklearn_version"], "runtime matches the saved scikit-learn model version")
    model = joblib.load(ARTIFACTS / "best_model.joblib")
    check(list(model.feature_names_in_) == config["feature_columns"], "saved pipeline feature order matches the app")
    check(np.isfinite(model.predict(records[config["feature_columns"]].head(10))).all(), "deployment pipeline predicts finite consumption")
    for preset in config["presets"]:
        row = records.loc[records.forecast_date.eq(pd.Timestamp(preset["forecast_date"]))].iloc[0]
        np.testing.assert_allclose([preset["inputs"][c] for c in config["input_features"]], row[config["input_features"]].to_numpy(dtype=float))
    print("PASS: application presets are actual measured-history rows")
    print("\nValidation passed.")


if __name__ == "__main__":
    main()
