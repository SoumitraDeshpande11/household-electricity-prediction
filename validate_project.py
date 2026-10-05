#!/usr/bin/env python3
"""Verify UCI source lineage, causal features, splits, metrics, and artifacts."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

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
SOURCE = ROOT / "data/source"
RAW = ROOT / "data/raw/uci_load_diagrams"
HORIZON = 30
ORIGIN_SPACING = 7
SEASONS = {"Winter", "Spring", "Summer", "Autumn"}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print("PASS:", message)


def score(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return {
        "R2": float(r2_score(actual, predicted)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def check_scores(actual, predicted, saved, description):
    for name, value in score(actual, predicted).items():
        check(np.isclose(value, saved[name], rtol=1e-8, atol=1e-5), f"{description} {name} matches exported predictions")


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def main():
    metadata = json.loads((PROCESSED / "dataset_metadata.json").read_text())
    config = json.loads((ARTIFACTS / "feature_config.json").read_text())
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    manifest = json.loads((SOURCE / "source_metadata.json").read_text())

    check(metadata["synthetic"] is False, "metadata marks the source as real")
    check(metadata["dataset"] == "ElectricityLoadDiagrams20112014", "metadata identifies the UCI multi-client dataset")
    check(metadata["raw_rows"] == 140_256 and metadata["clients"] == 370, "source counts are 140,256 intervals and 370 clients")
    check(metadata["interval_minutes"] == 15 and metadata["missing_intervals"] == 0, "source cadence and complete readings are recorded")
    for key in ("download_url", "archive_sha256", "raw_sha256", "doi", "license"):
        check(manifest.get(key) == metadata.get(key), f"source manifest {key} matches processed metadata")
    check(not (SOURCE / "household_power_consumption.zip").exists(), "obsolete one-home source archive is absent")

    archive_file = RAW / "electricityloaddiagrams20112014.zip"
    if archive_file.exists():
        check(hashlib.sha256(archive_file.read_bytes()).hexdigest() == metadata["archive_sha256"], "available UCI archive matches pinned archive hash")
    raw_file = RAW / "LD2011_2014.txt"
    if raw_file.exists():
        check(hashlib.sha256(raw_file.read_bytes()).hexdigest() == metadata["raw_sha256"], "available extracted source matches pinned raw hash")
    else:
        print("INFO: raw UCI archive is not present; hash is pinned in the manifest and notebook downloads it when run")

    daily = pd.read_csv(PROCESSED / "daily_consumption.csv", parse_dates=["date"])
    records = pd.read_csv(PROCESSED / "forecast_records.csv", parse_dates=["forecast_date", "forecast_end"])
    tests = pd.read_csv(ARTIFACTS / "test_predictions.csv", parse_dates=["forecast_date", "forecast_end"])
    folds = pd.read_csv(ARTIFACTS / "cv_fold_metrics.csv")

    check(len(daily) == metadata["daily_rows"] == metadata["daily_dates"] * metadata["clients"], "daily processed row count matches metadata")
    check(daily.date.nunique() == metadata["daily_dates"], "daily complete-date count matches metadata")
    check(daily.client_id.nunique() == metadata["clients"] == 370, "daily table contains all 370 clients")
    check(daily.groupby("date").size().eq(370).all() and daily.observed_intervals.eq(96).all(), "every complete date has 96 readings for every client")
    check(daily.usable_energy.eq(True).all() and daily.energy_kwh.notna().all(), "daily energy values are complete and usable")
    check(len(records) == metadata["forecast_examples"] == 71_040, "forecast row count matches metadata")
    origin_dates = pd.DatetimeIndex(sorted(records.forecast_date.unique()))
    check(len(origin_dates) == metadata["weekly_origins"] == 192, "forecast origins are weekly and complete")
    check(np.all(np.diff(origin_dates.values).astype("timedelta64[D]") == np.timedelta64(7, "D")), "origins are seven days apart")
    check(records.groupby("forecast_date").size().eq(370).all() and records.client_id.nunique() == 370, "each origin has one row per client")
    check(records.forecast_end.sub(records.forecast_date).dt.days.eq(29).all(), "every target spans exactly 30 days")
    check(records.next_30_day_kwh.notna().all() and set(records.season) <= SEASONS, "targets and seasons are valid")

    # Reconstruct daily targets and all history features from the saved daily table.
    wide = daily.pivot(index="date", columns="client_id", values="energy_kwh").sort_index()
    expected_target = wide.rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1))
    expected_previous = {
        "previous_30_day_kwh": wide.rolling(30, min_periods=30).sum().shift(1),
        "previous_7_day_mean_kwh": wide.rolling(7, min_periods=7).mean().shift(1),
        "previous_day_kwh": wide.shift(1),
        "previous_90_day_mean_kwh": wide.rolling(90, min_periods=90).mean().shift(1),
    }
    for client_id, group in records.groupby("client_id", sort=False):
        group = group.sort_values("forecast_date")
        dates = group.forecast_date
        np.testing.assert_allclose(group.next_30_day_kwh, expected_target.loc[dates, client_id], rtol=1e-5, atol=1e-4)
        for name, frame in expected_previous.items():
            np.testing.assert_allclose(group[name], frame.loc[dates, client_id], rtol=1e-5, atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(records.month_sin, np.sin(2 * np.pi * records.month / 12), atol=1e-10)
    np.testing.assert_allclose(records.month_cos, np.cos(2 * np.pi * records.month / 12), atol=1e-10)
    check(True, "targets and four history features reconstruct from preceding daily readings")

    test_start = origin_dates[int(len(origin_dates) * 0.8)]
    train_origins = origin_dates[origin_dates < test_start - pd.Timedelta(days=HORIZON - 1)]
    test_origins = origin_dates[origin_dates >= test_start]
    expected_train = records.forecast_date.isin(train_origins)
    expected_test = records.forecast_date.isin(test_origins)
    check(int(expected_train.sum()) == metrics["train_rows"] and int(expected_test.sum()) == metrics["test_rows"], "chronological row split matches metrics")
    check(tests.forecast_date.equals(records.loc[expected_test, "forecast_date"].reset_index(drop=True)), "holdout dates match the final origins")
    check(records.loc[expected_train, "forecast_end"].max() < records.loc[expected_test, "forecast_date"].min(), "training targets end before holdout starts")
    check(tests.client_id.equals(records.loc[expected_test, "client_id"].reset_index(drop=True)), "holdout client order matches forecast records")
    np.testing.assert_allclose(tests.actual, records.loc[expected_test, "next_30_day_kwh"])
    np.testing.assert_allclose(tests.residual, tests.actual - tests.predicted)
    check_scores(tests.actual, tests.predicted, metrics["test_metrics"], "Selected holdout")

    baseline = tests.previous_30_day_kwh.notna()
    check(int(baseline.sum()) == metrics["baseline_comparable_rows"], "baseline uses the same eligible holdout rows")
    check_scores(tests.loc[baseline, "actual"], tests.loc[baseline, "previous_30_day_kwh"], metrics["baseline_metrics"], "Persistence")
    check_scores(tests.loc[baseline, "actual"], tests.loc[baseline, "predicted"], metrics["model_on_baseline_rows"], "Comparable model")

    cv = TimeSeriesSplit(n_splits=5, gap=5)
    for training, validation in cv.split(train_origins):
        check(train_origins[training].max() + pd.Timedelta(days=HORIZON - 1) < train_origins[validation].min(), "CV training targets end before validation starts")
    required_models = {"Linear Regression", "Polynomial Regression", "Decision Tree Regressor", "Random Forest Regressor", "Gradient Boosting Regressor"}
    check(set(metrics["all_model_metrics"]) == required_models, "all five required regressors are evaluated")
    check(len(folds) == 25 and folds.groupby("Model").size().eq(5).all(), "all five models have five CV folds")
    for name, group in folds.groupby("Model"):
        check(np.isclose(group.RMSE.mean(), metrics["all_model_metrics"][name]["CV_RMSE"], rtol=1e-8, atol=1e-5), f"{name} CV RMSE matches fold results")
    winner = min(metrics["all_model_metrics"], key=lambda name: metrics["all_model_metrics"][name]["CV_RMSE"])
    check(winner == metrics["selected_model"], "selected model is the lowest-CV-RMSE model")

    notebook = nbformat.read(ROOT / "household_electricity_prediction.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    check(all(cell.execution_count is not None for cell in code_cells), "all notebook code cells have executed")
    check(not any(output.output_type == "error" for cell in code_cells for output in cell.outputs), "notebook completed without cell errors")
    code = "\n".join("".join(cell.source) for cell in code_cells)
    check("make_regression" not in code and "np.random" not in code, "notebook contains no synthetic-data generator")
    check(not (ROOT / "scripts/build_dataset.py").exists() and not (PROCESSED / "household_monthly_panel.csv").exists(), "obsolete synthetic generator and table are absent")

    ast.parse((ROOT / "app.py").read_text())
    check(sklearn.__version__ == metrics["sklearn_version"], "runtime matches saved scikit-learn version")
    model = joblib.load(ARTIFACTS / "best_model.joblib")
    check(list(model.feature_names_in_) == config["feature_columns"], "saved pipeline feature order matches app configuration")
    check(np.isfinite(model.predict(records[config["feature_columns"]].head(10))).all(), "deployment pipeline predicts finite values")
    for preset in config["presets"]:
        row = records.loc[records.forecast_date.eq(pd.Timestamp(preset["forecast_date"])) & records.client_id.eq(preset["client_id"])].iloc[0]
        np.testing.assert_allclose([preset["inputs"][column] for column in config["input_features"]], row[config["input_features"]].to_numpy(dtype=float), rtol=1e-5, atol=1e-4)
    check(True, "application presets are actual measured client-history rows")
    print("\nValidation passed.")


if __name__ == "__main__":
    main()
