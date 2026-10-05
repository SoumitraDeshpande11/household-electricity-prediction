#!/usr/bin/env python3
"""Small, dependency-light release check for the household-energy project.

This script deliberately checks the public contract of the project rather than
retraining anything: the notebook, processed table, saved model, metrics, and
Streamlit entry point should all be present and internally coherent.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL: {message}")


def ok(message: str) -> None:
    print(f"PASS: {message}")


def main() -> int:
    failures: list[str] = []

    notebook = ROOT / "household_electricity_prediction.ipynb"
    app = ROOT / "app.py"
    if notebook.is_file():
        try:
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            cells = payload.get("cells", [])
            code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
            if not code_cells:
                fail("the notebook contains no code cells", failures)
            else:
                ok(f"notebook is valid JSON with {len(code_cells)} code cells")
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"notebook cannot be read as JSON ({exc})", failures)
    else:
        fail("missing household_electricity_prediction.ipynb", failures)

    if app.is_file():
        try:
            ast.parse(app.read_text(encoding="utf-8"), filename=str(app))
            ok("app.py parses successfully")
        except (OSError, SyntaxError) as exc:
            fail(f"app.py does not parse ({exc})", failures)
    else:
        fail("missing app.py", failures)

    csv_files = sorted((ROOT / "data" / "processed").glob("*.csv"))
    frame = None
    if not csv_files:
        fail("no processed CSV found under data/processed", failures)
    else:
        csv_path = csv_files[0]
        try:
            import pandas as pd  # type: ignore

            frame = pd.read_csv(csv_path)
            if len(frame) < 100:
                fail(f"processed CSV has only {len(frame)} rows", failures)
            else:
                ok(f"processed CSV has {len(frame):,} rows ({csv_path.name})")

            names = {str(name).strip().lower() for name in frame.columns}
            aliases = {
                "target": {"monthly_consumption_kwh", "monthly_consumption", "consumption_kwh", "target"},
                "season": {"season"},
                "previous": {"previous_month_consumption", "previous_consumption", "prev_month_consumption"},
            }
            for label, options in aliases.items():
                if not names.intersection(options):
                    fail(f"processed CSV has no {label} column (found: {sorted(names)})", failures)
            if "season" in names:
                ok("processed CSV includes a season feature")
            if names.intersection(aliases["target"]):
                ok("processed CSV includes a consumption target")
        except ImportError:
            fail("pandas is unavailable; cannot inspect processed CSV", failures)
        except Exception as exc:  # pandas parser errors vary by version
            fail(f"processed CSV cannot be read ({exc})", failures)

    metrics_path = ROOT / "artifacts" / "metrics.json"
    model_candidates = sorted((ROOT / "artifacts").glob("*.joblib"))
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            text = json.dumps(metrics).lower()
            metric_names = ("r2", "mse", "rmse", "mae")
            missing = [name for name in metric_names if name not in text]
            if missing:
                fail(f"metrics.json is missing expected metric names: {missing}", failures)
            else:
                ok("metrics.json contains R2, MSE, RMSE, and MAE")
            all_model_metrics = metrics.get("all_model_metrics")
            expected_models = {
                "Linear Regression",
                "Polynomial Regression",
                "Decision Tree Regressor",
                "Random Forest Regressor",
                "Gradient Boosting Regressor",
            }
            if isinstance(all_model_metrics, dict):
                missing_models = sorted(expected_models.difference(all_model_metrics))
                if missing_models:
                    fail(f"metrics.json is missing required model rows: {missing_models}", failures)
                else:
                    ok("metrics.json compares all five required regressors")
            else:
                fail("metrics.json has no all_model_metrics comparison", failures)
        except (OSError, json.JSONDecodeError) as exc:
            fail(f"metrics.json is invalid ({exc})", failures)
    else:
        fail("missing artifacts/metrics.json", failures)

    if not model_candidates:
        fail("no .joblib model found under artifacts", failures)
    else:
        try:
            import joblib  # type: ignore

            model = joblib.load(model_candidates[0])
            if not hasattr(model, "predict"):
                fail(f"saved artifact {model_candidates[0].name} has no predict method", failures)
            else:
                ok(f"saved model loads and exposes predict() ({model_candidates[0].name})")
        except ImportError:
            fail("joblib is unavailable; cannot load saved model", failures)
        except Exception as exc:
            fail(f"saved model cannot be loaded ({exc})", failures)

    # The app's season selector comes from feature_config.json.  Every season
    # represented in the processed data should therefore be selectable.
    config_path = ROOT / "artifacts" / "feature_config.json"
    if frame is not None and config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            observed = {str(value) for value in frame["season"].dropna().unique()}
            configured = {str(value) for value in config.get("season_options", [])}
            uncovered = sorted(observed.difference(configured))
            if uncovered:
                fail(f"feature_config.json omits observed seasons: {uncovered}", failures)
            else:
                ok("feature_config.json covers every observed season")
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            fail(f"feature_config.json cannot be checked ({exc})", failures)

    if failures:
        print(f"\nValidation failed with {len(failures)} issue(s).")
        return 1
    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
