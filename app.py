"""Streamlit interface for the household electricity consumption model.

The trained model and the small JSON metadata files are produced by the
notebook.  Keeping the app dependent on those artifacts makes the prediction
path identical to the one used during evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "best_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
CONFIG_PATH = ARTIFACTS / "feature_config.json"

DEFAULT_FEATURES = [
    "household_size",
    "number_of_rooms",
    "appliance_count",
    "ac_usage_hours",
    "season",
    "previous_month_consumption",
]
DEFAULT_SEASONS = ["Winter", "Spring", "Summer", "Autumn", "Monsoon"]


def _read_json(path: Path) -> dict[str, Any]:
    """Read an artifact JSON file and return an empty mapping when absent."""

    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Could not read `{path.name}`: {exc}")
        return {}


@st.cache_resource(show_spinner=False)
def _load_model(path_string: str) -> Any:
    """Load the serialized estimator once per Streamlit session."""

    return joblib.load(path_string)


def _config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value for a set of compatible metadata keys."""

    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def _get_feature_columns(config: dict[str, Any]) -> list[str]:
    columns = _config_value(
        config,
        "feature_columns",
        "features",
        "input_features",
        "feature_names",
        default=DEFAULT_FEATURES,
    )
    if isinstance(columns, (tuple, list)) and columns:
        return [str(column) for column in columns]
    return DEFAULT_FEATURES.copy()


def _get_season_options(config: dict[str, Any]) -> list[str]:
    options = _config_value(
        config,
        "season_options",
        "season_categories",
        "seasons",
        default=DEFAULT_SEASONS,
    )
    if isinstance(options, (tuple, list)) and options:
        return [str(option) for option in options]
    return DEFAULT_SEASONS.copy()


def _metric(metrics: dict[str, Any], name: str, default: float | None = None) -> float | None:
    """Find a metric in common flat or nested metrics JSON layouts."""

    aliases = {
        "r2": ["r_squared", "r-squared", "R²"],
        "mse": ["mean_squared_error"],
        "rmse": ["root_mean_squared_error"],
        "mae": ["mean_absolute_error"],
    }
    candidates = [name, name.upper(), name.lower(), name.replace("_", " "), *aliases.get(name, [])]
    containers: list[dict[str, Any]] = [metrics]
    for key in ("test_metrics", "validation_metrics", "metrics", "best_model_metrics"):
        nested = metrics.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in candidates:
            value = container.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return default


def _model_name(metrics: dict[str, Any], model: Any) -> str:
    for key in ("best_model", "best_model_name", "selected_model", "model_name", "algorithm"):
        value = metrics.get(key)
        if isinstance(value, str) and value.strip():
            return value
    # Pipelines usually expose the final estimator in a useful repr/name.
    if hasattr(model, "named_steps"):
        steps = getattr(model, "named_steps")
        if isinstance(steps, dict) and steps:
            return type(next(reversed(steps.values()))).__name__
    return type(model).__name__


def _input_frame(values: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Build a one-row frame while preserving the training-column order."""

    # Canonical names are used by the notebook.  The fallbacks keep the app
    # usable if an equivalent, human-readable metadata spelling is supplied.
    aliases = {
        "household_size": values["household_size"],
        "number_of_rooms": values["number_of_rooms"],
        "rooms": values["number_of_rooms"],
        "appliance_count": values["appliance_count"],
        "ac_usage_hours": values["ac_usage_hours"],
        "ac_usage": values["ac_usage_hours"],
        "season": values["season"],
        "previous_month_consumption": values["previous_month_consumption"],
        "previous_consumption": values["previous_month_consumption"],
    }
    row = {column: aliases.get(column, values.get(column, 0)) for column in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def main() -> None:
    st.set_page_config(page_title="Household Electricity Prediction", page_icon="⚡", layout="centered")
    st.title("Household Electricity Consumption Prediction")
    st.write("Enter the household details to estimate the next month’s electricity use.")

    config = _read_json(CONFIG_PATH)
    metrics = _read_json(METRICS_PATH)
    feature_columns = _get_feature_columns(config)
    seasons = _get_season_options(config)

    if not MODEL_PATH.exists():
        st.error("The trained model is not available yet. Run the notebook to create `artifacts/best_model.joblib`.")
        st.info("Expected artifact folder: " + str(ARTIFACTS))
        return
    try:
        model = _load_model(str(MODEL_PATH))
    except Exception as exc:  # joblib can raise several pickle-related exceptions.
        st.error(f"The saved model could not be loaded: {exc}")
        return

    with st.form("prediction_form"):
        st.subheader("Household details")
        left, right = st.columns(2)
        with left:
            household_size = st.number_input("Household size (people)", min_value=1, max_value=20, value=4, step=1)
            rooms = st.number_input("Number of rooms", min_value=1, max_value=20, value=3, step=1)
            appliance_count = st.number_input("Number of appliances", min_value=0, max_value=50, value=8, step=1)
        with right:
            ac_usage_hours = st.number_input(
                "Air-conditioner usage (hours/month)", min_value=0.0, max_value=744.0, value=80.0, step=1.0
            )
            season = st.selectbox("Season", options=seasons, index=0)
            previous_consumption = st.number_input(
                "Previous month’s consumption (kWh)", min_value=0.0, max_value=10000.0, value=250.0, step=1.0
            )
        submitted = st.form_submit_button("Predict monthly consumption", type="primary")

    if not submitted:
        st.caption("The prediction uses the selected model saved in the artifacts folder.")
        return

    values = {
        "household_size": household_size,
        "number_of_rooms": rooms,
        "appliance_count": appliance_count,
        "ac_usage_hours": ac_usage_hours,
        "season": season,
        "previous_month_consumption": previous_consumption,
    }
    frame = _input_frame(values, feature_columns)
    try:
        try:
            prediction = float(model.predict(frame)[0])
        except Exception:
            # A model saved without the notebook's DataFrame preprocessor may
            # expect the already ordered numeric array instead.
            prediction = float(model.predict(frame.to_numpy())[0])
    except Exception as exc:
        st.error(f"Prediction failed because the model expected different input columns: {exc}")
        st.caption("Expected columns from feature_config.json: " + ", ".join(feature_columns))
        return

    rmse = _metric(metrics, "rmse")
    unit = str(_config_value(config, "target_unit", "prediction_unit", "unit", default="kWh"))
    # The notebook metadata explains the synthetic unit in prose; keep the
    # metric and range labels concise in the application.
    if "units/kWh" in unit:
        unit = "units/kWh"
    model_label = _model_name(metrics, model)
    st.subheader("Prediction")
    st.metric("Predicted monthly consumption", f"{prediction:,.2f} {unit}")
    if rmse is not None:
        lower = max(0.0, prediction - rmse)
        upper = prediction + rmse
        st.info(f"Expected range (± validation RMSE): **{lower:,.2f} to {upper:,.2f} {unit}**")
        st.caption(f"The range uses the held-out validation RMSE of {rmse:,.2f} {unit}.")
    else:
        st.warning("Validation RMSE was not found in metrics.json, so an error range cannot be shown.")

    st.write(f"**Selected model:** {model_label}")
    metric_values = [("R²", _metric(metrics, "r2")), ("RMSE", rmse), ("MAE", _metric(metrics, "mae")), ("MSE", _metric(metrics, "mse"))]
    available = {label: value for label, value in metric_values if value is not None}
    if available:
        st.write("**Validation metrics**")
        st.dataframe(pd.DataFrame([available]), hide_index=True, use_container_width=True)

    with st.expander("Prediction inputs"):
        st.dataframe(frame, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
