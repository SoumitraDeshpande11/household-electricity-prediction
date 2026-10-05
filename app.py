"""Interactive Streamlit application for the household electricity model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_PATH = ROOT / "data" / "processed" / "household_monthly_panel.csv"
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
DEFAULT_SEASONS = ["Winter", "Summer", "Monsoon", "Autumn"]
PALETTE = ["#18a999", "#f4a261", "#e76f51", "#457b9d", "#9b5de5"]


st.set_page_config(
    page_title="Household Energy Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_data(path_string: str) -> pd.DataFrame:
    frame = pd.read_csv(path_string)
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    return frame


@st.cache_resource(show_spinner=False)
def load_model(path_string: str) -> Any:
    return joblib.load(path_string)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


def feature_columns(config: dict[str, Any]) -> list[str]:
    columns = config_value(config, "feature_columns", "features", default=DEFAULT_FEATURES)
    if isinstance(columns, (list, tuple)) and columns:
        return [str(column) for column in columns]
    return DEFAULT_FEATURES.copy()


def season_options(config: dict[str, Any]) -> list[str]:
    options = config_value(
        config,
        "season_options",
        "season_categories",
        "seasons",
        default=DEFAULT_SEASONS,
    )
    if isinstance(options, (list, tuple)) and options:
        return [str(option) for option in options]
    return DEFAULT_SEASONS.copy()


def metric_value(metrics: dict[str, Any], name: str, default: float | None = None) -> float | None:
    keys = [name, name.upper(), name.lower(), name.replace("_", " ")]
    containers: list[dict[str, Any]] = [metrics]
    for key in ("test_metrics", "validation_metrics", "metrics", "best_model_metrics"):
        nested = metrics.get(key)
        if isinstance(nested, dict):
            containers.append(nested)
    for container in containers:
        for key in keys:
            value = container.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return default


def model_label(metrics: dict[str, Any], model: Any = None) -> str:
    for key in ("selected_model", "best_model", "model_name", "algorithm"):
        value = metrics.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if model is not None and hasattr(model, "named_steps"):
        steps = model.named_steps
        if steps:
            return type(next(reversed(steps.values()))).__name__
    return type(model).__name__ if model is not None else "Saved model"


def target_unit(config: dict[str, Any]) -> str:
    value = str(config_value(config, "target_unit", "prediction_unit", "unit", default="kWh"))
    return "units/kWh" if "units/kWh" in value else value


def make_input_frame(values: dict[str, Any], columns: list[str]) -> pd.DataFrame:
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
    row = {column: aliases.get(column, values.get(column, 0)) for column in columns}
    return pd.DataFrame([row], columns=columns)


def show_metric_cards(metrics: dict[str, Any], frame: pd.DataFrame, label: str) -> None:
    cv_rmse = metric_value(metrics, "CV_RMSE")
    test_rmse = metric_value(metrics, "RMSE")
    r2 = metric_value(metrics, "R2")
    cards = st.columns(4)
    cards[0].metric("Records", f"{len(frame):,}")
    cards[1].metric("Households", f"{frame['household_id'].nunique():,}")
    cards[2].metric("Selected model", label)
    cards[3].metric("Held-out RMSE", f"{test_rmse:.2f}" if test_rmse is not None else "—")
    if cv_rmse is not None or r2 is not None:
        st.caption(
            f"Grouped CV RMSE: {cv_rmse:.2f} · Held-out R²: {r2:.3f}"
            if cv_rmse is not None and r2 is not None
            else "Validation metrics are stored in artifacts/metrics.json."
        )


def page_overview(frame: pd.DataFrame, metrics: dict[str, Any], config: dict[str, Any], label: str) -> None:
    st.title("⚡ Household Energy Lab")
    st.write("A quick view of the household consumption dataset, the winning model, and the seasonal pattern.")
    show_metric_cards(metrics, frame, label)

    seasons = season_options(config)
    seasonal = (
        frame.groupby("season", as_index=False)["monthly_consumption"]
        .mean()
        .set_index("season")
        .reindex([season for season in seasons if season in frame["season"].unique()])
        .dropna()
    )
    monthly = frame.groupby("month", as_index=False)["monthly_consumption"].mean().set_index("month")

    left, right = st.columns(2)
    with left:
        st.subheader("Average consumption by season")
        st.bar_chart(seasonal.rename(columns={"monthly_consumption": "Average units/kWh"}), color="#18a999")
    with right:
        st.subheader("Monthly demand trend")
        st.line_chart(monthly.rename(columns={"monthly_consumption": "Average units/kWh"}), color="#f4a261")

    st.subheader("What the data says")
    summer_mean = seasonal.loc["Summer", "monthly_consumption"] if "Summer" in seasonal.index else None
    winter_mean = seasonal.loc["Winter", "monthly_consumption"] if "Winter" in seasonal.index else None
    if summer_mean is not None and winter_mean is not None:
        st.info(
            f"Summer averages {summer_mean:.1f} units/kWh versus {winter_mean:.1f} in winter. "
            "The model uses season and AC usage to capture this change."
        )
    st.caption("Use the sidebar to make a prediction, explore the records, or inspect residuals.")


def page_predict(model: Any | None, metrics: dict[str, Any], config: dict[str, Any], columns: list[str]) -> None:
    st.title("🔮 Predict next month")
    st.write("Enter a household profile and receive a model-backed monthly estimate.")
    if model is None:
        st.error("The saved model is unavailable. Run the notebook to create artifacts/best_model.joblib.")
        return

    seasons = season_options(config)
    with st.form("prediction_form"):
        st.subheader("Household profile")
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
                "Previous month's consumption (kWh)", min_value=0.0, max_value=10000.0, value=250.0, step=1.0
            )
        submitted = st.form_submit_button("Predict monthly consumption", type="primary")

    if not submitted:
        st.caption("The prediction uses the saved model and preprocessing pipeline.")
        return

    values = {
        "household_size": household_size,
        "number_of_rooms": rooms,
        "appliance_count": appliance_count,
        "ac_usage_hours": ac_usage_hours,
        "season": season,
        "previous_month_consumption": previous_consumption,
    }
    input_frame = make_input_frame(values, columns)
    try:
        prediction = float(model.predict(input_frame)[0])
    except Exception as exc:
        st.error(f"Prediction failed because the saved pipeline expected different columns: {exc}")
        return

    unit = target_unit(config)
    rmse = metric_value(metrics, "RMSE")
    lower = max(0.0, prediction - rmse) if rmse is not None else None
    upper = prediction + rmse if rmse is not None else None
    st.subheader("Prediction result")
    result_left, result_mid, result_right = st.columns(3)
    result_left.metric("Predicted monthly consumption", f"{prediction:,.2f} {unit}")
    result_mid.metric("Selected model", model_label(metrics, model))
    result_right.metric("Validation RMSE", f"{rmse:,.2f} {unit}" if rmse is not None else "—")
    if lower is not None and upper is not None:
        st.success(f"Typical planning range: **{lower:,.2f} to {upper:,.2f} {unit}**")
        st.caption("This is prediction ± validation RMSE, not a formal confidence interval.")
    st.subheader("Inputs used")
    st.dataframe(input_frame, hide_index=True, width="stretch")


def page_explore(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    st.title("📊 Explore the records")
    st.write("Use the filters to inspect how household attributes and season relate to consumption.")
    available_seasons = season_options(config)
    selected = st.multiselect("Show seasons", available_seasons, default=available_seasons)
    filtered = frame[frame["season"].isin(selected)].copy()
    if filtered.empty:
        st.warning("Choose at least one season to display the charts.")
        return

    st.caption(f"Showing {len(filtered):,} of {len(frame):,} household-month records.")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        box_data = [filtered.loc[filtered["season"] == season, "monthly_consumption"] for season in selected]
        box = ax.boxplot(box_data, patch_artist=True, showfliers=False)
        ax.set_xticks(range(1, len(selected) + 1))
        ax.set_xticklabels(selected)
        for patch, color in zip(box["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("")
        ax.set_ylabel("Monthly consumption (units/kWh)")
        ax.set_title("Consumption spread by season")
        st.pyplot(fig, clear_figure=True)
    with right:
        sample = filtered.sample(min(3500, len(filtered)), random_state=42)
        fig, ax = plt.subplots(figsize=(7, 4))
        for season, color in zip(selected, PALETTE):
            points = sample[sample["season"] == season]
            ax.scatter(
                points["previous_month_consumption"],
                points["monthly_consumption"],
                label=season,
                color=color,
                alpha=0.45,
                s=18,
                linewidths=0,
            )
        ax.set_xlabel("Previous month consumption")
        ax.set_ylabel("Current month consumption")
        ax.set_title("Previous month versus current month")
        ax.legend(title="Season", fontsize=8)
        st.pyplot(fig, clear_figure=True)

    feature_summary = (
        filtered.groupby("season")[["household_size", "number_of_rooms", "appliance_count", "ac_usage_hours", "monthly_consumption"]]
        .mean()
        .reindex(selected)
        .round(2)
    )
    st.subheader("Season summary")
    st.dataframe(feature_summary, width="stretch")


def page_diagnostics(frame: pd.DataFrame, metrics: dict[str, Any], model: Any | None, columns: list[str]) -> None:
    st.title("🧪 Model diagnostics")
    st.write("Compare the five regressors and inspect errors from the selected saved pipeline.")
    all_metrics = metrics.get("all_model_metrics", {})
    if isinstance(all_metrics, dict) and all_metrics:
        comparison = pd.DataFrame.from_dict(all_metrics, orient="index")
        comparison.index.name = "Model"
        comparison = comparison.sort_values("CV_RMSE")
        st.subheader("Model comparison")
        st.dataframe(comparison.round(3), width="stretch")
        st.bar_chart(comparison[["CV_RMSE"]].rename(columns={"CV_RMSE": "Grouped CV RMSE"}), color="#e76f51")

    if model is None:
        st.warning("The model artifact is unavailable, so residual diagnostics cannot be shown.")
        return

    diagnostics = frame[["household_id", "month", "season", "monthly_consumption"]].copy()
    diagnostics["predicted"] = model.predict(frame[columns])
    diagnostics["residual"] = diagnostics["monthly_consumption"] - diagnostics["predicted"]

    left, right = st.columns(2)
    sample = diagnostics.sample(min(3500, len(diagnostics)), random_state=42)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(sample["predicted"], sample["monthly_consumption"], alpha=0.35, s=18, color="#457b9d", linewidths=0)
        limits = [diagnostics[["predicted", "monthly_consumption"]].min().min(), diagnostics[["predicted", "monthly_consumption"]].max().max()]
        ax.plot(limits, limits, linestyle="--", color="#e76f51")
        ax.set_title("Actual versus predicted")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, clear_figure=True)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(sample["predicted"], sample["residual"], alpha=0.35, s=18, color="#18a999", linewidths=0)
        ax.axhline(0, linestyle="--", color="#e76f51")
        ax.set_title("Residuals versus predicted")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Residual summary by season")
    st.dataframe(diagnostics.groupby("season")["residual"].agg(["mean", "std"]).round(2), width="stretch")

    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.named_steps.get("model")
        if preprocessor is not None and estimator is not None and hasattr(preprocessor, "get_feature_names_out"):
            names = preprocessor.get_feature_names_out()
            if hasattr(estimator, "coef_"):
                values = np.abs(np.asarray(estimator.coef_).ravel())
                importance_label = "absolute coefficient"
            elif hasattr(estimator, "feature_importances_"):
                values = np.asarray(estimator.feature_importances_)
                importance_label = "importance"
            else:
                values = np.zeros(len(names))
                importance_label = "importance"
            influence = pd.DataFrame({"feature": names, importance_label: values}).sort_values(importance_label, ascending=False).head(10)
            st.subheader("Most influential transformed terms")
            st.bar_chart(influence.set_index("feature").sort_values(importance_label), color="#9b5de5")


def main() -> None:
    config = read_json(CONFIG_PATH)
    metrics = read_json(METRICS_PATH)
    columns = feature_columns(config)
    label = model_label(metrics)

    try:
        frame = load_data(str(DATA_PATH))
    except Exception as exc:
        frame = pd.DataFrame()
        st.error(f"The processed dataset could not be loaded: {exc}")

    model = None
    if MODEL_PATH.exists():
        try:
            model = load_model(str(MODEL_PATH))
            label = model_label(metrics, model)
        except Exception as exc:
            st.error(f"The saved model could not be loaded: {exc}")

    st.sidebar.title("⚡ Energy Lab")
    st.sidebar.caption("Household consumption prediction")
    page = st.sidebar.radio("Navigate", ["Overview", "Predict", "Explore data", "Model diagnostics"], index=0)
    st.sidebar.divider()
    st.sidebar.caption("Notebook-backed case study")
    st.sidebar.caption(f"Selected model: {label}")

    if frame.empty and page != "Predict":
        st.warning("The processed data is not available. Run scripts/build_dataset.py first.")
        return
    if page == "Overview":
        page_overview(frame, metrics, config, label)
    elif page == "Predict":
        page_predict(model, metrics, config, columns)
    elif page == "Explore data":
        page_explore(frame, config)
    else:
        page_diagnostics(frame, metrics, model, columns)


if __name__ == "__main__":
    main()
