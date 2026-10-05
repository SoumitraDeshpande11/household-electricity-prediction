import json
from html import escape
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_PATH = ROOT / "data/processed/household_monthly_panel.csv"
COLORS = ["#5288af", "#e3a044", "#0d9da3", "#ba8368"]
LABELS = {
    "household_size": "Household size",
    "number_of_rooms": "Number of rooms",
    "appliance_count": "Appliance count",
    "ac_usage_hours": "AC usage",
    "previous_month_consumption": "Previous consumption",
    "season": "Season",
    "monthly_consumption": "Monthly consumption",
}

st.set_page_config(page_title="Household Electricity Predictor", page_icon="⚡", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/app.css').read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def read_csv(path, modified):
    return pd.read_csv(path)


@st.cache_data
def read_json(path, modified):
    return json.loads(Path(path).read_text())


@st.cache_resource
def read_model(path, modified):
    return joblib.load(path)


def csv_file(path):
    return read_csv(str(path), path.stat().st_mtime_ns)


def json_file(path):
    return read_json(str(path), path.stat().st_mtime_ns)


def chart_style(chart, height=270):
    return (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#607485", titleColor="#607485", labelFontSize=11,
            titleFontSize=11, titleFontWeight=500, domain=False, tickSize=0,
            gridColor="#eaf0f4", labelPadding=9, titlePadding=12,
        )
        .configure_legend(labelColor="#607485", title=None, orient="bottom")
    )


def show_chart(chart, height=270):
    st.altair_chart(chart_style(chart, height), width="stretch", theme=None)


def note(text):
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)


def apply_preset(presets):
    selected = st.session_state["household_preset"]
    if selected != "Custom profile":
        for key, value in presets[selected].items():
            st.session_state[f"input_{key}"] = value
        st.session_state.pop("forecast", None)


def profile_presets(frame):
    def sample_profile(size, season):
        rows = frame[(frame["household_size"] == size) & (frame["season"] == season)]
        values = rows.median(numeric_only=True)
        return {
            "household_size": size,
            "number_of_rooms": int(round(values["number_of_rooms"])),
            "appliance_count": int(round(values["appliance_count"])),
            "ac_usage_hours": round(float(values["ac_usage_hours"]), 1),
            "season": season,
            "previous_month_consumption": round(float(values["previous_month_consumption"]), 1),
        }
    return {
        "Typical household": sample_profile(4, "Summer"),
        "Small household": sample_profile(2, "Winter"),
        "Family with AC": sample_profile(6, "Summer"),
        "Custom profile": {},
    }


def forecast_tab(frame, model, metrics, config):
    presets = profile_presets(frame)
    for key, value in presets["Typical household"].items():
        st.session_state.setdefault(f"input_{key}", value)
    seasons = config["season_options"]
    inputs, output = st.columns([1.25, 1], gap="large")
    with inputs:
        st.subheader("Your household, next month")
        note("Load an example household or adjust the details to explore a forecast.")
        st.selectbox(
            "Load an example household", list(presets), key="household_preset",
            on_change=apply_preset, args=(presets,),
        )
        values = {}
        with st.form("electricity_forecast"):
            left, right = st.columns(2)
            specifications = [
                ("household_size", "People at home", 1),
                ("number_of_rooms", "Number of rooms", 1),
                ("appliance_count", "Appliance count", 1),
                ("ac_usage_hours", "AC usage · hours/month", 1.0),
                ("previous_month_consumption", "Previous month · kWh", 5.0),
            ]
            for index, (key, label, step) in enumerate(specifications):
                with (left if index < 3 else right):
                    bounds = frame[key].dropna()
                    if isinstance(step, int):
                        minimum, maximum = int(bounds.min()), int(bounds.max())
                    else:
                        minimum, maximum = float(bounds.min()), float(bounds.max())
                    values[key] = st.number_input(
                        label, min_value=minimum, max_value=maximum, step=step,
                        key=f"input_{key}",
                        help=f"Observed range: {minimum:g}–{maximum:g}.",
                    )
            with right:
                values["season"] = st.selectbox("Season", seasons, key="input_season")
            submitted = st.form_submit_button("Predict monthly consumption", type="primary", width="stretch")

        if submitted or "forecast" not in st.session_state:
            prediction = float(model.predict(pd.DataFrame([values], columns=config["feature_columns"]))[0])
            if not np.isfinite(prediction) or prediction < 0:
                st.error("This combination produced an invalid estimate. Try a profile closer to the examples.")
                return
            st.session_state["forecast"] = {"value": prediction, "inputs": values.copy(), "example": not submitted}
        st.caption("1 electricity unit = 1 kWh. Predictions follow the synthetic dataset's assumptions.")

    result = st.session_state["forecast"]
    prediction, used = result["value"], result["inputs"]
    rmse = metrics["test_metrics"]["RMSE"]
    previous = used["previous_month_consumption"]
    change = prediction - previous
    with output:
        st.subheader("Your monthly forecast")
        note("Example preview" if result["example"] else "Forecast from your submitted household profile")
        direction = "above" if change >= 0 else "below"
        st.markdown(
            f'<div class="forecast-card"><div class="forecast-label">Predicted electricity consumption</div>'
            f'<div class="forecast-value">{prediction:,.1f}<span>kWh / month</span></div>'
            f'<div class="forecast-range">Expected error range<br><strong>'
            f'{max(0, prediction-rmse):,.1f} – {prediction+rmse:,.1f} kWh</strong> '
            f'<span style="opacity:.7"> · ±{rmse:.1f} kWh</span></div>'
            f'<div class="forecast-change">{abs(change):.1f} kWh {direction} the previous month</div></div>',
            unsafe_allow_html=True,
        )
        comparison = pd.DataFrame({
            "Month": ["Previous month", "Predicted month"],
            "kWh": [previous, prediction],
            "color": ["#b6c8d4", "#087f8c"],
        })
        show_chart(
            alt.Chart(comparison).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=55).encode(
                x=alt.X("Month:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("kWh:Q", title="Consumption · kWh"),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=["Month:N", alt.Tooltip("kWh:Q", format=".1f")],
            ), 145,
        )
        st.caption(f"Selected model: {metrics['selected_model']}. Range = prediction ± held-out RMSE; coverage is not guaranteed.")

    st.divider()
    seasonal_inputs = pd.DataFrame([{**used, "season": season} for season in seasons], columns=config["feature_columns"])
    seasonal_inputs["Forecast"] = model.predict(seasonal_inputs)
    seasonal_inputs["Season"] = seasons
    usage_inputs = pd.DataFrame(
        [{**used, "ac_usage_hours": hours} for hours in np.linspace(0, frame["ac_usage_hours"].max(), 45)],
        columns=config["feature_columns"],
    )
    usage_inputs["Forecast"] = model.predict(usage_inputs)
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.subheader("The same home, across seasons")
        note("Only the season changes. AC hours and the previous month's reading stay fixed.")
        show_chart(
            alt.Chart(seasonal_inputs).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=45).encode(
                x=alt.X("Season:N", sort=seasons, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Forecast:Q", title="Predicted consumption · kWh"),
                color=alt.Color("Season:N", scale=alt.Scale(domain=seasons, range=COLORS), legend=None),
                tooltip=["Season:N", alt.Tooltip("Forecast:Q", format=".1f", title="Forecast · kWh")],
            ), 230,
        )
    with right, st.container(border=True):
        st.subheader("What if AC usage changes?")
        note("The selected household stays fixed while monthly AC hours change.")
        show_chart(
            alt.Chart(usage_inputs).mark_line(color="#087f8c", strokeWidth=3).encode(
                x=alt.X("ac_usage_hours:Q", title="AC usage · hours/month"),
                y=alt.Y("Forecast:Q", title="Predicted consumption · kWh", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("ac_usage_hours:Q", format=".0f", title="AC hours"), alt.Tooltip("Forecast:Q", format=".1f", title="Forecast · kWh")],
            ).interactive(), 230,
        )
    with st.expander("View the inputs used for this forecast"):
        st.dataframe(
            pd.DataFrame([used]).rename(columns=LABELS), hide_index=True, width="stretch"
        )


def explorer_tab(frame, config):
    st.subheader("Explore household consumption")
    note("Compare seasons, inspect consumption patterns, and follow one household over time.")
    filters, years = st.columns([2, 1])
    with filters:
        selected = st.multiselect("Seasons", config["season_options"], default=config["season_options"])
    with years:
        year = st.selectbox("Year", ["All years", *sorted(frame["month"].dt.year.unique().tolist())])
    filtered = frame[frame["season"].isin(selected)].copy()
    if year != "All years":
        filtered = filtered[filtered["month"].dt.year == year]
    if filtered.empty:
        st.info("Select at least one season with records in the chosen year.")
        return
    cards = st.columns(3)
    cards[0].metric("Household-month records", f"{len(filtered):,}")
    cards[1].metric("Average consumption", f"{filtered['monthly_consumption'].mean():.1f} kWh")
    cards[2].metric("Median consumption", f"{filtered['monthly_consumption'].median():.1f} kWh")
    st.write("")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Seasonal demand")
        note("Average monthly consumption across the selected records.")
        summary = filtered.groupby("season", as_index=False)["monthly_consumption"].mean()
        show_chart(
            alt.Chart(summary).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=48).encode(
                x=alt.X("season:N", title=None, sort=config["season_options"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("monthly_consumption:Q", title="Average consumption · kWh"),
                color=alt.Color("season:N", scale=alt.Scale(domain=config["season_options"], range=COLORS), legend=None),
                tooltip=["season:N", alt.Tooltip("monthly_consumption:Q", format=".1f", title="Average · kWh")],
            )
        )
    with right, st.container(border=True):
        st.subheader("Monthly demand trend")
        note("The average household reading for each month.")
        monthly = filtered.groupby("month", as_index=False)["monthly_consumption"].mean()
        base = alt.Chart(monthly).encode(
            x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=-25)),
            y=alt.Y("monthly_consumption:Q", title="Average consumption · kWh"),
            tooltip=[alt.Tooltip("month:T", title="Month", format="%b %Y"), alt.Tooltip("monthly_consumption:Q", title="Average · kWh", format=".1f")],
        )
        show_chart(base.mark_area(color="#087f8c", opacity=.09) + base.mark_line(color="#087f8c", strokeWidth=2.5))

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Which attribute tracks consumption?")
        feature = st.selectbox("Compare consumption with", list(LABELS)[:5], format_func=LABELS.get)
        sample = filtered.dropna(subset=[feature]).sample(min(2200, len(filtered.dropna(subset=[feature]))), random_state=42)
        show_chart(
            alt.Chart(sample).mark_circle(size=22, opacity=.4).encode(
                x=alt.X(f"{feature}:Q", title=LABELS[feature]),
                y=alt.Y("monthly_consumption:Q", title="Monthly consumption · kWh"),
                color=alt.Color("season:N", scale=alt.Scale(domain=config["season_options"], range=COLORS)),
                tooltip=["household_id:N", "season:N", alt.Tooltip(f"{feature}:Q", format=".1f", title=LABELS[feature]), alt.Tooltip("monthly_consumption:Q", format=".1f", title="Consumption · kWh")],
            ).interactive()
        )
        st.caption("The chart displays a reproducible sample of up to 2,200 readings.")
    with right, st.container(border=True):
        st.subheader("One household, month by month")
        household = st.selectbox("Household", sorted(filtered["household_id"].unique()))
        history = filtered[filtered["household_id"] == household]
        show_chart(
            alt.Chart(history).mark_line(point=True, color="#087f8c", strokeWidth=2.5).encode(
                x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=-25)),
                y=alt.Y("monthly_consumption:Q", title="Monthly consumption · kWh", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("monthly_consumption:Q", format=".1f", title="Consumption · kWh"), "season:N"],
            )
        )
    with st.expander("Browse records and download the filtered dataset"):
        st.dataframe(filtered.head(200).rename(columns=LABELS), hide_index=True, width="stretch")
        st.caption("Preview shows the first 200 records. The download contains every selected record.")
        st.download_button("Download filtered CSV", filtered.to_csv(index=False), "household_consumption.csv", "text/csv")
    missing = frame.isna().sum()
    with st.expander("Missing readings before preprocessing"):
        st.dataframe(missing[missing > 0].rename_axis("Feature").reset_index(name="Missing values").replace({"Feature": LABELS}), hide_index=True, width="stretch")
        st.caption("The notebook repairs feature gaps within each household, then trains the preprocessing pipeline.")


def insights_tab(metrics, config):
    st.subheader("Model performance")
    note("Five regressors, grouped five-fold cross-validation, and a test split of complete households.")
    performance = metrics["test_metrics"]
    for column, key, label in zip(st.columns(4), ["R2", "RMSE", "MAE", "MSE"], ["Test R²", "Test RMSE · kWh", "Test MAE · kWh", "Test MSE · kWh²"]):
        column.metric(label, f"{performance[key]:.3f}" if key == "R2" else f"{performance[key]:,.2f}")

    comparison = pd.DataFrame.from_dict(metrics["all_model_metrics"], orient="index").reset_index(names="Model")
    comparison = comparison.sort_values("CV_RMSE")
    cv_best = comparison.iloc[0]["CV_RMSE"]
    insight(f"<strong>{escape(metrics['selected_model'])}</strong> has the lowest mean CV RMSE: <strong>{cv_best:.2f} kWh</strong>. The held-out test RMSE is <strong>{performance['RMSE']:.2f} kWh</strong> across {metrics['test_households']} households.")
    left, right = st.columns([1.1, 1.2])
    with left, st.container(border=True):
        st.subheader("Which model has the lowest error?")
        note("Shorter bars mean lower average error across five grouped folds.")
        comparison["Winner"] = comparison["Model"] == metrics["selected_model"]
        bars = alt.Chart(comparison).encode(
            y=alt.Y("Model:N", sort=alt.SortField(field="CV_RMSE", order="ascending"), title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("CV_RMSE:Q", title="Mean cross-validation RMSE · kWh", scale=alt.Scale(domain=[0, float(comparison["CV_RMSE"].max()) * 1.2])),
        )
        show_chart(
            bars.mark_bar(cornerRadiusEnd=5, height=24).encode(color=alt.condition(alt.datum.Winner, alt.value("#087f8c"), alt.value("#b5c8d4")), tooltip=["Model:N", alt.Tooltip("CV_RMSE:Q", format=".2f")])
            + bars.mark_text(align="left", dx=7, color="#244354").encode(text=alt.Text("CV_RMSE:Q", format=".2f")), 230,
        )
    with right, st.container(border=True):
        st.subheader("All evaluation metrics")
        note("Test performance and grouped CV error, in one comparison.")
        display = comparison[["Model", "R2", "MSE", "RMSE", "MAE", "CV_RMSE"]].rename(columns={"R2": "R²", "CV_RMSE": "CV RMSE"})
        st.dataframe(display.style.format({name: "{:.2f}" for name in display.columns if name != "Model"}), hide_index=True, width="stretch", height=235)

    predictions_path = ARTIFACTS / "test_predictions.csv"
    if not predictions_path.exists():
        st.info("Run the notebook to create the held-out diagnostic charts.")
        return
    tests = csv_file(predictions_path)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Actual versus predicted")
        note("Each point is a held-out household-month reading. The line marks a perfect forecast.")
        bounds = [float(tests[["actual", "predicted"]].min().min()), float(tests[["actual", "predicted"]].max().max())]
        points = alt.Chart(tests).mark_circle(size=16, opacity=.3, color="#087f8c").encode(
            x=alt.X("predicted:Q", title="Predicted consumption · kWh"),
            y=alt.Y("actual:Q", title="Actual consumption · kWh"),
            tooltip=["household_id:N", "season:N", alt.Tooltip("actual:Q", format=".1f"), alt.Tooltip("predicted:Q", format=".1f")],
        )
        diagonal = alt.Chart(pd.DataFrame({"x": bounds, "y": bounds})).mark_line(strokeDash=[5, 5], color="#bd8c51").encode(x="x:Q", y="y:Q")
        show_chart(points + diagonal)
    with right, st.container(border=True):
        st.subheader("Residuals versus predicted")
        note("Residual = actual − predicted. An even spread around zero is desirable.")
        points = alt.Chart(tests).mark_circle(size=16, opacity=.3, color="#5288af").encode(
            x=alt.X("predicted:Q", title="Predicted consumption · kWh"),
            y=alt.Y("residual:Q", title="Residual · kWh"),
            tooltip=[alt.Tooltip("predicted:Q", format=".1f"), alt.Tooltip("residual:Q", format=".1f"), "season:N"],
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(strokeDash=[5, 5], color="#bd8c51").encode(y="zero:Q")
        show_chart(points + zero)

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Which inputs matter most?")
        note("Increase in test RMSE when an input is shuffled. Higher values mean greater predictive reliance.")
        importance_path = ARTIFACTS / "feature_importance.csv"
        if importance_path.exists():
            influence = csv_file(importance_path)
            influence["Input"] = influence["feature"].map(LABELS)
            show_chart(
                alt.Chart(influence).mark_bar(cornerRadiusEnd=5, color="#087f8c", height=22).encode(
                    y=alt.Y("Input:N", sort=alt.SortField(field="rmse_increase", order="descending"), title=None),
                    x=alt.X("rmse_increase:Q", title="Increase in RMSE · kWh"),
                    tooltip=["Input:N", alt.Tooltip("rmse_increase:Q", format=".2f", title="RMSE increase")],
                ), 220,
            )
            st.caption("Five shuffles per input. Correlated attributes can share predictive information; this is not a causal ranking.")
    with right, st.container(border=True):
        st.subheader("Seasonal performance")
        note("How closely seasonal test averages match the forecasts.")
        season_means = tests.groupby("season", as_index=False)[["actual", "predicted"]].mean().melt("season", var_name="Reading", value_name="kWh")
        show_chart(
            alt.Chart(season_means).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("season:N", sort=config["season_options"], title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="Reading:N",
                y=alt.Y("kWh:Q", title="Mean monthly consumption · kWh"),
                color=alt.Color("Reading:N", scale=alt.Scale(domain=["actual", "predicted"], range=["#b5c8d4", "#087f8c"])),
                tooltip=["season:N", "Reading:N", alt.Tooltip("kWh:Q", format=".1f")],
            ), 220,
        )
    bins = pd.qcut(tests["predicted"], 5, duplicates="drop")
    spread = tests.groupby(bins, observed=True)["residual"].std()
    ratio = float(spread.max() / spread.min())
    insight(f"Residual spread is {ratio:.2f}× greater in the widest prediction band than the narrowest. This suggests variance changes with demand; inspect the residual plot when discussing capacity planning.")
    with st.expander("See test predictions"):
        st.dataframe(tests.head(100).round(2), hide_index=True, width="stretch")


def guide_tab(frame, metrics):
    st.subheader("From a household profile to a forecast")
    note("A complete regression workflow for Case Study 27.")
    steps = [
        ("01 · INPUT", "Six household attributes", "People, rooms, appliances, AC hours, season, and the previous monthly reading."),
        ("02 · MODEL", metrics["selected_model"], "A saved preprocessing and regression pipeline converts the household profile into an estimate."),
        ("03 · OUTPUT", "Consumption + error range", "Monthly energy in kWh, alongside a planning range using the held-out RMSE."),
    ]
    for column, (number, title, body) in zip(st.columns(3), steps):
        with column:
            st.markdown(f'<div class="step-card"><div class="step-number">{number}</div><h3>{escape(title)}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Dataset & generation")
        st.write(f"{len(frame):,} synthetic household-month records: 500 households over 36 months, generated with NumPy/Pandas and seed 42.")
        st.write("Consumption combines household attributes, summer AC demand, previous consumption, a household efficiency factor, and noise.")
        st.write("The committed table uses the generator's offline scale of 650 kWh. A local UCI/Kaggle download can optionally calibrate that scale.")
        st.link_button("Kaggle reference dataset", "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set")
    with right, st.container(border=True):
        st.subheader("What the results support")
        st.write(f"Five required regressors are compared. {metrics['selected_model']} has the lowest mean grouped CV RMSE.")
        st.write("One-hot season encoding and previous-month consumption help capture repeated patterns. Whole households are reserved for test evaluation.")
        st.write("Synthetic relationships and missing weather data limit real-world accuracy. Utility forecasting requires real meter and weather validation.")
    with st.expander("How to read the metrics"):
        st.write("RMSE: typical error magnitude in kWh, with larger errors weighted more heavily. MAE: average absolute error in kWh. MSE: squared error in kWh². R²: fit relative to predicting the test average.")
        st.write("The displayed range is prediction ± RMSE. It has no guaranteed probability coverage.")
    with st.expander("Missing readings & model selection"):
        st.write("Missing feature values are injected reproducibly. The notebook repairs gaps, encodes season, trains Linear, Polynomial, Decision Tree, Random Forest, and Gradient Boosting regressors, then reports test and grouped five-fold results.")


def main():
    required = [DATA_PATH, ARTIFACTS / "best_model.joblib", ARTIFACTS / "metrics.json", ARTIFACTS / "feature_config.json"]
    if not all(path.exists() for path in required):
        st.error("Project data or model artifacts are missing. Run the dataset builder and notebook first.")
        st.stop()
    try:
        frame = csv_file(DATA_PATH)
        frame["month"] = pd.to_datetime(frame["month"])
        metrics = json_file(ARTIFACTS / "metrics.json")
        config = json_file(ARTIFACTS / "feature_config.json")
        path = ARTIFACTS / "best_model.joblib"
        model = read_model(str(path), path.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"The project files could not be loaded: {error}")
        st.stop()

    with st.sidebar:
        st.markdown('<div class="brand">⚡ Household<span> Energy</span></div>', unsafe_allow_html=True)
        st.caption("Monthly consumption, made clearer.")
        st.divider()
        st.write("Explore household demand and estimate the next month's electricity use.")
        for label, value in [
            ("SELECTED MODEL", metrics["selected_model"]),
            ("DATASET", "18,000 synthetic records"),
            ("VALIDATION", "5-fold grouped cross-validation"),
        ]:
            st.markdown(f'<div class="side-label">{label}</div><div class="side-value">{escape(value)}</div>', unsafe_allow_html=True)
        st.divider()
        st.metric("Test RMSE", f"{metrics['test_metrics']['RMSE']:.2f} kWh")
        st.markdown('<div class="side-note">Academic case study · predictions follow the generated household data.</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><div class="eyebrow">Machine Learning · Case Study 27</div>'
        '<h1>Household Electricity<br>Predictor</h1>'
        '<p>Understand your household’s energy demand.<br>Predict the next month and explore what drives the difference.</p>'
        '<div class="hero-chips"><span>500 households</span><span>36 months</span><span>5 regression models</span></div>'
        '<svg class="hero-art" viewBox="0 0 300 220" aria-hidden="true">'
        '<circle cx="200" cy="120" r="95" fill="none" stroke="#a7e5e5" stroke-width="1"/>'
        '<circle cx="200" cy="120" r="75" fill="none" stroke="#a7e5e5" stroke-width="1"/>'
        '<path d="M100 145 L100 90 L160 42 L220 90 L220 145 M135 145 L135 108 L185 108 L185 145" fill="none" stroke="#e7ffff" stroke-width="4" stroke-linejoin="round"/>'
        '<path d="M30 170 H110 L128 148 L145 189 L168 150 L185 170 H270" fill="none" stroke="#a7e5e5" stroke-width="4" stroke-linejoin="round"/>'
        '</svg></div>', unsafe_allow_html=True,
    )
    forecast, explorer, insights, guide = st.tabs(["Predict consumption", "Data explorer", "Model insights", "Project guide"])
    with forecast:
        forecast_tab(frame, model, metrics, config)
    with explorer:
        explorer_tab(frame, config)
    with insights:
        insights_tab(metrics, config)
    with guide:
        guide_tab(frame, metrics)
    st.markdown('<div class="footer">Household Electricity Consumption Prediction · Machine Learning, Semester V · Synthetic household data</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
