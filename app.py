import json
from datetime import date, timedelta
from html import escape
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA_PATH = ROOT / "data/processed/daily_consumption.csv"
COLORS = ["#86b8e2", "#efb65f", "#40c9bd", "#cf9e86"]
LABELS = {
    "previous_30_day_kwh": "Previous 30 days · kWh",
    "previous_7_day_mean_kwh": "Past 7 days · average kWh/day",
    "previous_day_kwh": "Previous day · kWh",
    "kitchen_7_day_mean_kwh": "Kitchen · past-week kWh/day",
    "laundry_7_day_mean_kwh": "Laundry · past-week kWh/day",
    "heating_ac_7_day_mean_kwh": "Water heater + AC · past-week kWh/day",
    "month_sin": "Month · sine",
    "month_cos": "Month · cosine",
    "season": "Season",
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
            labelColor="#a1b4c7", titleColor="#a1b4c7", labelFontSize=11,
            titleFontSize=11, titleFontWeight=500, domain=False, tickSize=0,
            gridColor="#243244", labelPadding=9, titlePadding=12,
        )
        .configure_legend(labelColor="#a1b4c7", title=None, orient="bottom")
    )


def show_chart(chart, height=270):
    st.altair_chart(chart_style(chart, height), width="stretch", theme=None)


def note(text):
    st.markdown(f'<div class="section-note">{text}</div>', unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)



def calendar_features(forecast_date):
    month = forecast_date.month
    season = (
        "Winter" if month in [12, 1, 2] else
        "Spring" if month in [3, 4, 5] else
        "Summer" if month in [6, 7, 8] else "Autumn"
    )
    return {"month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12), "season": season}


def apply_preset(presets):
    selected = st.session_state["history_preset"]
    if selected != "Custom history":
        preset = presets[selected]
        for key, value in preset["inputs"].items():
            st.session_state[f"input_{key}"] = float(value)
        st.session_state["forecast_date"] = date.fromisoformat(preset["forecast_date"])
        st.session_state.pop("real_forecast", None)


def forecast_tab(model, metrics, config):
    presets = {preset["name"]: preset for preset in config["presets"]}
    default = config["presets"][0]
    for key, value in default["inputs"].items():
        st.session_state.setdefault(f"input_{key}", float(value))
    st.session_state.setdefault("forecast_date", date.fromisoformat(default["forecast_date"]))
    inputs, output = st.columns([1.25, 1], gap="large")
    with inputs:
        st.subheader("From recent readings to a forecast")
        note("Load an actual recorded history or enter meter totals. Season follows the forecast date.")
        st.selectbox(
            "Load a measured example", [*presets, "Custom history"],
            key="history_preset", on_change=apply_preset, args=(presets,),
        )
        values = {}
        with st.form("electricity_forecast"):
            left, right = st.columns(2)
            for index, key in enumerate(config["input_features"]):
                with (left if index < 3 else right):
                    limits = config["bounds"][key]
                    help_text = f"Recorded range: {limits['min']:.1f}–{limits['max']:.1f}."
                    if key == "heating_ac_7_day_mean_kwh":
                        help_text += " Combined water heater and AC circuit; separate AC hours are unavailable."
                    values[key] = st.number_input(
                        LABELS[key], min_value=0.0, step=1.0 if index == 0 else .1,
                        key=f"input_{key}", help=help_text,
                    )
            forecast_date = st.date_input(
                "First day of the forecast", key="forecast_date",
                min_value=date(2000, 1, 1), max_value=date(2100, 12, 31),
                help="Use readings from the days immediately before this date.",
            )
            submitted = st.form_submit_button("Predict next 30 days", type="primary", width="stretch")
        values.update(calendar_features(forecast_date))
        if submitted or "real_forecast" not in st.session_state:
            prediction = float(model.predict(pd.DataFrame([values], columns=config["feature_columns"]))[0])
            if not np.isfinite(prediction) or prediction < 0:
                st.error("This history produced an invalid estimate. Try values closer to a measured example.")
                return
            st.session_state["real_forecast"] = {
                "value": prediction, "inputs": values.copy(),
                "date": forecast_date, "example": not submitted,
            }
        st.caption("1 electricity unit = 1 kWh. This is a 30-day horizon, not an exact calendar billing month.")
        outside = [
            LABELS[key] for key in config["input_features"]
            if not config["bounds"][key]["min"] <= values[key] <= config["bounds"][key]["max"]
        ]
        if outside:
            st.info("Outside the recorded range: " + ", ".join(outside) + ". Accuracy here has not been evaluated.")

    result = st.session_state["real_forecast"]
    prediction, used = result["value"], result["inputs"]
    rmse = metrics["test_metrics"]["RMSE"]
    previous = used["previous_30_day_kwh"]
    change = prediction - previous
    with output:
        st.subheader("Your next 30 days")
        end = result["date"] + timedelta(days=29)
        note(f"{result['date']:%d %b %Y} – {end:%d %b %Y} · {used['season']}")
        direction = "above" if change >= 0 else "below"
        st.markdown(
            f'<div class="forecast-card"><div class="forecast-label">Predicted electricity consumption</div>'
            f'<div class="forecast-value">{prediction:,.1f}<span>kWh / 30 days</span></div>'
            f'<div class="forecast-range">Expected error range<br><strong>'
            f'{max(0, prediction-rmse):,.1f} – {prediction+rmse:,.1f} kWh</strong>'
            f'<span style="opacity:.7"> · ±{rmse:.1f} kWh</span></div>'
            f'<div class="forecast-change">{abs(change):.1f} kWh {direction} the previous 30 days</div></div>',
            unsafe_allow_html=True,
        )
        comparison = pd.DataFrame({
            "Period": ["Previous 30 days", "Forecast 30 days"],
            "kWh": [previous, prediction], "color": ["#68849d", "#40d6c5"],
        })
        show_chart(
            alt.Chart(comparison).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=55).encode(
                x=alt.X("Period:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("kWh:Q", title="Consumption · kWh"),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=["Period:N", alt.Tooltip("kWh:Q", format=".1f")],
            ), 145,
        )
        st.caption(f"Selected model: {metrics['selected_model']}. Range = prediction ± holdout RMSE; probability coverage is not guaranteed.")
        if result["example"]:
            st.caption("Preview uses an actual historical input row and the refitted deployment model. Test performance is shown separately.")

    st.divider()
    seasonal_rows = []
    for season, month in zip(config["season_options"], [1, 4, 7, 10]):
        row = {**used, **calendar_features(date(result["date"].year, month, 15))}
        seasonal_rows.append({**row, "Season": season})
    seasonal_inputs = pd.DataFrame(seasonal_rows)
    seasonal_inputs["Forecast"] = model.predict(seasonal_inputs[config["feature_columns"]])
    key = "previous_7_day_mean_kwh"
    limits = config["bounds"][key]
    sensitivity = pd.DataFrame([{**used, key: value} for value in np.linspace(limits["min"], limits["max"], 45)])
    sensitivity["Forecast"] = model.predict(sensitivity[config["feature_columns"]])
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("How calendar inputs change the estimate")
        note("January, April, July, and October scenarios hold historical readings fixed.")
        show_chart(
            alt.Chart(seasonal_inputs).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=45).encode(
                x=alt.X("Season:N", sort=config["season_options"], title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Forecast:Q", title="Predicted next 30 days · kWh"),
                color=alt.Color("Season:N", scale=alt.Scale(domain=config["season_options"], range=COLORS), legend=None),
                tooltip=["Season:N", alt.Tooltip("Forecast:Q", format=".1f")],
            ), 230,
        )
    with right, st.container(border=True):
        st.subheader("Sensitivity to recent daily demand")
        note("Only the past-week average changes. This explores the model, not a causal effect.")
        show_chart(
            alt.Chart(sensitivity).mark_line(color="#40d6c5", strokeWidth=3).encode(
                x=alt.X(f"{key}:Q", title="Past-week average · kWh/day"),
                y=alt.Y("Forecast:Q", title="Predicted next 30 days · kWh", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip(f"{key}:Q", format=".1f"), alt.Tooltip("Forecast:Q", format=".1f")],
            ).interactive(), 230,
        )
    st.caption("These scenarios vary correlated inputs independently. They illustrate sensitivity, not a full-year forecast.")
    with st.expander("View the submitted inputs"):
        st.dataframe(pd.DataFrame([used]).rename(columns=LABELS), hide_index=True, width="stretch")


def explorer_tab(frame, config, metadata):
    st.subheader("Explore the real meter readings")
    note("One measured home in Sceaux, France · December 2006 to November 2010.")
    filters, years = st.columns([2, 1])
    with filters:
        selected = st.multiselect("Seasons", config["season_options"], default=config["season_options"])
    with years:
        year = st.selectbox("Year", ["All years", *sorted(frame["year"].unique().tolist())])
    filtered = frame[frame["season"].isin(selected)].copy()
    if year != "All years":
        filtered = filtered[filtered["year"] == year]
    usable = filtered.dropna(subset=["energy_kwh"])
    if usable.empty:
        st.info("Select at least one season with usable readings in the chosen year.")
        return
    cards = st.columns(3)
    cards[0].metric("Usable recorded days", f"{len(usable):,}")
    cards[1].metric("Average daily demand", f"{usable.energy_kwh.mean():.1f} kWh")
    cards[2].metric("Original power gaps", f"{metadata['missing_power_minutes']:,} minutes")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Observed seasonal demand")
        note("Mean daily energy across the selected real observations.")
        summary = usable.groupby("season", as_index=False).energy_kwh.mean()
        show_chart(
            alt.Chart(summary).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=48).encode(
                x=alt.X("season:N", title=None, sort=config["season_options"], axis=alt.Axis(labelAngle=0)),
                y=alt.Y("energy_kwh:Q", title="Average daily consumption · kWh"),
                color=alt.Color("season:N", scale=alt.Scale(domain=config["season_options"], range=COLORS), legend=None),
                tooltip=["season:N", alt.Tooltip("energy_kwh:Q", format=".1f")],
            )
        )
    with right, st.container(border=True):
        st.subheader("Daily demand over time")
        note("Gaps mark excluded days; long outages remain missing.")
        base = alt.Chart(filtered).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %y", labelAngle=-25)),
            y=alt.Y("energy_kwh:Q", title="Daily demand · kWh"),
            tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("energy_kwh:Q", format=".1f")],
        )
        show_chart(base.mark_line(color="#40d6c5", strokeWidth=1.5).interactive())
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Where the measured energy goes")
        note("Three monitored circuits; these do not cover every household load.")
        circuit_names = {"kitchen_kwh": "Kitchen", "laundry_kwh": "Laundry", "heating_ac_kwh": "Water heater + AC"}
        circuits = filtered[list(circuit_names)].mean().rename_axis("circuit").reset_index(name="kWh")
        circuits["Circuit"] = circuits.circuit.map(circuit_names)
        show_chart(
            alt.Chart(circuits).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#40d6c5", size=45).encode(
                x=alt.X("Circuit:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("kWh:Q", title="Mean circuit demand · kWh/day"),
                tooltip=["Circuit:N", alt.Tooltip("kWh:Q", format=".2f")],
            )
        )
    with right, st.container(border=True):
        st.subheader("Meter coverage")
        note("Percentage of the day's 1,440 minutes observed before repair.")
        coverage = filtered.assign(coverage=filtered.observed_minutes / 1440 * 100)
        show_chart(
            alt.Chart(coverage).mark_line(color="#86b8e2", strokeWidth=1.5).encode(
                x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %y")),
                y=alt.Y("coverage:Q", title="Observed minutes · %", scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("coverage:Q", format=".1f"), "repaired_minutes:Q"],
            ).interactive()
        )
    insight(
        f"Of {metadata['missing_power_minutes']:,} missing power minutes, <strong>{metadata['repaired_power_minutes']:,}</strong> "
        f"were filled from the preceding reading for at most five minutes. Longer gaps stay missing. "
        f"<strong>{metadata['usable_energy_days']:,}</strong> days support energy targets."
    )
    with st.expander("Browse and download the selected daily readings"):
        st.dataframe(filtered.head(200), hide_index=True, width="stretch")
        st.download_button("Download selected daily CSV", filtered.to_csv(index=False), "measured_daily_consumption.csv", "text/csv")


def insights_tab(metrics, config):
    st.subheader("Model performance")
    note("Five regressors, five expanding time-series folds with a 30-day gap, and a later-period holdout from the same home.")
    performance = metrics["test_metrics"]
    for column, key, label in zip(st.columns(4), ["R2", "RMSE", "MAE", "MSE"], ["Test R²", "Test RMSE · kWh", "Test MAE · kWh", "Test MSE · kWh²"]):
        column.metric(label, f"{performance[key]:.3f}" if key == "R2" else f"{performance[key]:,.2f}")

    comparison = pd.DataFrame.from_dict(metrics["all_model_metrics"], orient="index").reset_index(names="Model")
    comparison = comparison.sort_values("CV_RMSE")
    cv_best = comparison.iloc[0]["CV_RMSE"]
    insight(f"<strong>{escape(metrics['selected_model'])}</strong> has the lowest mean CV RMSE: <strong>{cv_best:.2f} kWh</strong>. The held-out test RMSE is <strong>{performance['RMSE']:.2f} kWh</strong> across {metrics['test_rows']} forecast origins.")
    left, right = st.columns([1.1, 1.2])
    with left, st.container(border=True):
        st.subheader("Which model wins cross-validation?")
        note("Shorter bars mean lower average error across five chronological folds.")
        comparison["Winner"] = comparison["Model"] == metrics["selected_model"]
        bars = alt.Chart(comparison).encode(
            y=alt.Y("Model:N", sort=alt.SortField(field="CV_RMSE", order="ascending"), title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("CV_RMSE:Q", title="Mean cross-validation RMSE · kWh", scale=alt.Scale(domain=[0, float(comparison["CV_RMSE"].max()) * 1.2])),
        )
        show_chart(
            bars.mark_bar(cornerRadiusEnd=5, height=24).encode(color=alt.condition(alt.datum.Winner, alt.value("#40d6c5"), alt.value("#68849d")), tooltip=["Model:N", alt.Tooltip("CV_RMSE:Q", format=".2f")])
            + bars.mark_text(align="left", dx=7, color="#d6e5ef").encode(text=alt.Text("CV_RMSE:Q", format=".2f")), 230,
        )
    with right, st.container(border=True):
        st.subheader("All evaluation metrics")
        note("Test performance and time-series CV error, in one comparison.")
        display = comparison[["Model", "R2", "MSE", "RMSE", "MAE", "CV_RMSE"]].rename(columns={"R2": "R²", "CV_RMSE": "CV RMSE"})
        st.dataframe(display.style.format({name: "{:.2f}" for name in display.columns if name != "Model"}), hide_index=True, width="stretch", height=235)


    baseline = metrics["baseline_metrics"]["RMSE"]
    comparable = metrics["model_on_baseline_rows"]["RMSE"]
    outcome = "lower" if metrics["beats_persistence_baseline"] else "higher"
    insight(
        f"On the same <strong>{metrics['baseline_comparable_rows']} origins</strong> with complete prior history, "
        f"ML RMSE is <strong>{comparable:.2f} kWh</strong>, {outcome} than repeating the previous 30 days "
        f"(<strong>{baseline:.2f} kWh</strong>). Adjacent targets overlap, making successive errors correlated."
    )
    with st.expander("Cross-validation fold variability"):
        st.dataframe(comparison[["Model", "CV_RMSE", "CV_RMSE_std", "CV_MAE", "CV_R2"]].round(2), hide_index=True, width="stretch")

    predictions_path = ARTIFACTS / "test_predictions.csv"
    if not predictions_path.exists():
        st.info("Run the notebook to create the held-out diagnostic charts.")
        return
    tests = csv_file(predictions_path)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Actual versus predicted")
        note("Each point is a held-out next-30-day energy total. The line marks a perfect forecast.")
        bounds = [float(tests[["actual", "predicted"]].min().min()), float(tests[["actual", "predicted"]].max().max())]
        points = alt.Chart(tests).mark_circle(size=16, opacity=.3, color="#40d6c5").encode(
            x=alt.X("predicted:Q", title="Predicted consumption · kWh"),
            y=alt.Y("actual:Q", title="Actual consumption · kWh"),
            tooltip=["forecast_date:N", "season:N", alt.Tooltip("actual:Q", format=".1f"), alt.Tooltip("predicted:Q", format=".1f")],
        )
        diagonal = alt.Chart(pd.DataFrame({"x": bounds, "y": bounds})).mark_line(strokeDash=[5, 5], color="#efb65f").encode(x="x:Q", y="y:Q")
        show_chart(points + diagonal)
    with right, st.container(border=True):
        st.subheader("Residuals versus predicted")
        note("Residual = actual − predicted. A constant spread is not established here.")
        points = alt.Chart(tests).mark_circle(size=16, opacity=.3, color="#86b8e2").encode(
            x=alt.X("predicted:Q", title="Predicted consumption · kWh"),
            y=alt.Y("residual:Q", title="Residual · kWh"),
            tooltip=[alt.Tooltip("predicted:Q", format=".1f"), alt.Tooltip("residual:Q", format=".1f"), "season:N"],
        )
        zero = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(strokeDash=[5, 5], color="#efb65f").encode(y="zero:Q")
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
                alt.Chart(influence).mark_bar(cornerRadiusEnd=5, color="#40d6c5", height=22).encode(
                    y=alt.Y("Input:N", sort=alt.SortField(field="rmse_increase", order="descending"), title=None, axis=alt.Axis(labelLimit=230)),
                    x=alt.X("rmse_increase:Q", title="Increase in RMSE · kWh"),
                    tooltip=["Input:N", alt.Tooltip("rmse_increase:Q", format=".2f", title="RMSE increase")],
                ), 280,
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
                y=alt.Y("kWh:Q", title="Mean 30-day consumption · kWh"),
                color=alt.Color("Reading:N", scale=alt.Scale(domain=["actual", "predicted"], range=["#68849d", "#40d6c5"])),
                tooltip=["season:N", "Reading:N", alt.Tooltip("kWh:Q", format=".1f")],
            ), 220,
        )
    bins = pd.qcut(tests["predicted"], 5, duplicates="drop")
    spread = tests.groupby(bins, observed=True)["residual"].std()
    ratio = float(spread.max() / spread.min())
    insight(f"Residual spread is {ratio:.2f}× greater in the widest prediction band than the narrowest. This suggests variance changes with demand; inspect the residual plot when discussing capacity planning.")

    history = tests.assign(forecast_date=pd.to_datetime(tests.forecast_date)).melt(
        "forecast_date", value_vars=["actual", "predicted"], var_name="Reading", value_name="kWh",
    )
    st.subheader("Holdout forecasts over time")
    show_chart(
        alt.Chart(history).mark_line(strokeWidth=2).encode(
            x=alt.X("forecast_date:T", title="Forecast start date"),
            y=alt.Y("kWh:Q", title="Next 30-day energy · kWh", scale=alt.Scale(zero=False)),
            color=alt.Color("Reading:N", scale=alt.Scale(domain=["actual", "predicted"], range=["#68849d", "#40d6c5"])),
            tooltip=["Reading:N", alt.Tooltip("forecast_date:T", format="%d %b %Y"), alt.Tooltip("kWh:Q", format=".1f")],
        ).interactive(), 230,
    )
    st.caption(f"Lag-one residual correlation: {metrics['residual_lag1_correlation']:.3f}. Errors are not independent.")
    with st.expander("See test predictions and repair sensitivity"):
        sensitivity = metrics["fully_observed_test_metrics"]
        if sensitivity:
            st.caption(f"{metrics['fully_observed_test_rows']} targets contain no repaired power minutes; RMSE on this small subset is {sensitivity['RMSE']:.2f} kWh.")
        st.download_button("Download holdout predictions", tests.to_csv(index=False), "holdout_predictions.csv", "text/csv")
        st.dataframe(tests.head(100).round(2), hide_index=True, width="stretch")



def guide_tab(metrics, metadata):
    st.subheader("Real readings, explained end to end")
    note("Case Study 27 · the notebook contains the complete ML workflow.")
    steps = [
        ("01 · SOURCE", "Measured energy history", "Daily totals from actual minute readings. Six history inputs and date-derived calendar features."),
        ("02 · MODEL", metrics["selected_model"], "Training-only chronological cross-validation selects one of the five required regressors."),
        ("03 · OUTPUT", "Next 30 days + error range", "Energy in kWh and a rough ±holdout-RMSE planning range; no guaranteed interval coverage."),
    ]
    for column, (number, title, body) in zip(st.columns(3), steps):
        with column:
            st.markdown(f'<div class="step-card"><div class="step-number">{number}</div><h3>{escape(title)}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Dataset & provenance")
        st.write(f"{metadata['raw_rows']:,} real minute readings from one home in Sceaux, France, collected in 2006–2010.")
        st.write(f"{metadata['usable_energy_days']:,} usable energy days give {metadata['forecast_examples']:,} labelled forecast origins. These are overlapping periods of one home.")
        st.write("The exact Kaggle archive is included in the repository. The notebook verifies its hash and never generates substitute data.")
        st.link_button("Kaggle dataset", "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set")
        st.link_button("Original UCI documentation", "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption")
        st.caption("Hebrail & Berard (2006), UCI, DOI 10.24432/C58K54 · CC BY 4.0.")
    with right, st.container(border=True):
        st.subheader("What the results support")
        st.write("Validation measures later periods of the recorded home. Accuracy on new households remains unknown.")
        st.write("Household size, rooms, appliance count, separate AC hours, and weather are absent. The model uses available measured history instead.")
        st.write("Weather, occupancy, recent readings, and validation across many homes are needed for utility deployment. A 30-day energy total does not estimate peak grid load.")
    with st.expander("Missing readings, season encoding, and leakage prevention"):
        st.write("Only short gaps are forward-filled, for up to five minutes. Unresolved daily energy gaps exclude those days from targets. Historical feature gaps are median-imputed inside each training fold.")
        st.write("Month uses sine/cosine and season uses one-hot encoding. Every history feature ends before its forecast starts. A 30-day gap separates training targets from validation targets.")
        st.write("Adjacent validation targets still overlap and their errors correlate. Fold variability and the one-home limitation matter when interpreting accuracy.")
    with st.expander("How to read the metrics"):
        st.write("RMSE: root mean squared error in kWh, sensitive to large errors. MAE: mean absolute error in kWh. MSE: squared error in kWh². R²: fit relative to predicting the holdout mean.")
        st.write("The displayed range is prediction ± holdout RMSE, with the lower limit clipped at zero. It is not a calibrated prediction interval.")
    st.link_button("View the complete notebook", "https://github.com/SoumitraDeshpande11/household-electricity-prediction/blob/main/household_electricity_prediction.ipynb")


def main():
    required = [
        DATA_PATH, ARTIFACTS / "best_model.joblib", ARTIFACTS / "metrics.json",
        ARTIFACTS / "feature_config.json", ROOT / "data/processed/dataset_metadata.json",
        ARTIFACTS / "test_predictions.csv", ARTIFACTS / "feature_importance.csv",
    ]
    if not all(path.exists() for path in required):
        st.error("Data or artifacts are missing. Run all cells in household_electricity_prediction.ipynb first.")
        st.stop()
    frame = csv_file(DATA_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    metrics = json_file(ARTIFACTS / "metrics.json")
    config = json_file(ARTIFACTS / "feature_config.json")
    metadata = json_file(ROOT / "data/processed/dataset_metadata.json")
    path = ARTIFACTS / "best_model.joblib"
    model = read_model(str(path), path.stat().st_mtime_ns)
    with st.sidebar:
        st.markdown('<div class="brand">⚡ Household<span> Energy</span></div>', unsafe_allow_html=True)
        st.caption("Real readings. A clearer forecast.")
        st.divider()
        st.write("Explore measured demand and estimate energy over the next 30 days.")
        for label, value in [
            ("SELECTED MODEL", metrics["selected_model"]),
            ("DATASET", f"{metadata['raw_rows']:,} real minute readings"),
            ("VALIDATION", "5-fold time-series CV · 30-day gap"),
        ]:
            st.markdown(f'<div class="side-label">{label}</div><div class="side-value">{escape(value)}</div>', unsafe_allow_html=True)
        st.divider()
        st.metric("Holdout RMSE", f"{metrics['test_metrics']['RMSE']:.2f} kWh")
        st.markdown('<div class="side-note">Academic case study · one French home, 2006–2010. Accuracy on other homes has not been measured.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><div class="eyebrow">Machine Learning · Case Study 27</div>'
        '<h1>Household Electricity<br>Predictor</h1>'
        '<p>Explore real meter readings.<br>Forecast the next 30 days and see what drives the estimate.</p>'
        '<div class="hero-chips"><span>2.07M meter readings</span><span>1 measured home</span><span>5 regression models</span></div>'
        '<svg class="hero-art" viewBox="0 0 300 220" aria-hidden="true">'
        '<circle cx="200" cy="120" r="95" fill="none" stroke="#a7e5e5" stroke-width="1"/>'
        '<circle cx="200" cy="120" r="75" fill="none" stroke="#a7e5e5" stroke-width="1"/>'
        '<path d="M100 145 L100 90 L160 42 L220 90 L220 145 M135 145 L135 108 L185 108 L185 145" fill="none" stroke="#e7ffff" stroke-width="4" stroke-linejoin="round"/>'
        '<path d="M30 170 H110 L128 148 L145 189 L168 150 L185 170 H270" fill="none" stroke="#a7e5e5" stroke-width="4" stroke-linejoin="round"/>'
        '</svg></div>', unsafe_allow_html=True,
    )
    forecast, explorer, insights, guide = st.tabs(["Predict consumption", "Data explorer", "Model insights", "Project guide"])
    with forecast:
        forecast_tab(model, metrics, config)
    with explorer:
        explorer_tab(frame, config, metadata)
    with insights:
        insights_tab(metrics, config)
    with guide:
        guide_tab(metrics, metadata)
    st.markdown('<div class="footer">Household Electricity Consumption Prediction · Semester V · Real Kaggle/UCI meter readings</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
