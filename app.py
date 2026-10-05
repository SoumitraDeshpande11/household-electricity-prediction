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
COLORS = ["#40d6c5", "#86b8e2", "#efb65f", "#cf9e86"]
LABELS = {"previous_30_day_kwh":"Previous 30 days · kWh","previous_7_day_mean_kwh":"Previous 7-day average · kWh/day","previous_day_kwh":"Previous day · kWh","previous_90_day_mean_kwh":"Previous 90-day average · kWh/day"}

st.set_page_config(page_title="Electricity Load Studio", page_icon="⚡", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/app.css').read_text()}</style>", unsafe_allow_html=True)

@st.cache_data
def read_csv(path, modified, **kwargs): return pd.read_csv(path, **kwargs)
@st.cache_data
def read_json(path, modified): return json.loads(Path(path).read_text())
@st.cache_resource
def read_model(path, modified): return joblib.load(path)
def csv_file(path, **kwargs): return read_csv(str(path), path.stat().st_mtime_ns, **kwargs)
def json_file(path): return read_json(str(path), path.stat().st_mtime_ns)
def style(c, h=280):
    return c.properties(height=h, background="transparent").configure_view(stroke=None).configure_axis(labelColor="#a1b4c7", titleColor="#a1b4c7", domain=False, tickSize=0, gridColor="#243244", labelPadding=8).configure_legend(labelColor="#a1b4c7", title=None, orient="bottom")
def chart(c, h=280): st.altair_chart(style(c,h), width="stretch", theme=None)
def note(v): st.markdown(f'<div class="section-note">{v}</div>', unsafe_allow_html=True)
def insight(v): st.markdown(f'<div class="insight">{v}</div>', unsafe_allow_html=True)
def calendar_features(d):
    m=d.month; s="Winter" if m in (12,1,2) else "Spring" if m in (3,4,5) else "Summer" if m in (6,7,8) else "Autumn"
    return {"month_sin":np.sin(2*np.pi*m/12),"month_cos":np.cos(2*np.pi*m/12),"season":s}

def apply_preset(presets):
    name=st.session_state["client_preset"]
    if name!="Custom history":
        p=presets[name]
        for k,v in p["inputs"].items(): st.session_state[f"input_{k}"]=float(v)
        st.session_state["forecast_date"]=date.fromisoformat(p["forecast_date"])
        st.session_state["client_label"]=p["client_id"]; st.session_state["forecast_client"]=p["client_id"]; st.session_state.pop("forecast_result",None)

def predict_tab(model, metrics, config, tests):
    presets={p["name"]:p for p in config["presets"]}; default=config["presets"][0]
    for k,v in default["inputs"].items(): st.session_state.setdefault(f"input_{k}",float(v))
    st.session_state.setdefault("forecast_date",date.fromisoformat(default["forecast_date"]))
    st.session_state.setdefault("client_label",default["client_id"])
    left,right=st.columns([1.2,1],gap="large")
    with left:
        st.subheader("Forecast a measured client"); note("Use a real client-history preset or enter four history summaries. Calendar features follow the forecast date.")
        st.selectbox("Load a measured example",[*(presets),"Custom history"],key="client_preset",on_change=apply_preset,args=(presets,))
        clients = config.get("client_ids", [])
        selected_client = st.session_state.get("forecast_client", default["client_id"])
        if selected_client not in clients:
            selected_client = default["client_id"]
        preset_is_custom = st.session_state.get("client_preset", "Custom history") == "Custom history"
        client = st.selectbox("Measured client", clients, index=clients.index(selected_client), disabled=not preset_is_custom, help="Preset profiles use their recorded client. Choose Custom history to select another client.")
        if not preset_is_custom:
            st.caption(f"Preset inputs are measured history from {presets[st.session_state['client_preset']]['client_id']}. Choose Custom history to change the client.")
        with st.form("forecast_form"):
            a,b=st.columns(2); vals={}
            for i,k in enumerate(config["input_features"]):
                with (a if i<2 else b):
                    lim=config["bounds"][k]; vals[k]=st.number_input(LABELS[k],min_value=0.0,step=1.0,key=f"input_{k}",help=f"Observed range: {lim['min']:.1f} to {lim['max']:.1f} kWh.")
            d=st.date_input("First day of forecast",key="forecast_date",min_value=date(2011,1,1),max_value=date(2035,12,31))
            submitted=st.form_submit_button("Forecast next 30 days",type="primary",width="stretch")
        vals.update(calendar_features(d))
        if submitted or "forecast_result" not in st.session_state:
            pred=max(0.0,float(model.predict(pd.DataFrame([vals],columns=config["feature_columns"]))[0]))
            st.session_state["forecast_result"]={"prediction":pred,"inputs":vals.copy(),"date":d,"client":client,"example":not submitted}
    result=st.session_state["forecast_result"]; pred,used=result["prediction"],result["inputs"]
    client_tests=tests[tests.client_id.astype(str)==str(result["client"])] if "client_id" in tests.columns else tests.iloc[0:0]
    if len(client_tests)>=3:
        rmse=float(np.sqrt(np.mean((client_tests.actual-client_tests.predicted)**2)))
        error_scope=f"{result['client']} holdout ({len(client_tests)} origins)"
    else: rmse=float(metrics["test_metrics"]["RMSE"]); error_scope="pooled holdout"
    with right:
        st.subheader("Projected demand"); note(f"{result['client']} · {result['date']:%d %b %Y} – {(result['date']+timedelta(days=29)):%d %b %Y} · {used['season']}")
        st.markdown(f'<div class="forecast-card"><div class="forecast-label">Predicted electricity consumption</div><div class="forecast-value">{pred:,.0f}<span>kWh / 30 days</span></div><div class="forecast-range">Planning range<br><strong>{max(0,pred-rmse):,.0f} – {pred+rmse:,.0f} kWh</strong><span style="opacity:.7"> · ±{rmse:,.0f} kWh · {error_scope}</span></div></div>',unsafe_allow_html=True)
        comp=pd.DataFrame({"Period":["Previous 30 days","Forecast 30 days"],"kWh":[used["previous_30_day_kwh"],pred],"color":["#68849d","#40d6c5"]})
        chart(alt.Chart(comp).mark_bar(cornerRadiusTopLeft=6,cornerRadiusTopRight=6,size=60).encode(x=alt.X("Period:N",title=None),y=alt.Y("kWh:Q",title="Energy · kWh"),color=alt.Color("color:N",scale=None,legend=None),tooltip=["Period:N",alt.Tooltip("kWh:Q",format=",.0f")]),180)
        st.caption(f"Selected by CV: {metrics['selected_model']}. Final holdout winner: {metrics['holdout_winner']}.")
        if result["example"]: st.caption("Preset values come from an actual measured client-history row.")
    st.divider(); scenarios=[]
    for s,m in zip(config["season_options"],[1,4,7,10]):
        row={**used,**calendar_features(date(result["date"].year,m,15))}; scenarios.append({**row,"Season":s})
    sf=pd.DataFrame(scenarios); sf["Forecast"]=model.predict(sf[config["feature_columns"]]); key="previous_7_day_mean_kwh"; lim=config["bounds"][key]
    sens=pd.DataFrame([{**used,key:v} for v in np.linspace(lim["min"],lim["max"],40)]); sens["Forecast"]=model.predict(sens[config["feature_columns"]])
    a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader("Season scenarios"); note("Same measured history; only the forecast calendar month changes.")
        chart(alt.Chart(sf).mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(x=alt.X("Season:N",sort=config["season_options"],title=None),y=alt.Y("Forecast:Q",title="Next 30 days · kWh"),color=alt.Color("Season:N",scale=alt.Scale(domain=config["season_options"],range=COLORS),legend=None),tooltip=["Season:N",alt.Tooltip("Forecast:Q",format=",.0f")]),240)
    with b,st.container(border=True):
        st.subheader("History sensitivity"); note("Model sensitivity to recent daily average; not a causal experiment.")
        chart(alt.Chart(sens).mark_line(color="#40d6c5",strokeWidth=3).encode(x=alt.X(f"{key}:Q",title="Previous 7-day average · kWh/day"),y=alt.Y("Forecast:Q",title="Next 30 days · kWh"),tooltip=[alt.Tooltip(f"{key}:Q",format=",.1f"),alt.Tooltip("Forecast:Q",format=",.0f")]).interactive(),240)

def explorer_tab(frame,config,metadata):
    st.subheader("Explore 370 real client meters"); note(f"{metadata['clients']:,} UCI clients · 15-minute kW readings aggregated to {metadata['daily_dates']:,} complete dates.")
    a,b,c=st.columns(3)
    with a: year=st.selectbox("Year",["All years",*sorted(frame.year.unique())])
    with b: seasons=st.multiselect("Season",config["season_options"],default=config["season_options"])
    with c: client=st.selectbox("Client",["All clients",*sorted(frame.client_id.unique())])
    view=frame[frame.season.isin(seasons)].copy()
    if year!="All years": view=view[view.year==year]
    if client!="All clients": view=view[view.client_id==client]
    cards=st.columns(3); cards[0].metric("Client-days",f"{len(view):,}"); cards[1].metric("Mean daily demand",f"{view.energy_kwh.mean():,.1f} kWh"); cards[2].metric("Clients in view",f"{view.client_id.nunique():,}")
    a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader("Demand over time"); daily=view.groupby("date",as_index=False).energy_kwh.mean()
        chart(alt.Chart(daily.sample(min(700,len(daily)),random_state=42)).mark_line(color="#40d6c5").encode(x=alt.X("date:T",title=None),y=alt.Y("energy_kwh:Q",title="Mean daily energy · kWh"),tooltip=[alt.Tooltip("date:T",format="%d %b %Y"),alt.Tooltip("energy_kwh:Q",format=",.1f")]),280)
    with b,st.container(border=True):
        st.subheader("Seasonal demand"); summary=view.groupby("season",as_index=False).energy_kwh.mean()
        chart(alt.Chart(summary).mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(x=alt.X("season:N",sort=config["season_options"],title=None),y=alt.Y("energy_kwh:Q",title="Mean daily energy · kWh"),color=alt.Color("season:N",scale=alt.Scale(domain=config["season_options"],range=COLORS),legend=None),tooltip=["season:N",alt.Tooltip("energy_kwh:Q",format=",.1f")]),280)
    if client=="All clients":
        client_means=frame.groupby("client_id",as_index=False).energy_kwh.mean(); client_means["tier"]=pd.qcut(client_means.energy_kwh,3,labels=["Lower demand","Middle demand","Higher demand"])
        a,b=st.columns(2)
        with a,st.container(border=True):
            st.subheader("Client demand distribution")
            chart(alt.Chart(client_means).mark_bar(color="#86b8e2").encode(x=alt.X("energy_kwh:Q",bin=alt.Bin(maxbins=30),title="Mean client daily kWh"),y=alt.Y("count():Q",title="Clients"),tooltip=[alt.Tooltip("count():Q",title="Clients")]),260)
        with b,st.container(border=True):
            st.subheader("Demand tiers"); tiers=client_means.groupby("tier",observed=True,as_index=False).energy_kwh.mean()
            chart(alt.Chart(tiers).mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(x=alt.X("tier:N",title=None),y=alt.Y("energy_kwh:Q",title="Mean daily kWh"),color=alt.Color("tier:N",scale=alt.Scale(range=COLORS),legend=None),tooltip=["tier:N",alt.Tooltip("energy_kwh:Q",format=",.1f")]),260)
    with st.expander("Browse and download filtered daily data"):
        st.dataframe(view.head(250),hide_index=True,width="stretch"); st.download_button("Download filtered CSV",view.to_csv(index=False),"uci_daily_consumption.csv","text/csv")

def insights_tab(metrics,config):
    st.subheader("Model lab"); note("Five regressors, chronological cross-validation, later-date holdout, and an excluded-client check.")
    m=metrics["test_metrics"]; cols=st.columns(4)
    for col,k,label in zip(cols,["R2","RMSE","MAE","MSE"],["Holdout R²","Holdout RMSE · kWh","Holdout MAE · kWh","Holdout MSE · kWh²"]): col.metric(label,f"{m[k]:,.3f}" if k=="R2" else f"{m[k]:,.0f}")
    table=pd.DataFrame.from_dict(metrics["all_model_metrics"],orient="index").reset_index(names="Model").sort_values("CV_RMSE")
    insight(f"<strong>{escape(metrics['selected_model'])}</strong> has the lowest CV RMSE ({table.iloc[0].CV_RMSE:,.0f} kWh). <strong>{escape(metrics['holdout_winner'])}</strong> has the lowest final holdout RMSE ({metrics['all_model_metrics'][metrics['holdout_winner']]['RMSE']:,.0f} kWh).")
    a,b=st.columns([1.1,1.2])
    with a,st.container(border=True):
        st.subheader("Cross-validation RMSE")
        chart(alt.Chart(table).mark_bar(cornerRadiusEnd=5).encode(y=alt.Y("Model:N",sort=alt.SortField(field="CV_RMSE",order="ascending"),title=None),x=alt.X("CV_RMSE:Q",title="Mean CV RMSE · kWh"),color=alt.condition(alt.datum.Model==metrics["selected_model"],alt.value("#40d6c5"),alt.value("#68849d")),tooltip=["Model:N",alt.Tooltip("CV_RMSE:Q",format=",.0f")]),260)
    with b,st.container(border=True):
        st.subheader("All model metrics"); display=table[["Model","R2","MSE","RMSE","MAE","CV_RMSE"]].rename(columns={"R2":"R²","CV_RMSE":"CV RMSE"})
        st.dataframe(display.style.format({c:"{:,.2f}" for c in display.columns if c!="Model"}),hide_index=True,width="stretch",height=245)
    insight(f"On the same {metrics['baseline_comparable_rows']:,} holdout rows, selected-model RMSE is {metrics['model_on_baseline_rows']['RMSE']:,.0f} kWh versus {metrics['baseline_metrics']['RMSE']:,.0f} kWh for repeating the previous 30 days. Excluded-client check: {metrics['heldout_client_count']} clients, RMSE {metrics['heldout_client_metrics']['RMSE']:,.0f} kWh.")
    tests=csv_file(ARTIFACTS/"test_predictions.csv",parse_dates=["forecast_date","forecast_end"]); sample=tests.sample(min(2500,len(tests)),random_state=42)
    a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader("Actual versus predicted"); chart(alt.Chart(sample).mark_circle(size=18,opacity=.35,color="#40d6c5").encode(x=alt.X("predicted:Q",title="Predicted · kWh"),y=alt.Y("actual:Q",title="Actual · kWh"),tooltip=["client_id:N","forecast_date:T",alt.Tooltip("actual:Q",format=",.0f"),alt.Tooltip("predicted:Q",format=",.0f")]),280)
    with b,st.container(border=True):
        st.subheader("Residuals versus predicted"); chart(alt.Chart(sample).mark_circle(size=18,opacity=.35,color="#86b8e2").encode(x=alt.X("predicted:Q",title="Predicted · kWh"),y=alt.Y("residual:Q",title="Residual · kWh"),tooltip=["client_id:N","forecast_date:T",alt.Tooltip("residual:Q",format=",.0f")]),280)
    importance=csv_file(ARTIFACTS/"feature_importance.csv"); a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader("Permutation importance"); chart(alt.Chart(importance).mark_bar(color="#40d6c5").encode(y=alt.Y("feature:N",sort="-x",title=None),x=alt.X("rmse_increase:Q",title="RMSE increase · kWh"),tooltip=["feature:N",alt.Tooltip("rmse_increase:Q",format=",.0f")]),260)
    with b,st.container(border=True):
        st.subheader("Error by season"); season=tests.groupby("season",as_index=False).agg(actual=("actual","mean"),predicted=("predicted","mean")).melt("season",var_name="Reading",value_name="kWh")
        chart(alt.Chart(season).mark_bar().encode(x=alt.X("season:N",sort=config["season_options"],title=None),xOffset="Reading:N",y=alt.Y("kWh:Q",title="Mean next-30-day kWh"),color=alt.Color("Reading:N",scale=alt.Scale(range=["#68849d","#40d6c5"]),legend=None),tooltip=["season:N","Reading:N",alt.Tooltip("kWh:Q",format=",.0f")]),260)
    with st.expander("Holdout predictions"):
        view=tests.head(200).copy(); num=view.select_dtypes(include="number").columns; view[num]=view[num].round(2); st.dataframe(view,hide_index=True,width="stretch"); st.download_button("Download holdout predictions",tests.to_csv(index=False),"holdout_predictions.csv","text/csv")

def guide_tab(metrics,metadata):
    st.subheader("Project guide"); note("Real multi-client readings → daily kWh → causal history features → five regressors → Streamlit forecast.")
    for col,item in zip(st.columns(3),[("01 · SOURCE","370 client meters","UCI 15-minute readings from 2011–2014."),("02 · MODEL",metrics["selected_model"],"Chosen by chronological CV RMSE."),("03 · OUTPUT","Next 30 days","Forecast plus a pooled holdout RMSE planning range.")]):
        with col: st.markdown(f'<div class="step-card"><div class="step-number">{item[0]}</div><h3>{escape(item[1])}</h3><p>{item[2]}</p></div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a,st.container(border=True):
        st.subheader("Source and method"); st.write(f"{metadata['raw_rows']:,} quarter-hour readings across {metadata['clients']:,} clients. Values are average kW; daily kWh is the sum divided by four."); st.write(f"{metadata['daily_dates']:,} complete dates create {metadata['forecast_examples']:,} client-date forecast examples."); st.link_button("Open UCI dataset","https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014")
    with b,st.container(border=True):
        st.subheader("Interpretation"); st.write(f"Cross-validation selected {metrics['selected_model']}; {metrics['holdout_winner']} had the best final holdout RMSE. Client IDs are not model features."); st.write("This is energy forecasting, not peak-load forecasting. The source has no household demographics, weather, occupancy, or appliance inventory.")
    with st.expander("Reproducibility"):
        st.code("python -m jupyter nbconvert --to notebook --execute --inplace household_electricity_prediction.ipynb\npython validate_project.py\nstreamlit run app.py"); st.write("The notebook is the complete training and preprocessing source. The raw archive is ignored because it exceeds GitHub's file-size limit.")

def main():
    required=[DATA_PATH,ARTIFACTS/"best_model.joblib",ARTIFACTS/"metrics.json",ARTIFACTS/"feature_config.json",ROOT/"data/processed/dataset_metadata.json",ARTIFACTS/"test_predictions.csv",ARTIFACTS/"feature_importance.csv"]
    if not all(p.exists() for p in required): st.error("Run all notebook cells before starting the dashboard."); st.stop()
    frame=csv_file(DATA_PATH); frame["date"]=pd.to_datetime(frame["date"])
    metrics=json_file(ARTIFACTS/"metrics.json"); config=json_file(ARTIFACTS/"feature_config.json"); metadata=json_file(ROOT/"data/processed/dataset_metadata.json")
    path=ARTIFACTS/"best_model.joblib"; model=read_model(str(path),path.stat().st_mtime_ns)
    with st.sidebar:
        st.markdown('<div class="brand">⚡ Load<span> Studio</span></div>',unsafe_allow_html=True); st.caption("Real meters. Clearer forecasts."); st.divider()
        for label,value in [("DATASET",f"{metadata['clients']:,} clients"),("READINGS",f"{metadata['raw_rows']:,} quarter-hour"),("SELECTED MODEL",metrics["selected_model"]),("CV GAP","35 days")]: st.markdown(f'<div class="side-label">{label}</div><div class="side-value">{escape(value)}</div>',unsafe_allow_html=True)
        st.divider(); st.metric("Holdout RMSE",f"{metrics['test_metrics']['RMSE']:,.0f} kWh"); st.markdown('<div class="side-note">UCI ElectricityLoadDiagrams20112014 · real multi-client readings · 30-day horizon.</div>',unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="eyebrow">Machine Learning · Case Study 27</div><h1>Electricity Load<br>Studio</h1><p>Forecast client demand from measured history.<br>Explore the patterns behind the next 30 days.</p><div class="hero-chips"><span>370 real clients</span><span>140k interval readings</span><span>5 regression models</span></div><svg class="hero-art" viewBox="0 0 300 220" aria-hidden="true"><circle cx="200" cy="120" r="95" fill="none" stroke="#a7e5e5" stroke-width="1"/><circle cx="200" cy="120" r="75" fill="none" stroke="#a7e5e5" stroke-width="1"/><path d="M30 170 H110 L128 148 L145 189 L168 150 L185 170 H270" fill="none" stroke="#a7e5e5" stroke-width="4"/></svg></div>',unsafe_allow_html=True)
    tabs=st.tabs(["Forecast","Data explorer","Model lab","Project guide"])
    with tabs[0]: predict_tab(model,metrics,config,csv_file(ARTIFACTS/"test_predictions.csv"))
    with tabs[1]: explorer_tab(frame,config,metadata)
    with tabs[2]: insights_tab(metrics,config)
    with tabs[3]: guide_tab(metrics,metadata)
    st.markdown('<div class="footer">Electricity Load Studio · Semester V Machine Learning · UCI real multi-client readings</div>',unsafe_allow_html=True)

if __name__=="__main__": main()
