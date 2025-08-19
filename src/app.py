import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

# Absolute imports from src package
from src.utils import read_data, basic_clean, add_time_features, summarize
from src.features import kmeans_hotspots, engineer_classifier_features
from src.config import REPORTS_DIR, MODELS_DIR, N_CLUSTERS, DATA_DIR, DATA_FILE

st.set_page_config(page_title="Crime Analytics & Forecasting", layout="wide")
st.title("🔎 Crime Pattern Analytics & Forecasting")

# Load & prep
df = read_data(os.path.join(DATA_DIR, DATA_FILE))
df = basic_clean(df)
df = add_time_features(df)
meta = summarize(df)

# Ensure datetime column is proper type
df["datetime"] = pd.to_datetime(df["datetime"])

# Sidebar filters
st.sidebar.header("Filters")
if not df.empty:
    y0, y1 = int(df["year"].min()), int(df["year"].max())
    yr_range = st.sidebar.slider("Year range", y0, y1, (y0, y1))
    df = df[(df["year"] >= yr_range[0]) & (df["year"] <= yr_range[1])]

    types = ["All"] + sorted(df["crime_type"].dropna().unique().tolist())
    type_choice = st.sidebar.selectbox("Crime type", options=types, index=0)
    if type_choice != "All":
        df = df[df["crime_type"] == type_choice]

# Dataset summary
st.subheader("Dataset summary")
st.json(meta)

# Time series
if not df.empty:
    ts = df.set_index("datetime").resample("M")["crime_type"].count().reset_index()
    fig_ts = px.line(ts, x="datetime", y="crime_type", title="Monthly Incident Counts")
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("No data.")

# Hotspots
st.subheader("📍 Hotspots")
coords, meta_k = kmeans_hotspots(df, k=N_CLUSTERS)
if not coords.empty:
    fig_map = px.scatter_mapbox(coords, lat="latitude", lon="longitude", color="cluster", zoom=10, height=500)
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No coordinates available for hotspot map.")

# Heatmap
st.subheader("🔥 Crime Heatmap")
if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
    fig_heat = px.density_mapbox(
        df, lat="latitude", lon="longitude", radius=10,
        center=dict(lat=df["latitude"].mean(), lon=df["longitude"].mean()),
        zoom=10, mapbox_style="stamen-terrain"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# Top 5 Risky Areas
st.subheader("🏙️ Top 5 Risky Areas")
if "area" in df.columns:
    area_counts = df["area"].value_counts().head(5).reset_index()
    area_counts.columns = ["Area", "Incidents"]
    st.bar_chart(area_counts.set_index("Area"))

# High-risk prediction demo
st.subheader("⚠️ High-Risk Prediction Demo")
try:
    clf = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
    X, y = engineer_classifier_features(df)
    if len(X) > 0:
        y_hat = clf.predict(X)
        share = round((y_hat.sum() / len(y_hat)) * 100, 2)
        st.metric("Predicted High-Risk Share", f"{share}%")
    else:
        st.info("Not enough rows to score.")
except Exception:
    st.warning("⚠️ Train the classifier first: `py -m src.train`")

# Forecasts
st.subheader("📈 Forecasts")
overall_csv = os.path.join(REPORTS_DIR, "forecast_overall.csv")
if os.path.exists(overall_csv):
    fco = pd.read_csv(overall_csv, parse_dates=["date"])
    fig_fc = px.line(fco, x="date", y="forecast_count", title="Forecast — Overall")
    st.plotly_chart(fig_fc, use_container_width=True)
else:
    st.info("Run forecasting: `py -m src.forecast`")

# Download filtered dataset
st.subheader("⬇️ Download Reports")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data (CSV)", csv, "filtered_data.csv", "text/csv")
