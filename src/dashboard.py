"""
dashboard.py
-------------
Streamlit app showing predicted demand, baseline vs. optimised depots,
and the quantified improvement. Run with:

    streamlit run src/dashboard.py

Kept intentionally light (per the project's analysis-first weighting) —
this is meant to demonstrate the pipeline's output clearly in a viva, not
to be a polished product in its own right.
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be resolved when running via Streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pydeck as pdk
import pandas as pd

from src.demand_model import train_and_evaluate, predict_demand_all
from src.depot_siting import naive_baseline_depots, greedy_pmedian_depots, demand_weighted_avg_distance

st.set_page_config(page_title="ADV Last-Mile Delivery — Demand & Depot Siting", layout="wide")
st.title("Autonomous Last-Mile Delivery: Demand Prediction & Depot Siting")
st.caption("Birmingham case study — running on official ONS Census & IMD 2025 data (659 LSOAs)")

k = st.sidebar.slider("Number of depots", min_value=2, max_value=10, value=5)

@st.cache_data
def load_everything(k: int):
    results = train_and_evaluate()
    df = predict_demand_all(results["df_with_demand"], results["model"])
    baseline_depots = naive_baseline_depots(df, k)
    optimised_depots = greedy_pmedian_depots(df, k)
    baseline_dist = demand_weighted_avg_distance(df, baseline_depots)
    optimised_dist = demand_weighted_avg_distance(df, optimised_depots)
    return results, df, baseline_depots, optimised_depots, baseline_dist, optimised_dist

results, df, baseline_depots, optimised_depots, baseline_dist, optimised_dist = load_everything(k)

col1, col2, col3 = st.columns(3)
col1.metric("Demand model MAE improvement vs. baseline",
            f"{(1 - results['model_mae']/results['baseline_mae'])*100:.1f}%")
col2.metric("Naive depot placement — avg distance", f"{baseline_dist:.2f} km")
col3.metric("Optimised depot placement — avg distance", f"{optimised_dist:.2f} km",
            delta=f"-{(1 - optimised_dist/baseline_dist)*100:.1f}%")

st.subheader("Predicted demand & depot locations")

depot_df = df.loc[optimised_depots].copy()
demand_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["longitude", "latitude"],
    get_radius="predicted_demand * 2",
    get_fill_color=[30, 120, 200, 140],
    pickable=True,
)
depot_layer = pdk.Layer(
    "ScatterplotLayer",
    data=depot_df,
    get_position=["longitude", "latitude"],
    get_radius=250,
    get_fill_color=[220, 40, 40, 220],
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=float(df["latitude"].mean()),
    longitude=float(df["longitude"].mean()),
    zoom=10,
)

st.pydeck_chart(pdk.Deck(
    layers=[demand_layer, depot_layer],
    initial_view_state=view_state,
    tooltip={"text": "{lsoa_name}\nPredicted demand: {predicted_demand}"},
))
st.caption("Blue = predicted demand (size = relative demand). Red = optimised depot locations.")

st.subheader("Depot comparison")
st.dataframe(pd.DataFrame({
    "Strategy": ["Naive (top population)", "Optimised (greedy p-median)"],
    "Avg. demand-weighted distance (km)": [round(baseline_dist, 2), round(optimised_dist, 2)],
}))
