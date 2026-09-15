"""
Rossmann Sales Forecasting — Streamlit Serving App
Task 2.7 (model serving, using Streamlit instead of MLflow) + Task 3 (dashboard)

Run with: streamlit run streamlit_app.py
"""
import glob
import os
import sys

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import gzip
sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import (
    engineer_features, apply_store_target_encoding, to_model_matrix
)

st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

DATA_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.dirname(__file__)


@st.cache_resource
def load_bundle():
    candidates = sorted(glob.glob(os.path.join(MODELS_DIR, "serving-bundle-*.pkl.")))
    if not candidates:
        st.error("No serving bundle found in ../models. Run build_serving_bundle.py first.")
        st.stop()
    bundle = joblib.load(candidates[-1])
    bundle["_path"] = candidates[-1]
    return bundle


@st.cache_data
def load_store_table():
    path = os.path.join(DATA_DIR, "store_clean_for_app.csv")
    return pd.read_csv(path)


def predict(df_raw, bundle, store_table):
    """df_raw needs: Store, Date, DayOfWeek (optional, derived), Promo, SchoolHoliday,
    StateHoliday, Open (optional, default 1)."""
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1  # Monday=1 ... Sunday=7
    if "Open" not in df.columns:
        df["Open"] = 1
    if "StateHoliday" not in df.columns:
        df["StateHoliday"] = "0"
    if "SchoolHoliday" not in df.columns:
        df["SchoolHoliday"] = 0
    if "Promo" not in df.columns:
        df["Promo"] = 0
    df["StateHoliday"] = df["StateHoliday"].astype(str)

    df = df.merge(store_table, on="Store", how="left")
    missing_stores = df["StoreType"].isna().sum()
    if missing_stores:
        st.warning(f"{missing_stores} row(s) reference a Store ID not in store.csv — dropped.")
        df = df.dropna(subset=["StoreType"])

    df_fe = engineer_features(df, bundle["holiday_dates"])
    df_fe = apply_store_target_encoding(df_fe, bundle["store_enc_map"], bundle["global_mean"])
    X = to_model_matrix(df_fe)

    sales_pred = bundle["sales_booster"].predict(
        X, num_iteration=getattr(bundle["sales_booster"], "best_iteration", None))
    sales_lower = bundle["sales_model_lower"].predict(X)
    sales_upper = bundle["sales_model_upper"].predict(X)
    cust_pred = bundle["customers_booster"].predict(
        X, num_iteration=bundle["customers_booster"].best_iteration)

    is_closed = df_fe["Open"].values == 0
    sales_pred = np.where(is_closed, 0, sales_pred)
    sales_lower = np.where(is_closed, 0, sales_lower)
    sales_upper = np.where(is_closed, 0, sales_upper)
    cust_pred = np.where(is_closed, 0, cust_pred)

    out = df_fe[["Store", "Date", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "Open"]].copy()
    out["PredictedSales"] = sales_pred.round(0)
    out["PredictedSales_Low90"] = np.clip(sales_lower, 0, None).round(0)
    out["PredictedSales_High90"] = sales_upper.round(0)
    out["PredictedCustomers"] = np.clip(cust_pred, 0, None).round(0)
    return out


# ----------------------------- UI -----------------------------
st.title("🏪 Rossmann Store Sales Forecast")
st.caption(
    "Serves predictions from the project's trained LightGBM models "
    "(Task 2.7 — served here via Streamlit instead of MLflow)."
)

bundle = load_bundle()
store_table = load_store_table()

with st.sidebar:
    st.header("Model info")
    st.metric("Sales model validation RMSPE", f"{bundle['sales_val_rmspe']:.3f}")
    st.metric("Customers model validation RMSPE", f"{bundle['customers_val_rmspe']:.3f}")
    st.caption(f"Bundle: `{os.path.basename(bundle['_path'])}`")

tab1, tab2 = st.tabs(["🔢 Single store / date range", "📄 Upload CSV (batch)"])

# ---- Tab 1: manual input ----
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        store_id = st.selectbox("Store_id", sorted(store_table["Store"].unique()), index=0)
    with col2:
        date_range = st.date_input(
            "Date range",
            value=(pd.Timestamp("2015-08-01"), pd.Timestamp("2015-08-14")),
        )
    with col3:
        promo = st.selectbox("Running a promo?", [0, 1], index=1)

    col4, col5, col6 = st.columns(3)
    with col4:
        state_holiday = st.selectbox("State holiday", ["0", "a", "b", "c"], index=0,
                                      help="0=None, a=Public, b=Easter, c=Christmas")
    with col5:
        school_holiday = st.selectbox("School holiday", [0, 1], index=0)
    with col6:
        is_open = st.selectbox("Store open?", [1, 0], index=0)

    if st.button("Predict", type="primary"):
        if isinstance(date_range, tuple) and len(date_range) == 2:
            dates = pd.date_range(date_range[0], date_range[1])
        else:
            dates = pd.date_range(date_range, date_range)

        req = pd.DataFrame({
            "Store": store_id, "Date": dates, "Promo": promo,
            "StateHoliday": state_holiday, "SchoolHoliday": school_holiday,
            "Open": is_open,
        })
        result = predict(req, bundle, store_table)

        st.subheader("Predicted Sales & Customers")
        st.line_chart(result.set_index("Date")[["PredictedSales", "PredictedCustomers"]])
        st.dataframe(result, use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions as CSV", csv,
                            file_name=f"predictions_store{store_id}.csv", mime="text/csv")

# ---- Tab 2: CSV upload ----
with tab2:
    st.write(
        "Upload a CSV with columns: **Store, Date**, and optionally "
        "**Promo, StateHoliday, SchoolHoliday, Open, DayOfWeek**. "
        "Missing optional columns default to 0 / open."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        st.write("Preview of uploaded file:")
        st.dataframe(input_df.head(), use_container_width=True)

        if st.button("Run batch prediction", type="primary"):
            result = predict(input_df, bundle, store_table)
            st.subheader(f"Predictions for {len(result)} rows")

            daily_total = result.groupby("Date")[["PredictedSales", "PredictedCustomers"]].sum()
            st.line_chart(daily_total)
            st.dataframe(result, use_container_width=True)

            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions as CSV", csv,
                                file_name="batch_predictions.csv", mime="text/csv")

st.divider()
st.caption(
    "Model: LightGBM + leak-safe Store target encoding (Task 2). "
    "Sales interval is a 90% band from LightGBM quantile regression (5th-95th percentile)."
)
