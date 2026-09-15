```python
"""
Rossmann Sales Forecasting — Streamlit Serving App
Task 2.7 (model serving, using Streamlit instead of MLflow) + Task 3 (dashboard)

Run with:
    streamlit run streamlit_app.py
"""

import os
import sys
import gzip
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Import project modules
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR))

from feature_engineering import (
    engineer_features,
    apply_store_target_encoding,
    to_model_matrix,
)


# ---------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Rossmann Sales Forecast",
    layout="wide"
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"

# Compressed serving bundle
BUNDLE_PATTERN = "serving-bundle-*.pkl.gz"


# ---------------------------------------------------------
# Load model bundle
# ---------------------------------------------------------
@st.cache_resource
def load_bundle():

    # Check whether models directory exists
    if not MODELS_DIR.exists():
        st.error(
            f"Models directory not found: {MODELS_DIR}"
        )
        st.info(
            "Create a 'models' folder in your GitHub repository "
            "and place the serving bundle inside it."
        )
        st.stop()

    # Find compressed serving bundles
    candidates = sorted(
        MODELS_DIR.glob(BUNDLE_PATTERN)
    )

    if not candidates:
        st.error(
            "No serving bundle found in the models folder."
        )

        st.write("Expected location:")
        st.code(
            str(MODELS_DIR / "serving-bundle-*.pkl.gz")
        )

        st.write("Files found in models folder:")

        try:
            files = list(MODELS_DIR.iterdir())

            if files:
                for file in files:
                    st.write(f"- {file.name}")
            else:
                st.write("The models folder is empty.")

        except Exception as e:
            st.write(f"Could not read models folder: {e}")

        st.info(
            "Make sure the .pkl.gz serving bundle is committed "
            "to GitHub and deployed with the Streamlit app."
        )

        st.stop()

    # Use the newest/last bundle
    bundle_path = candidates[-1]

    try:
        # -------------------------------------------------
        # Load gzip-compressed pickle/joblib bundle
        # -------------------------------------------------
        with gzip.open(bundle_path, "rb") as f:
            bundle = joblib.load(f)

    except Exception as e:

        st.error(
            "Unable to load the serving bundle."
        )

        st.exception(e)

        st.info(
            "The model may have been created with pickle instead "
            "of joblib. If so, replace joblib.load(f) with "
            "pickle.load(f)."
        )

        st.stop()

    # Store path for UI
    bundle["_path"] = str(bundle_path)

    return bundle


# ---------------------------------------------------------
# Load store table
# ---------------------------------------------------------
@st.cache_data
def load_store_table():

    path = DATA_DIR / "store_clean_for_app.csv"

    if not path.exists():

        # Also check models folder
        models_path = MODELS_DIR / "store_clean_for_app.csv"

        if models_path.exists():
            path = models_path
        else:
            st.error(
                "store_clean_for_app.csv was not found."
            )

            st.write("Checked:")
            st.code(str(DATA_DIR / "store_clean_for_app.csv"))
            st.code(str(models_path))

            st.stop()

    return pd.read_csv(path)


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict(df_raw, bundle, store_table):
    """
    df_raw needs:
        Store
        Date

    Optional:
        DayOfWeek
        Promo
        SchoolHoliday
        StateHoliday
        Open
    """

    df = df_raw.copy()

    # -----------------------------------------------------
    # Date
    # -----------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"])

    # -----------------------------------------------------
    # Optional columns
    # -----------------------------------------------------
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = (
            df["Date"].dt.dayofweek + 1
        )

    if "Open" not in df.columns:
        df["Open"] = 1

    if "StateHoliday" not in df.columns:
        df["StateHoliday"] = "0"

    if "SchoolHoliday" not in df.columns:
        df["SchoolHoliday"] = 0

    if "Promo" not in df.columns:
        df["Promo"] = 0

    df["StateHoliday"] = df["StateHoliday"].astype(str)

    # -----------------------------------------------------
    # Merge store information
    # -----------------------------------------------------
    df = df.merge(
        store_table,
        on="Store",
        how="left"
    )

    missing_stores = df["StoreType"].isna().sum()

    if missing_stores:

        st.warning(
            f"{missing_stores} row(s) reference a Store ID "
            "not present in store_clean_for_app.csv — dropped."
        )

        df = df.dropna(
            subset=["StoreType"]
        )

    if df.empty:
        st.error(
            "No valid Store IDs remain after merging "
            "with store_clean_for_app.csv."
        )
        return pd.DataFrame()

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------
    df_fe = engineer_features(
        df,
        bundle["holiday_dates"]
    )

    df_fe = apply_store_target_encoding(
        df_fe,
        bundle["store_enc_map"],
        bundle["global_mean"]
    )

    X = to_model_matrix(df_fe)

    # -----------------------------------------------------
    # Sales prediction
    # -----------------------------------------------------
    sales_booster = bundle["sales_booster"]

    try:
        sales_pred = sales_booster.predict(
            X,
            num_iteration=getattr(
                sales_booster,
                "best_iteration",
                None
            )
        )
    except TypeError:
        sales_pred = sales_booster.predict(X)

    # -----------------------------------------------------
    # Sales confidence interval
    # -----------------------------------------------------
    sales_lower = bundle[
        "sales_model_lower"
    ].predict(X)

    sales_upper = bundle[
        "sales_model_upper"
    ].predict(X)

    # -----------------------------------------------------
    # Customer prediction
    # -----------------------------------------------------
    customers_booster = bundle[
        "customers_booster"
    ]

    try:
        cust_pred = customers_booster.predict(
            X,
            num_iteration=getattr(
                customers_booster,
                "best_iteration",
                None
            )
        )
    except TypeError:
        cust_pred = customers_booster.predict(X)

    # -----------------------------------------------------
    # Closed stores = zero sales/customers
    # -----------------------------------------------------
    is_closed = (
        df_fe["Open"].values == 0
    )

    sales_pred = np.where(
        is_closed,
        0,
        sales_pred
    )

    sales_lower = np.where(
        is_closed,
        0,
        sales_lower
    )

    sales_upper = np.where(
        is_closed,
        0,
        sales_upper
    )

    cust_pred = np.where(
        is_closed,
        0,
        cust_pred
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------
    output_columns = [
        "Store",
        "Date",
        "DayOfWeek",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "Open",
    ]

    out = df_fe[
        output_columns
    ].copy()

    out["PredictedSales"] = (
        sales_pred.round(0)
    )

    out["PredictedSales_Low90"] = (
        np.clip(
            sales_lower,
            0,
            None
        ).round(0)
    )

    out["PredictedSales_High90"] = (
        np.clip(
            sales_upper,
            0,
            None
        ).round(0)
    )

    out["PredictedCustomers"] = (
        np.clip(
            cust_pred,
            0,
            None
        ).round(0)
    )

    return out


# =========================================================
# UI
# =========================================================

st.title(
    "🏪 Rossmann Store Sales Forecast"
)

st.caption(
    "Serves predictions from the project's trained "
    "LightGBM models (Task 2.7 — served via Streamlit)."
)


# ---------------------------------------------------------
# Load model and store data
# ---------------------------------------------------------
bundle = load_bundle()

store_table = load_store_table()


# ---------------------------------------------------------
# Model information
# ---------------------------------------------------------
with st.sidebar:

    st.header("Model info")

    st.metric(
        "Sales model validation RMSPE",
        f"{bundle['sales_val_rmspe']:.3f}"
    )

    st.metric(
        "Customers model validation RMSPE",
        f"{bundle['customers_val_rmspe']:.3f}"
    )

    st.caption(
        f"Bundle: `{os.path.basename(bundle['_path'])}`"
    )


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(
    [
        "🔢 Single store / date range",
        "📄 Upload CSV (batch)"
    ]
)


# =========================================================
# TAB 1 — Manual prediction
# =========================================================

with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:

        store_ids = sorted(
            store_table["Store"].dropna().unique()
        )

        store_id = st.selectbox(
            "Store ID",
            store_ids,
            index=0
        )

    with col2:

        date_range = st.date_input(
            "Date range",
            value=(
                pd.Timestamp("2015-08-01"),
                pd.Timestamp("2015-08-14")
            ),
        )

    with col3:

        promo = st.selectbox(
            "Running a promo?",
            [0, 1],
            index=1
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        state_holiday = st.selectbox(
            "State holiday",
            ["0", "a", "b", "c"],
            index=0,
            help=(
                "0=None, a=Public, "
                "b=Easter, c=Christmas"
            )
        )

    with col5:

        school_holiday = st.selectbox(
            "School holiday",
            [0, 1],
            index=0
        )

    with col6:

        is_open = st.selectbox(
            "Store open?",
            [1, 0],
            index=0
        )

    if st.button(
        "Predict",
        type="primary"
    ):

        if (
            isinstance(date_range, tuple)
            and len(date_range) == 2
        ):

            dates = pd.date_range(
                date_range[0],
                date_range[1]
            )

        else:

            dates = pd.date_range(
                date_range,
                date_range
            )

        req = pd.DataFrame(
            {
                "Store": store_id,
                "Date": dates,
                "Promo": promo,
                "StateHoliday": state_holiday,
                "SchoolHoliday": school_holiday,
                "Open": is_open,
            }
        )

        result = predict(
            req,
            bundle,
            store_table
        )

        if not result.empty:

            st.subheader(
                "Predicted Sales & Customers"
            )

            st.line_chart(
                result.set_index("Date")[
                    [
                        "PredictedSales",
                        "PredictedCustomers"
                    ]
                ]
            )

            st.dataframe(
                result,
                use_container_width=True
            )

            csv = result.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download predictions as CSV",
                csv,
                file_name=(
                    f"predictions_store"
                    f"{store_id}.csv"
                ),
                mime="text/csv"
            )


# =========================================================
# TAB 2 — CSV batch prediction
# =========================================================

with tab2:

    st.write(
        "Upload a CSV with columns: **Store, Date**, "
        "and optionally **Promo, StateHoliday, "
        "SchoolHoliday, Open, DayOfWeek**. "
        "Missing optional columns default to 0 / open."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded is not None:

        input_df = pd.read_csv(
            uploaded
        )

        st.write(
            "Preview of uploaded file:"
        )

        st.dataframe(
            input_df.head(),
            use_container_width=True
        )

        if st.button(
            "Run batch prediction",
            type="primary"
        ):

            result = predict(
                input_df,
                bundle,
                store_table
            )

            if not result.empty:

                st.subheader(
                    f"Predictions for "
                    f"{len(result)} rows"
                )

                daily_total = (
                    result
                    .groupby("Date")[
                        [
                            "PredictedSales",
                            "PredictedCustomers"
                        ]
                    ]
                    .sum()
                )

                st.line_chart(
                    daily_total
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

                csv = result.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download predictions as CSV",
                    csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Model: LightGBM + leak-safe Store target encoding "
    "(Task 2). Sales interval is a 90% band from "
    "LightGBM quantile regression "
    "(5th–95th percentile)."
)
```
