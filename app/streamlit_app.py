import streamlit as st
import os
import sys

# Compute project root and src directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Prepend to sys.path so that imports from src.* work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from src.live_data import (
    get_live_all_stocks,
    get_live_all_mutuals,
    get_live_stock_quotes,
    get_live_mutual_nav,
)
from src.recommendation_engine import get_recommendations

st.set_page_config(page_title="AI Investment Advisor", layout="wide")

# ── Sidebar for user inputs ─────────────────────────────────────────
with st.sidebar:
    budget = st.number_input("Investment budget (₹)", 50_000, 1_00_00_000, step=5_000)
    horizon = st.selectbox("Time horizon", [1, 3, 5], format_func=lambda x: f"{x} year")
    risk = st.radio("Risk appetite", ["Conservative", "Moderate", "Aggressive"])
    products = st.multiselect("Product type", ["Stocks", "Mutual Funds"], default=["Stocks", "Mutual Funds"])
    run_button = st.button("Get recommendations")

# ── TABS: Live Market | Recommendations ────────────────────────────
live_tab, reco_tab = st.tabs(["📡 Live Market", "🎯 Recommendations"])

with live_tab:
    st.subheader("Live Market – All NSE Stocks")
    # Fetch all stocks once per run (cached inside live_data)
    stocks_df = get_live_all_stocks(limit=15)

    # Optional filter box
    sym_filter = st.text_input("Filter by symbol / company name").upper().strip()
    if sym_filter:
        stocks_df = stocks_df[
            stocks_df["symbol"].str.contains(sym_filter)
            | stocks_df["symbol"].str.contains(sym_filter)
        ]

    st.dataframe(
        stocks_df.sort_values("symbol").reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    st.divider()

    st.subheader("Live Mutual‑Fund NAVs (All Schemes)")
    mfs_df = get_live_all_mutuals(limit=10)
    mf_filter = st.text_input("Filter by scheme code / name").upper().strip()
    if mf_filter:
        mfs_df = mfs_df[
            mfs_df["scheme_code"].str.contains(mf_filter)
            | mfs_df["scheme_name"].str.upper().str.contains(mf_filter)
        ]

    st.dataframe(
        mfs_df.sort_values("scheme_name").reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    st.caption("Data refreshed on each run. Stock quotes: yfinance; MF NAV: MFAPI.in")

with reco_tab:
    if run_button:
        recs_df, alloc_fig, forecast_fig, pdf_bytes = get_recommendations(
            budget, horizon, risk, products
        )
        st.header("Recommended Portfolio")
        st.dataframe(
            recs_df[
                [
                    "symbol",
                    "alloc_%",
                    "invest_INR",
                    "value_future_INR",
                    "exp_CAGR",
                    "risk_score",
                ]
            ].style.format(
                {
                    "alloc_%": "{:.2f}%",
                    "invest_INR": "₹{:.0f}",
                    "value_future_INR": "₹{:.0f}",
                    "exp_CAGR": "{:.2%}",
                }
            ),
            use_container_width=True,
        )
        st.plotly_chart(alloc_fig)
        st.plotly_chart(forecast_fig)
        st.download_button("Download PDF report", pdf_bytes, "investment_report.pdf")
    else:
        st.info("Fill the sidebar and press **Get recommendations**")