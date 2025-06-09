"""
src/models/train_all.py
=======================

One‑stop script to (re)train **Prophet, LSTM, and LightGBM** models
for every ticker in your universe.  It also writes JSON metadata
(val_mape) so ensemble.py can weight models by inverse error.

Usage
-----
    # Activate your venv first
    python -m src.models.train_all           # default universe
    python -m src.models.train_all --tickers RELIANCE TCS INFY
    python -m src.models.train_all --horizon 252 --start 2012-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from src.data_ingestion import (
    fetch_stock,
    fetch_fundamentals,
    fetch_macro,
)
from src.feature_engineering import full_pipeline
from src.models.prophet_model import train_prophet
from src.models.lstm_model import train_lstm
from src.models.lightgbm_model import train_lgbm

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DEFAULT_TICKERS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def train_all_for_ticker(
    ticker: str,
    start: str,
    horizon_days: int,
):
    print(f"\n🚀  {ticker}: pulling data …")
    price_df = fetch_stock(ticker, start=start)

    if price_df.empty:
        print(f"⚠️  No price data for {ticker}. Skipping.")
        return

    print("🔧  engineering features …")
    fund_series = fetch_fundamentals(ticker)
    macro_df = fetch_macro()
    feat_df = full_pipeline(price_df, fund_series, macro_df)

    if feat_df.empty:
        print(f"⚠️  Feature DF empty for {ticker}. Skipping.")
        return

    # ── Prophet ───────────────────────────────────────────────
    print("📈  training Prophet …")
    _, p_mape = train_prophet(feat_df, ticker)
    print(f"   → val MAPE: {p_mape:.3f}")

    # ── LSTM ──────────────────────────────────────────────────
    print("🤖  training LSTM …")
    _, l_mape = train_lstm(price_df, ticker, horizon_days=horizon_days)
    print(f"   → val MAPE: {l_mape:.3f}")

    # ── LightGBM ──────────────────────────────────────────────
    print("🌳  training LightGBM …")
    _, g_mape = train_lgbm(feat_df, ticker, horizon_days=horizon_days)
    print(f"   → val MAPE: {g_mape:.3f}")

    print(f"✅  {ticker} done.")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description="Train all models for tickers.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="List of NSE symbols (without .NS)",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Historical start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=252,
        help="Forecast horizon in trading days (default 252 ≈ 1y)",
    )
    args = parser.parse_args(argv)

    print(
        f"\n🗓  {date.today()} | Training horizon: {args.horizon} days "
        f"| Start: {args.start}"
    )
    for t in args.tickers:
        train_all_for_ticker(t.upper(), args.start, args.horizon)

    print("\n🏁  All tickers processed.")


if __name__ == "__main__":
    main()