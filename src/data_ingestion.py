# src/data_ingestion.py
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Ensure data directory
Path("data").mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────
# Price series fetchers
# ──────────────────────────────────────────────────────────
def fetch_stock(ticker: str, start: str = "2005-01-01") -> pd.DataFrame:
    """
    Download OHLCV (adjusted) for an NSE symbol via yfinance and
    save a local parquet copy.  Ensures the 'Close' column is a
    flat 1-D float Series (yfinance can return a (N,1) DataFrame).
    """
    df = yf.download(f"{ticker}.NS", start=start, auto_adjust=True)

    # ── Sanity-fix: flatten Close to 1-D Series ───────────────────
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.Series(np.asarray(close).ravel(), index=df.index, dtype="float64")
        df["Close"] = close

    df.to_parquet(Path("data") / f"{ticker}.parquet")
    return df


def fetch_mutual_fund(code: str) -> pd.DataFrame:
    """
    Pull NAV history for an AMFI scheme code via mfapi.in and
    store a parquet copy.
    """
    url = f"https://api.mfapi.in/mf/{code}"
    js = requests.get(url, timeout=15).json()
    df = pd.DataFrame(js["data"])[["date", "nav"]].astype({"nav": float})
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.to_parquet(Path("data") / f"MF-{code}.parquet")
    return df


# ──────────────────────────────────────────────────────────
# Fundamental & macro helpers
# ──────────────────────────────────────────────────────────
def fetch_fundamentals(ticker: str) -> pd.Series:
    """Fetch basic valuation ratios from Yahoo Finance."""
    yf_ticker = yf.Ticker(f"{ticker}.NS")
    info = yf_ticker.info
    fundamentals = {
        "symbol": ticker,
        "pe_ratio": info.get("trailingPE"),
        "pb_ratio": info.get("priceToBook"),
        "eps": info.get("trailingEps"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "de_to_eq": info.get("debtToEquity"),
        "sector": info.get("sector"),
        "fetch_date": dt.date.today().isoformat(),
    }
    return pd.Series(fundamentals)


def fetch_macro() -> pd.DataFrame:
    """
    Retrieve latest macro-economic indicators (CPI, repo rate).
    """
    urls = {
        "cpi": "https://rbidocs.rbi.org.in/rdocs/Content/DOCs/CPI_Excel.xlsx",
        "repo_rate": "https://rbidocs.rbi.org.in/rdocs/Content/DOCs/REPO_Excel.xlsx",
    }
    records = []
    for name, url in urls.items():
        try:
            tbl = pd.read_excel(url, skiprows=5)
            latest_row = tbl.dropna().iloc[-1]
            records.append(
                {"series": name, "date": latest_row[0], "value": latest_row[1]}
            )
        except Exception:
            records.append({"series": name, "date": None, "value": None})
    return pd.DataFrame(records)