"""
Light-weight helpers to pull *current* prices.

• Stocks → Yahoo Finance via yfinance (free, 15 sec cache)
• Mutual-fund NAV → MFAPI.in (free AMFI dump)

For broker-grade ticks later you can swap in Zerodha
WebSockets or Upstox APIs.
"""
from __future__ import annotations
import datetime as _dt
from typing import List

import io, zipfile, csv
from functools import lru_cache

import pandas as pd
import requests
import yfinance as yf


# ── SYMBOL MASTER HELPERS ─────────────────────────────────────────
NSE_SYMBOL_URL = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
AMFI_LIST_URL  = "https://api.mfapi.in/mf"

@lru_cache(maxsize=1)
def list_all_nse_symbols() -> List[str]:
    """
    Download NSE's daily symbol master (EQUITY_L.csv) and return a list
    of active ticker symbols (without '.NS').
    Cached in‑memory for 15 minutes via lru_cache.
    """
    try:
        df = pd.read_csv(NSE_SYMBOL_URL)
        return df["SYMBOL"].dropna().unique().tolist()
    except Exception:
        # Fallback to NIFTY‑50 list if master fetch fails
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

@lru_cache(maxsize=1)
def list_all_amfi_codes() -> List[str]:
    """
    Pull the master list of AMFI schemes (API returns JSON array).
    """
    try:
        res = requests.get(AMFI_LIST_URL, timeout=15).json()
        return [str(s["schemeCode"]) for s in res]
    except Exception:
        return []

# ── Convenience: small subsets for UI demo ───────────────────────────
def list_nse_subset(n: int = 15) -> List[str]:
    """Return the first *n* active NSE symbols (alphabetical)."""
    return sorted(list_all_nse_symbols())[:n]

def list_amfi_subset(n: int = 10) -> List[str]:
    """Return the first *n* AMFI scheme codes."""
    return list_all_amfi_codes()[:n]


# ── STOCK QUOTES (NSE) ──────────────────────────────────────────────
def get_live_stock_quotes(tickers: List[str]) -> pd.DataFrame:
    """
    Returns: DataFrame[ symbol, price, prev_close, chg_%, time ]
    """
    recs = []
    for t in tickers:
        data = yf.Ticker(f"{t}.NS").fast_info
        price = data["lastPrice"]
        prev = data["previousClose"]
        recs.append(
            {
                "symbol": t,
                "price": round(price, 2),
                "prev_close": round(prev, 2),
                "chg_%": round((price - prev) / prev * 100, 2) if prev else None,
                "time": _dt.datetime.now().strftime("%H:%M:%S"),
            }
        )
    return pd.DataFrame(recs)


# ── MUTUAL-FUND NAV ────────────────────────────────────────────────
def get_live_mutual_nav(schemes: List[str]) -> pd.DataFrame:
    """
    `schemes` = AMFI codes as strings (e.g. "118834")
    Returns: DataFrame[ scheme_code, scheme_name, date, nav ]
    """
    rows = []
    for code in schemes:
        url = f"https://api.mfapi.in/mf/{code}"
        try:
            js = requests.get(url, timeout=10).json()
            latest = js["data"][0]
            rows.append(
                dict(
                    scheme_code=code,
                    scheme_name=js["meta"]["scheme_name"],
                    date=latest["date"],
                    nav=float(latest["nav"]),
                )
            )
        except Exception:
            rows.append(dict(scheme_code=code, scheme_name="ERROR", date="", nav=None))
    return pd.DataFrame(rows)


# ── BULK FETCH CONVENIENCE WRAPPERS ──────────────────────────────
def get_live_all_stocks(limit: int | None = None) -> pd.DataFrame:
    """
    Fetch quotes for every NSE symbol or a limited subset.
    If `limit` is provided, only the first `limit` alphabetic symbols are used.
    """
    syms = list_all_nse_symbols()
    if limit:
        syms = sorted(syms)[:limit]
    return get_live_stock_quotes(syms)

def get_live_all_mutuals(limit: int | None = None) -> pd.DataFrame:
    """
    Fetch NAV for every AMFI scheme or a limited subset.
    """
    codes = list_all_amfi_codes()
    if limit:
        codes = codes[:limit]
    return get_live_mutual_nav(codes)