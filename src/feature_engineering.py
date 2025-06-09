from __future__ import annotations
"""
Feature‑engineering utilities.

Each helper takes in a *price‑indexed* DataFrame (Date as index, one
symbol’s OHLCV) and appends new columns, returning the enriched frame.

For live inference we expose `transform_live()` which builds only the
fast features that don’t require long rolling windows.
"""

import numpy as np
import pandas as pd
import ta  # pip install ta

# ────────────────────────────────────────────────────────────────
# Technical indicators
# ────────────────────────────────────────────────────────────────


def add_technical(df: pd.DataFrame) -> pd.DataFrame:
    # DEBUG print removed
    # ── Flatten multi‑index & ensure 1‑D float Series ───────────────
    if df.columns.nlevels > 1:                      # yfinance multi‑index
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    if isinstance(close, pd.DataFrame):             # shape (N,1)
        close = close.iloc[:, 0]

    close = pd.Series(
        np.asarray(close).reshape(-1),              # flatten any nested array
        index=df.index,
        dtype="float64",
    )
    df["Close"] = close

    # Returns & volatility
    df["ret"] = np.log(df["Close"]).diff()
    df["vol_21d"] = df["ret"].rolling(21).std() * np.sqrt(252)

    # Moving averages
    for w in (50, 100, 200):
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    # RSI 14
    df["rsi_14"] = ta.momentum.rsi(df["Close"], window=14)

    # MACD
    macd = ta.trend.macd(df["Close"])
    macd_signal = ta.trend.macd_signal(df["Close"])
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    # Bollinger Bands width
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_hi"] = bb.bollinger_hband()
    df["bb_lo"] = bb.bollinger_lband()
    df["bb_width"] = df["bb_hi"] - df["bb_lo"]

    # Sharpe (rolling 1Y, rf≈0)
    df["sharpe_252"] = df["ret"].rolling(252).mean() / df["ret"].rolling(252).std()

    return df


# ────────────────────────────────────────────────────────────────
# Fundamentals & macro join
# ────────────────────────────────────────────────────────────────


def add_fundamentals(
    price_df: pd.DataFrame, fund_series: pd.Series
) -> pd.DataFrame:
    """
    Broadcast one‑row fundamentals Series across every row in price_df.
    Assumes Series has index like ['pe_ratio', 'pb_ratio', ...].
    """
    for col, val in fund_series.items():
        if col == "symbol":
            continue
        price_df[col] = val
    return price_df


def add_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward‑fill latest macro value to all rows.

    macro_df must have columns ['series','value'] and a single fetch date.
    """
    for _, row in macro_df.iterrows():
        price_df[row["series"]] = row["value"]
    return price_df


# ────────────────────────────────────────────────────────────────
# Pipeline helpers
# ────────────────────────────────────────────────────────────────


def full_pipeline(
    price_df: pd.DataFrame,
    fund_series: pd.Series | None = None,
    macro_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply all transforms; drop rows with freshly introduced NaNs."""
    feat = add_technical(price_df)

    if fund_series is not None:
        feat = add_fundamentals(feat, fund_series)

    if macro_df is not None:
        feat = add_macro(feat, macro_df)

    # Final drop of early NaNs from rolling windows
    return feat.dropna()


# ────────────────────────────────────────────────────────────────
# Live‑inference shortcut (no long rolling windows)
# ────────────────────────────────────────────────────────────────


def transform_live(price_row: pd.Series) -> pd.Series:
    """
    Build instantaneous features for a single latest row of live data.
    Expects price_row to have 'price', 'prev_close', 'chg_%' and 'time'.
    """
    out = price_row.copy()

    # % change from previous close
    price = price_row.get("price")
    prev  = price_row.get("prev_close")
    out["pct_chg"] = (price / prev - 1) if (price and prev) else None

    # raw change % as provided by the live API
    out["raw_chg_pct"] = price_row.get("chg_%")

    # encode timestamp as seconds since midnight
    ts = price_row.get("time")
    out["time_sec"] = None
    if ts:
        try:
            h, m, s = map(int, ts.split(":"))
            out["time_sec"] = h*3600 + m*60 + s
        except ValueError:
            pass

    return out