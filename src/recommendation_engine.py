"""
Recommendation engine – v1.

Pipeline
--------
1. Pull live prices for a predefined universe.
2. Build fast features (RSI, pct change …) – transform_live().
3. Load latest Prophet model for each ticker and forecast to user horizon.
4. Compute expected CAGR and simple volatility proxy.
5. Rank by risk‑adjusted score & build portfolio allocation.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go

from src.portfolio.optimizer import allocate_from_forecasts

from src.live_data import get_live_stock_quotes
from src.feature_engineering import transform_live
from src.models.prophet_model import load_prophet, forecast_prophet

# ── Universe & horizon mapping ─────────────────────────────────────
STOCK_UNIVERSE: List[str] = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
HORIZON_MAP = {1: 252, 3: 756, 5: 1260}  # years → trading days


# ── Core helper functions ──────────────────────────────────────────
def _expected_cagr(curr_price: float, future_price: float, years: int) -> float:
    """Compounded annual growth rate."""
    if curr_price <= 0 or future_price <= 0:
        return 0.0
    return (future_price / curr_price) ** (1 / years) - 1


def _risk_proxy(df_quote: pd.Series) -> float:
    """Use 20‑day vol as quick risk proxy (placeholder)."""
    return df_quote.get("vol_21d", 0.0)  # already annualised in features


# ── Public API ─────────────────────────────────────────────────────
def get_recommendations(
    budget: int,
    horizon: int,
    risk: str,
    products: list[str],
):
    """
    Parameters
    ----------
    budget : int
        Amount in INR user wishes to deploy.
    horizon : int
        1, 3, or 5 (years) for now.
    risk : str
        Conservative | Moderate | Aggressive
    products : list[str]
        Currently only 'Stocks' supported in this v1 scaffold.

    Returns
    -------
    (portfolio_df, alloc_pie_fig, forecast_fig, pdf_bytes)
    """
    # 1. Live quotes -------------------------------------------------
    live_df = get_live_stock_quotes(STOCK_UNIVERSE)
    live_df.set_index("symbol", inplace=True)

    # 2. Build quick features ---------------------------------------
    feats = {sym: transform_live(row) for sym, row in live_df.iterrows()}

    # 3. Forecast with stored Prophet models ------------------------
    horizon_days = HORIZON_MAP.get(horizon, 252)
    forecast_records = []
    today = _dt.date.today()

    for sym in STOCK_UNIVERSE:
        price_now = live_df.loc[sym, "price"]
        try:
            model = load_prophet(sym)
            future = pd.DataFrame({"ds": [today + _dt.timedelta(days=horizon_days)]})
            yhat = forecast_prophet(model, future)["yhat"].iloc[-1]
        except FileNotFoundError:
            # Fallback: naive one-day momentum forecast
            raw_change = feats[sym].get("raw_chg_pct", feats[sym].get("pct_chg", 0)) or 0.0
            yhat = price_now * (1 + raw_change)

        cagr = _expected_cagr(price_now, yhat, horizon)
        risk_score = _risk_proxy(feats[sym])

        forecast_records.append(
            {
                "symbol": sym,
                "price_now": price_now,
                "price_forecast": yhat,
                "exp_CAGR": cagr,
                "risk_score": risk_score,
            }
        )

    forecast_df = pd.DataFrame(forecast_records)

    # 4. Portfolio allocation via optimiser --------------------------
    portfolio_df, perf = allocate_from_forecasts(
        forecast_df,
        budget=budget,
        risk=risk,
    )

    # 5. Build charts -----------------------------------------------
    alloc_fig = go.Figure(
        go.Pie(labels=portfolio_df["symbol"], values=portfolio_df["alloc_%"], hole=0.4)
    )
    alloc_fig.update_layout(title="Suggested Allocation")

    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Bar(
            x=portfolio_df["symbol"],
            y=portfolio_df["exp_CAGR"] * 100,
            name="Expected CAGR (%)",
        )
    )
    forecast_fig.update_layout(title="Expected CAGR (simple Prophet forecast)")

    # 6. Placeholder PDF (to be implemented) ------------------------
    pdf_bytes = b""

    return portfolio_df, alloc_fig, forecast_fig, pdf_bytes