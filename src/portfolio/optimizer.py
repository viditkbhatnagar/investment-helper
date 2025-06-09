# src/portfolio/optimizer.py
"""
Minimal allocator for the live-recommendation engine.

Inputs
------
forecast_df : DataFrame with columns
    ['symbol', 'price_now', 'price_forecast', 'exp_CAGR', 'risk_score']
budget      : int  (INR the user wants to invest)
risk        : str  ("Conservative" | "Moderate" | "Aggressive")

Outputs
-------
portfolio_df : DataFrame
    ['symbol', 'alloc_%', 'invest_INR',
     'price_now', 'price_forecast',
     'value_future_INR', 'exp_CAGR', 'risk_score']
perf_summary : dict with expected CAGR, risk proxy, future value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def allocate_from_forecasts(
    forecast_df: pd.DataFrame,
    budget: int,
    risk: str = "Moderate",
) -> tuple[pd.DataFrame, dict]:
    if forecast_df.empty:
        raise ValueError("forecast_df is empty – nothing to allocate")

    # 1. Rank list by expected CAGR ----------------------------------------------------------------
    ranked = (
        forecast_df.sort_values("exp_CAGR", ascending=False)
        .reset_index(drop=True)
        .copy()
    )

    # 2. Pick the slice width based on risk ---------------------------------------------------------
    picks = {
        "Conservative": 3,
        "Moderate": 5,
        "Aggressive": min(8, len(ranked)),
    }.get(risk, 5)
    ranked = ranked.head(picks)

    # 3. Simple equal-weight (you can swap for PyPortfolioOpt later) -------------------------------
    w_equal = 1 / len(ranked)
    ranked["alloc_%"] = w_equal * 100
    ranked["invest_INR"] = ranked["alloc_%"] * 0.01 * budget

    # 4. Future value @ forecast price --------------------------------------------------------------
    growth_factor = ranked["price_forecast"] / ranked["price_now"]
    ranked["value_future_INR"] = ranked["invest_INR"] * growth_factor

    # 5. Portfolio-level numbers -------------------------------------------------------------------
    future_total = ranked["value_future_INR"].sum()
    port_cagr = (future_total / budget) ** (1 / 1) - 1  # 1 = horizon in yrs (overwrite later)

    perf = {
        "invest_now": budget,
        "value_future": round(future_total, 2),
        "exp_port_cagr": round(port_cagr, 4),
    }

    return ranked[
        [
            "symbol",
            "alloc_%",
            "invest_INR",
            "price_now",
            "price_forecast",
            "value_future_INR",
            "exp_CAGR",
            "risk_score",
        ]
    ], perf