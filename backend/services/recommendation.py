from __future__ import annotations

from typing import List, Optional

import pandas as pd

from backend.schemas import RecommendationResponse
from src.recommendation_engine import get_recommendations as _get_recommendations


def compute_recommendations(
    budget: int,
    horizon: int,
    risk: str,
    products: List[str],
    expected_cagr_min: Optional[float] = None,
    max_risk_score: Optional[float] = None,
) -> RecommendationResponse:
    # Delegate to existing engine and convert to API schema
    result = _get_recommendations(
        budget, horizon, risk, products
    )
    # Backward compatibility with old return signature
    if isinstance(result, tuple) and len(result) == 5:
        portfolio_df, alloc_fig, forecast_fig, _pdf, perf = result
    else:
        portfolio_df, alloc_fig, forecast_fig, _pdf = result
        perf = {}

    # Optional filters
    if expected_cagr_min is not None:
        portfolio_df = portfolio_df[portfolio_df["exp_CAGR"] >= expected_cagr_min]
    if max_risk_score is not None:
        portfolio_df = portfolio_df[portfolio_df["risk_score"] <= max_risk_score]

    rows = portfolio_df.to_dict(orient="records")
    return RecommendationResponse(portfolio=rows, perf=perf)


