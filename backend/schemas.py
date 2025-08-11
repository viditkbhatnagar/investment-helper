from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]


class SymbolsResponse(BaseModel):
    symbols: List[str]


class QuotesResponse(BaseModel):
    rows: List[dict]


class MFNavResponse(BaseModel):
    rows: List[dict]


class RecommendationRequest(BaseModel):
    budget: int = Field(..., ge=1000)
    horizon: int = Field(..., description="Years: 1, 3, or 5")
    risk: Literal["Conservative", "Moderate", "Aggressive"]
    products: List[Literal["Stocks", "Mutual Funds", "Bonds", "IPOs"]] = Field(
        default_factory=lambda: ["Stocks", "Mutual Funds"]
    )
    expected_cagr_min: Optional[float] = Field(
        default=None, description="Minimum expected CAGR (e.g., 0.12 for 12%)"
    )
    max_risk_score: Optional[float] = Field(default=None)


class RecommendationResponse(BaseModel):
    portfolio: List[dict]
    perf: dict


class InsightRequest(BaseModel):
    query: str
    tickers: Optional[List[str]] = None


class InsightResponse(BaseModel):
    insight: str
    sources: List[str] = Field(default_factory=list)


# Research (web-scraped + LLM summarized)
class ResearchRequest(BaseModel):
    query: str
    tickers: Optional[List[str]] = None
    competitors: Optional[List[str]] = None
    limit_sources: int = 6


class ResearchResponse(BaseModel):
    insight: str
    sources: List[str] = Field(default_factory=list)


