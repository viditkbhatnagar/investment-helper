from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.schemas import (
    HealthResponse,
    SymbolsResponse,
    QuotesResponse,
    MFNavResponse,
    RecommendationRequest,
    RecommendationResponse,
    InsightRequest,
    InsightResponse,
)
from backend.services import market_data, recommendation, insights
from backend.services import catalog, user as user_service
from backend.services import research
from backend.schemas import ResearchRequest, ResearchResponse
from backend.db import get_db


def create_app() -> FastAPI:
    app = FastAPI(title="Investment Advisor API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve a minimal demo UI from /ui (development convenience)
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.isdir(static_dir):
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/api/symbols", response_model=SymbolsResponse)
    async def get_symbols(limit: Optional[int] = None) -> SymbolsResponse:
        syms = market_data.list_symbols(limit=limit)
        return SymbolsResponse(symbols=syms)

    @app.get("/api/mutuals", response_model=SymbolsResponse)
    async def get_mutuals(limit: Optional[int] = None) -> SymbolsResponse:
        codes = market_data.list_mutual_schemes(limit=limit)
        return SymbolsResponse(symbols=codes)

    @app.get("/api/quotes", response_model=QuotesResponse)
    async def quotes(tickers: str) -> QuotesResponse:
        """tickers is a comma-separated list of NSE symbols (without .NS)."""
        tickers_list: List[str] = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not tickers_list:
            raise HTTPException(status_code=400, detail="No tickers provided")
        df = market_data.get_quotes(tickers_list)
        return QuotesResponse(rows=df.to_dict(orient="records"))

    @app.get("/api/mfnv", response_model=MFNavResponse)
    async def mf_nav(schemes: str) -> MFNavResponse:
        """schemes is a comma-separated list of AMFI scheme codes."""
        codes: List[str] = [c.strip() for c in schemes.split(",") if c.strip()]
        if not codes:
            raise HTTPException(status_code=400, detail="No scheme codes provided")
        df = market_data.get_mf_nav(codes)
        return MFNavResponse(rows=df.to_dict(orient="records"))

    @app.post("/api/recommendations", response_model=RecommendationResponse)
    async def get_recommendations(req: RecommendationRequest) -> RecommendationResponse:
        result = recommendation.compute_recommendations(
            budget=req.budget,
            horizon=req.horizon,
            risk=req.risk,
            products=req.products,
            expected_cagr_min=req.expected_cagr_min,
            max_risk_score=req.max_risk_score,
        )
        return result

    @app.post("/api/insights", response_model=InsightResponse)
    async def get_insights(req: InsightRequest) -> InsightResponse:
        return await insights.generate_market_insights(req)

    @app.post("/api/research", response_model=ResearchResponse)
    async def run_market_research(req: ResearchRequest) -> ResearchResponse:
        return await research.run_research(req)

    # Catalog: IPOs and Bonds (placeholder data for now)
    @app.get("/api/ipos")
    async def get_ipos():
        return {"rows": catalog.list_ipos()}

    @app.get("/api/bonds")
    async def get_bonds():
        return {"rows": catalog.list_bonds()}

    # User profile and portfolios (MongoDB)
    @app.get("/api/users/{user_id}/profile")
    async def get_user_profile(user_id: str):
        doc = await user_service.get_profile(user_id)
        return doc or {}

    @app.post("/api/users/{user_id}/profile")
    async def update_user_profile(user_id: str, profile: dict):
        await user_service.save_profile(user_id, profile)
        return {"status": "ok"}

    @app.post("/api/users/{user_id}/portfolios")
    async def save_user_portfolio(user_id: str, payload: dict):
        pid = await user_service.save_portfolio(user_id, payload.get("portfolio", []), payload.get("perf", {}))
        return {"portfolio_id": pid}

    @app.get("/api/users/{user_id}/portfolios")
    async def list_user_portfolios(user_id: str):
        rows = await user_service.list_portfolios(user_id)
        return {"rows": rows}

    return app


app = create_app()


