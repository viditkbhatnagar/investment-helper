from typing import List

from fastapi import APIRouter, Path, Query

from app.schemas import Recommendation, StockQuote
from app.services.recommendations import (
    get_recommendations,
    get_recommendations_from_quotes,
    predict_price_range,
    predict_enriched,
)
from app.services.recommendations import run_research
from app.services.langgraph_pipeline import run_research_graph
from app.db import get_cache_database_sync
from fastapi import Body


router = APIRouter()


@router.get("/predict")
async def predict(
    name: str = Query(..., description="Stock name"),
    days: str = Query("1,5,10", description="Comma-separated horizons in days"),
    enriched: bool = Query(True, description="Use enriched model with DB signals"),
):
    horizons = [int(x) for x in days.split(",") if x.strip().isdigit()]
    if not horizons:
        horizons = [1, 5, 10]
    if enriched:
        return predict_enriched(name, horizons)
    result = predict_price_range(name, horizons)
    return {"symbol": name, "predictions": result}


@router.get("/research")
async def research(
    name: str = Query(..., description="Stock name (company name)"),
    days: str = Query("1,5,10", description="Comma-separated horizons in days"),
):
    horizons = [int(x) for x in days.split(",") if x.strip().isdigit()]
    if not horizons:
        horizons = [1, 5, 10]
    return await run_research(name, horizons)


@router.post("/blend-weight")
async def set_blend_weight(
    name: str = Query(..., description="Stock name"),
    horizon: int = Query(..., description="Horizon in days"),
    alpha: float = Body(..., embed=True, description="Blend weight for model vs AI (0..1). 1 = use model only"),
):
    lname = name.strip().lower()
    alpha = max(0.0, min(1.0, float(alpha)))
    db = get_cache_database_sync()
    db.models_cache.update_one(
        {"_norm_name": lname, "horizon": int(horizon), "model_type": "blend_alpha"},
        {"$set": {"alpha": alpha}},
        upsert=True,
    )
    return {"ok": True, "alpha": alpha}


@router.get("/research-graph")
async def research_graph(name: str = Query(..., description="Stock name")):
    return await run_research_graph(name)


@router.get("/", response_model=List[Recommendation])
async def recommendations_for_symbols(symbols: str = Query(..., description="Comma separated symbols")):
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    # Temporary: synthesize quotes as placeholders for live data
    quotes = [
        StockQuote(symbol=s, price=100.0, change=0.0, percent_change=0.7 if i % 2 == 0 else -0.2)
        for i, s in enumerate(symbols_list)
    ]
    return get_recommendations_from_quotes(quotes)


@router.get("/{user_id}", response_model=List[Recommendation])
async def recommendations_for_user(user_id: str = Path(...)):
    return get_recommendations(user_id)


