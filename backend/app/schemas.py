from pydantic import BaseModel
from typing import Optional


class StockQuote(BaseModel):
    symbol: str
    price: float
    change: float
    percent_change: float


class Recommendation(BaseModel):
    symbol: str
    rating: str
    rationale: Optional[str] = None


class Insight(BaseModel):
    title: str
    description: str


