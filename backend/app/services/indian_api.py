from typing import Any, Dict, List

import httpx

from app.settings import settings


def _headers() -> Dict[str, str]:
    if not settings.INDIAN_STOCK_API_KEY:
        return {}
    key = (settings.INDIAN_STOCK_API_KEY or "").strip()
    return {
        "x-api-key": key,
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": "investment-advisor/0.1 httpx",
    }


async def _get(path: str, params: Dict[str, Any] | None = None, timeout: int = 20):
    async with httpx.AsyncClient(base_url=settings.INDIAN_STOCK_API_BASE, timeout=timeout) as client:
        r = await client.get(path, params=params or {}, headers=_headers())
        r.raise_for_status()
        return r.json()


async def fetch_quote(symbol: str) -> Dict[str, Any]:
    return await _get(f"/quotes/{symbol}")


async def fetch_stock_detail(symbol: str) -> Dict[str, Any]:
    return await _get("/stock", params={"symbol": symbol})


async def fetch_stock_by_name(name: str) -> Dict[str, Any]:
    return await _get("/stock", params={"name": name})


async def fetch_historical_data(stock_name: str, period: str = "10yr", filter_: str = "price") -> Dict[str, Any]:
    return await _get(
        "/historical_data",
        params={"stock_name": stock_name, "period": period, "filter": filter_},
    )


async def fetch_historical_stats(stock_name: str, stats: str = "all") -> Dict[str, Any]:
    return await _get(
        "/historical_stats",
        params={"stock_name": stock_name, "stats": stats},
    )


async def fetch_corporate_actions(stock_name: str) -> Dict[str, Any]:
    return await _get("/corporate_actions", params={"stock_name": stock_name})


async def fetch_recent_announcements(stock_name: str) -> Dict[str, Any]:
    return await _get("/recent_announcements", params={"stock_name": stock_name})


async def industry_search(query: str) -> List[Dict[str, Any]]:
    return await _get("/industry_search", params={"query": query})


async def fetch_quotes(symbols: List[str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for s in symbols:
        try:
            results.append(await _get("/stock", params={"symbol": s}))
        except Exception:
            continue
    return results


async def fetch_trending() -> List[Dict[str, Any]]:
    return await _get("/trending")


async def fetch_bse_most_active() -> List[Dict[str, Any]]:
    return await _get("/BSE_most_active")


async def fetch_nse_most_active() -> List[Dict[str, Any]]:
    return await _get("/NSE_most_active")


async def fetch_mutual_funds() -> List[Dict[str, Any]]:
    return await _get("/mutual_funds")


async def fetch_ipos() -> List[Dict[str, Any]]:
    return await _get("/ipo")


async def fetch_commodities() -> List[Dict[str, Any]]:
    return await _get("/commodities")


async def fetch_news(symbol: str, limit: int = 5) -> List[Dict[str, Any]]:
    return await _get("/news", params={"symbol": symbol, "limit": limit})

