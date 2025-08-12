from typing import List

from datetime import datetime
from typing import List, Dict, Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.schemas import StockQuote
from app.db import get_cache_database
from app.services import indian_api


def aggregate_quotes(quotes: List[StockQuote]) -> List[StockQuote]:
    # Placeholder for future aggregation logic
    return quotes


async def sync_news(db: AsyncIOMotorDatabase | None = None) -> int:
    db = db or get_cache_database()
    items = await indian_api.fetch_news(symbol="NIFTY")  # provider supports broad news feed
    if not isinstance(items, list):
        return 0
    for doc in items:
        doc["_kind"] = "news"
        doc["_fetched_at"] = datetime.utcnow()
    if items:
        await db.news.delete_many({})
        await db.news.insert_many(items)
    return len(items)


async def sync_price_shockers(db: AsyncIOMotorDatabase | None = None) -> int:
    db = db or get_cache_database()
    items = await indian_api._get("/price_shockers")
    items = _to_list(items)
    if not items:
        return 0
    for doc in items:
        doc["_kind"] = "price_shockers"
        doc["_fetched_at"] = datetime.utcnow()
    await db.price_shockers.delete_many({})
    if items:
        await db.price_shockers.insert_many(items)
    return len(items)


async def sync_52w(db: AsyncIOMotorDatabase | None = None) -> int:
    db = db or get_cache_database()
    items = await indian_api._get("/fetch_52_week_high_low_data")
    items = _to_list(items)
    if not items:
        return 0
    for doc in items:
        doc["_kind"] = "fiftytwo_week"
        doc["_fetched_at"] = datetime.utcnow()
    await db.fiftytwo_week.delete_many({})
    if items:
        await db.fiftytwo_week.insert_many(items)
    return len(items)


def _to_list(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        result: list[dict] = []
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                result.extend(v)
            elif isinstance(v, dict):
                for v2 in v.values():
                    if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                        result.extend(v2)
        return result
    return []


STOCK_NAME_SEED: list[str] = [
    "Adani Enterprises",
    "Tata Motors",
    "Eternal",
    "Hero Motocorp",
    "Bharat Electronics",
    "Tech Mahindra",
    "Tata Steel",
    "State Bank Of India",
    "Reliance Industries",
    "Federal Bank",
    "Ashok Leyland",
    "Bharat Petroleum Corp",
    "Coal India",
    "Indian Oil Corp",
    "Samvardhana Motherson Internatio",
    "Punjab National Bank",
    "Canara Bank",
    "Ntpc",
    "Indian Railway Finance Corporati",
    "Jio Financial Services",
]


async def sync_stock_details(names: list[str] | None = None, db: AsyncIOMotorDatabase | None = None) -> int:
    """Fetch /stock?name= for each provided company name and store one doc per company.
    Upserts by normalized name to avoid duplicates.
    """
    db = db or get_cache_database()
    names = names or STOCK_NAME_SEED
    saved = 0
    for name in names:
        try:
            doc = await indian_api.fetch_stock_by_name(name)
            if not isinstance(doc, dict) or not doc:
                continue
            doc["_kind"] = "stock_detail"
            doc["_fetched_at"] = datetime.utcnow()
            key = {"_norm_name": name.strip().lower()}
            doc.update(key)
            await db.stock_details.replace_one(key, doc, upsert=True)
            saved += 1
        except Exception:
            continue
    return saved


async def sync_historical_data(names: list[str] | None = None, period: str = "10yr", filter_: str = "price", db: AsyncIOMotorDatabase | None = None) -> int:
    db = db or get_cache_database()
    names = names or STOCK_NAME_SEED
    saved = 0
    for name in names:
        try:
            doc = await indian_api.fetch_historical_data(name, period=period, filter_=filter_)
            if not isinstance(doc, dict) or not doc:
                continue
            key = {"_norm_name": name.strip().lower(), "_period": period, "_filter": filter_}
            doc.update({"_kind": "historical_data", "_fetched_at": datetime.utcnow(), **key})
            await db.historical_data.replace_one(key, doc, upsert=True)
            saved += 1
        except Exception:
            continue
    return saved


async def sync_historical_stats(names: list[str] | None = None, stats: str = "all", db: AsyncIOMotorDatabase | None = None) -> int:
    db = db or get_cache_database()
    names = names or STOCK_NAME_SEED
    saved = 0
    for name in names:
        try:
            doc = await indian_api.fetch_historical_stats(name, stats=stats)
            if not isinstance(doc, dict) or not doc:
                continue
            key = {"_norm_name": name.strip().lower(), "_stats": stats}
            doc.update({"_kind": "historical_stats", "_fetched_at": datetime.utcnow(), **key})
            await db.historical_stats.replace_one(key, doc, upsert=True)
            saved += 1
        except Exception:
            continue
    return saved


async def sync_corporate_actions(names: list[str] | None = None, db: AsyncIOMotorDatabase | None = None) -> int:
    db = db or get_cache_database()
    names = names or STOCK_NAME_SEED
    saved = 0
    for name in names:
        try:
            doc = await indian_api.fetch_corporate_actions(name)
            if not isinstance(doc, dict) or not doc:
                continue
            key = {"_norm_name": name.strip().lower()}
            doc.update({"_kind": "corporate_actions", "_fetched_at": datetime.utcnow(), **key})
            await db.corporate_actions.replace_one(key, doc, upsert=True)
            saved += 1
        except Exception:
            continue
    return saved


async def sync_recent_announcements(names: list[str] | None = None, db: AsyncIOMotorDatabase | None = None) -> int:
    db = db or get_cache_database()
    names = names or STOCK_NAME_SEED
    saved = 0
    for name in names:
        try:
            data = await indian_api.fetch_recent_announcements(name)
            # API returns a list of announcements; persist as one doc per company
            if isinstance(data, list):
                doc = {"items": data}
            elif isinstance(data, dict):
                doc = data
            else:
                continue
            key = {"_norm_name": name.strip().lower()}
            doc.update({"_kind": "recent_announcements", "_fetched_at": datetime.utcnow(), **key})
            await db.recent_announcements.replace_one(key, doc, upsert=True)
            saved += 1
        except Exception:
            continue
    return saved


async def sync_all_for_names(names: list[str] | None = None) -> dict:
    """Run all syncs for provided names with basic retries.
    Returns a dict of counters.
    """
    names = names or STOCK_NAME_SEED
    summary = {"stock_details": 0, "historical_data": 0, "historical_stats": 0, "corporate_actions": 0, "recent_announcements": 0}

    async def _retry(coro, label):
        nonlocal summary
        got = 0
        for _ in range(3):
            try:
                got = await coro(names=names)
                if got:
                    break
            except Exception:
                continue
        summary[label] += int(got or 0)

    await _retry(sync_stock_details, "stock_details")
    await _retry(sync_historical_data, "historical_data")
    await _retry(sync_historical_stats, "historical_stats")
    await _retry(sync_corporate_actions, "corporate_actions")
    await _retry(sync_recent_announcements, "recent_announcements")
    return summary


