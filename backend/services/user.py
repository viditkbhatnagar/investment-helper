from __future__ import annotations

from typing import Any, Dict, List

from backend.db import get_db


async def save_profile(user_id: str, profile: Dict[str, Any]) -> None:
    db = get_db()
    await db.users.update_one({"_id": user_id}, {"$set": profile}, upsert=True)


async def get_profile(user_id: str) -> Dict[str, Any] | None:
    db = get_db()
    doc = await db.users.find_one({"_id": user_id})
    return doc


async def save_portfolio(user_id: str, portfolio: List[Dict[str, Any]], perf: Dict[str, Any]) -> str:
    db = get_db()
    doc = {"user_id": user_id, "portfolio": portfolio, "perf": perf}
    res = await db.portfolios.insert_one(doc)
    return str(res.inserted_id)


async def list_portfolios(user_id: str) -> List[Dict[str, Any]]:
    db = get_db()
    rows = []
    async for d in db.portfolios.find({"user_id": user_id}).sort("_id", -1):
        d["_id"] = str(d["_id"])
        rows.append(d)
    return rows


