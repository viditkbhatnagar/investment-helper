from __future__ import annotations

import os
from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


@lru_cache(maxsize=1)
def get_client() -> AsyncIOMotorClient:
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    return AsyncIOMotorClient(uri)


def get_db() -> AsyncIOMotorDatabase:
    db_name = os.getenv("MONGODB_DB", "investment_advisor")
    return get_client()[db_name]


