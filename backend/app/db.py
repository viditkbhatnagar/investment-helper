from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from app.settings import settings


_mongo_client: Optional[AsyncIOMotorClient] = None
_sync_client: Optional[MongoClient] = None


def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
    return _mongo_client


def get_database():
    client = get_mongo_client()
    return client[settings.MONGODB_DB]


def get_cache_database():
    # For sync access, motor returns a database that can be used synchronously for simple find_one
    client = get_mongo_client()
    return client[settings.MONGODB_CACHE_DB]


def get_cache_database_sync():
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(settings.MONGODB_URI)
    return _sync_client[settings.MONGODB_CACHE_DB]


