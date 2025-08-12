from typing import Optional

from redis.asyncio import Redis

from app.settings import settings


_redis: Optional[Redis] = None


async def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    return _redis


