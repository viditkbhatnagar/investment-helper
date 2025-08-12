from fastapi import APIRouter
from app.db import get_cache_database_sync
from datetime import datetime


router = APIRouter()


@router.get("/ping")
async def ping():
    return {"status": "ok"}


@router.get("/version")
async def version():
    return {"service": "investment-advisor", "version": 1}


@router.get("/models-cache")
async def models_cache_status():
    db = get_cache_database_sync()
    try:
        n = db.models_cache.count_documents({})
    except Exception:
        n = 0
    return {"models_cached": int(n), "time": datetime.utcnow().isoformat()}


