from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import settings
from app.middleware.logging import RequestLoggingMiddleware
from app.routers import health, recommendations
from app.routers import market


app = FastAPI(title="Investment Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.add_middleware(RequestLoggingMiddleware)


app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(
    recommendations.router, prefix="/api/recommendations", tags=["recommendations"]
)
app.include_router(market.router, prefix="/api/market", tags=["market"])

# Simple in-process scheduler using asyncio tasks
import asyncio
# Scheduler disabled per request
# from app.services.aggregation import sync_news, sync_price_shockers, sync_52w, sync_stock_details
# async def _scheduler():
#     await sync_news(); await sync_price_shockers(); await sync_52w(); await sync_stock_details()
#     while True:
#         await asyncio.sleep(60 * 60 * 24)
#         await sync_news(); await sync_price_shockers(); await sync_52w(); await sync_stock_details()
# @app.on_event("startup")
# async def start_background_tasks():
#     asyncio.create_task(_scheduler())


@app.get("/")
def root():
    return {"status": "ok", "service": "investment-advisor"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)


