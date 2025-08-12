from pydantic_settings import BaseSettings
from pydantic import AliasChoices, Field
from typing import List, Optional
from pathlib import Path


class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    # LangChain / LangGraph
    LANGCHAIN_API_KEY: Optional[str] = None
    INDIAN_STOCK_API_BASE: str = Field(
        default="https://stock.indianapi.in",
        validation_alias=AliasChoices("INDIAN_STOCK_API_BASE", "INDIAN_API_BASE"),
    )
    INDIAN_STOCK_API_KEY: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "INDIAN_STOCK_API_KEY",
            "INDIAN_API_KEY",
            "X_API_KEY",
            "X-API-KEY",
        ),
    )

    # Database
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "investment_advisor"
    MONGODB_CACHE_DB: str = "investment_advisor_cache"
    # BigQuery
    BIGQUERY_PROJECT_ID: Optional[str] = None
    BIGQUERY_DATASET: str = "investment_advisor"
    BIGQUERY_TABLE_EVENTS: str = "events"

    # Server
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8010

    # Cache
    QUOTES_CACHE_TTL_SECONDS: int = 60
    REDIS_URL: str = "redis://localhost:6379"

    # Security
    SECRET_KEY: str = "your_secret_key_here"
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8010",
    ]

    class Config:
        # Load .env from backend/ by default so it's unambiguous
        env_file = str(Path(__file__).resolve().parents[1] / ".env")
        case_sensitive = True


settings = Settings()


