from __future__ import annotations

from typing import Any, Dict, Optional

import json
from datetime import datetime

try:
    from google.cloud import bigquery  # type: ignore
except Exception:  # pragma: no cover
    bigquery = None  # type: ignore

from app.settings import settings


_client: Optional["bigquery.Client"] = None


def _get_client():
    global _client
    if bigquery is None:
        return None
    if _client is None:
        try:
            _client = bigquery.Client(project=settings.BIGQUERY_PROJECT_ID or None)  # type: ignore
        except Exception:
            _client = None
    return _client


def _table_ref():
    if not settings.BIGQUERY_PROJECT_ID:
        return None
    dataset = settings.BIGQUERY_DATASET
    table = settings.BIGQUERY_TABLE_EVENTS
    return f"{settings.BIGQUERY_PROJECT_ID}.{dataset}.{table}"


def log_event(kind: str, payload: Dict[str, Any] | None = None, symbol: str | None = None, meta: Dict[str, Any] | None = None) -> bool:
    client = _get_client()
    table = _table_ref()
    if not client or not table:
        return False
    row = {
        "ts": datetime.utcnow().isoformat(),
        "kind": kind,
        "symbol": symbol,
        "payload": json.dumps(payload or {}, default=str)[:900000],  # keep under 1MB
        "meta": json.dumps(meta or {}, default=str)[:200000],
    }
    try:
        errors = client.insert_rows_json(table, [row])  # type: ignore
        return not errors
    except Exception:
        return False



