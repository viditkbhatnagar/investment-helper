from typing import Any, Dict, List

from fastapi import APIRouter

from app.services import indian_api
from app.services.aggregation import (
    sync_news,
    sync_price_shockers,
    sync_52w,
    sync_stock_details,
    sync_historical_data,
    sync_historical_stats,
    sync_corporate_actions,
    sync_recent_announcements,
    sync_all_for_names,
)
from app.services.recommendations import predict_enriched
from app.db import get_cache_database_sync
from datetime import datetime, timedelta
import pandas as pd
from app.services import bq
from app.services.feature_store import build_features_for_symbol
from app.services.train import train_symbol, train_global


router = APIRouter()


@router.get("/stocks")
async def stocks_overview() -> Dict[str, List[Dict[str, Any]]]:
    trending_raw = await _safe(indian_api.fetch_trending)
    bse_raw = await _safe(indian_api.fetch_bse_most_active)
    nse_raw = await _safe(indian_api.fetch_nse_most_active)
    trending = _as_list(trending_raw)
    bse = _as_list(bse_raw)
    nse = _as_list(nse_raw)
    return {"trending": trending, "bse_most_active": bse, "nse_most_active": nse}


@router.get("/mutual-funds")
async def mutual_funds_list() -> List[Dict[str, Any]]:
    return _as_list(await _safe(indian_api.fetch_mutual_funds))


@router.get("/ipos")
async def ipos_list() -> List[Dict[str, Any]]:
    return _as_list(await _safe(indian_api.fetch_ipos))


@router.get("/commodities")
async def commodities_list() -> List[Dict[str, Any]]:
    return _as_list(await _safe(indian_api.fetch_commodities))


async def _safe(fn):
    try:
        return await fn()
    except Exception:
        return []


def _as_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data  # type: ignore[return-value]
    if isinstance(data, dict):
        collected: List[Dict[str, Any]] = []
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                collected.extend(v)
            elif isinstance(v, dict):
                for v2 in v.values():
                    if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                        collected.extend(v2)
        return collected
    return []


@router.post("/sync-news")
async def run_sync_news() -> Dict[str, Any]:
    count = await sync_news()
    return {"synced": count}


@router.post("/sync-price-shockers")
async def run_sync_price_shockers() -> Dict[str, Any]:
    count = await sync_price_shockers()
    return {"synced": count}


@router.post("/sync-52w")
async def run_sync_52w() -> Dict[str, Any]:
    count = await sync_52w()
    return {"synced": count}


@router.post("/sync-stocks")
async def run_sync_stocks() -> Dict[str, Any]:
    count = await sync_stock_details()
    return {"synced": count}


@router.post("/sync-historical-data")
async def run_sync_historical_data() -> Dict[str, Any]:
    count = await sync_historical_data()
    return {"synced": count}


@router.post("/sync-historical-stats")
async def run_sync_historical_stats() -> Dict[str, Any]:
    count = await sync_historical_stats()
    return {"synced": count}


@router.post("/sync-corporate-actions")
async def run_sync_corporate_actions() -> Dict[str, Any]:
    count = await sync_corporate_actions()
    return {"synced": count}


@router.post("/sync-recent-announcements")
async def run_sync_recent_announcements() -> Dict[str, Any]:
    count = await sync_recent_announcements()
    return {"synced": count}


@router.post("/sync-all")
async def run_sync_all() -> Dict[str, Any]:
    return await sync_all_for_names()


@router.get("/backtest")
async def backtest(
    name: str,
    horizon: int = 5,
    window_days: int = 365,
):
    """Expanding-window backtest that retrains enriched model at each step.
    We use the same feature builder as predict_enriched, but at each time t we train on
    data up to t and score t+h. This is slower but higher fidelity.
    """
    from app.services.recommendations import _extract_price_series, _build_features, _train_lgbm_enriched, _append_exog
    from app.services.recommendations import predict_enriched as _predict_once

    db = get_cache_database_sync()
    lname = name.strip().lower()
    key = {"_norm_name": lname, "_period": "10yr", "_filter": "price"}
    hist = db.historical_data.find_one(key) or db.historical_data.find_one({"_norm_name": lname}) or {}
    df = _extract_price_series(hist) if isinstance(hist, dict) else pd.DataFrame()
    if df.empty:
        return {"error": "no historical data"}

    # limit window and build base features
    df = df.tail(max(window_days + horizon + 200, 400))
    feats_full = _build_features(df)
    if feats_full.empty:
        return {"error": "insufficient features"}

    # Get current exogenous vector by calling enriched once (fast path) and reuse for all t
    enriched = _predict_once(name, [horizon])
    explain = enriched.get("explain", {}) if isinstance(enriched, dict) else {}
    # flatten exog similarly to predict_enriched
    exog = {}
    for k in ["prox_52w", "shock", "corp_events", "sales_growth", "risk_meter", "sentiment"]:
        v = explain.get(k)
        if isinstance(v, (int, float)):
            exog[f"feat_{k}"] = float(v)
    for subkey in (explain.get("ai_factors") or {}).keys():
        exog[f"ai_{subkey}"] = float((explain["ai_factors"] or {}).get(subkey) or 0.0)
    for subkey in (explain.get("peer_exog") or {}).keys():
        exog[f"peer_{subkey}"] = float((explain["peer_exog"] or {}).get(subkey) or 0.0)

    # Walk-forward loop
    idx = feats_full.index
    start_i = int(len(idx) * 0.3)  # burn-in
    errors: List[float] = []
    dates: List[str] = []
    y_true_vals: List[float] = []
    y_pred_vals: List[float] = []

    for i in range(start_i, len(idx) - horizon):
        train_slice = feats_full.iloc[:i]
        if len(train_slice) < 150:
            continue
        # train per step
        model, _ = _train_lgbm_enriched(train_slice, horizon, exog)
        last = _append_exog(feats_full.iloc[i : i + 1].copy(), exog)
        try:
            y_hat = float(model.predict(last.values)[0])  # type: ignore[attr-defined]
        except Exception:
            y_hat = float(model.predict(last[[c for c in train_slice.columns if c in last.columns]].values)[0])  # type: ignore[attr-defined]
        # true target at t+h
        true_price = float(df["close"].reindex(feats_full.index).iloc[i + horizon])
        errors.append(abs(true_price - y_hat))
        dates.append(str(idx[i].date()))
        y_true_vals.append(true_price)
        y_pred_vals.append(y_hat)

    if not errors:
        return {"error": "not enough data for backtest"}

    mae = float(pd.Series(errors).mean())
    result = {
        "horizon": int(horizon),
        "window_days": int(window_days),
        "mae": mae,
        "n": int(len(errors)),
        "series": {"date": dates, "y_true": y_true_vals, "y_pred": y_pred_vals},
        "train_metrics": enriched.get("metrics"),
    }
    # Quantile series using saved quantile models and features_daily
    try:
        db = get_cache_database_sync()
        lname = name.strip().lower()
        rows = list(db.features_daily.find({"_norm_name": lname}).sort("date", 1))
        if rows:
            dfF = pd.DataFrame(rows)
            dfF["date"] = pd.to_datetime(dfF["date"])  # type: ignore
            # restrict window
            cutoff = dfF["date"].max() - pd.Timedelta(days=window_days)
            dfF = dfF[dfF["date"] >= cutoff]
            yq = dfF[f"target_{horizon}"] if f"target_{horizon}" in dfF.columns else pd.Series(dtype=float)
            # Try local first
            import base64, pickle
            qdoc = db.models_cache.find_one({"_norm_name": lname, "horizon": int(horizon), "model_type": "lgbm_quantile"})
            if not qdoc:
                qdoc = db.models_cache.find_one({"_norm_name": "__GLOBAL__", "horizon": int(horizon), "model_type": "lgbm_quantile_global"})
            if qdoc and qdoc.get("blob_b64"):
                bundle = pickle.loads(base64.b64decode(qdoc["blob_b64"].encode("ascii")))
                cols = bundle.get("feature_cols") or []
                Xq = dfF.reindex(columns=cols, fill_value=0.0)
                models = bundle.get("models", {})
                def _pred(mkey: str):
                    try:
                        m = models.get(mkey)
                        return m.predict(Xq.values).tolist()
                    except Exception:
                        return []
                q20 = _pred("q20")
                q50 = _pred("q50")
                q80 = _pred("q80")
                dts = dfF["date"].dt.date.astype(str).tolist()
                result["quantile_series"] = {"date": dts, "q20": q20, "q50": q50, "q80": q80}
                # Quantile MAE where target present
                try:
                    mask = yq.notna()
                    mae_q = float(pd.Series(q50)[mask.values].reset_index(drop=True).sub(yq[mask].reset_index(drop=True)).abs().mean())
                    result["quantile_backtest"] = {"mae": mae_q, "n": int(mask.sum())}
                except Exception:
                    pass
    except Exception:
        pass
    try:
        bq.log_event("backtest", payload=result, symbol=name)
    except Exception:
        pass
    return result


@router.post("/features")
async def build_features(name: str) -> Dict[str, Any]:
    n = build_features_for_symbol(name)
    return {"built": n}


@router.post("/train")
async def train(name: str) -> Dict[str, Any]:
    res = train_symbol(name)
    return res


@router.post("/train-global")
async def train_global_models() -> Dict[str, Any]:
    return train_global()

