from __future__ import annotations

from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd

from app.db import get_cache_database_sync


def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["dma_20"] = out["close"].rolling(20).mean()
    out["dma_50"] = out["close"].rolling(50).mean()
    out["vol_20"] = out["ret_1d"].rolling(20).std()
    out["mom_10"] = out["close"].pct_change(10)
    out["roll_max_60"] = out["close"].rolling(60).max()
    out["roll_min_60"] = out["close"].rolling(60).min()
    out["prox_roll_60"] = (out["close"] - (out["roll_min_60"] + out["roll_max_60"]) / 2) / (out["roll_max_60"] - out["roll_min_60"]).replace(0, np.nan)
    return out


def _extract_price_series(historical_doc: dict) -> pd.DataFrame:
    datasets = historical_doc.get("datasets", [])
    price_ds = next((ds for ds in datasets if ds.get("metric", "").lower() == "price" or ds.get("label", "").lower().startswith("price")), None)
    if not price_ds:
        return pd.DataFrame()
    rows = price_ds.get("values", [])
    dt, val = [], []
    for r in rows:
        if len(r) >= 2:
            dt.append(pd.to_datetime(r[0]))
            try:
                val.append(float(r[1]))
            except Exception:
                val.append(np.nan)
    df = pd.DataFrame({"date": dt, "close": val}).dropna()
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df


def build_features_for_symbol(symbol: str, horizons: List[int] = [1, 5, 10]) -> int:
    db = get_cache_database_sync()
    lname = symbol.strip().lower()
    key = {"_norm_name": lname, "_period": "10yr", "_filter": "price"}
    hist = db.historical_data.find_one(key) or db.historical_data.find_one({"_norm_name": lname}) or {}
    df = _extract_price_series(hist) if isinstance(hist, dict) else pd.DataFrame()
    if df.empty:
        return 0
    feats = _price_features(df).dropna()

    # Gather DB signals
    details = db.stock_details.find_one({"_norm_name": lname}) or {}
    risk = _safe_num(((details.get("riskMeter") or {}).get("score") if isinstance(details.get("riskMeter"), dict) else details.get("riskMeter")), 0.0)
    stats_doc = db.historical_stats.find_one({"_norm_name": lname}) or {}
    sales_growth = 0.0
    if isinstance(stats_doc, dict):
        q = (stats_doc.get("quarter_results") or stats_doc.get("Sales") or stats_doc.get("quarterResults"))
        if isinstance(q, dict):
            try:
                series = list(q.get("Sales", {}).items()) if "Sales" in q else list(q.items())
                vals = [_safe_num(v) for _, v in series[-6:]]
                if len(vals) >= 2:
                    sales_growth = (vals[-1] - vals[0]) / max(1e-6, abs(vals[0]))
            except Exception:
                pass

    # Recent intel / AI factors and peer aggregates (if present)
    today = df.index[-1].date().isoformat()
    intel = db.intel_daily.find_one({"_norm_name": lname, "date": today}) or {}
    signals = intel.get("signals", {}) if isinstance(intel, dict) else {}
    ai_factors = signals.get("ai_factors", {}) if isinstance(signals, dict) else {}
    peer_exog = signals.get("peer_exog", {}) if isinstance(signals, dict) else {}

    # Create one doc per date with aligned targets for each horizon
    saved = 0
    for idx, row in feats.iterrows():
        doc: Dict[str, Any] = {
            "_norm_name": lname,
            "symbol": symbol,
            "date": idx.date().isoformat(),
            "close": _safe_num(row.get("close")),
            "ret_1d": _safe_num(row.get("ret_1d")),
            "dma_20": _safe_num(row.get("dma_20")),
            "dma_50": _safe_num(row.get("dma_50")),
            "vol_20": _safe_num(row.get("vol_20")),
            "mom_10": _safe_num(row.get("mom_10")),
            "prox_roll_60": _safe_num(row.get("prox_roll_60")),
            "risk": risk,
            "sales_growth": sales_growth,
        }
        # Attach AI and peer factors (static for now; could be dated if stored per-day)
        for k, v in (ai_factors or {}).items():
            doc[f"ai_{k}"] = _safe_num(v)
        for k, v in (peer_exog or {}).items():
            doc[f"peer_{k}"] = _safe_num(v)
        # Targets
        for h in horizons:
            j = feats.index.get_loc(idx)
            if j + h < len(df.index):
                tdate = df.index[j + h]
                doc[f"target_{h}"] = _safe_num(df.loc[tdate, "close"])  # future close
            else:
                doc[f"target_{h}"] = None
        db.features_daily.update_one({"_norm_name": lname, "date": doc["date"]}, {"$set": doc}, upsert=True)
        saved += 1
    return saved


def build_features_for_all(symbols: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in symbols:
        try:
            out[s] = build_features_for_symbol(s)
        except Exception:
            out[s] = 0
    return out


