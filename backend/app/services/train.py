from __future__ import annotations

from typing import Dict, Any, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from app.db import get_cache_database_sync


def _load_features(symbol: str, horizon: int) -> Tuple[pd.DataFrame, pd.Series]:
    db = get_cache_database_sync()
    lname = symbol.strip().lower()
    rows = list(db.features_daily.find({"_norm_name": lname}).sort("date", 1))
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    df = df.sort_values("date")
    y = df[f"target_{horizon}"].astype(float)
    # Feature columns: exclude id/meta and targets
    drop_cols = {"_id", "_norm_name", "symbol", "date"}
    drop_cols |= {c for c in df.columns if c.startswith("target_")}
    X = df.drop(columns=list(drop_cols), errors="ignore").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def _train_quantile_lgbm(X: pd.DataFrame, y: pd.Series, quantiles: List[float]) -> Dict[str, Any]:
    try:
        import lightgbm as lgb
    except Exception:
        return {"models": {}, "metrics": {}}
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error

    models: Dict[str, Any] = {}
    metrics: Dict[str, float] = {}
    tscv = TimeSeriesSplit(n_splits=3)
    for q in quantiles:
        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "verbosity": -1,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
        }
        model = lgb.LGBMRegressor(**params, n_estimators=600)
        # simple time-series fit with last split for validation
        got_split = False
        for tr, va in tscv.split(X):
            if len(tr) == 0 or len(va) == 0:
                continue
            got_split = True
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[va])
            try:
                metrics[f"mae_q{int(q*100)}"] = float(mean_absolute_error(y.iloc[va], pred))
            except Exception:
                pass
        model.fit(X, y)
        models[f"q{int(q*100)}"] = model
    # SHAP on q50
    shap_summary: Dict[str, float] = {}
    try:
        import shap
        mdl = models.get("q50")
        if mdl is not None:
            Xs = X.tail(min(len(X), 300))
            explainer = shap.TreeExplainer(mdl)
            sv = explainer.shap_values(Xs)
            # sv can be ndarray
            import numpy as np
            vals = np.abs(sv).mean(axis=0)
            shap_summary = {col: float(vals[i]) for i, col in enumerate(Xs.columns)}
    except Exception:
        shap_summary = {}
    return {"models": models, "metrics": metrics, "feature_cols": list(X.columns), "shap_summary": shap_summary}
    return {"models": models, "metrics": metrics, "feature_cols": list(X.columns)}


def train_symbol(symbol: str, horizons: List[int] = [1, 5, 10]) -> Dict[str, Any]:
    db = get_cache_database_sync()
    lname = symbol.strip().lower()
    out: Dict[str, Any] = {}
    for h in horizons:
        X, y = _load_features(symbol, h)
        if X.empty or y.dropna().empty:
            out[str(h)] = {"trained": False}
            continue
        bundle = _train_quantile_lgbm(X, y, quantiles=[0.2, 0.5, 0.8])
        # persist
        try:
            import base64, pickle
            blob = base64.b64encode(pickle.dumps(bundle)).decode("ascii")
            db.models_cache.update_one(
                {"_norm_name": lname, "horizon": int(h), "model_type": "lgbm_quantile"},
                {"$set": {"blob_b64": blob, "metrics": bundle.get("metrics"), "feature_cols": bundle.get("feature_cols"), "shap_summary": bundle.get("shap_summary"), "saved_at": datetime.utcnow()}},
                upsert=True,
            )
            out[str(h)] = {"trained": True, "metrics": bundle.get("metrics")}
        except Exception:
            out[str(h)] = {"trained": False}
    return out


def _load_features_all(horizon: int) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    db = get_cache_database_sync()
    rows = list(db.features_daily.find({f"target_{horizon}": {"$ne": None}}).sort([("date", 1)]))
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float), {}
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])  # ensure datetime
    df = df.sort_values("date")
    # Build symbol id map
    symbols = list(df["symbol"].astype(str).unique())
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    df["symbol_id"] = df["symbol"].astype(str).map(symbol_to_id)
    y = df[f"target_{horizon}"].astype(float)
    drop_cols = {"_id", "_norm_name", "symbol", "date"}
    drop_cols |= {c for c in df.columns if c.startswith("target_")}
    X = df.drop(columns=list(drop_cols), errors="ignore").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y, symbol_to_id


def train_global(horizons: List[int] = [1, 5, 10]) -> Dict[str, Any]:
    db = get_cache_database_sync()
    out: Dict[str, Any] = {}
    for h in horizons:
        X, y, symbol_to_id = _load_features_all(h)
        if X.empty or y.dropna().empty:
            out[str(h)] = {"trained": False}
            continue
        bundle = _train_quantile_lgbm(X, y, quantiles=[0.2, 0.5, 0.8])
        bundle["symbol_to_id"] = symbol_to_id
        try:
            import base64, pickle
            blob = base64.b64encode(pickle.dumps(bundle)).decode("ascii")
            db.models_cache.update_one(
                {"_norm_name": "__GLOBAL__", "horizon": int(h), "model_type": "lgbm_quantile_global"},
                {"$set": {"blob_b64": blob, "metrics": bundle.get("metrics"), "feature_cols": bundle.get("feature_cols"), "shap_summary": bundle.get("shap_summary"), "saved_at": datetime.utcnow()}},
                upsert=True,
            )
            out[str(h)] = {"trained": True, "metrics": bundle.get("metrics"), "num_symbols": len(symbol_to_id)}
        except Exception:
            out[str(h)] = {"trained": False}
    return out


