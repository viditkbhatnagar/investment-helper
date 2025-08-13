from typing import List, Tuple, Dict, Any

from anthropic import Anthropic
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import pickle
import base64
import hashlib

from app.schemas import Recommendation, StockQuote
from app.settings import settings
from app.db import get_cache_database, get_cache_database_sync
from datetime import datetime, timedelta
import json
from app.services import indian_api
from app.services import aggregation
from app.services import bq
from app.services import model_store
import asyncio
import math


def _rate(percent_change: float) -> str:
    if percent_change >= 1.0:
        return "strong_buy"
    if percent_change >= 0.3:
        return "buy"
    if percent_change <= -1.0:
        return "strong_sell"
    if percent_change <= -0.3:
        return "sell"
    return "hold"


def get_recommendations(user_id: str) -> List[Recommendation]:
    return [Recommendation(symbol="TCS", rating="hold")]


def get_recommendations_from_quotes(quotes: List[StockQuote]) -> List[Recommendation]:
    recs: List[Recommendation] = []
    for q in quotes:
        recs.append(
            Recommendation(
                symbol=q.symbol,
                rating=_rate(q.percent_change),
                rationale=f"Intraday move {q.percent_change:.2f}%",
            )
        )
    return recs


def summarize_news_sentiment(symbol: str, news_items: List[dict]) -> str:
    if not settings.ANTHROPIC_API_KEY:
        return ""
    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    joined = "\n\n".join(
        [f"Title: {n.get('title','')}\nSummary: {n.get('summary','')}" for n in news_items[:5]]
    )
    prompt = (
        "You are a financial analyst. Given recent news for the stock, provide a short, neutral sentiment "
        "summary (Bullish, Bearish, Neutral) with 1-line rationale.\n\n" + joined
    )
    msg = client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    # type: ignore[attr-defined]
    return getattr(msg, "content", "") or ""


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


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price-derived technical features including MACD, RSI, and moving averages.
    Ensures no NaNs/Infs are returned.
    """
    if df.empty:
        return df
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["dma_20"] = out["close"].rolling(20).mean()
    out["dma_50"] = out["close"].rolling(50).mean()
    out["dma_100"] = out["close"].rolling(100).mean()
    out["dma_200"] = out["close"].rolling(200).mean()
    out["vol_20"] = out["ret_1d"].rolling(20).std()
    out["mom_10"] = out["close"].pct_change(10)
    # price relative to moving averages
    out["close_over_dma20"] = out["close"] / out["dma_20"]
    out["close_over_dma50"] = out["close"] / out["dma_50"]
    out["close_over_dma100"] = out["close"] / out["dma_100"]
    out["close_over_dma200"] = out["close"] / out["dma_200"]
    # momentum indicators
    out["mom_5"] = out["close"].pct_change(5)
    out["mom_20"] = out["close"].pct_change(20)
    # rolling max/min proximity (approximate 52w within our window if needed)
    out["roll_max_60"] = out["close"].rolling(60).max()
    out["roll_min_60"] = out["close"].rolling(60).min()
    denom = (out["roll_max_60"] - out["roll_min_60"]).replace(0, np.nan)
    out["prox_roll_60"] = (out["close"] - (out["roll_min_60"] + out["roll_max_60"]) / 2) / denom
    # RSI(14)
    window_rsi = 14
    delta = out["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window_rsi).mean()
    avg_loss = loss.rolling(window_rsi).mean().replace(0, np.nan)
    rs = avg_gain / avg_loss
    out["rsi_14"] = 100 - (100 / (1 + rs))
    # MACD (12, 26, 9)
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal
    # Cleanup
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def _train_model(df: pd.DataFrame, horizon_days: int) -> Tuple[Pipeline, float]:
    data = df.copy()
    data["target"] = data["close"].shift(-horizon_days)
    data = data.dropna()
    if len(data) < 100:
        model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())])
    else:
        model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("rf", RandomForestRegressor(n_estimators=200, random_state=42))])
    X = data[["close", "ret_1d", "dma_20", "dma_50", "vol_20", "mom_10"]].values
    y = data["target"].values
    split = int(len(data) * 0.8)
    model.fit(X[:split], y[:split])
    if split < len(data):
        pred = model.predict(X[split:])
        mae = float(mean_absolute_error(y[split:], pred))
    else:
        mae = float(np.nan)
    return model, mae


def _train_lgbm(df: pd.DataFrame, horizon_days: int) -> Tuple[object, float]:
    """Train a LightGBM regressor if available; fallback to sklearn pipeline elsewhere."""
    try:
        import lightgbm as lgb
    except Exception:
        return _train_model(df, horizon_days)
    data = df.copy()
    data["target"] = data["close"].shift(-horizon_days)
    data = data.dropna()
    if len(data) < 150:
        return _train_model(df, horizon_days)
    X = data[["close", "ret_1d", "dma_20", "dma_50", "vol_20", "mom_10"]].values
    y = data["target"].values
    split = int(len(data) * 0.8)
    train = lgb.Dataset(X[:split], label=y[:split])
    valid = lgb.Dataset(X[split:], label=y[split:]) if split < len(data) else None
    params = {"objective": "regression", "metric": "l1", "verbosity": -1, "num_leaves": 31, "learning_rate": 0.05}
    model = lgb.train(params, train, num_boost_round=400, valid_sets=[valid] if valid else None)
    if split < len(data):
        pred = model.predict(X[split:])
        mae = float(mean_absolute_error(y[split:], pred))
    else:
        mae = float(np.nan)
    return model, mae


def _prepare_supervised(df: pd.DataFrame, horizon_days: int, exog: Dict[str, float]):
    data = _append_exog(df.copy(), exog)
    data["target"] = data["close"].shift(-horizon_days)
    data = data.dropna()
    feature_cols = [c for c in data.columns if c not in ("target",)]
    X = data[feature_cols].values
    y = data["target"].values
    return data, feature_cols, X, y


def _train_lgbm_enriched(df: pd.DataFrame, horizon_days: int, exog: Dict[str, float]) -> Tuple[object, Dict[str, float]]:
    """Train LightGBM with constant exogenous features appended. Returns model and metrics.
    metrics contains: mae, mape, cv_mae (TimeSeriesSplit), n_samples, horizon
    """
    try:
        import lightgbm as lgb
    except Exception:
        model, mae = _train_model(_append_exog(df, exog), horizon_days)
        return model, {"mae": mae, "mape": float("nan"), "cv_mae": float("nan"), "n_samples": float(len(df)), "horizon": float(horizon_days), "features": list(_append_exog(df, exog).columns)}

    data, feature_cols, X, y = _prepare_supervised(df, horizon_days, exog)
    metrics = {"mae": float("nan"), "mape": float("nan"), "cv_mae": float("nan"), "n_samples": float(len(data)), "horizon": float(horizon_days)}
    metrics["features"] = feature_cols
    if len(data) < 150:
        # small sample fallback to baseline
        model, mae = _train_model(data.drop(columns=["target"]), horizon_days)
        metrics["mae"] = mae
        return model, metrics

    split = int(len(data) * 0.8)
    train = lgb.Dataset(X[:split], label=y[:split])
    valid = lgb.Dataset(X[split:], label=y[split:]) if split < len(data) else None
    params = {"objective": "regression", "metric": "l1", "verbosity": -1, "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.9}
    model = lgb.train(params, train, num_boost_round=600, valid_sets=[valid] if valid else None)
    if split < len(data):
        pred = model.predict(X[split:])
        mae = float(mean_absolute_error(y[split:], pred))
        try:
            mape = float(mean_absolute_percentage_error(y[split:], pred))
        except Exception:
            mape = float("nan")
        metrics["mae"] = mae
        metrics["mape"] = mape

    # simple time-series CV
    try:
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=3)
        fold_mae: List[float] = []
        for tr_idx, va_idx in tscv.split(X):
            tr = lgb.Dataset(X[tr_idx], label=y[tr_idx])
            va = lgb.Dataset(X[va_idx], label=y[va_idx])
            m = lgb.train(params, tr, num_boost_round=400, valid_sets=[va])
            p = m.predict(X[va_idx])
            fold_mae.append(float(mean_absolute_error(y[va_idx], p)))
        if fold_mae:
            metrics["cv_mae"] = float(np.mean(fold_mae))
    except Exception:
        pass

    return model, metrics


def _append_exog(df: pd.DataFrame, exog: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for k, v in (exog or {}).items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = 0.0
    return out


# ============================
# LSTM + Keras Tuner pipeline
# ============================

def _build_seq_dataset(data: pd.DataFrame, target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    vals = data.values
    tgt_idx = list(data.columns).index(target_col)
    for i in range(seq_len, len(data)):
        X_list.append(vals[i - seq_len : i, :])
        y_list.append(vals[i, tgt_idx])
    if not X_list:
        return np.empty((0, seq_len, data.shape[1])), np.empty((0,))
    return np.stack(X_list), np.array(y_list)


def _train_lstm_tuned(feats: pd.DataFrame, horizon_days: int, exog: Dict[str, float]) -> Tuple[object, Dict[str, Any], List[str], int, object]:
    """Train a tuned LSTM on normalized technical features predicting RETURNS.
    Target is r_t = close_{t+h}/close_t - 1 to keep outputs near 0 and anchored to last price later.
    Returns (model, metrics, feature_cols, seq_len, scaler).
    If TensorFlow/Keras Tuner are unavailable or data is too small, falls back to LightGBM enriched model.
    """
    try:
        import tensorflow as tf  # type: ignore
        import keras_tuner as kt  # type: ignore
        from tensorflow import keras  # type: ignore
    except Exception:
        mdl, met = _train_lgbm_enriched(feats, horizon_days, exog)
        return mdl, met, met.get("features") or list(feats.columns), 30, None

    data = _append_exog(feats.copy(), exog)
    # target as forward return over horizon
    data["target"] = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    data = data.dropna()
    if len(data) < 250:
        mdl, met = _train_lgbm_enriched(feats, horizon_days, exog)
        return mdl, met, met.get("features") or list(feats.columns), 30, None

    feature_cols = [c for c in data.columns if c != "target"]
    X_full = data[feature_cols].astype(float)
    y_full = data["target"].astype(float)

    # Time-ordered split: 70% train, 15% val, 15% test
    n = len(data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_df = X_full.iloc[:n_train]
    val_df = X_full.iloc[n_train : n_train + n_val]
    test_df = X_full.iloc[n_train + n_val :]
    y_train = y_full.iloc[:n_train]
    y_val = y_full.iloc[n_train : n_train + n_val]
    y_test = y_full.iloc[n_train + n_val :]

    # Normalize using training stats only
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    train_n = pd.DataFrame(scaler.transform(train_df.values), index=train_df.index, columns=feature_cols)
    val_n = pd.DataFrame(scaler.transform(val_df.values), index=val_df.index, columns=feature_cols)
    test_n = pd.DataFrame(scaler.transform(test_df.values), index=test_df.index, columns=feature_cols)

    seq_len = 30
    # assemble datasets with the target aligned at end of each sequence
    train_n["y"] = y_train
    val_n["y"] = y_val
    test_n["y"] = y_test

    X_train, y_train_seq = _build_seq_dataset(train_n[feature_cols + ["y"]], target_col="y", seq_len=seq_len)
    X_val, y_val_seq = _build_seq_dataset(val_n[feature_cols + ["y"]], target_col="y", seq_len=seq_len)
    X_test, y_test_seq = _build_seq_dataset(test_n[feature_cols + ["y"]], target_col="y", seq_len=seq_len)

    if X_train.shape[0] < 50 or X_val.shape[0] < 10:
        mdl, met = _train_lgbm_enriched(feats, horizon_days, exog)
        return mdl, met, met.get("features") or list(feats.columns), seq_len, None

    input_shape = (seq_len, len(feature_cols))

    def build_model(hp: "kt.HyperParameters"):
        model = keras.Sequential()
        units1 = hp.Int("units1", min_value=32, max_value=256, step=32)
        model.add(keras.layers.LSTM(units1, return_sequences=hp.Boolean("return_seq", default=True), input_shape=input_shape))
        model.add(keras.layers.Dropout(hp.Float("drop1", 0.0, 0.5, step=0.1)))
        if hp.Boolean("stack_second", default=True):
            units2 = hp.Int("units2", min_value=16, max_value=128, step=16)
            model.add(keras.layers.LSTM(units2))
            model.add(keras.layers.Dropout(hp.Float("drop2", 0.0, 0.5, step=0.1)))
        model.add(keras.layers.Dense(hp.Int("dense", 16, 128, step=16), activation="relu"))
        # Predict bounded return via tanh, then scale to a reasonable daily band (~15%)
        model.add(keras.layers.Dense(1, activation="tanh"))
        model.add(keras.layers.Lambda(lambda x: 0.15 * x))
        lr = hp.Choice("lr", values=[1e-4, 3e-4, 1e-3])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mae", metrics=[keras.metrics.MAE, keras.metrics.MAPE])
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective=kt.Objective("val_mae", direction="min"),
        max_trials=6,
        overwrite=True,
        directory="/tmp/keras_tuner",
        project_name=f"lstm_{horizon_days}d",
    )

    early = keras.callbacks.EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True)
    tuner.search(
        X_train,
        y_train_seq,
        epochs=30,
        validation_data=(X_val, y_val_seq),
        callbacks=[early],
        verbose=0,
    )
    model = tuner.get_best_models(num_models=1)[0]

    # Evaluate (MAE/MAPE on returns)
    def _eval(x, y):
        res = model.evaluate(x, y, verbose=0)
        # res aligns with [loss, mae, mape]
        mae_val = float(res[1] if len(res) > 1 else res)
        mape_val = float(res[2]) if len(res) > 2 else float("nan")
        return mae_val, mape_val

    train_mae, train_mape = _eval(X_train, y_train_seq)
    val_mae, val_mape = _eval(X_val, y_val_seq)
    test_mae, test_mape = _eval(X_test, y_test_seq)
    val_gap = float(val_mae - train_mae)

    metrics = {
        "model": "lstm",
        "horizon": float(horizon_days),
        "n_train": int(len(y_train_seq)),
        "n_val": int(len(y_val_seq)),
        "n_test": int(len(y_test_seq)),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "train_mape": float(train_mape),
        "val_mape": float(val_mape),
        "test_mape": float(test_mape),
        "val_gap": float(val_gap),
        "features": feature_cols,
        "seq_len": int(seq_len),
    }

    return model, metrics, feature_cols, seq_len, scaler


def _train_lgbm_return_enriched(df: pd.DataFrame, horizon_days: int, exog: Dict[str, float]) -> Tuple[object, Dict[str, Any], List[str], StandardScaler]:
    """LightGBM on forward returns r = close(t+h)/close(t) - 1. More stable and anchored.
    Returns: (model, metrics) where metrics includes mae_ret, mape_ret, and features.
    """
    try:
        import lightgbm as lgb
    except Exception:
        # Fallback: linear model on returns
        data = _append_exog(df.copy(), exog)
        data["target"] = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
        data = data.dropna()
        feature_cols = [c for c in data.columns if c != "target"]
        X = data[feature_cols].values
        y = data["target"].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())])
        split = int(len(data) * 0.8)
        model.fit(Xs[:split], y[:split])
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        if split < len(data):
            pred = model.predict(Xs[split:])
            mae = float(mean_absolute_error(y[split:], pred))
            try:
                mape = float(mean_absolute_percentage_error(y[split:], pred))
            except Exception:
                mape = float("nan")
        else:
            mae, mape = float("nan"), float("nan")
        return model, {"mae_ret": mae, "mape_ret": mape, "features": feature_cols}, feature_cols, scaler

    data = _append_exog(df.copy(), exog)
    data["target"] = (data["close"].shift(-horizon_days) / data["close"]) - 1.0
    data = data.dropna()
    feature_cols = [c for c in data.columns if c != "target"]
    X = data[feature_cols].values
    y = data["target"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if len(data) < 150:
        # use baseline regression pipeline
        model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("lin", LinearRegression())])
        split = int(len(data) * 0.8)
        model.fit(Xs[:split], y[:split])
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
        if split < len(data):
            pred = model.predict(Xs[split:])
            mae = float(mean_absolute_error(y[split:], pred))
            try:
                mape = float(mean_absolute_percentage_error(y[split:], pred))
            except Exception:
                mape = float("nan")
        else:
            mae, mape = float("nan"), float("nan")
        return model, {"mae_ret": mae, "mape_ret": mape, "features": feature_cols}, feature_cols, scaler

    split = int(len(data) * 0.8)
    train = lgb.Dataset(Xs[:split], label=y[:split])
    valid = lgb.Dataset(Xs[split:], label=y[split:]) if split < len(data) else None
    params = {"objective": "regression", "metric": "l1", "verbosity": -1, "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.9}
    model = lgb.train(params, train, num_boost_round=600, valid_sets=[valid] if valid else None)
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    mae = float("nan")
    mape = float("nan")
    if split < len(data):
        pred = model.predict(Xs[split:])
        mae = float(mean_absolute_error(y[split:], pred))
        try:
            mape = float(mean_absolute_percentage_error(y[split:], pred))
        except Exception:
            mape = float("nan")
    metrics = {"mae_ret": mae, "mape_ret": mape, "n_samples": float(len(data)), "horizon": float(horizon_days), "features": feature_cols}
    return model, metrics, feature_cols, scaler


def _get_blend_alpha(lname_key: str, horizon_key: int) -> float:
    """Fetch blend alpha from cache DB: weight for model vs AI. Default 0.7."""
    try:
        doc = get_cache_database_sync().models_cache.find_one({
            "_norm_name": lname_key,
            "horizon": int(horizon_key),
            "model_type": "blend_alpha",
        })
        a = doc.get("alpha") if isinstance(doc, dict) else None
        if a is None:
            return 0.7
        a = float(a)
        if a < 0.0 or a > 1.0:
            return 0.7
        return a
    except Exception:
        return 0.7


def predict_price_range(symbol: str, horizons: List[int]) -> dict:
    db = get_cache_database_sync()
    key = {"_norm_name": symbol.strip().lower(), "_period": "10yr", "_filter": "price"}
    # Motor may return awaitables depending on context; guard against Future
    try:
        hist = db.historical_data.find_one(key)
        if hasattr(hist, "result"):
            hist = hist.result()
    except Exception:
        hist = None
    if not isinstance(hist, dict) or not hist:
        try:
            hist = db.historical_data.find_one({"_norm_name": symbol.strip().lower()})
            if hasattr(hist, "result"):
                hist = hist.result()
        except Exception:
            hist = None
    if not hist:
        return {"error": "historical data not found"}
    df = _extract_price_series(hist)
    feats = _build_features(df)
    results = {}
    for h in horizons:
        if feats.empty:
            results[str(h)] = {"point": None, "range": None}
            continue
        model, mae = _train_model(feats, h)
        last = feats[["close", "ret_1d", "dma_20", "dma_50", "vol_20", "mom_10"]].iloc[-1:].values
        point = float(model.predict(last)[0])
        # Use MAE as uncertainty; widen a bit for longer horizons
        spread = (mae if not np.isnan(mae) else feats["close"].std()) * (1 + h / 10)
        results[str(h)] = {"point": point, "range": [max(0.0, point - spread), point + spread]}
    return results


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fetch_company_rows(lst: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    nm = name.strip().lower()
    out: List[Dict[str, Any]] = []
    for row in lst or []:
        for key in ("company", "company_name", "companyName", "commonName"):
            if key in row and str(row[key]).strip().lower().startswith(nm[:6]):
                out.append(row)
                break
    return out


def predict_enriched(symbol: str, horizons: List[int]) -> Dict[str, Any]:
    """Predict with extra features (52w proximity, shockers, corp actions, stats, announcements)."""
    db = get_cache_database_sync()
    lname = symbol.strip().lower()

    # Prepare historical series and technical features
    key = {"_norm_name": lname, "_period": "10yr", "_filter": "price"}
    hist = db.historical_data.find_one(key) or db.historical_data.find_one({"_norm_name": lname}) or {}
    df_series = _extract_price_series(hist) if isinstance(hist, dict) else pd.DataFrame()
    feats = _build_features(df_series) if not df_series.empty else pd.DataFrame()

    # 52-week proximity
    fifty = db.fiftytwo_week.find_one({}) or {}
    company_hits: List[Dict[str, Any]] = []
    for sec in ["BSE_52WeekHighLow", "NSE_52WeekHighLow", "BSE_52WeekHighLow".lower(), "NSE_52WeekHighLow".lower()]:
        block = fifty.get(sec, {}) if isinstance(fifty, dict) else {}
        for cat in ("high52Week", "low52Week", "high52week", "low52week"):
            rows = block.get(cat, []) if isinstance(block, dict) else []
            company_hits.extend(_fetch_company_rows(rows, symbol))
    prox_52w = 0.0
    if company_hits:
        # use first hit
        r = company_hits[0]
        price = _numeric(r.get("price") or r.get("Price"), 0.0)
        h = _numeric(r.get("52_week_high") or r.get("high"), price)
        l = _numeric(r.get("52_week_low") or r.get("low"), price)
        if h and l:
            prox_52w = 2 * (price - (l + h) / 2) / max(1e-6, (h - l))

    # Price shocker flag
    shockers = list(db.price_shockers.find({}))
    shock_flag = 0
    for doc in shockers:
        if isinstance(doc, dict):
            rows = doc if isinstance(doc, list) else doc.get("items") or doc
    rows = []
    for doc in shockers:
        if isinstance(doc, dict):
            rows = doc.get("items") if "items" in doc else doc
        if isinstance(rows, list) and _fetch_company_rows(rows, symbol):
            shock_flag = 1
            break

    # Corporate actions count (approx)
    corp = db.corporate_actions.find_one({"_norm_name": lname}) or {}
    corp_count = 0
    if isinstance(corp, dict):
        for k, v in corp.items():
            if k.endswith("meetings") and isinstance(v, dict):
                data = v.get("data", [])
                if isinstance(data, list):
                    corp_count += len(data)

    # Historical stats (quarter_results): slope of Sales over last few points
    stats_doc = db.historical_stats.find_one({"_norm_name": lname}) or {}
    sales_growth = 0.0
    if isinstance(stats_doc, dict):
        # try several nestings
        q = (stats_doc.get("quarter_results") or stats_doc.get("Sales") or stats_doc.get("quarterResults"))
        if isinstance(q, dict):
            try:
                series = list(q.get("Sales", {}).items()) if "Sales" in q else list(q.items())
                vals = [ _numeric(v) for _, v in series[-6:] ]
                if len(vals) >= 2:
                    sales_growth = (vals[-1] - vals[0]) / max(1e-6, abs(vals[0]))
            except Exception:
                pass

    # Stock details risk meter approx and industry
    details = db.stock_details.find_one({"_norm_name": lname}) or {}
    details = details or {}
    risk = _numeric(((details.get("riskMeter") or {}).get("score") if isinstance(details.get("riskMeter"), dict) else details.get("riskMeter")), 0.0)
    industry = details.get("industry") or details.get("mgIndustry") or ""

    # Recent announcements sentiment (Claude) or keyword-only score, broaden to industry news
    ann = db.recent_announcements.find_one({"_norm_name": lname}) or {}
    items = ann.get("items", []) if isinstance(ann, dict) else []
    titles = "\n".join([str(i.get("title", "")) for i in items[:10] if isinstance(i, dict)])
    # Pull fresh company news via API if available
    try:
        fresh = indian_api._get("/news", params={"symbol": symbol})  # type: ignore
        if isinstance(fresh, list):
            titles2 = "\n".join([str(i.get("title", "")) for i in fresh[:10] if isinstance(i, dict)])
            if titles2:
                titles = titles2 + ("\n" + titles if titles else "")
    except Exception:
        pass
    # try industry-wide items from news collection if industry present
    if not titles:
        if industry:
            # naive search: collect news items that contain the industry keyword
            news_doc = db.news.find_one({}) or {}
            pool = []
            if isinstance(news_doc, dict):
                for k, v in news_doc.items():
                    if isinstance(v, list):
                        pool.extend(v)
            filt = [n for n in pool if isinstance(n, dict) and industry.lower() in str(n.get("title", "")).lower()]
            titles = "\n".join([str(i.get("title", "")) for i in filt[:10]])
    sentiment_score = 0.0
    if titles and settings.ANTHROPIC_API_KEY:
        try:
            client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            msg = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=128,
                messages=[{"role": "user", "content": f"Rate the sentiment of these headlines on a -1 to +1 scale and return only a number.\n{titles}"}],
            )
            text = getattr(msg, "content", "") or ""
            sentiment_score = _numeric(str(text).strip().split()[0], 0.0)
        except Exception:
            sentiment_score = 0.0
    elif titles:
        text = titles.lower()
        sentiment_score = text.count("up") - text.count("down") - text.count("loss") + text.count("profit")
        sentiment_score = float(sentiment_score) / 10.0

    # Peer-based aggregates from DB, by industry
    def _peer_aggregates_from_db(industry_name: str) -> Dict[str, float]:
        if not industry_name:
            return {}
        peer_docs = list(db.stock_details.find({"$or": [{"industry": industry_name}, {"mgIndustry": industry_name}]}).limit(30))
        risks: List[float] = []
        sales: List[float] = []
        corp_counts: List[int] = []
        # flatten shockers company names for quick membership
        shockers_docs = list(db.price_shockers.find({}))
        shock_names: List[str] = []
        for sd in shockers_docs:
            rows = sd.get("items") if isinstance(sd, dict) else None
            if isinstance(rows, list):
                for r in rows:
                    for key in ("company", "company_name", "companyName"):
                        if isinstance(r, dict) and key in r:
                            shock_names.append(str(r[key]).strip().lower())
        peer_count = 0
        shock_hits = 0
        for d in peer_docs:
            if not isinstance(d, dict):
                continue
            nm = str(d.get("_norm_name") or "").strip().lower()
            if not nm or nm == lname:
                continue
            peer_count += 1
            # risk
            rv = (d.get("riskMeter") or {})
            rv = rv.get("score") if isinstance(rv, dict) else rv
            risks.append(_numeric(rv, 0.0))
            # corp actions
            cdoc = db.corporate_actions.find_one({"_norm_name": nm}) or {}
            ccount = 0
            if isinstance(cdoc, dict):
                for k, v in cdoc.items():
                    if k.endswith("meetings") and isinstance(v, dict):
                        data = v.get("data", [])
                        if isinstance(data, list):
                            ccount += len(data)
            corp_counts.append(ccount)
            # sales growth
            sdoc = db.historical_stats.find_one({"_norm_name": nm}) or {}
            sg = 0.0
            if isinstance(sdoc, dict):
                q = (sdoc.get("quarter_results") or sdoc.get("Sales") or sdoc.get("quarterResults"))
                if isinstance(q, dict):
                    try:
                        series = list(q.get("Sales", {}).items()) if "Sales" in q else list(q.items())
                        vals = [_numeric(v) for _, v in series[-6:]]
                        if len(vals) >= 2:
                            sg = (vals[-1] - vals[0]) / max(1e-6, abs(vals[0]))
                    except Exception:
                        sg = 0.0
            sales.append(sg)
            # shock membership
            if nm in shock_names:
                shock_hits += 1
        out: Dict[str, float] = {}
        if peer_count:
            out["peer_risk_mean"] = float(np.nanmean(risks) if risks else 0.0)
            out["peer_sales_growth_mean"] = float(np.nanmean(sales) if sales else 0.0)
            out["peer_corp_events_mean"] = float(np.nanmean(corp_counts) if corp_counts else 0.0)
            out["peer_shock_rate"] = float(shock_hits) / float(peer_count)
        return out

    peer_exog = _peer_aggregates_from_db(industry)

    # AI numeric factors from details + stats + headlines
    ai_factors: Dict[str, float] = {}
    if settings.ANTHROPIC_API_KEY:
        try:
            client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            # keep stats digest small to avoid token bloat
            stats_digest = {}
            if isinstance(stats_doc, dict):
                for k in list(stats_doc.keys())[:5]:
                    stats_digest[k] = stats_doc.get(k)
            prompt = (
                "Extract numeric forward-looking factors for this stock as STRICT JSON object with keys: "
                "forward_eps_growth (-1..1), demand_outlook (-1..1), cost_pressure (-1..1), regulatory_risk (0..1), "
                "management_guidance (0..1), upcoming_catalyst_score (0..1). No commentary.\n\n"
                f"Details: {json.dumps(details)[:1500]}\n\n"
                f"Stats: {json.dumps(stats_digest)[:1500]}\n\n"
                f"Headlines: {titles[:1500]}\n"
            )
            msg = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = getattr(msg, "content", "") or ""
            parsed = {}
            try:
                parsed = json.loads(str(raw))
            except Exception:
                parsed = {}
            for k in [
                "forward_eps_growth",
                "demand_outlook",
                "cost_pressure",
                "regulatory_risk",
                "management_guidance",
                "upcoming_catalyst_score",
            ]:
                ai_factors[k] = _numeric(parsed.get(k), 0.0) if isinstance(parsed, dict) else 0.0
        except Exception:
            ai_factors = {}

    # Adjust base predictions with explanatory signals
    explain: Dict[str, Any] = {
        "prox_52w": prox_52w,
        "shock": shock_flag,
        "corp_events": corp_count,
        "sales_growth": sales_growth,
        "risk_meter": risk,
        "sentiment": sentiment_score,
        "ai_factors": ai_factors,
        "peer_exog": peer_exog,
    }

    adjusted: Dict[str, Any] = {}
    # today price and intraday range
    today_price = None
    today_date = datetime.utcnow().date().isoformat()
    # derive today price and volatility from historical series
    df0 = df_series
    if not df0.empty:
        today_price = float(df0["close"].iloc[-1])
        ret = df0["close"].pct_change().dropna()
        vol = float(ret.rolling(20).std().iloc[-1]) if len(ret) > 20 else float(ret.std()) if len(ret) else 0.02
    else:
        vol = 0.02
    widen = 1.3 if shock_flag else 1.0
    intraday_spread = (today_price or 0.0) * (vol * 2.5) * widen
    intraday_range = None
    if today_price:
        intraday_range = [float(max(0.0, today_price - intraday_spread)), float(today_price + intraday_spread)]
    # Build exogenous feature vector from explain, clamp values
    exog_vec: Dict[str, float] = {
        "feat_prox_52w": float(np.clip(prox_52w, -3.0, 3.0)),
        "feat_shock": float(shock_flag),
        "feat_corp_events": float(corp_count),
        "feat_sales_growth": float(np.clip(sales_growth, -3.0, 3.0)),
        "feat_risk_meter": float(np.clip(risk, 0.0, 10.0)),
        "feat_sentiment": float(np.clip(sentiment_score, -1.0, 1.0)),
    }
    for k, v in (ai_factors or {}).items():
        try:
            exog_vec[f"ai_{k}"] = float(v)
        except Exception:
            exog_vec[f"ai_{k}"] = 0.0
    for k, v in (peer_exog or {}).items():
        try:
            exog_vec[f"peer_{k}"] = float(v)
        except Exception:
            exog_vec[f"peer_{k}"] = 0.0

    def _make_fingerprint(cols: List[str], exog_keys: List[str]) -> str:
        base = ",".join(sorted([c for c in cols])) + "|" + ",".join(sorted([k for k in exog_keys]))
        return hashlib.sha1(base.encode("utf-8")).hexdigest()

    def _load_model_cache(lname_key: str, horizon_key: int, fingerprint: str):
        try:
            dbs = get_cache_database_sync()
            doc = dbs.models_cache.find_one({"_norm_name": lname_key, "horizon": int(horizon_key), "model_type": "lgbm_enriched", "fingerprint": fingerprint})
            if not doc:
                return None
            saved_at = doc.get("saved_at")
            # optional TTL check (24h)
            try:
                age_hours = (datetime.utcnow() - saved_at).total_seconds() / 3600.0 if saved_at else 9999
            except Exception:
                age_hours = 9999
            if age_hours > 24:
                return None
            b64 = doc.get("blob_b64")
            if not b64:
                return None
            model = pickle.loads(base64.b64decode(b64.encode("ascii")))
            return {"model": model, "metrics": doc.get("metrics") or {}, "feature_cols": doc.get("feature_cols") or []}
        except Exception:
            return None

    def _save_model_cache(lname_key: str, horizon_key: int, fingerprint: str, model_obj: object, metrics: Dict[str, Any], feature_cols: List[str]):
        try:
            blob = base64.b64encode(pickle.dumps(model_obj)).decode("ascii")
            get_cache_database_sync().models_cache.update_one(
                {"_norm_name": lname_key, "horizon": int(horizon_key), "model_type": "lgbm_enriched", "fingerprint": fingerprint},
                {"$set": {"blob_b64": blob, "metrics": metrics, "feature_cols": feature_cols, "saved_at": datetime.utcnow()}},
                upsert=True,
            )
        except Exception:
            pass

    def _get_blend_alpha(lname_key: str, horizon_key: int) -> float:
        try:
            doc = get_cache_database_sync().models_cache.find_one({"_norm_name": lname_key, "horizon": int(horizon_key), "model_type": "blend_alpha"})
            a = doc.get("alpha") if isinstance(doc, dict) else None
            if a is None:
                return 0.7
            a = float(a)
            if a < 0.0 or a > 1.0:
                return 0.7
            return a
        except Exception:
            return 0.7

    horizon_by_date: Dict[str, Any] = {}
    metrics_by_h: Dict[str, Any] = {}
    if feats.empty:
        for h in horizons:
            d = (datetime.utcnow().date() + timedelta(days=int(h))).isoformat()
            horizon_by_date[d] = {"point": None, "range": None}
            metrics_by_h[str(h)] = {"mae": None, "mape": None, "cv_mae": None, "n_samples": 0}
    else:
        # build fingerprint from expected features
        feat_cols = list(feats.columns)
        exog_keys = list(exog_vec.keys())
        fp = _make_fingerprint(feat_cols, exog_keys)
        for h in horizons:
            # Base LightGBM enriched model on returns for stability
            cached = _load_model_cache(lname, h, fp)
            if cached and cached.get("model") is not None and cached.get("metrics", {}).get("mae_ret") is not None:
                model = cached["model"]
                met = cached.get("metrics") or {}
            else:
                model, met, norm_cols, scaler_feat = _train_lgbm_return_enriched(feats, h, exog_vec)
                # persist normalized feature columns; we will normalize last row the same way
                met = dict(met)
                met["norm_cols"] = norm_cols
                _save_model_cache(lname, h, fp, (model, scaler_feat), met, met.get("features") or feat_cols)
                # also persist to local disk for reuse across sessions
                try:
                    bundle = {"model": model, "scaler": scaler_feat, "metrics": met, "features": met.get("features") or feat_cols}
                    model_store.save_model_bundle(lname, h, bundle, kind="return_lgbm")
                except Exception:
                    pass
            # try loading from disk if cache missing
            if cached is None:
                try:
                    disk = model_store.load_model_bundle(lname, h, kind="return_lgbm")
                    if isinstance(disk, dict) and disk.get("model") is not None:
                        model = (disk.get("model"), disk.get("scaler"))
                        met = disk.get("metrics") or {}
                except Exception:
                    pass
            last = _append_exog(feats.iloc[-1:].copy(), exog_vec)
            # Choose feature matrix based on model type
            try:
                is_pipeline = isinstance(model, Pipeline)
            except Exception:
                is_pipeline = False
            if is_pipeline:
                base_cols = ["close", "ret_1d", "dma_20", "dma_50", "vol_20", "mom_10"]
                arr = last.reindex(columns=base_cols, fill_value=0.0)[base_cols].values
            else:
                feature_cols = met.get("features") if isinstance(met, dict) else None
                if isinstance(feature_cols, list) and feature_cols:
                    arr = last.reindex(columns=feature_cols, fill_value=0.0).values
                else:
                    arr = last.values
            try:
                # If model predicts returns, convert to price anchored on last close
                # Handle saved (model, scaler) tuple
                scal = None
                mdl = model
                try:
                    if isinstance(model, tuple) and len(model) == 2:
                        mdl, scal = model  # type: ignore[misc]
                except Exception:
                    mdl = model
                    scal = None
                if isinstance(met.get("norm_cols"), list) and scal is not None:
                    arr_df = pd.DataFrame(arr, columns=met.get("features") or last.columns)
                    arr_df = arr_df.reindex(columns=met["norm_cols"], fill_value=0.0)
                    arr = scal.transform(arr_df.values)
                ret_hat_lgbm = float(mdl.predict(arr)[0])  # type: ignore[attr-defined]
                base_close = float(feats["close"].iloc[-1])
                if "mae_ret" in (met or {}):
                    lgbm_point = float(base_close * (1.0 + ret_hat_lgbm))
                else:
                    lgbm_point = ret_hat_lgbm
            except Exception:
                lgbm_point = float(feats["close"].iloc[-1])
            # Spread from LGBM validation MAE as uncertainty
            spread_base = (met.get("mae") if met.get("mae") is not None and not np.isnan(met.get("mae", np.nan)) else float(feats["close"].std())) * (1 + h / 10)

            # LSTM tuned model for robustness
            lstm_model, lstm_metrics, lstm_feats, seq_len, scaler = _train_lstm_tuned(feats, h, exog_vec)
            try:
                # Prepare last sequence
                last_seq_df = _append_exog(feats.tail(seq_len).copy(), exog_vec).reindex(columns=lstm_feats, fill_value=0.0)
                if scaler is not None:
                    last_seq_norm = scaler.transform(last_seq_df.values)
                else:
                    # fall back to identity
                    last_seq_norm = last_seq_df.values
                X_last = np.expand_dims(last_seq_norm, axis=0)
                # LSTM outputs return; convert to price anchored on latest close
                ret_hat = float(lstm_model.predict(X_last, verbose=0)[0][0])  # type: ignore[attr-defined]
                base_close = float(feats["close"].iloc[-1])
                lstm_point = float(base_close * (1.0 + ret_hat))
            except Exception:
                lstm_point = float(lgbm_point)

            # Ensemble LGBM and LSTM using inverse validation MAE weights (convert LSTM return-MAE to price-MAE)
            def _safe_mae(v, default: float):
                try:
                    vv = float(v)
                    if np.isnan(vv) or np.isinf(vv):
                        return default
                    return max(1e-6, vv)
                except Exception:
                    return default

            price_scale = float(feats["close"].iloc[-1]) if not feats.empty else (today_price or 1.0)
            lgbm_mae_price = _safe_mae(met.get("mae_ret", met.get("mae")), default=price_scale * 0.02)
            lstm_mae_price = _safe_mae(lstm_metrics.get("val_mae", np.nan) * price_scale, default=price_scale * 0.03)
            w_lgbm = 1.0 / lgbm_mae_price
            w_lstm = 1.0 / lstm_mae_price
            if not np.isfinite(w_lgbm):
                w_lgbm = 1.0
            if not np.isfinite(w_lstm):
                w_lstm = 1.0
            ens_point = float((w_lgbm * lgbm_point + w_lstm * lstm_point) / max(1e-6, (w_lgbm + w_lstm)))
            # Clamp to realistic band based on recent volatility (sqrt time)
            try:
                ret_series = df0["close"].pct_change().dropna()
                daily_vol = float(ret_series.rolling(20).std().iloc[-1]) if len(ret_series) > 20 else float(ret_series.std())
                if not np.isfinite(daily_vol) or daily_vol == 0:
                    daily_vol = 0.02
            except Exception:
                daily_vol = 0.02
            # tighter band: use 2.5x instead of 4.0
            band = (today_price or float(feats["close"].iloc[-1])) * (daily_vol * max(1.0, (h ** 0.5)) * 2.5)
            lower_cap = max(0.0, (today_price or float(feats["close"].iloc[-1])) - band)
            upper_cap = (today_price or float(feats["close"].iloc[-1])) + band
            ens_point = float(np.clip(ens_point, lower_cap, upper_cap))

            d = (datetime.utcnow().date() + timedelta(days=int(h))).isoformat()
            horizon_by_date[d] = {"point": float(ens_point), "range": [max(0.0, float(ens_point - spread_base)), float(ens_point + spread_base)]}
            metrics_by_h[str(h)] = {"lgbm": met, "lstm": lstm_metrics, "ensemble": {"lgbm_point": float(lgbm_point), "lstm_point": float(lstm_point)}}

            # Try quantile model for improved ranges
            try:
                qdoc = get_cache_database_sync().models_cache.find_one({"_norm_name": lname, "horizon": int(h), "model_type": "lgbm_quantile"})
                if qdoc and qdoc.get("blob_b64"):
                    import base64, pickle
                    bundle = pickle.loads(base64.b64decode(qdoc["blob_b64"].encode("ascii")))
                    feature_cols_q = bundle.get("feature_cols") or list(last.columns)
                    arr_q = last.reindex(columns=feature_cols_q, fill_value=0.0).values
                    models_q = bundle.get("models", {})
                    q20 = float(models_q["q20"].predict(arr_q)[0]) if "q20" in models_q else None
                    q50 = float(models_q["q50"].predict(arr_q)[0]) if "q50" in models_q else None
                    q80 = float(models_q["q80"].predict(arr_q)[0]) if "q80" in models_q else None
                    metrics_by_h[str(h)]["quantiles"] = {"q20": q20, "q50": q50, "q80": q80}
                    if q20 is not None and q80 is not None:
                        # Intersect quantile band with volatility clamp around ensemble point for tighter bounds
                        lo = max(0.0, min(q20, ens_point))
                        hi = max(q80, ens_point)
                        horizon_by_date[d]["range"] = [lo, hi]
            except Exception:
                pass

            # Try global quantile model as fallback and ensemble
            try:
                gdoc = get_cache_database_sync().models_cache.find_one({"_norm_name": "__GLOBAL__", "horizon": int(h), "model_type": "lgbm_quantile_global"})
                if gdoc and gdoc.get("blob_b64"):
                    import base64, pickle
                    gbundle = pickle.loads(base64.b64decode(gdoc["blob_b64"].encode("ascii")))
                    gcols = gbundle.get("feature_cols") or list(last.columns)
                    arr_g = last.reindex(columns=gcols, fill_value=0.0).values
                    gmodels = gbundle.get("models", {})
                    gq50 = float(gmodels["q50"].predict(arr_g)[0]) if "q50" in gmodels else None
                    if gq50 is not None:
                        # simple ensemble: average with local point
                        horizon_by_date[d]["point"] = float(np.mean([horizon_by_date[d]["point"], gq50]))
            except Exception:
                pass

            # Blend with AI numeric forecast if available
            try:
                alpha_default = _get_blend_alpha(lname, h)
            except Exception:
                alpha_default = 0.7
            ai_point_here = None
            try:
                # Attempt a lightweight AI forecast using headlines and signals
                if settings.ANTHROPIC_API_KEY and titles:
                    client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                    prompt = (
                        f"Stock: {symbol}. Today price: {today_price}. Given these signals {json.dumps(explain)}, "
                        f"estimate the closing price in {h} trading days. Return only a number.\nHeadlines:\n{titles[:1000]}"
                    )
                    msg = client.messages.create(model=settings.ANTHROPIC_MODEL, max_tokens=64, messages=[{"role": "user", "content": prompt}])
                    txt = str(getattr(msg, "content", "") or "").strip().split()[0]
                    ai_point_here = float(txt)
            except Exception:
                ai_point_here = None
            if ai_point_here is not None and horizon_by_date[d]["point"] is not None:
                base_point = float(horizon_by_date[d]["point"])  # type: ignore[arg-type]
                blended_point = float(alpha_default) * base_point + (1.0 - float(alpha_default)) * float(ai_point_here)
                # Apply the same volatility clamp to blended value
                try:
                    # lower_cap/upper_cap defined earlier in the loop
                    blended_point = float(np.clip(blended_point, lower_cap, upper_cap))  # type: ignore[name-defined]
                except Exception:
                    pass
                rng = horizon_by_date[d].get("range")
                spread_now = (rng[1] - rng[0]) / 2 if isinstance(rng, list) and len(rng) == 2 else max(1.0, blended_point * 0.02)
                lo = float(blended_point - spread_now)
                hi = float(blended_point + spread_now)
                # Keep range within clamp as well if available
                try:
                    lo = max(lo, lower_cap)  # type: ignore[name-defined]
                    hi = min(hi, upper_cap)  # type: ignore[name-defined]
                except Exception:
                    pass
                horizon_by_date[d] = {"point": float(blended_point), "range": [lo, hi]}
                # Track components
                metrics_by_h[str(h)]["ensemble"]["ai_point"] = float(ai_point_here)
                metrics_by_h[str(h)]["ensemble"]["alpha_model_vs_ai"] = float(alpha_default)

    result = {
        "symbol": symbol,
        "today": {"date": today_date, "price": today_price, "intraday_range": intraday_range},
        "predictions": horizon_by_date,
        "explain": explain,
        "metrics": metrics_by_h,
    }

    # Ask Claude for a concise rationale and factor assessment; store plus return
    ai_report = ""
    todays_news = []
    try:
        if settings.ANTHROPIC_API_KEY:
            client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            prompt = (
                f"You are a financial assistant. Summarize key drivers for {symbol} given these signals: "
                f"{json.dumps(explain)} and recent headlines:\n{titles[:1500]}\n"
                "Return 3-5 bullets with risks and a one-line outlook."
            )
            msg = client.messages.create(model=settings.ANTHROPIC_MODEL, max_tokens=256, messages=[{"role": "user", "content": prompt}])
            ai_report = (getattr(msg, "content", "") or "")
            # also create a JSON list of top facts via a second short call
            jf = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": f"Extract up to 5 key facts as short bullets from these headlines. Respond as JSON list of strings only.\n{titles[:1500]}"}],
            )
            try:
                todays_news = json.loads(getattr(jf, "content", "") or "[]")
            except Exception:
                todays_news = []
    except Exception:
        ai_report = ""

    result["report"] = ai_report

    # Attach top drivers from last trained quantile model if available (for P50)
    try:
        shap_doc = get_cache_database_sync().models_cache.find_one({"_norm_name": lname, "model_type": "lgbm_quantile"})
        shap_summary = (shap_doc or {}).get("shap_summary") or {}
        if isinstance(shap_summary, dict):
            # top 10 features
            top = sorted(shap_summary.items(), key=lambda kv: float(kv[1] or 0.0), reverse=True)[:10]
            result["drivers"] = [{"feature": k, "importance": float(v)} for k, v in top]
    except Exception:
        pass

    # Persist to DB for audit trail and today's news/intelligence bucket
    try:
        db.predictions_daily.update_one(
            {"_norm_name": lname, "date": today_date},
            {"$set": {"symbol": symbol, "date": today_date, "result": result, "created_at": datetime.utcnow()}},
            upsert=True,
        )
        # BQ log
        bq.log_event("prediction_enriched", payload=result, symbol=symbol, meta={"horizons": horizons})
        db.intel_daily.update_one(
            {"_norm_name": lname, "date": today_date},
            {"$set": {
                "_norm_name": lname,
                "symbol": symbol,
                "date": today_date,
                "industry": details.get("industry") or details.get("mgIndustry"),
                "headlines": titles.split("\n") if titles else [],
                "facts": todays_news,
                "signals": explain,
                "saved_at": datetime.utcnow(),
            }},
            upsert=True,
        )
    except Exception:
        pass

    return _json_safe(result)


async def run_research(symbol: str, horizons: List[int]) -> Dict[str, Any]:
    """Full research workflow for a stock name.
    - Pull fresh data for this stock (details, historical, stats, corp actions, announcements)
    - Aggregate company + industry news
    - Generate model predictions and AI predictions, blend them
    - Save per-day artifacts in DB: todays news, prediction, information
    - Return consolidated payload
    """
    lname = symbol.strip().lower()
    db = get_cache_database_sync()

    # 1) Fresh syncs for just this name
    try:
        await aggregation.sync_stock_details(names=[symbol])
        await aggregation.sync_historical_data(names=[symbol])
        await aggregation.sync_historical_stats(names=[symbol])
        await aggregation.sync_corporate_actions(names=[symbol])
        await aggregation.sync_recent_announcements(names=[symbol])
    except Exception:
        # best-effort; continue
        pass

    # 2) Compute enriched predictions (also persists predictions_daily and intel_daily)
    enriched = predict_enriched(symbol, horizons)

    # 3) Build broader industry news pool
    details = db.stock_details.find_one({"_norm_name": lname}) or {}
    industry = (details.get("industry") or details.get("mgIndustry") or "").strip()
    news_items: List[Dict[str, Any]] = []
    # company-specific recent announcements headlines
    try:
        ann = db.recent_announcements.find_one({"_norm_name": lname}) or {}
        items = ann.get("items", []) if isinstance(ann, dict) else []
        for i in (items or [])[:20]:
            if isinstance(i, dict) and (t := i.get("title")):
                news_items.append({"title": str(t), "source": "announcement"})
    except Exception:
        pass
    # fetch direct news if available
    try:
        fresh_news = await indian_api.fetch_news(symbol=symbol, limit=10)
        if isinstance(fresh_news, list):
            for n in fresh_news:
                if isinstance(n, dict):
                    news_items.append({
                        "title": str(n.get("title", "")),
                        "summary": n.get("summary"),
                        "source": n.get("source") or "news",
                    })
    except Exception:
        pass
    # industry peers → gather their headlines too
    if industry:
        try:
            peers = await indian_api.industry_search(industry)
            peer_syms: List[str] = []
            for p in peers or []:
                if isinstance(p, dict):
                    sym = p.get("symbol") or p.get("ticker") or p.get("Symbol")
                    nm = p.get("company") or p.get("company_name") or p.get("name")
                    if sym:
                        peer_syms.append(str(sym))
                    elif nm:
                        peer_syms.append(str(nm))
            peer_syms = [s for s in peer_syms if s and s.strip().lower() != lname][:5]
            # fetch in parallel
            async def _peer_news(s: str):
                try:
                    return await indian_api.fetch_news(symbol=s, limit=5)
                except Exception:
                    return []
            results = await asyncio.gather(*[_peer_news(s) for s in peer_syms], return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    for n in res:
                        if isinstance(n, dict):
                            news_items.append({
                                "title": str(n.get("title", "")),
                                "summary": n.get("summary"),
                                "source": n.get("source") or "peer_news",
                            })
        except Exception:
            pass

    # dedupe by title
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for n in news_items:
        t = (n.get("title") or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(n)

    # 4) Ask Claude for an explicit numeric forecast to blend with our model
    ai_forecasts: Dict[str, Any] = {}
    ai_notes = ""
    if settings.ANTHROPIC_API_KEY:
        try:
            client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            joined = "\n".join([f"- {n.get('title','')}" for n in deduped[:15]])
            # Build a concise context of signals we computed
            base_explain = enriched.get("explain", {}) if isinstance(enriched, dict) else {}
            prompt = (
                f"You are an equity analyst. Given the stock {symbol} in industry '{industry}', recent signals {json.dumps(base_explain)}, "
                f"and headlines below, estimate closing prices for horizons in days {horizons}.\n"
                f"Return STRICT JSON with keys 'predictions' (map of day->price), 'rationale' (short text). No extra text.\n\n"
                f"Headlines:\n{joined}"
            )
            msg = client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=384,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = getattr(msg, "content", "") or ""
            try:
                parsed = json.loads(str(raw))
                ai_forecasts = parsed.get("predictions") or {}
                ai_notes = parsed.get("rationale") or ""
            except Exception:
                ai_forecasts = {}
                ai_notes = str(raw)[:1000]
        except Exception:
            ai_forecasts = {}
            ai_notes = ""

    # 5) Blend AI and model predictions if both present
    model_preds = enriched.get("predictions", {}) if isinstance(enriched, dict) else {}
    blended: Dict[str, Any] = {}
    alpha_default = 0.7
    try:
        alpha_default = _get_blend_alpha(lname, horizons[0] if horizons else 5)
    except Exception:
        alpha_default = 0.7
    for date_key, payload in model_preds.items():
        if not isinstance(payload, dict):
            blended[date_key] = payload
            continue
        point = payload.get("point")
        rng = payload.get("range")
        # Map date back to horizon in days relative to 'today'
        try:
            d_obj = pd.to_datetime(date_key).date()
            delta_days = (d_obj - datetime.utcnow().date()).days
        except Exception:
            delta_days = None
        ai_point = None
        if delta_days is not None and str(delta_days) in ai_forecasts:
            try:
                ai_point = float(ai_forecasts[str(delta_days)])
            except Exception:
                ai_point = None
        if point is not None and ai_point is not None:
            combined = float(alpha_default) * float(point) + (1.0 - float(alpha_default)) * float(ai_point)
            spread = (rng[1] - rng[0]) / 2 if isinstance(rng, list) and len(rng) == 2 else max(1.0, float(point) * 0.02)
            blended[date_key] = {"point": float(combined), "range": [float(combined - spread), float(combined + spread)]}
        else:
            blended[date_key] = payload

    # 6) Persist research document
    today_date = datetime.utcnow().date().isoformat()
    research_doc = {
        "_norm_name": lname,
        "symbol": symbol,
        "date": today_date,
        "todays_news": deduped,
        "information": enriched.get("report", ""),
        "prediction": {
            "model": model_preds,
            "ai": ai_forecasts,
            "blended": blended,
            "ai_notes": ai_notes,
        },
        "saved_at": datetime.utcnow(),
    }
    try:
        db.research_daily.update_one(
            {"_norm_name": lname, "date": today_date},
            {"$set": research_doc},
            upsert=True,
        )
        if ai_forecasts:
            db.ai_predictions_daily.update_one(
                {"_norm_name": lname, "date": today_date},
                {"$set": {"_norm_name": lname, "symbol": symbol, "date": today_date, "predictions": ai_forecasts, "notes": ai_notes, "created_at": datetime.utcnow()}},
                upsert=True,
            )
        # BQ log
        bq.log_event("research_run", payload={"enriched": enriched, "blended": blended, "ai": ai_forecasts}, symbol=symbol)
    except Exception:
        pass

    # 7) Return consolidated payload
    enriched_out = dict(enriched)
    enriched_out["predictions_blended"] = blended
    enriched_out["todays_news"] = deduped
    enriched_out["ai_predictions"] = ai_forecasts
    enriched_out["ai_notes"] = ai_notes
    return _json_safe(enriched_out)


def compute_accuracy(symbol: str, horizon: int = 5, window_days: int = 120) -> Dict[str, Any]:
    """Compute rolling MAE over last N days by comparing saved predictions vs realized price.
    - Uses `predictions_daily` for point forecasts at date d+h
    - Compares with historical close at that future date
    """
    db = get_cache_database_sync()
    lname = symbol.strip().lower()
    # load historical prices
    key = {"_norm_name": lname, "_period": "10yr", "_filter": "price"}
    hist = db.historical_data.find_one(key) or db.historical_data.find_one({"_norm_name": lname}) or {}
    df = _extract_price_series(hist) if isinstance(hist, dict) else pd.DataFrame()
    if df.empty:
        return {"mae": None, "n": 0, "series": []}
    # collect predictions in window
    cutoff = (datetime.utcnow() - timedelta(days=window_days)).date().isoformat()
    rows = list(db.predictions_daily.find({"_norm_name": lname, "date": {"$gte": cutoff}}).sort("date", -1))
    errs: List[Dict[str, Any]] = []
    for r in rows:
        try:
            day = r.get("date")
            pred_map = (r.get("result") or {}).get("predictions") or {}
            # future date = day + horizon
            d = (pd.to_datetime(day).date() + timedelta(days=int(horizon))).isoformat()
            pv = pred_map.get(d)
            point = pv.get("point") if isinstance(pv, dict) else None
            if point is None:
                continue
            # realized
            if d in df.index:
                realized = float(df.loc[d, "close"])  # type: ignore[index]
            else:
                # nearest
                try:
                    realized = float(df.loc[pd.to_datetime(d)].get("close"))  # type: ignore
                except Exception:
                    realized = None  # type: ignore[assignment]
            if realized is None:
                continue
            err = abs(float(point) - float(realized))
            errs.append({"date": d, "pred": float(point), "realized": float(realized), "abs_err": float(err)})
        except Exception:
            continue
    if not errs:
        return {"mae": None, "n": 0, "series": []}
    mae = float(np.mean([e["abs_err"] for e in errs]))
    return {"mae": mae, "n": len(errs), "series": errs}


def _json_safe(obj: Any) -> Any:
    """Recursively convert NaN/Inf to None and numpy types to Python primitives for JSON safety."""
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return [_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_json_safe(list(obj)))
    return obj

