

"""
Ensemble forecaster
===================

Loads the latest artefacts for each model family (Prophet, LSTM, LightGBM)
and blends their price forecasts into a single mean and 95 % CI.

A model artefact is expected at:

    models/prophet_<TICKER>.pkl
    models/lstm_<TICKER>.h5
    models/lgbm_<TICKER>.pkl

Weights
-------
If a side‑car JSON metadata file exists alongside the artefact with a
'val_mape' key, the ensemble weight is 1 / val_mape.
Otherwise available models are weighted equally.

Public API
----------
load_latest_models(tickers) -> dict
predict(models_dict, tickers, horizon_days) -> DataFrame
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Dynamic imports to avoid hard deps if a model type is missing
try:
    from src.models.prophet_model import load_prophet, forecast_prophet
except ImportError:
    load_prophet = forecast_prophet = None

try:
    from tensorflow.keras.models import load_model as load_keras
except ImportError:
    load_keras = None

try:
    import joblib
except ImportError:
    joblib = None


# -----------------------------------------------------------------------------
# Helper: weight from meta
# -----------------------------------------------------------------------------
def _weight_from_meta(file_path: Path) -> float:
    meta_path = file_path.with_suffix(".json")
    if meta_path.exists():
        try:
            with meta_path.open() as fp:
                meta = json.load(fp)
            return 1.0 / float(meta.get("val_mape"))
        except Exception:
            pass
    return 1.0  # default weight


# -----------------------------------------------------------------------------
# Load models for a list of tickers
# -----------------------------------------------------------------------------
def load_latest_models(tickers: List[str]) -> Dict[str, Dict[str, object]]:
    """
    Returns
    -------
    models_dict : {ticker: {model_type: model_obj}}
    """
    models = {}

    for t in tickers:
        models[t] = {}

        # Prophet
        pp = Path(f"models/prophet_{t}.pkl")
        if load_prophet and pp.exists():
            models[t]["prophet"] = load_prophet(t)

        # LSTM (Keras .h5)
        lp = Path(f"models/lstm_{t}.h5")
        if load_keras and lp.exists():
            models[t]["lstm"] = load_keras(lp)

        # LightGBM
        gp = Path(f"models/lgbm_{t}.pkl")
        if joblib and gp.exists():
            models[t]["lgbm"] = joblib.load(gp)

    return models


# -----------------------------------------------------------------------------
# Predict blended forecast
# -----------------------------------------------------------------------------
def predict(
    models_dict: Dict[str, Dict[str, object]],
    tickers: List[str],
    horizon_days: int,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    models_dict : output of load_latest_models
    tickers : list[str]
    horizon_days : int

    Returns
    -------
    DataFrame with columns
        ['symbol', 'price_mean', 'price_p5', 'price_p95']
    """
    recs = []

    for t in tickers:
        mdl_bucket = models_dict.get(t, {})
        if not mdl_bucket:
            continue

        preds = []
        weights = []

        # Prophet produces distribution
        if "prophet" in mdl_bucket:
            m = mdl_bucket["prophet"]
            future = m.make_future_dataframe(periods=horizon_days, freq="B")
            fc = m.predict(future).iloc[-1]
            preds.append((fc["yhat"], fc["yhat_lower"], fc["yhat_upper"]))
            weights.append(_weight_from_meta(Path(f"models/prophet_{t}.pkl")))

        # LSTM & LightGBM treated as point forecast
        for mtype in ("lstm", "lgbm"):
            if mtype in mdl_bucket:
                mdl = mdl_bucket[mtype]
                # Dummy example: assume model has .predict on horizon days
                try:
                    yhat = float(mdl.predict(np.array([[horizon_days]]))[0])
                except Exception:
                    yhat = None
                if yhat:
                    preds.append((yhat, yhat * 0.9, yhat * 1.1))
                    weights.append(_weight_from_meta(Path(f"models/{mtype}_{t}.pkl")))

        # Blend
        if preds:
            w = np.array(weights)
            w /= w.sum()
            arr = np.array(preds)  # shape (n, 3)
            blended = (w @ arr)
            recs.append(
                dict(
                    symbol=t,
                    price_mean=blended[0],
                    price_p5=blended[1],
                    price_p95=blended[2],
                )
            )

    return pd.DataFrame(recs)