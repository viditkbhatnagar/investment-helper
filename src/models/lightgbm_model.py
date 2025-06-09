

"""
LightGBM tabular forecaster
===========================

Trains a gradient‑boosting regression model on the engineered
feature matrix produced by `feature_engineering.full_pipeline()`.
The target is the *future log‑return* over a chosen horizon.

Artefacts
---------
• Model:  models/lgbm_<TICKER>.pkl
• Metadata (CV MAPE): models/lgbm_<TICKER>.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Training helper
# ──────────────────────────────────────────────────────────────
def train_lgbm(
    feat_df: pd.DataFrame,
    ticker: str,
    horizon_days: int = 252,  # 1‑year log‑return target
) -> Tuple[lgb.LGBMRegressor, float]:
    """
    Parameters
    ----------
    feat_df : DataFrame
        Output of full_pipeline; index must be chronological.
    ticker : str
        Symbol used in filename.
    horizon_days : int
        How far ahead (in rows) to compute the target.

    Returns
    -------
    (model, val_mape)
    """
    df = feat_df.sort_index().copy()

    # ── Build target: future log‑return ─────────────────────────
    df["y"] = np.log(df["Close"].shift(-horizon_days) / df["Close"])
    df = df.dropna(subset=["y"])

    X = df.drop(columns=["y"]).select_dtypes(include=["number"])
    y = df["y"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # keep chronology
    )

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    val_mape = mean_absolute_percentage_error(y_val, y_pred)

    # ── Persist artefacts ───────────────────────────────────────
    mpath = MODELS_DIR / f"lgbm_{ticker}.pkl"
    joblib.dump(model, mpath)

    meta = {"horizon_days": horizon_days, "val_mape": float(val_mape)}
    (MODELS_DIR / f"lgbm_{ticker}.json").write_text(json.dumps(meta, indent=2))

    return model, val_mape


# ──────────────────────────────────────────────────────────────
# Loader & quick predictor
# ──────────────────────────────────────────────────────────────
def load_lgbm(ticker: str):
    """Load saved LightGBM model for ticker."""
    mpath = MODELS_DIR / f"lgbm_{ticker}.pkl"
    if not mpath.exists():
        raise FileNotFoundError(mpath)
    return joblib.load(mpath)


def predict_price(
    model,
    latest_feat: pd.Series,
    price_now: float,
    horizon_years: int = 1,
):
    """
    Convert predicted log‑return back to price forecast.
    """
    log_ret = float(model.predict(latest_feat.values.reshape(1, -1))[0])
    price_future = price_now * np.exp(log_ret * horizon_years)
    return price_future