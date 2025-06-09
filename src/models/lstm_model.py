

"""
LSTM sequence forecaster
========================

A very light Keras implementation that learns to predict the *future price*
horizon_days ahead from the last SEQ_LEN daily closes (optionally a few
technical features).  Designed for rapid experimentation – tune hyper‑params
offline as needed.

Artefacts
---------
• Model (HDF5):  models/lstm_<TICKER>.h5
• Metadata JSON: models/lstm_<TICKER>.json  (contains val_mape)

Public helpers
--------------
train_lstm(feat_df, ticker, horizon_days=30)
load_lstm(ticker)
predict_price(model, window_array)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 60  # look‑back window (days)


# ──────────────────────────────────────────────────────────────
# Utils: build X / y sliding windows
# ──────────────────────────────────────────────────────────────
def _make_windows(
    series: np.ndarray, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(series) - SEQ_LEN - horizon):
        X.append(series[i : i + SEQ_LEN])
        y.append(series[i + SEQ_LEN + horizon])
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def train_lstm(
    price_df: pd.DataFrame,
    ticker: str,
    horizon_days: int = 30,
) -> Tuple[Sequential, float]:
    """
    price_df must have a 'Close' column indexed chronologically.
    """
    price_series = price_df["Close"].values

    X, y = _make_windows(price_series, horizon_days)
    # Convert y to log‑return target
    y = np.log(y / X[:, -1, None])

    # Simple train/val split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Reshape to (samples, timesteps, features)
    X_train = X_train.reshape((-1, SEQ_LEN, 1))
    X_val = X_val.reshape((-1, SEQ_LEN, 1))

    # ── Model ────────────────────────────────────────────────
    model = Sequential(
        [
            LSTM(32, input_shape=(SEQ_LEN, 1)),
            Dense(16, activation="relu"),
            Dense(1),  # log‑return output
        ]
    )
    model.compile(optimizer=Adam(0.001), loss="mse")

    es = EarlyStopping(patience=8, restore_best_weights=True, verbose=0)
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=0,
    )

    # ── Validation MAPE (converted back to price) ────────────
    y_pred_log = model.predict(X_val, verbose=0)
    y_pred_price = np.exp(y_pred_log) * X_val[:, -1, 0]
    y_val_price = np.exp(y_val) * X_val[:, -1, 0]
    val_mape = mean_absolute_percentage_error(y_val_price, y_pred_price)

    # ── Save artefacts ───────────────────────────────────────
    h5_path = MODELS_DIR / f"lstm_{ticker}.h5"
    model.save(h5_path)

    meta = {"horizon_days": horizon_days, "val_mape": float(val_mape)}
    (MODELS_DIR / f"lstm_{ticker}.json").write_text(json.dumps(meta, indent=2))

    return model, val_mape


# ──────────────────────────────────────────────────────────────
# Loader & inference
# ──────────────────────────────────────────────────────────────
def load_lstm(ticker: str):
    h5_path = MODELS_DIR / f"lstm_{ticker}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    return load_model(h5_path, compile=False)


def predict_price(
    model,
    recent_close_series: np.ndarray,
    price_now: float,
    horizon_years: int = 1,
):
    """
    Input
    -----
    recent_close_series : last SEQ_LEN closes as 1‑D array
    price_now           : latest close
    """
    x = recent_close_series.reshape((1, SEQ_LEN, 1))
    log_ret = float(model.predict(x, verbose=0)[0])
    price_future = price_now * np.exp(log_ret * horizon_years)
    return price_future