# src/models/prophet_model.py
from prophet import Prophet
import pandas as pd, joblib, mlflow
from pathlib import Path

# Ensure the models directory exists
Path("models").mkdir(exist_ok=True)

def train_prophet(df: pd.DataFrame, ticker: str, extra_cols: list[str] | None = None):
    """
    Train a Prophet model on OHLCV+feature DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have Date index and at least a 'Close' column plus
        engineered feature columns (e.g. 'rsi_14', 'vol_21d', etc.).
    ticker : str
        Stock or fund symbol.
    extra_cols : list[str] | None
        Names of additional feature columns to treat as regressors.
        Pass None to auto‑detect numeric cols except 'y'.

    Saves the fitted model to models/prophet_<ticker>.pkl and logs to MLflow.
    """
    df = df.copy()

    # Prophet expects ds / y plus numeric regressors
    df = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    if extra_cols is None:
        extra_cols = [c for c in df.columns if c not in {"ds", "y"} and pd.api.types.is_numeric_dtype(df[c])]

    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    for col in extra_cols:
        m.add_regressor(col)

    m.fit(df[["ds", "y", *extra_cols]])

    # MLflow logging (+ fallback if mlflow unavailable)
    try:
        mlflow.prophet.log_model(m, artifact_path=f"prophet_{ticker}")
    except Exception:
        pass  # skip if mlflow not set up

    joblib.dump(m, Path("models") / f"prophet_{ticker}.pkl")
    return m


# === Helper functions ===

def load_prophet(ticker: str):
    """Load a previously saved Prophet model."""
    path = Path("models") / f"prophet_{ticker}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model for {ticker}: {path}")
    return joblib.load(path)


def forecast_prophet(model, df_future: pd.DataFrame, periods: int = 252):
    """
    Generate a forecast using an already-fitted model.

    Parameters
    ----------
    model : Prophet
    df_future : DataFrame
        Must contain 'ds' plus the same regressor columns used at train‑time.
    periods : int
        Number of business days to forecast if df_future is None.

    Returns
    -------
    DataFrame with columns ['ds','yhat','yhat_lower','yhat_upper']
    """
    if df_future is None:
        df_future = model.make_future_dataframe(periods=periods, freq="B")

    forecast = model.predict(df_future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]