from __future__ import annotations

from typing import List, Optional

import pandas as pd

# Reuse existing live data functions without modifying them
from src.live_data import (
    list_all_nse_symbols,
    list_all_amfi_codes,
    get_live_stock_quotes,
    get_live_mutual_nav,
)


def list_symbols(limit: Optional[int] = None) -> List[str]:
    syms = list_all_nse_symbols()
    if limit:
        return sorted(syms)[:limit]
    return sorted(syms)


def list_mutual_schemes(limit: Optional[int] = None) -> List[str]:
    codes = list_all_amfi_codes()
    return codes[:limit] if limit else codes


def get_quotes(tickers: List[str]) -> pd.DataFrame:
    return get_live_stock_quotes(tickers)


def get_mf_nav(schemes: List[str]) -> pd.DataFrame:
    return get_live_mutual_nav(schemes)


