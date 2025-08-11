from __future__ import annotations

from typing import List

import pandas as pd

from src.live_data import list_all_nse_symbols, list_all_amfi_codes


def list_stocks(limit: int | None = None) -> List[str]:
    syms = sorted(list_all_nse_symbols())
    return syms[:limit] if limit else syms


def list_mutual_funds(limit: int | None = None) -> List[str]:
    codes = list_all_amfi_codes()
    return codes[:limit] if limit else codes


def list_ipos() -> list[dict]:
    # Placeholder; NSE/BSE public endpoints require scraping or paid APIs
    return [
        {"symbol": "ABCIPO", "exchange": "NSE", "status": "Upcoming", "price_band": "₹100-110"},
    ]


def list_bonds() -> list[dict]:
    # Placeholder; typically requires scraping RBI/NSE/BSE bond listings
    return [
        {"name": "Govt 2034 G-Sec", "coupon": 7.18, "maturity": "2034-06-15", "ytm": 7.22},
    ]


