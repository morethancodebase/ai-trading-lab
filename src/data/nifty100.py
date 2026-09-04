"""Current NIFTY 100 NSE symbols.

Snapshot used as a fallback if the live NSE constituent file cannot be fetched.
Updated 2026-09-04.
"""

from __future__ import annotations

import csv
import io

import requests

# Official-style constituent list (NSE tradingsymbols), 2026-09-04 snapshot.
NIFTY100_FALLBACK = [
    "ABB",
    "ADANIENSOL",
    "ADANIENT",
    "ADANIGREEN",
    "ADANIPORTS",
    "ADANIPOWER",
    "AMBUJACEM",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJAJFINSV",
    "BAJAJHLDNG",
    "BAJFINANCE",
    "BANKBARODA",
    "BEL",
    "BHARTIARTL",
    "BOSCHLTD",
    "BPCL",
    "BRITANNIA",
    "CANBK",
    "CGPOWER",
    "CHOLAFIN",
    "CIPLA",
    "COALINDIA",
    "CUMMINSIND",
    "DIVISLAB",
    "DLF",
    "DMART",
    "DRREDDY",
    "EICHERMOT",
    "ENRIN",
    "ETERNAL",
    "GAIL",
    "GODREJCP",
    "GRASIM",
    "HAL",
    "HCLTECH",
    "HDFCAMC",
    "HDFCBANK",
    "HDFCLIFE",
    "HINDALCO",
    "HINDUNILVR",
    "HINDZINC",
    "HYUNDAI",
    "ICICIBANK",
    "INDHOTEL",
    "INDIGO",
    "INFY",
    "IOC",
    "IRFC",
    "ITC",
    "JINDALSTEL",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LODHA",
    "LT",
    "LTM",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "MAZDOCK",
    "MOTHERSON",
    "MUTHOOTFIN",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "PFC",
    "PIDILITIND",
    "PNB",
    "POWERGRID",
    "RECLTD",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHREECEM",
    "SHRIRAMFIN",
    "SIEMENS",
    "SOLARINDS",
    "SUNPHARMA",
    "TATACAP",
    "TATACONSUM",
    "TATASTEEL",
    "TATAPOWER",
    "TCS",
    "TECHM",
    "TITAN",
    "TMCV",
    "TMPV",
    "TORNTPHARM",
    "TRENT",
    "TVSMOTOR",
    "ULTRACEMCO",
    "UNIONBANK",
    "UNITDSPR",
    "VBL",
    "VEDL",
    "WIPRO",
    "ZYDUSLIFE",
]

NSE_CONSTITUENT_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
]


def _parse_symbol_csv(text: str) -> list[str]:
    reader = csv.DictReader(io.StringIO(text))
    symbols: list[str] = []
    for row in reader:
        symbol = (row.get("Symbol") or row.get("SYMBOL") or row.get("symbol") or "").strip()
        if symbol:
            symbols.append(symbol)
    return symbols


def load_nifty100_symbols() -> tuple[list[str], str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
    }
    for url in NSE_CONSTITUENT_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            symbols = _parse_symbol_csv(response.text)
            if len(symbols) >= 90:
                return sorted(set(symbols)), url
        except Exception:
            continue
    return list(NIFTY100_FALLBACK), "fallback:src/data/nifty100.py"
