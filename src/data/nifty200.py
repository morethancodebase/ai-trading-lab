"""Current NIFTY 200 NSE symbols.

The full NIFTY 200 list (NIFTY 100 + the next 100 by free-float market cap).
``load_nifty200_symbols()`` tries the live NSE constituent CSV first and falls
back to the snapshot below. ``load_next100_symbols()`` returns the NIFTY-200
constituents that are NOT in NIFTY 100 (the "next 100", ranks 101-200).

Snapshot updated 2026-09-10 from NSE ``ind_nifty200list.csv``.
"""

from __future__ import annotations

import csv
import io

import requests

from data.nifty100 import NIFTY100_FALLBACK, load_nifty100_symbols

# The 100 NIFTY-200 constituents that are NOT in NIFTY 100 ("next 100",
# ranks 101-200), 2026-09-10 snapshot.
NEXT100_FALLBACK = [
    "360ONE",
    "ABCAPITAL",
    "ALKEM",
    "APLAPOLLO",
    "ASHOKLEY",
    "ASTRAL",
    "ATGL",
    "AUBANK",
    "AUROPHARMA",
    "BANKINDIA",
    "BDL",
    "BHARATFORG",
    "BHEL",
    "BIOCON",
    "BLUESTARCO",
    "BSE",
    "COCHINSHIP",
    "COFORGE",
    "COLPAL",
    "CONCOR",
    "COROMANDEL",
    "DABUR",
    "DIXON",
    "EXIDEIND",
    "FEDERALBNK",
    "FORTIS",
    "GLENMARK",
    "GMRAIRPORT",
    "GODFRYPHLP",
    "GODREJPROP",
    "GROWW",
    "GVT&D",
    "HAVELLS",
    "HEROMOTOCO",
    "HINDPETRO",
    "HUDCO",
    "ICICIAMC",
    "ICICIGI",
    "IDEA",
    "IDFCFIRSTB",
    "INDIANB",
    "INDUSINDBK",
    "INDUSTOWER",
    "IRCTC",
    "IREDA",
    "JSWENERGY",
    "JUBLFOOD",
    "KALYANKJIL",
    "KEI",
    "KPITTECH",
    "LAURUSLABS",
    "LENSKART",
    "LGEINDIA",
    "LICHSGFIN",
    "LTF",
    "LUPIN",
    "M&MFIN",
    "MANKIND",
    "MARICO",
    "MCX",
    "MFSL",
    "MOTILALOFS",
    "MPHASIS",
    "MRF",
    "NATIONALUM",
    "NAUKRI",
    "NHPC",
    "NMDC",
    "NYKAA",
    "OBEROIRLTY",
    "OFSS",
    "OIL",
    "PAGEIND",
    "PATANJALI",
    "PAYTM",
    "PERSISTENT",
    "PHOENIXLTD",
    "PIIND",
    "POLICYBZR",
    "POLYCAB",
    "POWERINDIA",
    "PREMIERENE",
    "PRESTIGE",
    "RADICO",
    "RVNL",
    "SAIL",
    "SBICARD",
    "SRF",
    "SUPREMEIND",
    "SUZLON",
    "SWIGGY",
    "TATACOMM",
    "TATAELXSI",
    "TATAINVEST",
    "TIINDIA",
    "UPL",
    "VMM",
    "VOLTAS",
    "WAAREEENER",
    "YESBANK",
]

# Full NIFTY 200 = NIFTY 100 + next 100 (union, de-duplicated, sorted).
NIFTY200_FALLBACK = sorted(set(NIFTY100_FALLBACK) | set(NEXT100_FALLBACK))

NSE_CONSTITUENT_URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
]


def _parse_symbol_csv(text: str) -> list[str]:
    reader = csv.DictReader(io.StringIO(text))
    symbols: list[str] = []
    for row in reader:
        symbol = (row.get("Symbol") or row.get("SYMBOL") or row.get("symbol") or "").strip()
        if symbol:
            symbols.append(symbol)
    return symbols


def load_nifty200_symbols() -> tuple[list[str], str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,text/plain,*/*",
    }
    for url in NSE_CONSTITUENT_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            symbols = _parse_symbol_csv(response.text)
            if len(symbols) >= 190:
                return sorted(set(symbols)), url
        except Exception:
            continue
    return list(NIFTY200_FALLBACK), "fallback:src/data/nifty200.py"


def load_next100_symbols() -> tuple[list[str], str]:
    """NIFTY-200 constituents not in NIFTY 100 (the "next 100", ranks 101-200)."""
    nifty200, src200 = load_nifty200_symbols()
    nifty100, src100 = load_nifty100_symbols()
    next100 = sorted(set(nifty200) - set(nifty100))
    return next100, f"{src200} minus {src100}"
