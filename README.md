# ai-trading-lab

Building an AI-powered trading research system from scratch: market data, backtesting, ML, and paper trading.

Local Python workspace for NSE research. **Day 1** sets up the environment and downloads **3 years of 1-minute OHLCV for NIFTY 100** via Zerodha Kite Connect, stored as Parquet and queryable with DuckDB.

No ML, indicators, backtesting, or live trading yet.

## Requirements

- Python 3.12
- uv
- Git

### Development environment

Developed and tested on macOS Apple Silicon.

## Setup

```bash
git clone https://github.com/morethancodebase/ai-trading-lab.git
cd ai-trading-lab
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12 && uv venv --python 3.12
source .venv/bin/activate
uv pip install pandas numpy jupyter duckdb yfinance matplotlib python-dotenv kiteconnect
```

**Kite Connect** (required for 1-minute data):

- Zerodha account + [Kite Connect app](https://developers.kite.trade/) with API subscription
- Redirect URL in app (e.g. `http://127.0.0.1:8000`)
- Create `.env` (gitignored):

```env
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
```

```bash
.venv/bin/python src/kite_client.py
.venv/bin/python src/kite_client.py --login --request-token YOUR_REQUEST_TOKEN
```

Access token is saved in `.kite_session` (gitignored) until it expires.

## Day 1: Project setup and market data pipeline

Download NIFTY 100 1-minute candles (**2023-09-04 → 2026-09-04**) from Kite, chunked at 59 days per request (Kite's 60-day limit), with retries and resume support.

| Script | Purpose |
|---|---|
| `src/kite_client.py` | Load `.env`, authenticate, test connectivity |
| `src/data/download_nifty100_1min.py` | Download all NIFTY 100 symbols |
| `src/data/kite_ohlcv.py` | Shared chunking, Parquet I/O, validation |

**Output:** `data/raw/{SYMBOL}_1min.parquet` (gitignored)

```bash
.venv/bin/python src/data/download_nifty100_1min.py --plan-only   # preview chunks
.venv/bin/python src/data/download_nifty100_1min.py               # full download
```

```sql
SELECT * FROM read_parquet('data/raw/RELIANCE_1min.parquet') LIMIT 5;
```

**Kite limits:** ~3 req/sec; max 60 calendar days per 1-minute request; daily login required.

**Never commit:** `.env`, `.kite_session`, or `data/raw/`.
