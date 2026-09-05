# ai-trading-lab

Building an AI-powered trading research system from scratch: market data, backtesting, ML, and paper trading.

Local Python workspace for NSE research.

- [Day 1](#day-1-project-setup-and-market-data-pipeline): Project setup and market data pipeline
- [Day 2](#day-2-bar-indicators-from-1-minute-ohlcv): Bar indicators from 1-minute OHLCV
- [Day 3](#day-3-future-return-targets-and-date-splits): Future-return targets and date splits

No ML, strategy discovery, backtesting, or live trading yet.

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

## Day 2: Bar indicators from 1-minute OHLCV

Convert each stock's raw 1-minute candles into analyzable indicator columns. Stocks are processed one at a time (never load all ~27M rows into memory). Raw Parquet files are left unchanged.

| Script | Purpose |
|---|---|
| `src/data/build_bar_indicators.py` | Compute indicators, validate, write processed Parquet |
| `src/data/kite_ohlcv.py` | Shared Parquet I/O (reused from Day 1) |

**Indicators:** returns (1/5/15/30/60m), candle range/body/ratios, volume change & SMA ratios, SMA (5/20/50/200), intraday VWAP + distance, rolling volatility (20/60), RSI (14).

**How each is calculated** (on 1-minute bars; windows are bar counts unless noted):

**Session boundaries:** Rolling and shift-based indicators are computed **within each trading session** (calendar date of `timestamp`). They never carry values from the previous session into the next — e.g. `return_5m` at today's open does not use yesterday's close, and `sma_20` uses the prior 20 one-minute bars **within the same session only**. VWAP resets at the start of each session.

| Column | Calculation |
|---|---|
| `return_Nm` | `(close / close.shift(N)) - 1` for N = 1, 5, 15, 30, 60; **within session** |
| `candle_range` | `high - low` |
| `candle_body` | `close - open` |
| `body_ratio` | `abs(close - open) / (high - low)`; NaN if range is 0 |
| `close_position` | `(close - low) / (high - low)`; NaN if range is 0 |
| `volume_change_1m` | `(volume / volume.shift(1)) - 1`; **within session** |
| `volume_sma_20` | rolling mean of `volume` over 20 bars **within session** |
| `volume_ratio_20` | `volume / volume_sma_20`; NaN if SMA is 0 |
| `sma_W` | rolling mean of `close` over W = 5, 20, 50, 200 bars **within session** |
| `vwap` | intraday cumulative: `sum(typical_price * volume) / sum(volume)` where `typical_price = (high + low + close) / 3`; resets each calendar day |
| `distance_from_vwap` | `(close - vwap) / vwap`; NaN if VWAP is 0 |
| `volatility_W` | rolling std of `return_1m` over W = 20, 60 bars **within session** |
| `rsi_14` | Wilder RSI(14): EWM gains/losses with `alpha = 1/14`; **within session** |

±Inf from zero denominators is replaced with NaN. Initial warm-up rows stay NaN (no forward-fill).

**Output:** `data/processed/{SYMBOL}_1min_indicators.parquet` (gitignored)

```bash
.venv/bin/python -m src.data.build_bar_indicators
.venv/bin/python -m src.data.build_bar_indicators --symbol RELIANCE
```

```sql
SELECT * FROM read_parquet('data/processed/RELIANCE_1min_indicators.parquet') LIMIT 5;
```

**Rules:** sort by timestamp; no look-ahead; no forward-fill; warm-up NaNs kept; original OHLCV preserved; indicators reset at each trading-session boundary. Re-runs overwrite processed files (no duplication).

## Day 3: Future-return targets and date splits

Append future-return target columns to each stock's processed indicators, and define reusable train/validation/test date splits. Each stock is processed independently (never load all ~27M rows at once). Indicator Parquet files are left unchanged.

| Script | Purpose |
|---|---|
| `src/data/build_targets.py` | Append future-return targets, validate, write targets Parquet |
| `src/data/date_splits.py` | Train/validation/test date ranges + timestamp classifier |
| `src/data/kite_ohlcv.py` | Shared Parquet I/O (reused from Day 1) |

**Targets:** `future_return_5m`, `future_return_15m`, `future_return_30m`, `future_return_60m`.

**How each is calculated** (on 1-minute bars):

**Session boundaries:** Future-return targets are computed **within each trading session** (calendar date of `timestamp`). `future_return_Nm` at row `t` uses the close `N` bars *later in the same session* — `close.groupby(day).shift(-N)`. The last `N` bars of a session have no future bar in the same session, so their target is NaN. A target **never** uses the next session's open or close, so there is no cross-session look-ahead leakage.

| Column | Calculation |
|---|---|
| `future_return_Nm` | `(close.shift(-N) / close) - 1` for N = 5, 15, 30, 60; **within session** (NaN when the future bar is missing or falls in another session) |

**Date splits** (inclusive, by calendar date; no rows are removed or duplicated — this is metadata for downstream filtering):

| Split | Range |
|---|---|
| train | 2023-09-04 → 2025-09-03 |
| validation | 2025-09-04 → 2026-03-03 |
| test | 2026-03-04 → 2026-09-04 |

**Output:** `data/processed/{SYMBOL}_1min_targets.parquet` (gitignored) — 31 columns (27 indicators + 4 targets).

```bash
.venv/bin/python -m src.data.build_targets
.venv/bin/python -m src.data.build_targets --symbol RELIANCE
```

```sql
SELECT timestamp, close, future_return_5m, future_return_60m
FROM read_parquet('data/processed/RELIANCE_1min_targets.parquet') LIMIT 5;
```

**Rules:** sort by timestamp; no look-ahead across sessions; existing indicator columns preserved unchanged; target NaNs at session tails kept (no forward-fill); re-runs overwrite targets files (no duplication).

**Never commit:** `.env`, `.kite_session`, `data/raw/`, or `data/processed/`.
