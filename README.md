# ai-trading-lab

Building an AI-powered trading research system from scratch: market data, backtesting, ML, and paper trading.

Local Python workspace for NSE research.

- [Day 1](#day-1-project-setup-and-market-data-pipeline): Project setup and market data pipeline
- [Day 2](#day-2-bar-indicators-from-1-minute-ohlcv): Bar indicators from 1-minute OHLCV
- [Day 3](#day-3-future-return-targets-and-date-splits): Future-return targets and date splits
- [Day 4](#day-4-predictive-analysis-of-bar-indicators): Predictive analysis of bar indicators

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

## Day 4: Predictive analysis of bar indicators

With indicators (Day 2) and future-return targets (Day 3) in place, the next step was to test whether the bar indicators contain **measurable predictive information** about the future-return targets — using only correlation and decile statistics. **No ML model, no strategy, no backtest, no threshold optimization.** Discovery happens on **TRAIN** only; **VALIDATION** is used solely to check whether discovered relationships persist out-of-sample; **TEST is never read.**

| Script | Purpose |
|---|---|
| `src/analysis/predictive_analysis.py` | Discover indicator → target relationships on TRAIN, check VALIDATION persistence, never read TEST |
| `src/analysis/generate_html_report.py` | Build a self-contained HTML report from the CSV/JSON outputs |

**What we analyzed** (100 NIFTY-100 stocks; 21 indicators × 4 targets = 84 relationships):

- Target distributions (count, mean, std, positive/negative %) per split.
- **Pearson** and **Spearman** correlation of each indicator vs each target.
- **Per-stock-mean** correlation (mean of the 100 per-stock correlations) as the primary metric, robust to the differing price scales of the 100 stocks; naive pooled (concatenate-all-stocks) correlations are reported alongside, with a cross-sectional caveat for level indicators.
- **Decile / bucket analysis**: 10 boundaries learned on TRAIN, reused verbatim on VALIDATION; per-bucket mean/median/positive% and a one-way ANOVA of the target across buckets.
- **TRAIN vs VALIDATION persistence**: same sign and magnitude ratio `|val| / |train|` for each relationship.
- **Per-stock** detail and **time-of-day** slices for the strongest relationships.
- **Basic statistical significance** (correlation p-value, ANOVA F/p) — reported but explicitly not treated as evidence of profitability.

**Output:** `reports/day4_predictive_analysis/` (gitignored) — `run_summary.json`, correlation CSVs, decile CSVs, `strongest_relationships.csv`, `time_of_day_analysis.csv`, `target_distributions.csv`, and `report.html`.

### Running Day 4 Analysis

The analysis reads each stock's `data/processed/{SYMBOL}_1min_targets.parquet` (produced by Day 3) and writes all results to `reports/day4_predictive_analysis/` (created automatically; gitignored). Run from the project root:

```bash
.venv/bin/python -m src.analysis.predictive_analysis                  # full 100-stock run
.venv/bin/python -m src.analysis.predictive_analysis --limit-stocks 5 # quick smoke test (first 5 stocks)
.venv/bin/python -m src.analysis.predictive_analysis --top 30         # report top-30 (default 20)
.venv/bin/python -m src.analysis.generate_html_report                 # build report.html from the outputs
```

Options: `--limit-stocks N` processes only the first N stocks (smoke test); `--top N` sets how many strongest relationships to report (default 20). The full run processes all 100 NIFTY-100 stocks (~16 minutes on an M-series Mac). Generated CSV/JSON/HTML outputs are written to `reports/day4_predictive_analysis/` and are gitignored — they are not committed.

Stdlib-only statistics (no scipy): p-values use `math.erfc` (normal) and a self-contained incomplete-gamma routine (chi-square/ANOVA). Only `pandas` / `numpy` / `duckdb` are used (already in the project stack).

### What we found

**Most relationships that rank high on TRAIN persist to VALIDATION** — a notably clean out-of-sample result. All of the top 10 keep their sign; 8 of 10 have a magnitude ratio ≥ 1.0.

Top relationships (by |TRAIN per-stock-mean Pearson|):

| # | Indicator → Target | Train psm | Val psm | Persists | Mag ratio |
|---|--------------------|-----------|---------|----------|-----------|
| 1 | `close_position` → `future_return_5m`  | −0.0415 | −0.0421 | ✓ | 1.01 |
| 2 | `vwap` → `future_return_60m`           | −0.0305 | −0.0415 | ✓ | 1.36 |
| 3 | `sma_20` → `future_return_60m`         | −0.0300 | −0.0387 | ✓ | 1.29 |
| 4 | `sma_5` → `future_return_60m`          | −0.0300 | −0.0404 | ✓ | 1.35 |
| 5 | `return_1m` → `future_return_5m`       | −0.0288 | −0.0274 | ✓ | 0.95 |

- **Short-term mean reversion is the strongest recurring relationship.** `close_position`, `return_1m`, `return_5m`, and `candle_body` all show a **negative** association with the 5-minute future return and persist with magnitude ratio ≈ 0.95–1.05. `close_position → 5m` is consistent in sign across 100% of the 98 stocks with data.
- **Mean reversion vs own moving averages.** `vwap` and `sma_5/20/50/200` show a **negative** association with the 60-minute future return and **strengthen** in validation (ratio 1.29–1.37). These are level indicators — their pooled Pearson is ≈0; only the per-stock-mean reveals the signal (e.g. `vwap → 60m`: pooled +0.0007, psm −0.0305).
- **Volatility does not persist.** The only top-20 relationships that fail validation are volatility ones: `volatility_60 → future_return_60m` **flips sign** (train +0.015 → val −0.005); `volatility_20 → future_return_30m` collapses (ratio 0.022). This is exactly why the validation step exists.
- **Time-of-day.** Reversal associations are present throughout the session and somewhat stronger in the afternoon (e.g. `return_1m → 5m`: −0.013 at open → −0.049 mid-afternoon on TRAIN, mirrored on VALIDATION).

These are **associations**, not strategies. No trading rule has been created. Any future use would require further testing.

### Important research caveats

- **Statistical significance does not imply profitability.** With ~17.8M TRAIN rows, even negligible effects are "statistically significant" (every ANOVA p ≈ 0), but the decile top-vs-bottom bucket spread is only ~1–2.4 basis points against a ~19 bps target std. Effect size and out-of-sample persistence matter far more than p-values.
- **Small effect sizes.** |psm| ≤ ~0.04; no single relationship is a strong predictor. Any future use would require further testing, ensembling, and explicit transaction-cost, slippage, and execution-timing modeling.
- **Multiple-testing / data-mining risk.** 84 relationships × several metrics were examined; treat the ranking as exploratory. The held-out VALIDATION persistence check is the main safeguard against cherry-picking.
- **Level-indicator cross-sectional caveat.** Indicators in absolute price units (`vwap, sma_*, candle_range, candle_body, volume_sma_20`) have cross-sectionally differing scales, so their naive pooled correlation is attenuated toward zero. Trust the per-stock-mean for these; ignore their pooled column.
- **Regime change.** VALIDATION had lower volatility (5m std 0.0015 vs 0.0019 on TRAIN), so a few magnitude ratios >1 partly reflect denominator shrinkage, not stronger signal.
- **Survivorship bias.** The current NIFTY 100 universe is today's index constituents; delisted or removed stocks are absent.
- **Linear / monotonic only.** Only Pearson and Spearman association is measured; interaction and conditional effects are not assessed.
- Day 4 is **exploratory research, not a trading strategy.**

**Rules:** every indicator is backward-looking (within-session); `future_return_*` targets are never used as inputs; every load/query filters `date ≤ VALIDATION_END` and a hard guard asserts no row has `date ≥ TEST_START` (2026-03-04); decile edges are fit on TRAIN only and reused verbatim on VALIDATION; discovery and ranking use TRAIN only. Re-runs overwrite the outputs (no duplication).

**Never commit:** `.env`, `.kite_session`, `data/raw/`, `data/processed/`, or `reports/day4_predictive_analysis/` (generated Day 4 outputs).
