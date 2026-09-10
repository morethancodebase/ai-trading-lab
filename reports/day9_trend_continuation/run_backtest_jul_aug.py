#!/usr/bin/env python3
"""VWAP Trend-Continuation backtest on JULY-AUGUST 2026 -> self-contained HTML journal.

This runner REUSES the already-validated engine in ``run_backtest_v2.py`` (which
implements the strategy EXACTLY as specified) and the per-trade chart generator
in ``gen_trade_charts.py``.  The only differences vs. the August-only BASELINE run
(``trend_continuation_report_aug_base.html``) are:

  * the back-test window is 2026-07-01 .. 2026-08-31  (July + August 2026), and
  * every output artefact is suffixed ``_jul_aug`` instead of ``_aug_base``.

The variant used is the BASELINE = original rules, with NO deviations:
  * 09:45 earliest entry, no notional cap, trail stop to breakeven once price
    reaches 1:2 (2R).

Strategy (implemented verbatim by run_backtest_v2.py):
  * Trend-continuation setup: stock in a one-sided move -- higher highs & higher
    lows (long) / lower highs & lower lows (short).  5-min candles, session VWAP.
  * Watchlist = stocks showing a clear one-directional move (filtered at signal).
  * Wait for price to bounce on VWAP: the 5-min candle wicks INTO VWAP and closes
    back on the trend side; it must NOT close on the wrong side of VWAP.
  * Entry: high (long) / low (short) of the bounce candle +/- a Rs 0.05 buffer,
    as a stop order.
  * Stop-loss: low (long) / high (short) of the bounce candle -/+ Rs 0.05 buffer.
  * Target: manually exit near 15:10-15:15 -- modelled as the CLOSE of the 15:10
    5-min candle (price at ~15:15).
  * Risk management: once price reaches 1:2 (2R), trail the stop to entry (BE).
  * Never take a trade before 09:45 or after 13:00 (1 pm).
  * Avoid big candles: skip if bounce-candle range > 2 x ATR(14).
  * At least 4-5 candles away from VWAP making higher highs (long) / lower lows
    (short) before the bounce, otherwise avoid the trade.
  * Pin bar / hammer bouncing off VWAP is a good sign (recorded as a flag).
  * Bounce candle touching VWAP must be GREEN for long / RED for short
    (close vs open -- the candle touching VWAP must be in the trade direction).
  * Max 5 trades per calendar day (first 5 entries by entry time).
  * Risk per trade Rs 1000 -> qty = floor(1000 / risk_per_share).

Outputs (under reports/day9_trend_continuation/, untracked):
  * trend_continuation_report_jul_aug.html   -- self-contained HTML journal
  * trade_journal_jul_aug.csv                -- one row per trade
  * daily_summary_jul_aug.csv                -- one row per trading day
  * run_summary_jul_aug.json                 -- config + overall metrics
  * charts_jul_aug/                          -- per-trade chart PNGs

Run from the project root:
    .venv/bin/python reports/day9_trend_continuation/run_backtest_jul_aug.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from datetime import time as dtime

ROOT = Path(__file__).resolve().parents[2]
REPDIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(REPDIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the validated engine + chart generator (import = no side effects; main()
# is guarded by __name__ == "__main__").
import run_backtest_v2 as eng          # noqa: E402
import gen_trade_charts as charts       # noqa: E402

START = "2026-07-01"                    # July 2026
END = "2026-08-31"                      # August 2026 (inclusive)
SUFFIX = "_jul_aug"


def main() -> int:
    # Point the shared engine at the July-August window.  These module globals
    # are read at call-time by load_5min(), build_html() and write_artifacts().
    eng.START_DATE = START
    eng.END_DATE = END
    eng.OUT_DIR.mkdir(parents=True, exist_ok=True)

    stocks = eng.discover_stocks()
    print("=" * 72)
    print("VWAP TREND-CONTINUATION BACKTEST (5-min) -- JUL-AUG 2026 (BASELINE rules)")
    print(f"Window: {eng.START_DATE} .. {eng.END_DATE}  |  Stocks: {len(stocks)}")
    print(f"Risk/trade: Rs {eng.RISK_PER_TRADE:.0f}  |  Buffer Rs {eng.BUFFER}  "
          f"|  max {eng.MAX_TRADES_PER_DAY} trades/day  |  trail-to-BE at 2R")
    print("=" * 72)

    t0 = time.time()
    cached: list[tuple[str, "pd.DataFrame"]] = []   # type: ignore[name-defined]
    for k, (sym, path) in enumerate(stocks, 1):
        df = eng.load_5min(sym, path)
        if df is not None:
            cached.append((sym, df))
        if k % 20 == 0 or k == len(stocks):
            print(f"  loaded {k}/{len(stocks)} stocks  ({time.time()-t0:.1f}s)")
    print(f"Data loaded: {len(cached)} stocks in {time.time()-t0:.1f}s\n")

    # BASELINE = original rules exactly as specified (no deviations).
    overall = eng.run_variant(
        "BASELINE (original rules: 09:45 start, no notional cap, trail-to-BE at 2R)",
        SUFFIX, None, "be", dtime(9, 45), cached)
    if overall is None:
        print("No trades generated; aborting before chart generation.")
        return 1

    # ---- per-trade charts: reuse gen_trade_charts with the jul_aug suffix/dates ----
    charts.SUFFIX = SUFFIX
    charts.START_DATE, charts.END_DATE = START, END
    charts.CHART_DIR = charts.OUT_DIR / f"charts{SUFFIX}"
    charts.JOURNAL = charts.OUT_DIR / f"trade_journal{SUFFIX}.csv"
    charts.REPORT = charts.OUT_DIR / f"trend_continuation_report{SUFFIX}.html"
    charts.main()

    print("\n" + "=" * 72)
    print("DONE.  Report + artefacts (untracked, not committed):")
    print(f"  {charts.REPORT}")
    print(f"  {charts.JOURNAL}")
    print(f"  {eng.OUT_DIR / ('daily_summary' + SUFFIX + '.csv')}")
    print(f"  {eng.OUT_DIR / ('run_summary' + SUFFIX + '.json')}")
    print(f"  {charts.CHART_DIR}/  (per-trade chart PNGs)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
