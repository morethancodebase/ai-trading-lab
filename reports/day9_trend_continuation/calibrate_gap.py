#!/usr/bin/env python3
"""Calibrate the new 'far away from VWAP' gap filter on July-August 2026.

Sweeps GAP_MIN_ATR (the minimum near-edge-to-VWAP gap, in ATR(14) units, that a
lookback candle must clear to count as 'far away') and reports trade count,
win rate, net P&L and the first few trades for each value.  Data is loaded once
and the simulation loop (simulate_stock + apply_daily_cap) is reused from the
engine so the numbers match a real run.

Run: .venv/bin/python reports/day9_trend_continuation/calibrate_gap.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = Path(__file__).resolve().parent
for _p in (str(ROOT), str(REP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_backtest_v2 as eng  # noqa: E402

eng.START_DATE, eng.END_DATE = "2026-07-01", "2026-08-31"


def sweep(cached, thresholds):
    print(f"{'GAP_MIN_ATR':>10} {'trades':>7} {'win%':>6} {'net':>14} {'avgR':>6}  first trades")
    print("-" * 90)
    for t in thresholds:
        eng.GAP_MIN_ATR = t
        all_trades, days = [], set()
        for sym, df in cached:
            days.update(df["date"].unique())
            trs = eng.simulate_stock(sym, df)
            if trs:
                all_trades.extend(trs)
        capped = eng.apply_daily_cap(all_trades)
        if not capped:
            print(f"{t:>10.2f} {0:>7} {'-':>6} {'-':>14} {'-':>6}")
            continue
        tdf = pd.DataFrame(capped).sort_values(["date", "entry_time", "symbol"]).reset_index(drop=True)
        net = tdf["net_pnl"].sum()
        win = (tdf["net_pnl"] > 0).mean() * 100
        avgR = tdf["r_multiple"].mean()
        first = tdf.head(6).apply(lambda r: f"{r['symbol']}:{r['direction'][:1]}", axis=1).tolist()
        print(f"{t:>10.2f} {len(tdf):>7} {win:>5.1f}% {net:>14,.0f} {avgR:>6.2f}  {', '.join(first)}")


def main() -> int:
    if not hasattr(eng, "GAP_MIN_ATR"):
        print("ERROR: engine has no GAP_MIN_ATR global -- apply the check_filters fix first.")
        return 1
    stocks = eng.discover_stocks()
    print(f"Loading Jul-Aug 2026 data for {len(stocks)} stocks ...")
    t0 = time.time()
    cached = []
    for sym, path in stocks:
        df = eng.load_5min(sym, path)
        if df is not None:
            cached.append((sym, df))
    print(f"Loaded {len(cached)} stocks in {time.time()-t0:.1f}s\n")
    sweep(cached, [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
