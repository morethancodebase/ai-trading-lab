#!/usr/bin/env python3
"""Inspect the lookback (watchlist) candles for each trade to calibrate the
'far away from VWAP' gap filter. Prints the near-edge distance from VWAP in
ATR(14) units for the 5 candles preceding each bounce.

Run: .venv/bin/python reports/day9_trend_continuation/inspect_gap.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = Path(__file__).resolve().parent
for _p in (str(ROOT), str(REP)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import run_backtest_v2 as eng  # noqa: E402

eng.START_DATE, eng.END_DATE = "2026-07-01", "2026-08-31"
JOURNAL = eng.OUT_DIR / "trade_journal_jul_aug.csv"
LOOKBACK = eng.LOOKBACK
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def main() -> int:
    tdf = pd.read_csv(JOURNAL).sort_values("trade_no").head(N)
    stocks = dict(eng.discover_stocks())   # engine returns list[tuple] -> dict
    cache: dict[str, pd.DataFrame] = {}
    for _, tr in tdf.iterrows():
        sym = tr["symbol"]
        if sym not in cache:
            cache[sym] = eng.load_5min(sym, stocks[sym]) if sym in stocks else None
        g = cache[sym]
        if g is None:
            print(f"#{int(tr['trade_no'])} {sym}: no data"); continue
        setup = pd.to_datetime(tr["setup_time"])
        g = g[g["date"] == setup.date()].reset_index(drop=True)
        idx = g.index[g["timestamp"] == setup]
        if len(idx) == 0:
            print(f"#{int(tr['trade_no'])} {sym} {setup}: setup candle not found"); continue
        i = int(idx[0])
        s = 1 if tr["direction"] == "LONG" else -1
        a = i - LOOKBACK
        print(f"\n#{int(tr['trade_no'])} {sym} {tr['direction']} {setup} "
              f"(vwap@bounce={tr['vwap_at_bounce']:.3f})  net={tr['net_pnl']:.0f} R={tr['r_multiple']:.2f}")
        print(f"  {'time':<6} {'open':>10} {'high':>10} {'low':>10} {'close':>10} {'vwap':>10} "
              f"{'closeVW%':>8} {'nearEdgeVW_atr':>13} {'sideOK':>7}")
        if a < 0:
            print("  not enough lookback"); continue
        seg = g.iloc[a:i]            # 5 lookback candles
        atr_at_setup = g["atr14"].iloc[i]
        near_dists = []
        for _, r in seg.iterrows():
            near = r["low"] if s == 1 else r["high"]
            near_dist_atr = (s * (near - r["vwap"])) / atr_at_setup if atr_at_setup else float("nan")
            near_dists.append(near_dist_atr)
            side_ok = (near > r["vwap"]) if s == 1 else (near < r["vwap"])
            print(f"  {str(r['timestamp'].time())[:5]:<6} {r['open']:>10.2f} {r['high']:>10.2f} "
                  f"{r['low']:>10.2f} {r['close']:>10.2f} {r['vwap']:>10.2f} "
                  f"{(r['close']-r['vwap'])/r['vwap']*100:>8.3f} {near_dist_atr:>13.3f} {str(side_ok):>7}")
        # bounce candle
        b = g.iloc[i]
        near_b = b["low"] if s == 1 else b["high"]
        print(f"  {str(b['timestamp'].time())[:5]:<6} {b['open']:>10.2f} {b['high']:>10.2f} "
              f"{b['low']:>10.2f} {b['close']:>10.2f} {b['vwap']:>10.2f} "
              f"{(b['close']-b['vwap'])/b['vwap']*100:>8.3f} {'(bounce)':>13}")
        import numpy as np
        nd = np.array(near_dists)
        gap_min = getattr(eng, "GAP_MIN_ATR", 0.20)
        print(f"  >> near-edge min ATR dist over lookback = {np.nanmin(nd):.3f}  "
              f"median = {np.nanmedian(nd):.3f}  "
              f"#sideOK (near edge on trend side) = {int(np.sum(nd>0))}/{LOOKBACK}  "
              f"#gapOK (>={gap_min:.2f}xATR) = {int(np.sum(nd>=gap_min))}/{LOOKBACK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
