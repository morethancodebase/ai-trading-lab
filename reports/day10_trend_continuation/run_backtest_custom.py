#!/usr/bin/env python3
"""VWAP Trend-Continuation backtest -- PARAMETERIZED runner (Day 10).

A thin, reusable harness around the validated Day-9 engine
(``reports/day9_trend_continuation/run_backtest_v2.py``) and the per-trade chart
generator (``gen_trade_charts.py``).  Instead of a hard-coded window it takes the
backtest window, risk-per-trade and max notional from the command line, so the
SAME strategy can be re-run on ANY date range without editing code.

All artefacts are written under ``reports/day10_trend_continuation/`` (with a
date-derived suffix) and kept local / untracked (see .gitignore).

Strategy (implemented verbatim by the Day-9 engine):
  * VWAP trend-continuation bounce setup on 5-min candles, session VWAP.
  * Entry on the bounce-candle high/low +/- Rs 0.05 buffer (stop order).
  * Stop-loss on the bounce-candle low/high -/+ Rs 0.05 buffer.
  * Exit at the CLOSE of the 15:10 candle (~15:15) unless SL / trail hits first.
  * Trail stop to breakeven once price reaches 2R (baseline "be" mode).
  * No entry before 09:45 or after 13:00 (1 pm).
  * Skip big candles (range > 2 x ATR14); >= 4 candles away from VWAP first.
  * Bounce candle must be GREEN (long) / RED (short).
  * Max 5 trades/day; qty = floor(risk_per_trade / risk_per_share), capped by
    max_notional when set.

Usage (from project root):
    .venv/bin/python reports/day10_trend_continuation/run_backtest_custom.py \
        --start 2026-07-01 --end 2026-08-31 --risk 1000 --max-notional 500000

    # uncapped notional:
    .venv/bin/python reports/day10_trend_continuation/run_backtest_custom.py \
        --start 2026-07-01 --end 2026-08-31 --risk 1000 --max-notional none

Optional:
    --suffix _jul_aug   override the auto-derived output suffix
    --trail ratchet     breakeven (be, default) or ratchet trail
    --not-before 09:45  earliest entry time (default 09:45)
    --not-after 13:00   latest entry time (default 13:00)
    --no-open           do not open the report in the browser
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import webbrowser
from pathlib import Path
from datetime import time as dtime

ROOT = Path(__file__).resolve().parents[2]
DAY9 = ROOT / "reports" / "day9_trend_continuation"
DAY10 = Path(__file__).resolve().parent
for _p in (str(ROOT), str(DAY9)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_backtest_v2 as eng      # noqa: E402
import gen_trade_charts as charts  # noqa: E402


def _parse_time(s: str) -> dtime:
    parts = s.split(":")
    return dtime(int(parts[0]), int(parts[1]))


def _auto_suffix(start: str, end: str) -> str:
    # 2026-07-01 / 2026-08-31 -> _202607_202608
    return "_" + start.replace("-", "")[:6] + "_" + end.replace("-", "")[:6]


def _clean_stale_pngs(chart_dir: Path, journal: Path) -> int:
    """Remove chart PNGs that no longer match a trade in the journal."""
    if not chart_dir.exists() or not journal.exists():
        return 0
    rows = list(csv.DictReader(open(journal)))
    expected = {
        f'trade_{int(r["trade_no"]):03d}_{r["symbol"]}_{r["date"]}.png'
        for r in rows
    }
    removed = 0
    for p in chart_dir.glob("*.png"):
        if p.name not in expected:
            p.unlink()
            removed += 1
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description="Parameterized VWAP trend-continuation backtest.")
    ap.add_argument("--start", required=True, help="Backtest start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="Backtest end date YYYY-MM-DD (inclusive)")
    ap.add_argument("--risk", type=float, default=1000.0, help="Rs risk per trade (default 1000)")
    ap.add_argument("--max-notional", default="none",
                    help="Max Rs to invest per trade ('none' = uncapped). Default none")
    ap.add_argument("--suffix", default=None, help="Output suffix (auto-derived if omitted)")
    ap.add_argument("--label", default=None, help="Variant label for the report")
    ap.add_argument("--trail", choices=["be", "ratchet"], default="be",
                    help="Trail mode (default be = breakeven at 2R)")
    ap.add_argument("--not-before", default="09:45", help="Earliest entry HH:MM (default 09:45)")
    ap.add_argument("--not-after", default="13:00", help="Latest entry HH:MM (default 13:00)")
    ap.add_argument("--no-open", action="store_true", help="Do not open the report in the browser")
    args = ap.parse_args()

    notional_raw = args.max_notional.strip().lower()
    max_notional = None if notional_raw in ("", "none", "uncapped", "0") else float(notional_raw)

    suffix = args.suffix or _auto_suffix(args.start, args.end)
    not_before = _parse_time(args.not_before)
    not_after = _parse_time(args.not_after)

    notional_txt = "no cap" if max_notional is None else f"cap Rs {max_notional:,.0f}"
    label = args.label or (
        f"CUSTOM ({args.not_before}-{args.not_after} window, {notional_txt}, "
        f"risk Rs {args.risk:,.0f}/trade, trail-{args.trail} at 2R)"
    )

    # Point the shared engine + chart generator at the requested window and the
    # Day-10 output directory.  These module globals are read at call time by
    # load_5min(), run_variant(), write_artifacts(), build_html() and charts.main().
    eng.START_DATE = args.start
    eng.END_DATE = args.end
    eng.RISK_PER_TRADE = args.risk
    eng.NOT_AFTER = not_after
    eng.OUT_DIR = DAY10
    charts.OUT_DIR = DAY10

    DAY10.mkdir(parents=True, exist_ok=True)

    stocks = eng.discover_stocks()
    print("=" * 72)
    print("VWAP TREND-CONTINUATION BACKTEST (5-min) -- CUSTOM WINDOW")
    print(f"Window: {args.start} .. {args.end}  |  Stocks: {len(stocks)}")
    print(f"Risk/trade: Rs {args.risk:,.0f}  |  Max notional: {notional_txt}  "
          f"|  entry {args.not_before}-{args.not_after}  |  trail: {args.trail}")
    print(f"Output suffix: {suffix}  ->  {DAY10}")
    print("=" * 72)

    t0 = time.time()
    cached: list[tuple[str, "pd.DataFrame"]] = []  # type: ignore[name-defined]
    for k, (sym, path) in enumerate(stocks, 1):
        df = eng.load_5min(sym, path)
        if df is not None:
            cached.append((sym, df))
        if k % 20 == 0 or k == len(stocks):
            print(f"  loaded {k}/{len(stocks)} stocks  ({time.time()-t0:.1f}s)")
    print(f"Data loaded: {len(cached)} stocks in {time.time()-t0:.1f}s\n")

    overall = eng.run_variant(label, suffix, max_notional, args.trail, not_before, cached)
    if overall is None:
        print("No trades generated; aborting before chart generation.")
        return 1

    # ---- per-trade charts (reuse gen_trade_charts with this suffix / window) ----
    charts.SUFFIX = suffix
    charts.START_DATE, charts.END_DATE = args.start, args.end
    charts.CHART_DIR = charts.OUT_DIR / f"charts{suffix}"
    charts.JOURNAL = charts.OUT_DIR / f"trade_journal{suffix}.csv"
    charts.REPORT = charts.OUT_DIR / f"trend_continuation_report{suffix}.html"
    charts.main()

    chart_dir = DAY10 / f"charts{suffix}"
    journal = DAY10 / f"trade_journal{suffix}.csv"
    removed = _clean_stale_pngs(chart_dir, journal)

    report = DAY10 / f"trend_continuation_report{suffix}.html"
    print("\n" + "=" * 72)
    print("DONE.")
    print(f"  Trades: {overall['total_trades']}  |  Win: {overall['win_rate']*100:.1f}%  "
          f"|  Net: Rs {overall['total_net']:,.2f}  |  PF: {overall['profit_factor']:.2f}  "
          f"|  Avg R: {overall['avg_r']:.2f}")
    print(f"  Stale PNGs cleaned: {removed}")
    print("Report + artefacts (untracked, not committed):")
    print(f"  {report}")
    print(f"  {journal}")
    print(f"  {DAY10 / ('daily_summary' + suffix + '.csv')}")
    print(f"  {DAY10 / ('monthly_summary' + suffix + '.csv')}")
    print(f"  {DAY10 / ('run_summary' + suffix + '.json')}")
    print(f"  {chart_dir}/  (per-trade chart PNGs)")
    print("=" * 72)

    if not args.no_open:
        webbrowser.open(report.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
