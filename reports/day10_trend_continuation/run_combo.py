#!/usr/bin/env python3
"""VWAP combination search -- find the global max-profit strategy config.

Builds on the ablation study.  Four levers, each with two states, are swept as
a full 2x2x2x2 grid (16 runs) on the NIFTY-200 universe (Jan-Aug 2026):

  LEVER                OFF (kept, baseline)        ON (dropped/opened)
  ----                 ---------------------       --------------------
  big_candle           BIG_CANDLE_MULT=2.0         BIG_CANDLE_MULT=1e9 (no skip)
  notional_cap         MAX_NOTIONAL=300000          MAX_NOTIONAL=None (uncapped)
  trail                _trail_stop = original BE   _trail_stop = None (let run)
  shorts_only          ALLOW_LONG=True (both)      ALLOW_LONG=False (shorts only)

Rationale (from the ablation + direction analysis):
  * big-candle skip HURTS  (drop => +Rs 12k single-rule)
  * notional cap HURTS     (drop => +Rs 11k -- bigger size on cheap, tight-stop names)
  * trail-to-BE HURTS      (drop => +Rs 1.7k -- let winners ride)
  * LONGS LOSE money in every config (long_net negative); shorts carry all the edge.

These four mechanisms are orthogonal (setups vs size vs exit vs side), so they
should compound.  The grid finds the true global max net P&L, then regenerates
the winner's full report (journal + per-trade charts + HTML) and writes a
comparison HTML showing the risk/reward (net P&L vs max drawdown) of all 16.

Usage (from project root):
    .venv/bin/python reports/day10_trend_continuation/run_combo.py
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import webbrowser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY10 = Path(__file__).resolve().parent
DAY9 = ROOT / "reports" / "day9_trend_continuation"
for _p in (str(ROOT), str(DAY10), str(DAY9)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_ablation as abl   # noqa: E402  (sets up eng patches + helpers)
import gen_trade_charts as charts  # noqa: E402

eng = abl.eng
START, END = "2026-01-01", "2026-08-31"
ABL_HTML = DAY10 / "combo_report_202601_202608.html"
WIN_SUFFIX = "_combo_best_202601_202608"


def set_combo(big_candle, notional_cap, trail, shorts_only):
    """Configure the engine for one grid cell. False = baseline state."""
    abl.set_baseline()
    eng.START_DATE, eng.END_DATE = START, END
    if big_candle:            # drop the big-candle skip
        eng.BIG_CANDLE_MULT = 1e9
    if notional_cap:         # drop the notional cap (uncapped size)
        eng.MAX_NOTIONAL = None
    if trail:                # disable trailing (let winners run to time-stop)
        eng._trail_stop = abl._no_trail
    # else: set_baseline() already restored the original BE trail
    if shorts_only:          # shorts only (longs are a net drag in every run)
        eng.ALLOW_LONG = False
        eng.ALLOW_SHORT = True


def combo_label(bc, nc, tr, so):
    parts = []
    if bc:
        parts.append("drop big-candle skip")
    if nc:
        parts.append("drop notional cap")
    if tr:
        parts.append("no trail (let winners run)")
    if so:
        parts.append("shorts-only")
    if not parts:
        return "Baseline (all rules ON)"
    return " + ".join(parts)


def _inr(v, dp=0):
    return "\u20b9" + f"{v:,.{dp}f}"


def _inr_s(v, dp=0):
    sign = "-" if v < 0 else ""
    return f"{sign}\u20b9{abs(v):,.{dp}f}"


def build_combo_html(rows, baseline_net, win_idx):
    """rows: list of dicts. win_idx: index of the max-net row."""
    s = []
    s.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    s.append("<title>VWAP Combination Search -- NIFTY 200 Jan-Aug 2026</title>")
    s.append("<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
             "background:#f7f8fa;color:#222;margin:0;padding:24px}"
             "h1{font-size:22px;margin:0 0 4px}h2{font-size:14px;font-weight:500;"
             "color:#666;margin:0 0 20px}"
             ".tbl{border-collapse:collapse;width:100%;background:#fff;"
             "box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:8px;overflow:hidden}"
             "th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #eee;"
             "font-size:13px}th{background:#fafbfc;color:#666;font-weight:600;"
             "text-transform:uppercase;font-size:11px;letter-spacing:.04em}"
             "td:first-child,th:first-child{text-align:left}"
             "tr.win{background:#e8f5e9;font-weight:600}tr.win td{color:#1b5e20}"
             ".neg{color:#c62828}.pos{color:#2e7d32}"
             ".top{display:flex;gap:16px;margin-bottom:20px}"
             ".card{flex:1;background:#fff;border-radius:8px;padding:16px 20px;"
             "box-shadow:0 1px 3px rgba(0,0,0,.08)}"
             ".card .k{font-size:11px;color:#888;text-transform:uppercase;"
             "letter-spacing:.05em}.card .v{font-size:22px;font-weight:600;margin-top:4px}"
             ".note{margin-top:18px;font-size:12px;color:#888}")
    s.append("</style></head><body>")
    s.append("<h1>VWAP Trend-Continuation -- Combination Search</h1>")
    s.append("<h2>NIFTY 200, 5-min, Jan-Aug 2026 &middot; 4 levers &times; 2 "
             "= 16 configs (full grid)</h2>")
    win = rows[win_idx]
    d_win = win["net"] - baseline_net
    s.append("<div class='top'>")
    s.append(f"<div class='card'><div class='k'>Baseline net P&L</div>"
             f"<div class='v'>{_inr_s(baseline_net)}</div></div>")
    s.append(f"<div class='card'><div class='k'>Best combo net P&L</div>"
             f"<div class='v pos'>{_inr_s(win['net'])}</div></div>")
    s.append(f"<div class='card'><div class='k'>Uplift vs baseline</div>"
             f"<div class='v pos'>{_inr_s(d_win)}</div></div>")
    s.append(f"<div class='card'><div class='k'>Best combo PF / max DD</div>"
             f"<div class='v' style='font-size:18px'>{win['pf']:.2f} / "
             f"{_inr_s(win['max_dd'])}</div></div>")
    s.append("</div>")
    s.append("<table class='tbl'><thead><tr><th>#</th><th>Configuration "
             "(dropped levers)</th><th>Trades</th><th>Win %</th><th>Long</th>"
             "<th>Short</th><th>Net P&L</th><th>&Delta; vs base</th>"
             "<th>PF</th><th>Max DD</th></tr></thead><tbody>")
    for i, r in enumerate(rows):
        cls = " class='win'" if i == win_idx else ""
        d = r["net"] - baseline_net
        dcls = "pos" if d >= 0 else "neg"
        ncls = "pos" if r["net"] >= 0 else "neg"
        s.append(f"<tr{cls}><td>{i+1}</td><td>{r['label']}</td>"
                 f"<td>{r['trades']}</td><td>{r['win_pct']:.1f}%</td>"
                 f"<td class='{'neg' if r['long_net']<0 else 'pos'}'>"
                 f"{_inr_s(r['long_net'])}</td>"
                 f"<td class='{'pos' if r['short_net']>=0 else 'neg'}'>"
                 f"{_inr_s(r['short_net'])}</td>"
                 f"<td class='{ncls}'>{_inr_s(r['net'])}</td>"
                 f"<td class='{dcls}'>{_inr_s(d)}</td>"
                 f"<td>{r['pf']:.2f}</td>"
                 f"<td class='neg'>{_inr_s(r['max_dd'])}</td></tr>")
    s.append("</tbody></table>")
    s.append("<div class='note'>Green row = max net P&L. &Delta; vs base = "
             "row net &minus; baseline net. Max DD is the worst peak-to-trough "
             "equity drawdown (risk side of each lever). Levers: big-candle "
             "skip / notional cap / trail-to-BE are dropped (off = kept = "
             "baseline); shorts-only disables long entries.</div>")
    s.append("</body></html>")
    return "".join(s)


def main() -> int:
    DAY10.mkdir(parents=True, exist_ok=True)
    set_combo(False, False, False, False)  # baseline
    eng.OUT_DIR = DAY10

    stocks = eng.discover_stocks()
    grid = list(itertools.product([False, True], repeat=4))  # 16 combos
    print("=" * 78)
    print("VWAP COMBINATION SEARCH -- NIFTY 200, Jan-Aug 2026")
    print("Levers: big-candle-skip x notional-cap x trail x shorts-only (2^4 = 16)")
    print("=" * 78)

    # ---- load 5-min data ONCE ----
    t0 = time.time()
    cached: list[tuple[str, pd.DataFrame]] = []
    for k, (sym, path) in enumerate(stocks, 1):
        df = eng.load_5min(sym, path)
        if df is not None:
            cached.append((sym, df))
        if k % 40 == 0 or k == len(stocks):
            print(f"  loaded {k}/{len(stocks)}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"Data loaded: {len(cached)} stocks in {time.time()-t0:.1f}s\n", flush=True)

    # ---- run the 16-cell grid ----
    rows = []
    base_net = None
    for n, (bc, nc, tr, so) in enumerate(grid, 1):
        set_combo(bc, nc, tr, so)
        label = combo_label(bc, nc, tr, so)
        print(f"[{n:2d}/16] {label}", flush=True)
        res = abl.simulate(cached)
        if res is None:
            print("        no trades -- skipped\n", flush=True)
            rows.append(dict(label=label, trades=0, win_pct=0.0, net=0.0,
                             long_net=0.0, short_net=0.0, pf=0.0, max_dd=0.0,
                             cfg=(bc, nc, tr, so)))
            continue
        o, *_ = res
        net = float(o["final_net"])
        if n == 1:  # first cell = all-False = baseline
            base_net = net
        print(f"        trades={o['total_trades']}  win={o['win_rate']*100:.1f}%  "
              f"net={_inr_s(net)}  PF={o['profit_factor']:.2f}  "
              f"DD={_inr_s(o['max_drawdown'])}\n", flush=True)
        rows.append(dict(label=label, trades=o["total_trades"],
                         win_pct=o["win_rate"] * 100, net=net,
                         long_net=float(o["long_net"]), short_net=float(o["short_net"]),
                         pf=float(o["profit_factor"]),
                         max_dd=float(o["max_drawdown"]),
                         cfg=(bc, nc, tr, so)))

    # ---- find global max net ----
    win_idx = max(range(len(rows)), key=lambda i: rows[i]["net"])
    win = rows[win_idx]
    print("=" * 78)
    print(f"BEST COMBO: {win['label']}")
    print(f"  net {_inr_s(win['net'])}  ({win['trades']} trades, "
          f"PF {win['pf']:.2f}, DD {_inr_s(win['max_dd'])}, "
          f"uplift {_inr_s(win['net']-base_net)} vs baseline)")
    print("=" * 78 + "\n", flush=True)

    # ---- regenerate the winner's full report + charts ----
    bc, nc, tr, so = win["cfg"]
    set_combo(bc, nc, tr, so)
    res = abl.simulate(cached)
    o, tdf, daily, monthly, trading_days = res
    eng.OUT_DIR = DAY10
    eng.OUT_SUFFIX = WIN_SUFFIX
    eng.VARIANT_LABEL = (
        f"COMBO BEST -- {win['label']} (NIFTY 200, Jan-Aug 2026)")
    if tr:
        eng.TRAIL_MODE = "off (disabled)"
    eng.write_artifacts(tdf, daily, monthly, o, len(trading_days))
    cfg = abl.current_cfg(eng.VARIANT_LABEL)
    cfg["combo"] = {
        "drop_big_candle": bc, "drop_notional_cap": nc,
        "no_trail": tr, "shorts_only": so}
    summary = {"config": cfg, "overall": o, "monthly": monthly.to_dict("records")}
    with (DAY10 / f"run_summary{WIN_SUFFIX}.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    html = eng.build_html(tdf, daily, monthly, o, summary)
    win_report = DAY10 / f"trend_continuation_report{WIN_SUFFIX}.html"
    win_report.write_text(html, encoding="utf-8")

    charts.OUT_DIR = DAY10
    charts.SUFFIX = WIN_SUFFIX
    charts.START_DATE, charts.END_DATE = START, END
    charts.CHART_DIR = DAY10 / f"charts{WIN_SUFFIX}"
    charts.JOURNAL = DAY10 / f"trade_journal{WIN_SUFFIX}.csv"
    charts.REPORT = win_report
    print("Generating per-trade charts for the best combo...", flush=True)
    try:
        charts.main()
    except Exception as e:
        print(f"  (chart generation warning: {e})", flush=True)

    # ---- combination comparison HTML ----
    ABL_HTML.write_text(build_combo_html(rows, base_net, win_idx),
                        encoding="utf-8")

    print("\n" + "=" * 78)
    print("DONE.")
    print(f"  Baseline net: {_inr_s(base_net)}  ({rows[0]['trades']} trades)")
    print(f"  Best combo: {win['label']} -> net {_inr_s(win['net'])} "
          f"({win['trades']} trades, PF {win['pf']:.2f}, "
          f"uplift {_inr_s(win['net']-base_net)})")
    print("  Combo report    : " + str(ABL_HTML))
    print("  Winner full report: " + str(win_report))
    print("  Winner journal   : " + str(DAY10 / f"trade_journal{WIN_SUFFIX}.csv"))
    print("  Winner charts    : " + str(DAY10 / f"charts{WIN_SUFFIX}/"))
    print("=" * 78)

    webbrowser.open(ABL_HTML.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

