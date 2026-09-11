#!/usr/bin/env python3
"""VWAP strategy ablation study (leave-one-rule-out) -- NIFTY 200, Jan-Aug 2026.

For the VWAP trend-continuation strategy on the full NIFTY 200 universe
(Jan-Aug 2026), drop ONE rule at a time, re-run the backtest, and record the
total net P&L.  The single rule whose removal yields the highest net P&L is
declared the winner; its full report (journal + per-trade charts + HTML) is
regenerated, and a comparison table of every variant is written to an
ablation HTML report.

Rules ablated (leave-one-out, 11 rules + 1 baseline = 12 runs):
  1. no-entry-before-09:45      NOT_BEFORE -> 09:15
  2. no-entry-after-13:00       NOT_AFTER  -> 15:15
  3. green/red-bounce-candle    REQUIRE_GREEN_BOUNCE -> False
  4. big-candle-skip (>2xATR)   BIG_CANDLE_MULT -> 1e9
  5. 4+-candles-away-from-VWAP ABOVE_VWAP_MIN -> 0
  6. higher-highs/lower-lows    patched check_filters bypass (ABLATE_TREND)
  7. extend-0.10%-away         EXTEND_MIN_DIST -> -1e9
  8. one-directional-day        patched check_filters bypass (ABLATE_ONE_DIR)
  9. max-5-trades/day          MAX_TRADES_PER_DAY -> 9999
 10. notional-cap-Rs-3L         MAX_NOTIONAL -> None
 11. trail-to-BE-at-2R          patched _trail_stop -> always None

Usage (from project root):
    .venv/bin/python reports/day10_trend_continuation/run_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
import webbrowser
from pathlib import Path
from datetime import time as dtime

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DAY9 = ROOT / "reports" / "day9_trend_continuation"
DAY10 = Path(__file__).resolve().parent
for _p in (str(ROOT), str(DAY9)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_backtest_v2 as eng      # noqa: E402
import gen_trade_charts as charts  # noqa: E402

START = "2026-01-01"
END = "2026-08-31"
ABL_HTML = DAY10 / "ablation_report_202601_202608.html"
WIN_SUFFIX = "_abl_best_202601_202608"

# ---- baseline config (matches the committed NIFTY-200 run) ----
BASELINE = dict(
    START_DATE=START, END_DATE=END,
    RISK_PER_TRADE=1000.0, MAX_NOTIONAL=300000.0,
    BUFFER=0.05, BIG_CANDLE_MULT=2.0, ATR_WINDOW=14,
    NOT_BEFORE=dtime(9, 45), NOT_AFTER=dtime(13, 0),
    ALLOW_LONG=True, ALLOW_SHORT=True,
    SKIP_LUNCH_START=None, SKIP_LUNCH_END=None,
    EXIT_BAR_OPEN=dtime(15, 10),
    LOOKBACK=5, ABOVE_VWAP_MIN=4, EXTEND_MIN_DIST=0.0010, GAP_MIN_ATR=0.20,
    REQUIRE_GREEN_BOUNCE=True, MAX_TRADES_PER_DAY=5, TRAIL_MODE="be",
)

# ---------------------------------------------------------------------------
# Monkey-patches for the three hardcoded rules (trend, one-dir, trail).
# With ABLATE_TREND / ABLATE_ONE_DIR False and the original _trail_stop, the
# patched check_filters is byte-for-byte identical to the engine's original.
# ---------------------------------------------------------------------------
eng.ABLATE_TREND = False
eng.ABLATE_ONE_DIR = False
_orig_trail_stop = eng._trail_stop


def _patched_check_filters(s, i, cl, hi, lo, vwap, dist, day_open, atr, rng,
                           times, is_green):
    """Engine check_filters + two ablation bypasses (trend / one-directional)."""
    if (pd.isna(atr[i]) or i < eng.ATR_WINDOW
            or times[i] < eng.NOT_BEFORE or times[i] >= eng.NOT_AFTER):
        return False
    if s == 1:
        if not (lo[i] <= vwap[i] and cl[i] > vwap[i]):
            return False
    else:
        if not (hi[i] >= vwap[i] and cl[i] < vwap[i]):
            return False
    if eng.REQUIRE_GREEN_BOUNCE:
        if s == 1 and not is_green[i]:
            return False
        if s == -1 and is_green[i]:
            return False
    if rng[i] > eng.BIG_CANDLE_MULT * atr[i]:
        return False
    a = i - eng.LOOKBACK
    if a < 0:
        return False
    seg_cl, seg_vw = cl[a:i], vwap[a:i]
    seg_hi, seg_lo = hi[a:i], lo[a:i]
    seg_dist = dist[a:i]
    if len(seg_cl) < eng.LOOKBACK:
        return False
    if s == 1:
        near = seg_lo
        gap_atr = (near - seg_vw) / atr[i]
        if int((gap_atr >= eng.GAP_MIN_ATR).sum()) < eng.ABOVE_VWAP_MIN:
            return False
        if not eng.ABLATE_TREND:
            if not (seg_hi[-1] >= seg_hi[0]):          # higher high
                return False
            if not (seg_lo[-1] >= seg_lo[0]):          # higher low
                return False
        if float(seg_dist.max()) < eng.EXTEND_MIN_DIST:
            return False
        if not eng.ABLATE_ONE_DIR:
            if not (cl[i] > day_open):                 # one-directional up day
                return False
    else:
        near = seg_hi
        gap_atr = (seg_vw - near) / atr[i]
        if int((gap_atr >= eng.GAP_MIN_ATR).sum()) < eng.ABOVE_VWAP_MIN:
            return False
        if not eng.ABLATE_TREND:
            if not (seg_hi[-1] <= seg_hi[0]):          # lower high
                return False
            if not (seg_lo[-1] <= seg_lo[0]):          # lower low
                return False
        if float((-seg_dist).max()) < eng.EXTEND_MIN_DIST:
            return False
        if not eng.ABLATE_ONE_DIR:
            if not (cl[i] < day_open):                 # one-directional down day
                return False
    return True


def _no_trail(s, entry_px, risk, fav_r, mode):
    """Disable trailing entirely (stop stays at the initial SL until SL/time)."""
    return None


eng.check_filters = _patched_check_filters   # always active; flags default off

def set_baseline():
    """Reset every engine knob to the baseline (all-rules-on) config."""
    for k, v in BASELINE.items():
        setattr(eng, k, v)
    eng.ABLATE_TREND = False
    eng.ABLATE_ONE_DIR = False
    eng._trail_stop = _orig_trail_stop


# ---------------------------------------------------------------------------
# Rule definitions: (key, human label, list of (attr, value) overrides).
# Special attr "_TRAIL_OFF" swaps in the no-trail patch.
# ---------------------------------------------------------------------------
RULES = [
    ("no_entry_before_0945", "No entry before 09:45",
     [("NOT_BEFORE", dtime(9, 15))]),
    ("no_entry_after_1300", "No entry after 13:00",
     [("NOT_AFTER", dtime(15, 15))]),
    ("green_bounce", "Bounce candle GREEN/RED",
     [("REQUIRE_GREEN_BOUNCE", False)]),
    ("big_candle", "Skip big candles (>2xATR)",
     [("BIG_CANDLE_MULT", 1e9)]),
    ("away_from_vwap", "4+ candles away from VWAP",
     [("ABOVE_VWAP_MIN", 0)]),
    ("trend_hh_ll", "Higher highs / lower lows",
     [("ABLATE_TREND", True)]),
    ("extend_min_dist", "Extended 0.10% away from VWAP",
     [("EXTEND_MIN_DIST", -1e9)]),
    ("one_directional", "One-directional day",
     [("ABLATE_ONE_DIR", True)]),
    ("max_trades_day", "Max 5 trades/day",
     [("MAX_TRADES_PER_DAY", 9999)]),
    ("notional_cap", "Notional cap Rs 3L/trade",
     [("MAX_NOTIONAL", None)]),
    ("trail_to_be", "Trail stop to breakeven at 2R",
     [("_TRAIL_OFF", True)]),
]


def apply_rule(rule):
    for attr, val in rule[2]:
        if attr == "_TRAIL_OFF":
            eng._trail_stop = _no_trail
        else:
            setattr(eng, attr, val)


def simulate(cached):
    """Run the simulation with the current engine globals; return metrics."""
    all_trades = []
    trading_days: set = set()
    for sym, df in cached:
        trading_days.update(df["date"].unique())
        trs = eng.simulate_stock(sym, df)
        if trs:
            all_trades.extend(trs)
    capped = eng.apply_daily_cap(all_trades)
    if not capped:
        return None
    tdf = pd.DataFrame(capped)
    tdf = tdf.sort_values(["date", "entry_time", "symbol"]).reset_index(drop=True)
    tdf["trade_no"] = range(1, len(tdf) + 1)
    daily = eng.build_daily(tdf)
    monthly = eng.build_monthly(tdf)
    overall = eng.build_overall(tdf, daily, trading_days)
    return overall, tdf, daily, monthly, trading_days


def current_cfg(label, ablated=None):
    return {
        "strategy": "VWAP Trend Continuation (5-min)", "variant": label,
        "start_date": eng.START_DATE, "end_date": eng.END_DATE,
        "risk_per_trade": eng.RISK_PER_TRADE,
        "max_trades_per_day": eng.MAX_TRADES_PER_DAY,
        "max_notional": (None if eng.MAX_NOTIONAL is None else float(eng.MAX_NOTIONAL)),
        "buffer": eng.BUFFER, "big_candle_mult": eng.BIG_CANDLE_MULT,
        "atr_window": eng.ATR_WINDOW,
        "not_before": str(eng.NOT_BEFORE), "not_after": str(eng.NOT_AFTER),
        "exit_bar_open": str(eng.EXIT_BAR_OPEN),
        "lookback": eng.LOOKBACK, "above_vwap_min": eng.ABOVE_VWAP_MIN,
        "extend_min_dist": eng.EXTEND_MIN_DIST, "gap_min_atr": eng.GAP_MIN_ATR,
        "require_green_bounce": eng.REQUIRE_GREEN_BOUNCE,
        "trail_mode": eng.TRAIL_MODE,
        "allow_long": eng.ALLOW_LONG, "allow_short": eng.ALLOW_SHORT,
        "skip_lunch_start": (None if eng.SKIP_LUNCH_START is None else str(eng.SKIP_LUNCH_START)),
        "skip_lunch_end": (None if eng.SKIP_LUNCH_END is None else str(eng.SKIP_LUNCH_END)),
        "ablated_rule": ablated,
    }


def _inr(v, dp=2):
    return "\u20b9" + f"{v:,.{dp}f}"


def _inr_signed(v, dp=2):
    sign = "-" if v < 0 else ""
    return f"{sign}\u20b9{abs(v):,.{dp}f}"


def build_ablation_html(rows, base_net, winner):
    """rows: list of dicts (key,label,trades,win,net,gross,charges,pf,avg_r,dd,...)."""
    css = """
    *{box-sizing:border-box}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
      margin:0;color:#1f2933;background:#f4f6f8;line-height:1.45}
    .wrap{max-width:1100px;margin:0 auto;padding:28px 22px 60px}
    h1{font-size:22px;margin:0 0 4px;color:#102a43}
    h2{font-size:16px;margin:26px 0 10px;color:#243b53;border-bottom:2px solid #d9e2ec;padding-bottom:6px}
    .sub{color:#486581;font-size:13px;margin-bottom:6px}
    table{border-collapse:collapse;width:100%;font-size:12px;background:#fff}
    th,td{padding:7px 9px;border-bottom:1px solid #eef1f4;text-align:right;white-space:nowrap}
    th{background:#f0f4f8;color:#334e68;font-weight:600}
    td.l,th.l{text-align:left}
    tr:hover td{background:#fbfcfd}
    .pos{color:#0a7c2a}.neg{color:#c02a2a}.muted{color:#627d98}
    tr.win td{background:#eef7ff;font-weight:700}
    tr.base td{background:#f0f4f8;font-weight:700}
    .small{font-size:12px;color:#486581}
    .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0}
    .card{background:#fff;border:1px solid #e0e6eb;border-radius:10px;padding:14px 16px}
    .card .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#829ab1}
    .card .v{font-size:18px;font-weight:700;margin-top:4px}
    """
    head = ("<tr><th class='l'>Variant (rule dropped)</th><th>Trades</th>"
           "<th>Win%</th><th>Net P&L</th><th>&Delta; vs base</th>"
           "<th>Gross</th><th>Charges</th><th>PF</th><th>Avg R</th>"
           "<th>Max DD</th><th>Long net</th><th>Short net</th></tr>")
    body = ""
    for r in rows:
        cls = "win" if r["key"] == winner else ("base" if r["key"] == "baseline" else "")
        ncls = "pos" if r["net"] >= 0 else "neg"
        delta = r["net"] - base_net
        dcls = "pos" if delta >= 0 else "neg"
        body += (
            f'<tr class="{cls}"><td class="l">{r["label"]}</td>'
            f'<td>{r["trades"]}</td><td>{r["win"] * 100:.1f}%</td>'
            f'<td class="{ncls}"><b>{_inr_signed(r["net"])}</b></td>'
            f'<td class="{dcls}">{_inr_signed(delta)}</td>'
            f'<td>{_inr(r["gross"])}</td><td class="muted">{_inr(r["charges"])}</td>'
            f'<td>{r["pf"]:.2f}</td><td>{r["avg_r"]:.2f}</td>'
            f'<td class="neg">{_inr_signed(r["dd"])}</td>'
            f'<td>{_inr_signed(r["long_net"])}</td>'
            f'<td>{_inr_signed(r["short_net"])}</td></tr>'
        )
    win_label = next(r["label"] for r in rows if r["key"] == winner)
    win_row = next(r for r in rows if r["key"] == winner)
    cards = "".join([
        f'<div class="card"><div class="k">Best rule to drop</div><div class="v" style="font-size:14px">{win_label}</div></div>',
        f'<div class="card"><div class="k">Winner net P&L</div><div class="v pos">{_inr_signed(win_row["net"])}</div></div>',
        f'<div class="card"><div class="k">Baseline net P&L</div><div class="v {"pos" if base_net>=0 else "neg"}">{_inr_signed(base_net)}</div></div>',
        f'<div class="card"><div class="k">Improvement</div><div class="v pos">{_inr_signed(win_row["net"] - base_net)}</div></div>',
    ])
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>VWAP Strategy Ablation Study</title><style>' + css + '</style></head>'
        '<body><div class="wrap">'
        '<h1>VWAP Strategy Ablation Study &mdash; leave-one-rule-out</h1>'
        f'<div class="sub">NIFTY 200 &bull; 5-min &bull; {START} to {END} &bull; '
        f'Rs 1,000 risk/trade &bull; cap Rs 3,00,000 &bull; trail-BE at 2R &bull; '
        f'11 rules ablated one at a time vs an all-rules-on baseline.</div>'
        f'<div class="cards">{cards}</div>'
        '<h2>Comparison table (each row = one rule dropped)</h2>'
        '<p class="small">Positive &Delta; = dropping this rule <b>improved</b> net P&L '
        '(the rule was hurting). Negative &Delta; = the rule was helping (dropping it hurts). '
        'The highlighted <span style="background:#eef7ff">blue row</span> is the winner.</p>'
        f'<table><thead>{head}</thead><tbody>{body}</tbody></table>'
        '<h2>Methodology</h2><p class="small">The same NIFTY-200 5-min dataset is loaded '
        'once and re-simulated 12 times. For each run exactly one strategy rule is disabled; '
        'all other rules stay on. No re-optimisation of thresholds &mdash; this measures each '
        'rule\'s marginal contribution to net P&L, not a parameter search. The winner\'s full '
        'report (journal, per-trade charts, equity curve) is regenerated separately.</p>'
        '<p class="small" style="margin-top:24px">Generated by '
        'reports/day10_trend_continuation/run_ablation.py. Untracked / not committed.</p>'
        '</div></body></html>'
    )


def main() -> int:
    DAY10.mkdir(parents=True, exist_ok=True)
    set_baseline()
    eng.START_DATE, eng.END_DATE = START, END
    eng.OUT_DIR = DAY10

    stocks = eng.discover_stocks()
    print("=" * 76)
    print("VWAP ABLATION STUDY (leave-one-rule-out) -- NIFTY 200, Jan-Aug 2026")
    print(f"Stocks: {len(stocks)}  |  Rules ablated: {len(RULES)}  |  Runs: {len(RULES)+1}")
    print("=" * 76)

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

    results = []

    # ---- baseline (all rules on) ----
    print("[ 0/12] BASELINE (all rules on)", flush=True)
    set_baseline()
    res = simulate(cached)
    if res is None:
        print("ERROR: baseline produced no trades.", flush=True)
        return 1
    o, *_ = res
    base_net = float(o["final_net"])
    print(f"        trades={o['total_trades']}  win={o['win_rate']*100:.1f}%  "
          f"net={_inr_signed(o['final_net'])}  PF={o['profit_factor']:.2f}\n", flush=True)
    results.append(dict(
        key="baseline", label="Baseline (all rules ON)", trades=o["total_trades"],
        win=o["win_rate"], net=float(o["final_net"]), gross=o["total_gross"],
        charges=o["total_charges"], pf=o["profit_factor"], avg_r=o["avg_r"],
        dd=o["max_drawdown"], long_net=o["long_net"], short_net=o["short_net"]))

    # ---- leave-one-out ----
    for n, rule in enumerate(RULES, 1):
        set_baseline()
        apply_rule(rule)
        print(f"[{n:2d}/12] drop: {rule[1]}", flush=True)
        res = simulate(cached)
        if res is None:
            print("        no trades -- skipped\n", flush=True)
            results.append(dict(key=rule[0], label=f"Drop: {rule[1]}", trades=0,
                                win=0.0, net=0.0, gross=0.0, charges=0.0, pf=0.0,
                                avg_r=0.0, dd=0.0, long_net=0.0, short_net=0.0))
            continue
        o, *_ = res
        print(f"        trades={o['total_trades']}  win={o['win_rate']*100:.1f}%  "
              f"net={_inr_signed(o['final_net'])}  PF={o['profit_factor']:.2f}\n", flush=True)
        results.append(dict(
            key=rule[0], label=f"Drop: {rule[1]}", trades=o["total_trades"],
            win=o["win_rate"], net=float(o["final_net"]), gross=o["total_gross"],
            charges=o["total_charges"], pf=o["profit_factor"], avg_r=o["avg_r"],
            dd=o["max_drawdown"], long_net=o["long_net"], short_net=o["short_net"]))

    # ---- find winner (max net P&L among non-baseline variants with trades) ----
    cand = [r for r in results if r["key"] != "baseline" and r["trades"] > 0]
    winner = max(cand, key=lambda r: r["net"])["key"]
    win_row = next(r for r in results if r["key"] == winner)
    print("=" * 76)
    print(f"WINNER: drop '{win_row['label']}' -> net {_inr_signed(win_row['net'])} "
          f"(baseline {_inr_signed(base_net)}, delta {_inr_signed(win_row['net']-base_net)})")
    print("=" * 76 + "\n", flush=True)

    # ---- regenerate the winner's full report + charts ----
    win_rule = next(r for r in RULES if r[0] == winner)
    set_baseline()
    apply_rule(win_rule)
    if winner == "trail_to_be":
        eng.TRAIL_MODE = "off (disabled)"
    win_label = f"ABLATION BEST -- drop: {win_rule[1]} (NIFTY 200, Jan-Aug 2026)"
    res = simulate(cached)
    o, tdf, daily, monthly, trading_days = res
    eng.OUT_DIR = DAY10
    eng.OUT_SUFFIX = WIN_SUFFIX
    eng.VARIANT_LABEL = win_label
    eng.write_artifacts(tdf, daily, monthly, o, len(trading_days))
    cfg = current_cfg(win_label, ablated=winner)
    summary = {"config": cfg, "overall": o, "monthly": monthly.to_dict("records")}
    with (DAY10 / f"run_summary{WIN_SUFFIX}.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    html = eng.build_html(tdf, daily, monthly, o, summary)
    win_report = DAY10 / f"trend_continuation_report{WIN_SUFFIX}.html"
    win_report.write_text(html, encoding="utf-8")

    # per-trade charts for the winner
    charts.OUT_DIR = DAY10
    charts.SUFFIX = WIN_SUFFIX
    charts.START_DATE, charts.END_DATE = START, END
    charts.CHART_DIR = DAY10 / f"charts{WIN_SUFFIX}"
    charts.JOURNAL = DAY10 / f"trade_journal{WIN_SUFFIX}.csv"
    charts.REPORT = win_report
    print("Generating per-trade charts for the winner...", flush=True)
    try:
        charts.main()
    except Exception as e:
        print(f"  (chart generation warning: {e})", flush=True)

    # ---- ablation comparison HTML ----
    ABL_HTML.write_text(build_ablation_html(results, base_net, winner), encoding="utf-8")

    print("\n" + "=" * 76)
    print("DONE.")
    print(f"  Baseline net: {_inr_signed(base_net)}  ({results[0]['trades']} trades)")
    print(f"  Winner: drop '{win_row['label']}' -> net {_inr_signed(win_row['net'])} "
          f"({win_row['trades']} trades, delta {_inr_signed(win_row['net']-base_net)})")
    print("  Ablation report  : " + str(ABL_HTML))
    print("  Winner full report: " + str(win_report))
    print("  Winner journal   : " + str(DAY10 / f"trade_journal{WIN_SUFFIX}.csv"))
    print("  Winner charts    : " + str(DAY10 / f"charts{WIN_SUFFIX}/"))
    print("=" * 76)

    webbrowser.open(ABL_HTML.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
