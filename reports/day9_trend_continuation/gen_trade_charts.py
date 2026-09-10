#!/usr/bin/env python3
"""Generate per-trade chart images (price + VWAP + entry/exit/SL) for every trade
in the August-2026 baseline journal and inject them into the HTML report.

Run from project root:
    .venv/bin/python reports/day9_trend_continuation/gen_trade_charts.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import time as dtime

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports" / "day9_trend_continuation"
DATA_DIR = ROOT / "data" / "processed"
SUFFIX = "_aug_base"                       # baseline (original rules) journal
CHART_DIR = OUT_DIR / f"charts{SUFFIX}"
JOURNAL = OUT_DIR / f"trade_journal{SUFFIX}.csv"
REPORT = OUT_DIR / f"trend_continuation_report{SUFFIX}.html"
START_DATE, END_DATE = "2026-08-01", "2026-08-31"
EXIT_BAR_OPEN = dtime(15, 10)
GREEN_C = "#26a69a"
RED_C = "#ef5350"


def load_5min(symbol: str, path: Path) -> pd.DataFrame | None:
    q = (
        "SELECT timestamp, open, high, low, close, volume, vwap "
        f"FROM read_parquet('{path.as_posix()}') "
        f"WHERE CAST(timestamp AS DATE) BETWEEN DATE '{START_DATE}' AND DATE '{END_DATE}' "
        "ORDER BY timestamp"
    )
    df = duckdb.execute(q).df()
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    agg = (
        df.resample("5min", on="timestamp", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"), vwap=("vwap", "last"))
        .dropna(subset=["open", "close"]).reset_index()
    )
    agg = agg[(agg["timestamp"].dt.time >= dtime(9, 15)) & (agg["timestamp"].dt.time <= dtime(15, 30))]
    if agg.empty:
        return None
    agg["date"] = agg["timestamp"].dt.date
    return agg.reset_index(drop=True)


def discover_stocks() -> dict[str, Path]:
    files = sorted(DATA_DIR.glob("*_1min_indicators.parquet"))
    return {p.name[: -len("_1min_indicators.parquet")]: p for p in files}


def _fmt_inr(x: float) -> str:
    s = "-" if x < 0 else ""
    return f"{s}₹{abs(x):,.0f}"


def draw_candles(ax, g):
    x = np.arange(len(g))
    op, hi, lo, cl = (g["open"].values, g["high"].values, g["low"].values, g["close"].values)
    for k in range(len(g)):
        color = GREEN_C if cl[k] >= op[k] else RED_C
        ax.plot([k, k], [lo[k], hi[k]], color=color, linewidth=0.7, zorder=2)
        body_lo, body_hi = min(op[k], cl[k]), max(op[k], cl[k])
        if body_hi - body_lo < 1e-9:
            body_hi = body_lo + 1e-6
        ax.plot([k, k], [body_lo, body_hi], color=color, linewidth=4.2,
                solid_capstyle="butt", zorder=3)

def chart_one(tr, g) -> str:
    """Render one trade's day chart; return the PNG filename."""
    sym = tr["symbol"]
    date = pd.to_datetime(tr["date"]).date()
    g = g[g["date"] == date].reset_index(drop=True)
    if g.empty:
        return ""
    g["t"] = g["timestamp"].dt.strftime("%H:%M")
    x = np.arange(len(g))
    s = 1 if tr["direction"] == "LONG" else -1

    setup_ts = pd.to_datetime(tr["setup_time"])
    entry_ts = pd.to_datetime(tr["entry_time"])
    exit_ts = pd.to_datetime(tr["exit_time"])
    gts = pd.to_datetime(g["timestamp"])
    i_setup = int(np.where(gts == setup_ts)[0][0]) if (gts == setup_ts).any() else None
    i_entry = int(np.where(gts == entry_ts)[0][0]) if (gts == entry_ts).any() else None
    i_exit = int(np.where(gts == exit_ts)[0][0]) if (gts == exit_ts).any() else None

    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=110)
    draw_candles(ax, g)
    ax.plot(x, g["vwap"].values, color="#ff8f00", linewidth=1.5, zorder=4, label="VWAP")

    ylo = float(g["low"].min()); yhi = float(g["high"].max())
    pad = (yhi - ylo) * 0.08
    ax.set_ylim(ylo - pad, yhi + pad)

    exit_bar = g[g["timestamp"].dt.time == EXIT_BAR_OPEN]
    if not exit_bar.empty:
        ax.axvline(exit_bar.index[0], color="#888", ls=":", lw=1, zorder=1)
        ax.text(exit_bar.index[0] + 0.3, yhi + pad * 0.9, "15:15 exit", fontsize=8, color="#666")

    entry_px = float(tr["entry_price"]); sl_px = float(tr["sl_initial"])
    risk = abs(entry_px - sl_px)
    ax.hlines(sl_px, 0, len(g) - 1, colors=RED_C, linestyles="--", linewidth=1.2, zorder=5)
    ax.text(len(g) - 1, sl_px, f"  SL {sl_px:.2f}", color=RED_C, fontsize=8, va="center")
    ax.hlines(entry_px, 0, len(g) - 1, colors="#1565c0", linestyles="--", linewidth=1.0, zorder=5)
    ax.text(len(g) - 1, entry_px, f"  Entry {entry_px:.2f}", color="#1565c0", fontsize=8, va="center")
    r2 = entry_px + s * 2.0 * risk
    ax.hlines(r2, 0, len(g) - 1, colors="#7e57c2", linestyles=":", linewidth=1.0, zorder=5)
    ax.text(0.3, r2, f"2R {r2:.2f}", color="#7e57c2", fontsize=8, va="bottom")

    if i_setup is not None:
        ax.axvspan(i_setup - 0.4, i_setup + 0.4, color="#fff59d", alpha=0.45, zorder=0)
        ax.text(i_setup, yhi + pad * 0.55, "setup", fontsize=7, color="#827717",
                ha="center", rotation=90)
    if i_entry is not None:
        mcol = "#2e7d32" if s == 1 else "#c62828"
        ax.scatter([i_entry], [entry_px], marker="^" if s == 1 else "v",
                   s=130, color=mcol, edgecolor="white", zorder=10)
        ax.annotate(f"BUY {entry_px:.2f}" if s == 1 else f"SELL {entry_px:.2f}",
                    (i_entry, entry_px), xytext=(8, 14), textcoords="offset points",
                    fontsize=8, color=mcol, fontweight="bold")
    if i_entry is not None and i_exit is not None:
        seg = g.iloc[i_entry:i_exit + 1]
        if not seg.empty:
            if s == 1:
                mk, mv = int(seg["high"].idxmax()), float(seg["high"].max())
            else:
                mk, mv = int(seg["low"].idxmin()), float(seg["low"].min())
            ax.scatter([mk], [mv], marker="*", s=120, color="#f9a825",
                       edgecolor="black", zorder=9)
            ax.annotate(f"MFE {mv:.2f} ({tr['mfe_r']:.1f}R)", (mk, mv),
                        xytext=(8, -16), textcoords="offset points", fontsize=7, color="#bf8c00")
    if i_exit is not None:
        exit_px = float(tr["exit_price"]); win = tr["net_pnl"] >= 0
        ecol = "#2e7d32" if win else "#c62828"
        ax.scatter([i_exit], [exit_px], marker="P" if win else "x",
                   s=120, color=ecol, linewidths=1.5, zorder=11)
        ax.annotate(f"EXIT {exit_px:.2f} ({tr['exit_reason']})",
                    (i_exit, exit_px), xytext=(8, -14), textcoords="offset points",
                    fontsize=8, color=ecol, fontweight="bold")
    step = max(1, len(g) // 14)
    ax.set_xticks(x[::step]); ax.set_xticklabels(g["t"].values[::step], rotation=45, fontsize=8)
    ax.set_xlim(-0.7, len(g) + 4.5); ax.grid(axis="y", ls=":", alpha=0.35)
    ax.set_ylabel("Price (Rs)", fontsize=9)
    leg = [Patch(facecolor=GREEN_C, label="up"), Patch(facecolor=RED_C, label="down"),
           plt.Line2D([], [], color="#ff8f00", label="VWAP"),
           plt.Line2D([], [], color="#1565c0", ls="--", label="entry"),
           plt.Line2D([], [], color=RED_C, ls="--", label="initial SL"),
           plt.Line2D([], [], color="#7e57c2", ls=":", label="2R trigger")]
    ax.legend(handles=leg, loc="upper left", fontsize=7, ncol=3, framealpha=0.9)
    title = (f"{sym}  {date}  {tr['direction']}  |  entry {entry_px:.2f}  "
             f"exit {float(tr['exit_price']):.2f}  |  R={tr['r_multiple']:.2f}  "
             f"MFE={tr['mfe_r']:.1f}R  qty={int(tr['qty'])}  |  {tr['exit_reason']}  "
             f"|  net {_fmt_inr(float(tr['net_pnl']))}")
    ax.set_title(title, fontsize=9.5, loc="left", fontfamily="monospace")
    fig.tight_layout()
    fname = f"trade_{int(tr['trade_no']):03d}_{sym}_{date}.png"
    fig.savefig(CHART_DIR / fname, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fname

def build_charts_section(items: list[tuple[dict, str]]) -> str:
    cards = []
    for tr, fn in items:
        if not fn:
            continue
        win = "win" if tr["net_pnl"] >= 0 else "loss"
        cls = "pos" if win == "win" else "neg"
        cards.append(
            f'<div class="chart-card {win}">'
            f'<img src="charts{SUFFIX}/{fn}" loading="lazy" alt="{tr["symbol"]} {tr["date"]}">'
            f'<div class="chart-cap"><b>#{int(tr["trade_no"])}</b> {tr["symbol"]} '
            f'{pd.to_datetime(tr["date"]).date()} {tr["direction"]} &bull; '
            f'R {tr["r_multiple"]:.2f} &bull; {tr["exit_reason"]} &bull; '
            f'<span class="{cls}">{_fmt_inr(float(tr["net_pnl"]))}</span></div></div>'
        )
    style = (
        "<style>#charts{max-width:1500px;margin:0 auto}"
        ".chart-grid{display:grid;grid-template-columns:1fr;gap:18px}"
        ".chart-card{border:1px solid #e0e0e0;border-radius:10px;overflow:hidden;background:#fff;"
        "box-shadow:0 1px 4px rgba(0,0,0,.08)}"
        ".chart-card.win{border-left:5px solid #43a047}.chart-card.loss{border-left:5px solid #e53935}"
        ".chart-card img{width:100%;height:auto;display:block}"
        ".chart-cap{padding:7px 12px;font-size:12px;background:#fafafa;font-family:monospace}"
        ".pos{color:#2e7d32}.neg{color:#c62828}</style>"
    )
    return (
        style
        + f'<section id="charts"><h2>Trade Charts -- every August 2026 trade '
        f'(price + VWAP + entry/exit/SL)</h2>'
        f'<p class="small" style="margin-bottom:14px">{len(cards)} trades. '
        f'Yellow band = setup (bounce) candle. Triangle = entry. Star = MFE. '
        f'Cross/plus = exit. Dashed blue = entry, dashed red = initial stop, '
        f'dotted purple = 2R (trail-trigger). Orange = session VWAP.</p>'
        f'<div class="chart-grid">{"".join(cards)}</div></section>'
    )


def inject_into_report(html: str, section: str) -> str:
    foot = '<p class="small" style="margin-top:30px">Generated by'
    # add a Charts link to the nav, then insert the section before the footer
    if '<a href="#notes">Methodology</a></nav>' in html and 'href="#charts"' not in html:
        html = html.replace(
            '<a href="#notes">Methodology</a></nav>',
            '<a href="#charts">Trade Charts</a><a href="#notes">Methodology</a></nav>')
    if foot in html:
        return html.replace(foot, section + foot)
    return html.replace("</body>", section + "</body>")


def main() -> int:
    if not JOURNAL.exists():
        print(f"ERROR: {JOURNAL} not found -- run run_backtest_v2.py first.")
        return 1
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    tdf = pd.read_csv(JOURNAL).sort_values("trade_no").reset_index(drop=True)
    print(f"Loaded {len(tdf)} trades from {JOURNAL.name}")
    stocks = discover_stocks()
    cache: dict[str, pd.DataFrame] = {}
    items: list[tuple[dict, str]] = []
    for k, tr in enumerate(tdf.to_dict("records"), 1):
        sym = tr["symbol"]
        if sym not in cache:
            if sym not in stocks:
                items.append((tr, "")); continue
            cache[sym] = load_5min(sym, stocks[sym])
        g = cache[sym]
        fn = chart_one(tr, g) if g is not None else ""
        items.append((tr, fn))
        if k % 10 == 0 or k == len(tdf):
            print(f"  charted {k}/{len(tdf)}")
    n_ok = sum(1 for _, fn in items if fn)
    print(f"Generated {n_ok} chart PNGs in {CHART_DIR.name}/")
    section = build_charts_section(items)
    if REPORT.exists():
        REPORT.write_text(inject_into_report(REPORT.read_text(encoding="utf-8"), section),
                          encoding="utf-8")
    else:
        REPORT.write_text(f"<!doctype html><html><body>{section}</body></html>",
                          encoding="utf-8")
    print(f"Updated report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

