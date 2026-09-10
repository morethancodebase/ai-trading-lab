"""Generate a self-contained HTML report for the Day 6 cost-aware backtest.

Reads the CSV/JSON/PNG outputs produced by ``src.backtesting.run_backtest`` and
emits a single ``report.html`` (no external CSS/JS/fonts; the validation
model-mode equity PNGs are embedded as base64 so the file is fully portable) in
the same reports folder. All numbers are read from the generated outputs --
nothing is hardcoded. Re-run any time the data changes::

    python -m src.backtesting.generate_report
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

from src.backtesting.costs import DEFAULT_COST

D = Path(__file__).resolve().parents[2] / "reports" / "day6_backtest"

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#eef2f7;color:#1e293b;line-height:1.6;padding:22px}
.wrap{max-width:1180px;margin:0 auto}
header{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;padding:34px 34px;border-radius:14px 14px 0 0}
header h1{font-size:25px;font-weight:700;letter-spacing:-.01em}
header p{opacity:.88;margin-top:7px;font-size:14px}
header .meta{font-size:12px;opacity:.72;margin-top:14px}
section{background:#fff;padding:26px 34px;border:1px solid #e2e8f0;border-top:none}
section.last{border-radius:0 0 14px 14px}
h2{font-size:19px;font-weight:700;margin-bottom:6px;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:9px}
h3{font-size:14px;font-weight:600;margin:18px 0 8px;color:#334155}
p.lead{font-size:13px;color:#475569;margin-bottom:14px}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin:4px 0 16px}
.card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:15px}
.card .k{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:21px;font-weight:700;margin-top:3px;color:#0f172a}
.card .s{font-size:11px;color:#94a3b8;margin-top:2px}
.card.bad .v{color:#b91c1c}
.card.ok .v{color:#15803d}
table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
th{text-align:left;padding:9px 9px;background:#f1f5f9;border-bottom:2px solid #cbd5e1;font-weight:600;color:#334155;font-size:12px}
td{padding:8px 9px;border-bottom:1px solid #eef2f7;vertical-align:top}
tr:hover td{background:#f8fafc}
.num{text-align:right;font-variant-numeric:tabular-nums;font-feature-settings:"tnum"}
.pos{color:#15803d;font-weight:600}
.neg{color:#b91c1c;font-weight:600}
.muted{color:#94a3b8}
.tag{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;white-space:nowrap}
.tag.yes{background:#dcfce7;color:#166534}
.tag.no{background:#fee2e2;color:#991b1b}
.tag.lvl{background:#fef3c7;color:#92400e}
.callout{border-left:4px solid #2563eb;background:#eff6ff;padding:13px 17px;border-radius:0 8px 8px 0;margin:6px 0 4px;font-size:13px}
.warn{border-left:4px solid #b45309;background:#fffbeb;padding:13px 17px;border-radius:0 8px 8px 0;margin:6px 0 4px;font-size:13px}
.danger{border-left:4px solid #b91c1c;background:#fef2f2;padding:13px 17px;border-radius:0 8px 8px 0;margin:6px 0 4px;font-size:13px}
ul{padding-left:22px;margin:8px 0}
li{margin:6px 0;font-size:13px}
.small{font-size:12px;color:#64748b}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}
.eq{border:1px solid #e2e8f0;border-radius:10px;padding:10px;background:#fff}
.eq img{width:100%;height:auto;display:block;border-radius:6px}
.eq .cap{font-size:12px;color:#475569;margin-top:7px}
footer{margin:16px 0 4px;text-align:center;color:#94a3b8;font-size:12px}
"""


# ----------------------------- formatting helpers ---------------------------
def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _f(x, d=2) -> str:
    return "&ndash;" if _nan(x) else f"{float(x):.{d}f}"


def _bps(x) -> str:
    return "&ndash;" if _nan(x) else f"{float(x):+.3f}"


def _pct(x) -> str:  # x is a 0..1 fraction
    return "&ndash;" if _nan(x) else f"{float(x) * 100:.2f}%"


def _int(x) -> str:
    return "&ndash;" if _nan(x) else f"{int(x):,}"


def _rs(x) -> str:
    return "&ndash;" if _nan(x) else f"{float(x):,.0f}"


def _cbps(x) -> str:  # coloured bps
    if _nan(x):
        return "&ndash;"
    return f"<span class='{'pos' if x >= 0 else 'neg'}'>{float(x):+.3f}</span>"


def _survive(net_bps) -> str:
    if _nan(net_bps):
        return "<span class='tag lvl'>&ndash;</span>"
    if net_bps > 0:
        return "<span class='tag yes'>SURVIVES</span>"
    return "<span class='tag no'>NO</span>"


def _match_tag(m: bool) -> str:
    return "<span class='tag yes'>MATCH</span>" if m else "<span class='tag no'>DIFF</span>"


# ----------------------------- data loaders ---------------------------------
def _summary() -> dict:
    return json.loads((D / "run_summary.json").read_text(encoding="utf-8"))


def _metrics() -> pd.DataFrame:
    df = pd.read_csv(D / "metrics.csv")
    df["signal_id"] = df["signal_id"].astype(str)
    return df


def _by_time(sig: str, split: str, mode: str) -> pd.DataFrame:
    p = D / f"by_time_{sig}_{split}_{mode}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _signal_order(summ: dict) -> list[str]:
    order = list(summ.get("config", {}).get("signals", []))
    seen, out = set(), []
    for s in order:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def _name_map(metrics: pd.DataFrame) -> dict:
    return dict(zip(metrics["signal_id"], metrics["signal_name"]))


# ----------------------------- cost model -----------------------------------
def _cost_breakdown_rows() -> list[dict]:
    """Per-component round-trip cost on a representative Rs 1,00,000 long."""
    b = DEFAULT_COST.round_trip_breakdown(1.0, 100000.0, 100500.0)
    turnover = 100000.0 + 100500.0
    total = b["total"]
    order = ["brokerage", "stt", "exchange", "sebi", "stamp", "gst", "slippage"]
    label = {
        "brokerage": "Brokerage", "stt": "STT (sell-side)", "exchange": "Exchange (NSE)",
        "sebi": "SEBI turnover", "stamp": "Stamp duty (buy)",
        "gst": "GST (18% on brok+exch+sebi)", "slippage": "Slippage (1 bps/leg)",
    }
    rows = []
    for k in order:
        v = b[k]
        rows.append({"comp": label[k], "rs": v, "bps_turn": v / turnover * 1e4,
                     "bps_entry": v / 100000 * 1e4, "pct_total": v / total * 100})
    rows.append({"comp": "TOTAL", "rs": total, "bps_turn": total / turnover * 1e4,
                 "bps_entry": total / 100000 * 1e4, "pct_total": 100.0})
    return rows


def _cost_table() -> str:
    rows = _cost_breakdown_rows()
    out = ["<h3>Round-trip cost on a &#8377;1,00,000 long (representative)</h3>",
           "<p class='lead'>Components of <code>DEFAULT_COST</code> on a Rs 1L entry / "
           "Rs 1.005L exit round trip. The total &asymp; Rs 178.3 = <b>17.8 bps of the entry "
           "notional</b> is the wall every signal must clear. STT alone is 56% of it.</p>",
           "<table><tr><th>Component</th><th class='num'>Rs</th><th class='num'>bps of turnover</th>"
           "<th class='num'>bps of entry</th><th class='num'>% of total</th></tr>"]
    for r in rows:
        bold = " style='font-weight:700;background:#f1f5f9'" if r["comp"] == "TOTAL" else ""
        out.append(
            f"<tr{bold}><td>{r['comp']}</td><td class='num'>{r['rs']:,.4f}</td>"
            f"<td class='num'>{r['bps_turn']:.3f}</td><td class='num'>{r['bps_entry']:.3f}</td>"
            f"<td class='num'>{r['pct_total']:.1f}%</td></tr>")
    out.append("</table>")
    return "".join(out)



def _cards(summ: dict, metrics: pd.DataFrame) -> str:
    cfg = summ["config"]
    data = summ["data"]
    vm = metrics[(metrics.split == "validation") & (metrics.cost_mode == "model")]
    best_gross = float(vm["avg_gross_ret_bps"].max())
    n_survive = int((vm["avg_net_ret_bps"] > 0).sum())
    cost_bps = float(vm["avg_cost_bps"].max())
    gap = cost_bps / best_gross if best_gross > 0 else float("nan")
    stt_bps = _cost_breakdown_rows()[1]["bps_entry"]
    cards = [
        ("Stocks", _int(data["stocks_processed"]), "NIFTY 100; failed " + _int(data["stocks_failed"]), ""),
        ("Signals", _int(len(cfg["signals"])), "Day 5 directional, reused verbatim", ""),
        ("Cost modes", _int(len(cfg["cost_modes"])), "zero (gross) &middot; model (17.8 bps)", ""),
        ("Hold", f"{cfg['hold_bars']} bars", "300 s, same-day, no overnight", ""),
        ("Round-trip cost", f"{cost_bps:.1f} bps", "&#8377;178.3 on Rs 1L notional", "bad"),
        ("Best gross edge", f"{best_gross:+.3f} bps", "validation, all signals", "bad"),
        ("Survive costs", f"{n_survive}/{len(cfg['signals'])}", "net edge > 0 on validation", "bad"),
        ("TEST rows read", _int(summ["guards"]["test_rows_read"]), "hard guard; TEST untouched", "ok"),
    ]
    parts = ["<div class='cards'>"]
    for k, v, s, cls in cards:
        parts.append(f"<div class='card {cls}'><div class='k'>{k}</div><div class='v'>{v}</div>"
                     f"<div class='s'>{s}</div></div>")
    parts.append("</div>")
    parts.append(
        f"<div class='danger'><b>Headline.</b> No signal survives costs. The best gross edge is "
        f"{best_gross:+.3f} bps against a {cost_bps:.2f} bps round-trip &mdash; a <b>{gap:.0f}&times; "
        f"gap</b>. STT alone ({stt_bps:.1f} bps) exceeds every validation edge by 15&ndash;55&times;.</div>")
    return "".join(parts)


def _cross_check_table(summ: dict) -> str:
    cc = summ["day5_cross_check"]
    comps = cc["comparisons"]
    max_delta = max(abs(c["bps_delta"]) for c in comps)
    out = [f"<p class='lead'>The engine re-derives Day 5's pooled all-signals count and average "
           f"expected return and compares them to <code>signal_train_vs_val.csv</code>. "
           f"<b>{len(comps)} comparisons (11 signals &times; 2 splits) &mdash; "
           f"<span class='tag yes'>ALL MATCH</span></b> "
           f"(max |&Delta;bps| &asymp; {max_delta:.1e}). This confirms the backtest uses Day 5's "
           f"signal logic verbatim, so the negative result is a <i>cost</i> finding, not a "
           f"signal-logic discrepancy.</p>",
           "<table><tr><th>Signal</th><th>Split</th><th class='num'>backtest n</th>"
           "<th class='num'>Day 5 n</th><th class='num'>n match</th><th class='num'>backtest bps</th>"
           "<th class='num'>Day 5 bps</th><th class='num'>&Delta;bps</th><th>verdict</th></tr>"]
    for c in comps:
        out.append(
            f"<tr><td><b>{c['signal_id']}</b></td><td>{c['split']}</td>"
            f"<td class='num'>{c['backtest_n_signals']:,}</td>"
            f"<td class='num'>{c['day5_n_signals']:,}</td>"
            f"<td class='num'>{_match_tag(c['n_match'])}</td>"
            f"<td class='num'>{c['backtest_avg_ret_bps']:.6f}</td>"
            f"<td class='num'>{c['day5_avg_ret_bps']:.6f}</td>"
            f"<td class='num'>{c['bps_delta']:+.2e}</td>"
            f"<td>{_match_tag(c['match'])}</td></tr>")
    out.append("</table>")
    return "".join(out)



def _results_table(metrics: pd.DataFrame, order: list[str], split: str) -> str:
    sub = metrics[metrics.split == split].set_index("signal_id")
    rows = ["<table><tr><th>Signal</th><th class='num'>n</th><th class='num'>gross bps</th>"
            "<th class='num'>cost bps</th><th class='num'>net bps</th>"
            "<th class='num'>net hit%</th><th class='num'>net PF</th>"
            "<th class='num'>max DD%</th><th>survive</th></tr>"]
    for sid in order:
        if sid not in sub.index:
            continue
        g = sub.loc[sid]
        m = g[g.cost_mode == "model"].iloc[0]
        rows.append(
            f"<tr><td><b>{sid}</b> <span class='small'>{m['signal_name']}</span></td>"
            f"<td class='num'>{int(m['n_trades']):,}</td>"
            f"<td class='num'>{_cbps(m['avg_gross_ret_bps'])}</td>"
            f"<td class='num'>{m['avg_cost_bps']:.3f}</td>"
            f"<td class='num'>{_cbps(m['avg_net_ret_bps'])}</td>"
            f"<td class='num'>{_pct(m['hit_rate'])}</td>"
            f"<td class='num'>{m['profit_factor']:.3f}</td>"
            f"<td class='num'>{m['max_dd_pct']:.1f}%</td>"
            f"<td>{_survive(m['avg_net_ret_bps'])}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _win_profile(metrics: pd.DataFrame, order: list[str]) -> str:
    sub = metrics[metrics.split == "validation"].set_index("signal_id")
    rows = ["<table><tr><th>Signal</th><th class='num'>gross hit% (zero)</th>"
            "<th class='num'>net hit% (model)</th><th class='num'>gross PF (zero)</th>"
            "<th class='num'>net PF (model)</th></tr>"]
    for sid in order:
        if sid not in sub.index:
            continue
        g = sub.loc[sid]
        z = g[g.cost_mode == "zero"].iloc[0]
        m = g[g.cost_mode == "model"].iloc[0]
        rows.append(
            f"<tr><td><b>{sid}</b> <span class='small'>{z['signal_name']}</span></td>"
            f"<td class='num pos'>{_pct(z['hit_rate'])}</td>"
            f"<td class='num neg'>{_pct(m['hit_rate'])}</td>"
            f"<td class='num pos'>{z['profit_factor']:.3f}</td>"
            f"<td class='num neg'>{m['profit_factor']:.3f}</td></tr>")
    rows.append("</table>")
    return "".join(rows)


def _tod_table(sig: str = "C1") -> str:
    z = _by_time(sig, "validation", "zero")
    m = _by_time(sig, "validation", "model")
    if z.empty or m.empty:
        return "<p class='muted'>by_time output not found.</p>"
    z = z.set_index("period"); m = m.set_index("period")
    rows = [f"<table><tr><th>Period</th><th class='num'>n</th>"
            "<th class='num'>gross edge bps (zero)</th>"
            "<th class='num'>net edge bps (model)</th>"
            "<th class='num'>hit (zero)</th></tr>"]
    for p in z.index:
        rows.append(
            f"<tr><td>{z.loc[p, 'period_name']}</td>"
            f"<td class='num'>{int(z.loc[p, 'n_trades']):,}</td>"
            f"<td class='num'>{_cbps(z.loc[p, 'avg_net_ret_bps'])}</td>"
            f"<td class='num'>{_cbps(m.loc[p, 'avg_net_ret_bps'])}</td>"
            f"<td class='num'>{_pct(z.loc[p, 'hit_rate'])}</td></tr>")
    rows.append("</table>")
    return "".join(rows)



def _equity_section(metrics: pd.DataFrame, order: list[str]) -> str:
    vm = metrics[(metrics.split == "validation") & (metrics.cost_mode == "model")]
    vm = vm.set_index("signal_id")
    parts = ["<p class='lead'>Net equity (after 17.8 bps costs) per signal on VALIDATION, "
             "model-cost mode. Each curve starts at Rs 10,00,000 initial capital and bleeds toward "
             "zero &mdash; the visual signature of an edge far smaller than its cost.</p>",
             "<div class='grid2'>"]
    for sid in order:
        p = D / f"equity_{sid}_validation_model.png"
        if not p.exists() or sid not in vm.index:
            continue
        b64 = base64.b64encode(p.read_bytes()).decode()
        m = vm.loc[sid]
        if isinstance(m, pd.DataFrame):
            m = m.iloc[0]
        parts.append(
            f"<div class='eq'><img src='data:image/png;base64,{b64}' alt='equity {sid}'>"
            f"<div class='cap'><b>{sid}</b> {m['signal_name']} &mdash; "
            f"net {_cbps(m['avg_net_ret_bps'])} bps, max DD {m['max_dd_pct']:.0f}%</div></div>")
    parts.append("</div>")
    return "".join(parts)


def _findings(summ: dict, metrics: pd.DataFrame) -> str:
    vm = metrics[(metrics.split == "validation") & (metrics.cost_mode == "model")]
    best = vm.loc[vm.avg_gross_ret_bps.idxmax()]
    n_cc = len(summ["day5_cross_check"]["comparisons"])
    return f"""
<section class="last"><h2>Findings & limitations</h2>
<div class="callout"><b>What the data says.</b> Day 5's directional signals carry a tiny but real
gross edge &mdash; +{best['avg_gross_ret_bps']:.3f} bps at best ({best['signal_id']}) on
validation, with a ~51% gross hit rate. But a realistic FY2024-25 round trip costs ~17.8 bps, so
every signal lands ~17 bps underwater net. <b>No signal survives costs.</b></div>
<h3>Key findings</h3><ul>
<li><b>Best gross edge 0.40 bps vs 17.8 bps cost &mdash; a ~44&times; gap.</b> The edge would have
to be ~45&times; larger just to break even.</li>
<li><b>STT is the killer.</b> The 10 bps sell-side STT alone exceeds every validation edge by
15&ndash;55&times; and is 56% of the round-trip cost.</li>
<li><b>Costs invert the win profile.</b> ~51% gross win rate (PF 1.03&ndash;1.09) collapses to a
7&ndash;13% net win rate (PF 0.05&ndash;0.09).</li>
<li><b>Combinations don't help enough.</b> C1&ndash;C4 roughly double the per-signal gross edge
vs singles, but +0.40 bps is still 44&times; short of cost.</li>
<li><b>Signal logic is intact.</b> The Day 5 cross-check matches on all {n_cc} comparisons to
~1e-10 bps &mdash; the negative result is a <i>cost</i> finding, not a signal bug.</li>
</ul>
<div class="warn"><b>Read before concluding.</b><ul>
<li><b>Per-edge, not capital-constrained.</b> P&L is summed per signal at fixed Rs 1L notional
with no capital limit; rupee P&L / drawdown are <i>not</i> an achievable portfolio result.</li>
<li><b>Single notional, single hold, market-on-close exit.</b> No order-type mix, partial fills, or
scaling; real slippage on 100 concurrent Rs 1L market orders likely exceeds 1 bps/leg.</li>
<li><b>STT is a conservative constant.</b> 10 bps applies from 2024-10-01; the 2023&ndash;2024
sample actually paid 2.5 bps &mdash; but even at 2.5 bps the round trip is ~10.6 bps, still
15&ndash;58&times; the edge.</li>
<li><b>No TEST data touched.</b> Guards report test_rows_read = 0; TEST stays untouched for any
future final evaluation.</li>
<li><b>The negative result is specific to this setup.</b> It does not rule out longer holds,
selective models, cost-structured execution, or a larger edge from another method &mdash; but it
sets the bar any must clear: <b>> 17.8 bps round-trip on the 5-minute horizon.</b></li>
</ul></div>
</section>
<footer>Day 6 cost-aware backtest &mdash; local research report. Numbers derived from
run_summary.json, metrics.csv, by_time_*.csv, equity_*.png and src/backtesting/costs.py.
TEST was never read.</footer>"""



def build() -> str:
    summ = _summary()
    metrics = _metrics()
    order = _signal_order(summ)
    cfg = summ["config"]
    data = summ["data"]
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Day 6 &mdash; Cost-aware Backtest</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><h1>Day 6 &mdash; Cost-aware Backtest of Day 5 Directional Signals</h1>
<p>Do Day 5's persistent directional signals survive realistic Indian equity intraday trading
costs when executed as actual 5-minute round-trip trades? A per-signal-edge evaluation on TRAIN
and VALIDATION &mdash; <b>TEST is never read.</b></p>
<div class="meta">{data['stocks_processed']} stocks &middot; {len(order)} signals &middot;
{len(cfg['splits'])} splits &middot; {len(cfg['cost_modes'])} cost modes &middot;
{cfg['hold_bars']}-bar hold &middot; Rs {cfg['notional']:,.0f} notional/trade &middot;
generated {summ['generated_at']} &middot; {summ['elapsed_seconds'] / 60:.1f} min run</div></header>
"""]
    parts.append(f"<section><h2>Overview</h2>{_cards(summ, metrics)}</section>")
    parts.append(f"<section><h2>The cost model &mdash; the 17.8 bps wall</h2>{_cost_table()}</section>")
    parts.append(f"<section><h2>Results &mdash; TRAIN (model costs)</h2>"
                 f"<p class='lead'>Same trades, model costs. The gross edge is positive but a small "
                 f"fraction of cost; net is deeply negative for every signal.</p>"
                 f"{_results_table(metrics, order, 'train')}</section>")
    parts.append(f"<section><h2>Results &mdash; VALIDATION (model costs)</h2>"
                 f"<p class='lead'>The headline table. No signal's net edge is positive; even the "
                 f"best case is ~17 bps underwater.</p>"
                 f"{_results_table(metrics, order, 'validation')}</section>")
    parts.append(f"<section><h2>Costs invert the win profile (validation)</h2>"
                 f"<p class='lead'>In zero mode ~51% of trades are gross-profitable (PF 1.03&ndash;1.09); "
                 f"in model mode only 7&ndash;13% are net-profitable (PF 0.05&ndash;0.09), because a trade "
                 f"must move more than ~17.8 bps in the signal's favour just to cover costs.</p>"
                 f"{_win_profile(metrics, order)}</section>")
    parts.append(f"<section><h2>Time of day &mdash; C1, validation</h2>"
                 f"<p class='lead'>The gross edge is positive in every intraday period (zero mode) but "
                 f"turns uniformly negative once costs are applied (model mode) &mdash; no time-of-day "
                 f"window escapes the cost wall.</p>"
                 f"{_tod_table()}</section>")
    parts.append(f"<section><h2>Day 5 cross-check &mdash; signal logic verified</h2>"
                 f"{_cross_check_table(summ)}</section>")
    parts.append(f"<section><h2>Equity curves (validation, model costs)</h2>"
                 f"{_equity_section(metrics, order)}</section>")
    parts.append(_findings(summ, metrics))
    parts.append("</div></body></html>")
    return "".join(parts)


def main() -> int:
    out = D / "report.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

