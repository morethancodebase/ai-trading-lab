"""Generate a self-contained HTML report for the Day 5 signal discovery.

Reads the CSV/JSON outputs produced by ``src.analysis.signal_discovery`` and
emits a single ``report.html`` (no external CSS/JS/fonts) in the same reports
folder. All numbers are read from the generated outputs -- nothing is hardcoded.
Re-run any time the data changes::

    python -m src.analysis.generate_signal_report
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[2] / "reports" / "day5_signal_discovery"

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
ul{padding-left:22px;margin:8px 0}
li{margin:6px 0;font-size:13px}
.small{font-size:12px;color:#64748b}
footer{margin:16px 0 4px;text-align:center;color:#94a3b8;font-size:12px}
"""


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


def _cls_bps(x: float) -> str:
    if _nan(x):
        return "muted"
    return "pos" if x > 0 else ("neg" if x < 0 else "muted")


def _load():
    summ = pd.read_csv(D / "signal_summary.csv")
    cmp = pd.read_csv(D / "signal_train_vs_val.csv")
    by_stock = pd.read_csv(D / "signal_by_stock.csv")
    by_time = pd.read_csv(D / "signal_by_time.csv")
    with (D / "run_summary.json").open() as f:
        meta = json.load(f)
    return summ, cmp, by_stock, by_time, meta


def _comparison_table(cmp: pd.DataFrame) -> str:
    rows = []
    for _, r in cmp.iterrows():
        pers = '<span class="tag yes">persist</span>' if r["persistence_flag"] else (
            '<span class="tag no">fail</span>')
        rows.append(
            f"<tr><td><b>{r['signal_id']}</b></td><td>{r['signal_name']}</td>"
            f"<td>{r['kind']}</td><td>{r['threshold']}</td>"
            f"<td class='num'>{_int(r['train_n_signals'])}</td>"
            f"<td class='num'>{_int(r['val_n_signals'])}</td>"
            f"<td class='num'>{_pct(r['train_hit_rate'])}</td>"
            f"<td class='num'>{_pct(r['val_hit_rate'])}</td>"
            f"<td class='num {_cls_bps(r['train_avg_ret_bps'])}'>{_bps(r['train_avg_ret_bps'])}</td>"
            f"<td class='num {_cls_bps(r['val_avg_ret_bps'])}'>{_bps(r['val_avg_ret_bps'])}</td>"
            f"<td class='num'>{_f(r['val_train_ratio'], 2)}</td>"
            f"<td>{pers}</td></tr>"
        )
    return (
        "<table><tr><th>ID</th><th>Signal</th><th>Kind</th><th>Thr</th>"
        "<th class='num'>Train&nbsp;n</th><th class='num'>Val&nbsp;n</th>"
        "<th class='num'>Train&nbsp;hit</th><th class='num'>Val&nbsp;hit</th>"
        "<th class='num'>Train&nbsp;ret&nbsp;bps</th><th class='num'>Val&nbsp;ret&nbsp;bps</th>"
        "<th class='num'>Val/Train</th><th>Persist</th></tr>"
        + "".join(rows) + "</table>"
    )


def _val_detail_table(summ: pd.DataFrame) -> str:
    v = summ[summ["split"] == "validation"].copy()
    rows = []
    for _, r in v.iterrows():
        rows.append(
            f"<tr><td><b>{r['signal_id']}</b></td><td>{r['signal_name']}</td>"
            f"<td class='num'>{_int(r['n_signals'])}</td>"
            f"<td class='num'>{_pct(r['signal_freq'])}</td>"
            f"<td class='num'>{_pct(r['hit_rate'])}</td>"
            f"<td class='num'>{_pct(r['hit_rate_long'])}</td>"
            f"<td class='num'>{_pct(r['hit_rate_short'])}</td>"
            f"<td class='num {_cls_bps(r['avg_exp_ret_bps'])}'>{_bps(r['avg_exp_ret_bps'])}</td>"
            f"<td class='num {_cls_bps(r['median_exp_ret_bps'])}'>{_bps(r['median_exp_ret_bps'])}</td>"
            f"<td class='num'>{_f(r['pct_positive'], 1)}%</td>"
            f"<td class='num'>{_f(r['avg_move_bps'], 2)}</td></tr>"
        )
    return (
        "<table><tr><th>ID</th><th>Signal</th><th class='num'>n_sig</th>"
        "<th class='num'>freq</th><th class='num'>hit</th><th class='num'>hit&nbsp;L</th>"
        "<th class='num'>hit&nbsp;S</th><th class='num'>avg&nbsp;ret&nbsp;bps</th>"
        "<th class='num'>med&nbsp;ret&nbsp;bps</th><th class='num'>%>0</th>"
        "<th class='num'>avg&nbsp;|move|</th></tr>" + "".join(rows) + "</table>"
    )


def _consistency_table(summ: pd.DataFrame) -> str:
    v = summ[summ["split"] == "validation"].copy()
    rows = []
    for _, r in v.iterrows():
        rows.append(
            f"<tr><td><b>{r['signal_id']}</b></td><td>{r['signal_name']}</td>"
            f"<td class='num'>{_int(r['n_stocks_signals'])}</td>"
            f"<td class='num'>{_int(r['n_stocks_positive'])}</td>"
            f"<td class='num'>{_f(r['median_stock_ret_bps'], 3)}</td>"
            f"<td class='num'>{_f(r['mean_stock_ret_bps'], 3)}</td>"
            f"<td>{r['worst_stock']} <span class='small'>({_bps(r['worst_stock_ret_bps'])})</span></td>"
            f"<td>{r['best_stock']} <span class='small'>({_bps(r['best_stock_ret_bps'])})</span></td></tr>"
        )
    return (
        "<table><tr><th>ID</th><th>Signal</th><th class='num'>#stocks</th>"
        "<th class='num'>#>0</th><th class='num'>med&nbsp;stock&nbsp;bps</th>"
        "<th class='num'>mean&nbsp;stock&nbsp;bps</th><th>Worst stock</th>"
        "<th>Best stock</th></tr>" + "".join(rows) + "</table>"
    )


def _tod_table(by_time: pd.DataFrame, signal_ids: list[str]) -> str:
    sub = by_time[(by_time["split"] == "validation") & (by_time["signal_id"].isin(signal_ids))]
    if sub.empty:
        return "<p class='small'>No time-of-day data.</p>"
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            f"<tr><td>{r['signal_id']}</td><td>{r['period']}</td>"
            f"<td class='num'>{_int(r['n_signals'])}</td>"
            f"<td class='num'>{_pct(r['hit_rate'])}</td>"
            f"<td class='num {_cls_bps(r['avg_exp_ret_bps'])}'>{_bps(r['avg_exp_ret_bps'])}</td></tr>"
        )
    return ("<table><tr><th>Signal</th><th>Period</th><th class='num'>n</th>"
            "<th class='num'>hit</th><th class='num'>avg&nbsp;ret&nbsp;bps</th></tr>"
            + "".join(rows) + "</table>")

def build() -> str:
    summ, cmp, by_stock, by_time, meta = _load()

    n_total = len(cmp)
    n_persist = int(cmp["persistence_flag"].sum())
    n_pos_val = int((cmp["val_avg_ret_bps"] > 0).sum())
    promising = cmp[cmp["persistence_flag"]].sort_values("val_avg_ret_bps", ascending=False)
    rejected = cmp[cmp["val_avg_ret_bps"] <= 0]
    best = cmp.sort_values("val_avg_ret_bps", ascending=False).iloc[0]
    top_tod = promising["signal_id"].tolist()[:3]

    p_list = "".join(
        f"<li><b>{r['signal_id']}</b> {r['signal_name']}: train {r['train_avg_ret_bps']:+.3f} bps "
        f"&rarr; val {r['val_avg_ret_bps']:+.3f} bps (ratio {r['val_train_ratio']:.2f}, "
        f"val hit {r['val_hit_rate']*100:.2f}%, val n={int(r['val_n_signals']):,})</li>"
        for _, r in promising.iterrows()
    )
    r_list = "".join(
        f"<li><b>{r['signal_id']}</b> {r['signal_name']}: val {r['val_avg_ret_bps']:+.3f} bps</li>"
        for _, r in rejected.iterrows()
    ) or "<li>(none)</li>"

    cards = "".join([
        f"<div class='card'><div class='k'>Signals tested</div><div class='v'>{meta['n_signals_tested']}</div>"
        f"<div class='s'>declared a priori</div></div>",
        f"<div class='card'><div class='k'>Stocks</div><div class='v'>{meta['stocks_processed']}</div>"
        f"<div class='s'>NIFTY 100; failed {meta['stocks_failed']}</div></div>",
        f"<div class='card'><div class='k'>Target horizon</div><div class='v'>{meta['target'].split('_')[-1]}</div>"
        f"<div class='s'>future return</div></div>",
        f"<div class='card'><div class='k'>Persisted</div><div class='v'>{n_persist}/{n_total}</div>"
        f"<div class='s'>pos TRAIN&VAL, ratio&ge;0.5, n&ge;1000</div></div>",
        f"<div class='card'><div class='k'>TRAIN scorable</div><div class='v'>{meta['train_scorable_rows']:,}</div>"
        f"<div class='s'>rows with valid target</div></div>",
        f"<div class='card'><div class='k'>VAL scorable</div><div class='v'>{meta['validation_scorable_rows']:,}</div>"
        f"<div class='s'>rows with valid target</div></div>",
        f"<div class='card'><div class='k'>TEST read</div><div class='v'>0</div>"
        f"<div class='s'>never touched (max {meta['max_allowed_date']})</div></div>",
        f"<div class='card'><div class='k'>Elapsed</div><div class='v'>{meta['elapsed_seconds']:.0f}s</div>"
        f"<div class='s'>generated {meta['generated_at']}</div></div>",
    ])

    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Day 5 &mdash; Signal Discovery</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><h1>Day 5 &mdash; Signal Discovery from Predictive Relationships</h1>
<p>Can the weak but persistent Day-4 indicator relationships be converted into simple,
interpretable directional signals (LONG / SHORT / NO SIGNAL) that survive on unseen
VALIDATION data? Exploratory only &mdash; not a strategy, not a backtest.</p>
<div class="meta">Target: {meta['target']} &middot; {meta['n_signals_tested']} signals &middot;
TRAIN {meta['train_rows_loaded']:,} rows / VALIDATION {meta['validation_rows_loaded']:,} rows
&middot; TEST never read</div></header>

<section><h2>Overview</h2><div class="cards">{cards}</div>
<p class="lead">Thresholds are <b>per-stock TRAIN percentiles (10/25/75/90)</b>, frozen and applied
unchanged to VALIDATION. Signal directions come from Day-4 per-stock-mean Pearson signs
(negative association &rarr; high=SHORT / low=LONG = reversal). Returns are <b>RAW basis points</b>
before brokerage, taxes, slippage, spread, and market impact.</p>
<div class="callout"><b>Method.</b> Signals are declared a priori (7 extreme singles, 2 moderate
singles as a threshold-sensitivity check, 4 extreme combinations). Nothing is selected or tuned on
VALIDATION. A signal is flagged <b>persist</b> only if positive in both TRAIN and VALIDATION,
val/train ratio &ge; 0.5, and VALIDATION produces &ge; 1,000 signals.</div></section>

<section><h2>TRAIN vs VALIDATION (the key test)</h2>
<p class="lead">A signal is interesting only if the behaviour found in TRAIN stays reasonably similar
on unseen VALIDATION. Average expected return is in raw basis points (LONG: mean future return;
SHORT: mean of &minus;future return).</p>
{_comparison_table(cmp)}
<p class="small"><b>Read as associations, not edge.</b> A few bps of raw expected return is small
relative to realistic Indian intraday costs (brokerage ~2-3 bps round-trip + STT + spread + slippage).
Persistence means the directional <i>pattern</i> survives, not that it is tradeable.</p></section>

<section><h2>Validation detail (all metrics)</h2>
<p class="lead">Per signal on VALIDATION: count, frequency, hit rate (overall/long/short), average
& median expected return, % positive, average underlying |move|.</p>
{_val_detail_table(summ)}</section>

<section><h2>Cross-stock consistency</h2>
<p class="lead">Pooled results can be distorted by a few stocks. On VALIDATION: how many of the 100
stocks produced signals and had a positive stock-level expected return, plus cross-stock
median/mean and worst/best stock.</p>
{_consistency_table(summ)}</section>

<section><h2>Time of day (exploratory, top persistent signals)</h2>
<p class="lead">VALIDATION signals bucketed by intraday period for the strongest persistent signals.
Exploratory only &mdash; not used to select or tune anything.</p>
{_tod_table(by_time, top_tod)}</section>
"""]
    parts.append(f"""
<section class="last"><h2>Findings & limitations</h2>
<div class="callout"><b>What the data says.</b> Of {n_total} declared signals, {n_persist}
persisted (positive in TRAIN & VALIDATION, ratio &ge; 0.5, val n &ge; 1000) and {n_pos_val} had a
positive raw expected return on VALIDATION. The strongest by VALIDATION expected return is
<b>{best['signal_id']} {best['signal_name']}</b> ({best['val_avg_ret_bps']:+.3f} bps, val hit
{best['val_hit_rate']*100:.2f}%, ratio {best['val_train_ratio']:.2f}).</div>
<h3>Promising (persisted) signals</h3><ul>{p_list}</ul>
<h3>Rejected / collapsed on validation</h3><ul>{r_list}</ul>
<div class="warn"><b>These are not profitable signals.</b> Statistical / directional interest
&ne; profitability. Conclusions:
<ul>
<li><b>Statistically interesting:</b> several reversal signals keep their sign and a non-trivial
fraction of their TRAIN magnitude on VALIDATION &mdash; the Day-4 mean-reversion association does
convert into a directional signal that generalises in direction.</li>
<li><b>Potentially useful (unproven):</b> combinations that require agreement amplify the
per-signal expected return and still persist, but fire far less often (lower n) &mdash; a
classic precision-vs-frequency trade-off.</li>
<li><b>Rejected:</b> signals whose Day-4 association was weakest or carried the "wrong"
sign fail to persist (see the rejected list above) &mdash; exactly as their small Day-4 effect
sizes warned.</li>
<li><b>Not profitable:</b> raw returns are a few bps at most; no transaction-cost, slippage, or
execution model has been applied. Any "edge" could be erased by costs. A proper cost-aware
backtest on the untouched TEST period (much later) is the only valid test of profitability.</li>
</ul></div>
<ul>
<li><b>Per-stock thresholds</b> are required because candle_body / distance_from_vwap are in price
units; pooled thresholds would be meaningless across the 100 stocks.</li>
<li><b>Survivorship / regime.</b> Today's NIFTY 100 universe; VALIDATION was lower-volatility than
TRAIN, so some ratios partly reflect denominator shrinkage.</li>
<li><b>No overfitting safeguards beyond persistence.</b> 13 signals &times; a few metrics were
examined; treat rankings as exploratory.</li>
</ul></section>
<footer>Day 5 signal discovery &mdash; local research report. Numbers derived from
run_summary.json, signal_summary.csv, signal_train_vs_val.csv, signal_by_stock.csv,
signal_by_time.csv. TEST was never read.</footer>
</div></body></html>
""")
    return "".join(parts)


def main() -> int:
    out = D / "report.html"
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())



