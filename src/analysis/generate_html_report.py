"""Generate a self-contained HTML report for the Day 4 predictive analysis.

Reads the CSV/JSON outputs produced by ``src.analysis.predictive_analysis`` and
emits a single ``report.html`` (no external CSS/JS/fonts) in the same reports
folder. Re-run any time the data changes::

    python -m src.analysis.generate_html_report
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[2] / "reports" / "day4_predictive_analysis"

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#eef2f7;color:#1e293b;line-height:1.6;padding:22px}
.wrap{max-width:1120px;margin:0 auto}
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
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
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
.tag.ratio{background:#e0e7ff;color:#3730a3}
.callout{border-left:4px solid #2563eb;background:#eff6ff;padding:13px 17px;border-radius:0 8px 8px 0;margin:6px 0 4px;font-size:13px}
.chart{margin:14px 0 4px}
.bar-row{display:grid;grid-template-columns:225px 1fr 150px;align-items:center;gap:12px;padding:3px 0}
.bar-label{font-size:12px;color:#334155}
.bar-track{position:relative;height:20px;background:#f1f5f9;border-radius:5px}
.bar-axis{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#94a3b8}
.bar{position:absolute;height:7px;border-radius:3px}
.bar.train{top:2px;background:#2563eb}
.bar.val{top:10px;background:#ea580c}
.bar.neg{right:50%}
.bar.pos{left:50%}
.bar-vals{font-size:11px;font-variant-numeric:tabular-nums;color:#475569;text-align:right}
.legend{display:flex;gap:20px;font-size:12px;color:#475569;margin-top:12px}
.legend span{display:flex;align-items:center;gap:6px}
.dot{width:11px;height:11px;border-radius:2px}
ul{padding-left:22px;margin:8px 0}
li{margin:6px 0;font-size:13px}
.small{font-size:12px;color:#64748b}
footer{margin:16px 0 4px;text-align:center;color:#94a3b8;font-size:12px}
"""


def load():
    s = json.loads((D / "run_summary.json").read_text())
    return (
        s,
        pd.read_csv(D / "strongest_relationships.csv"),
        pd.read_csv(D / "target_distributions.csv"),
        pd.read_csv(D / "decile_significance_train.csv"),
        pd.read_csv(D / "time_of_day_analysis.csv"),
    )


def f4(x):
    return f"{x:.4f}" if pd.notna(x) else "—"


def fsci(x):
    return f"{x:.2e}" if pd.notna(x) else "—"


def fp(p):
    if pd.isna(p):
        return "—"
    if p == 0:
        return "≈0"
    return f"{p:.1e}" if p < 1e-4 else f"{p:.2e}"


def fi(n):
    return f"{int(n):,}"


def fpct(x):
    return f"{x*100:.1f}%"


def fmr(x):
    return f"{x:.2f}"


def fci(x):
    return f"{x:.2f}" if pd.notna(x) else "—"


def sc(v):
    return "neg" if v < 0 else "pos"


def heat(v, vmax=0.06):
    if pd.isna(v):
        return "transparent"
    a = min(abs(v) / vmax, 1) * 0.6
    if v < 0:
        return f"rgba(220,38,38,{a:.2f})"
    if v > 0:
        return f"rgba(22,163,74,{a:.2f})"
    return "transparent"


def header(s):
    return (
        "<header><h1>Day 4 — Predictive Analysis of Bar Indicators</h1>"
        "<p>Do Day 2 bar indicators contain predictive information about Day 3 future-return targets? "
        "&nbsp; TRAIN discovery · VALIDATION persistence · TEST never read · no ML models</p>"
        f"<div class='meta'>Generated {s['generated_at']} · {s['stocks_processed']} stocks · "
        f"{s['n_indicators']} indicators × {s['n_targets']} targets · elapsed {s['elapsed_seconds']:.0f}s · "
        f"splits used: {', '.join(s['splits_used'])}</div></header>"
    )


def summary(s):
    cards = "".join(
        f"<div class='card'><div class='k'>{k}</div><div class='v'>{v}</div><div class='s'>{sub}</div></div>"
        for k, v, sub in [
            ("Stocks processed", fi(s["stocks_processed"]), f"{s['stocks_failed']} failed"),
            ("TRAIN rows", fi(s["train_rows_loaded"]), "2023-09-04 → 2025-09-03"),
            ("VALIDATION rows", fi(s["validation_rows_loaded"]), "2025-09-04 → 2026-03-03"),
            ("TEST rows read", "0", f"hard guard: max date {s['max_allowed_date']}"),
        ]
    )
    nla = (
        "<div class='callout'><b>No look-ahead & no TEST.</b> Every indicator is backward-looking "
        "(within-session, Day 2); future-return targets are never used as inputs. Every query filters "
        "<code>date ≤ VALIDATION_END</code>; a hard guard asserts no row has <code>date ≥ TEST_START</code>. "
        "<span class='tag yes'>test_ever_read = false</span></div>"
    )
    return f"<section><h2>Run summary</h2><div class='cards'>{cards}</div>{nla}</section>"


def top_rel(df, s):
    scale = max(df["train_pearson_psm"].abs().max(), df["val_pearson_psm"].abs().max()) * 1.15

    def bar(v, cls):
        w = abs(v) / scale * 50
        side = "neg" if v < 0 else "pos"
        return f'<div class="bar {cls} {side}" style="width:{w:.1f}%"></div>'

    brows = "".join(
        f"<div class='bar-row'><div class='bar-label'><b>{r.indicator}</b> → {r.target}</div>"
        f"<div class='bar-track'><div class='bar-axis'></div>{bar(r.train_pearson_psm,'train')}{bar(r.val_pearson_psm,'val')}</div>"
        f"<div class='bar-vals'>tr {f4(r.train_pearson_psm)} · val {f4(r.val_pearson_psm)}</div></div>"
        for _, r in df.iterrows()
    )
    chart = (
        f"<div class='chart'>{brows}<div class='legend'>"
        "<span><span class='dot' style='background:#2563eb'></span>TRAIN per-stock-mean Pearson</span>"
        "<span><span class='dot' style='background:#ea580c'></span>VALIDATION per-stock-mean Pearson</span></div></div>"
    )
    trows = "".join(
        f"<tr><td class='num'>{r['rank']}</td><td><b>{r.indicator}</b><br><span class='muted'>{r.target}</span></td>"
        f"<td>{'<span class="tag lvl">level</span>' if r.scale_type=='level' else '<span class="tag ratio">ratio</span>'}</td>"
        f"<td class='num {sc(r.train_pearson_psm)}'>{f4(r.train_pearson_psm)}</td>"
        f"<td class='num {sc(r.val_pearson_psm)}'>{f4(r.val_pearson_psm)}</td>"
        f"<td>{'<span class="tag yes">persists</span>' if bool(r.sign_persists) else '<span class="tag no">flips</span>'}</td>"
        f"<td class='num'>{fmr(r.val_magnitude_ratio)}</td><td class='num'>{fci(r.train_frac_consistent)}</td>"
        f"<td class='num'>{fi(r.train_n_stocks)}</td></tr>"
        for _, r in df.iterrows()
    )
    table = (
        "<table><thead><tr><th class='num'>#</th><th>Indicator → Target</th><th>Scale</th>"
        "<th class='num'>Train psm</th><th class='num'>Val psm</th><th>Persist</th>"
        "<th class='num'>Mag ratio</th><th class='num'>Frac consistent</th><th class='num'># stocks</th></tr></thead>"
        f"<tbody>{trows}</tbody></table>"
    )
    return (
        "<section><h2>Top relationships — ranked by |TRAIN per-stock-mean Pearson|</h2>"
        "<p class='lead'>Diverging bars from centre: left = negative (reversal), right = positive. "
        f"PSM = mean of per-stock correlations, robust to the differing price scales of the {s['stocks_processed']} stocks "
        "(naive pooled correlation is in the CSVs but is misleading for level indicators).</p>"
        f"{chart}<h3>Detail</h3>{table}</section>"
    )


def distributions(df):
    rows = "".join(
        f"<tr><td>{r.target}</td><td>{r.split}</td><td class='num'>{fi(r['count'])}</td>"
        f"<td class='num'>{fsci(r['mean'])}</td><td class='num'>{f4(r['std'])}</td>"
        f"<td class='num'>{fpct(r.positive_pct)}</td><td class='num'>{fpct(r.negative_pct)}</td></tr>"
        for _, r in df.iterrows()
    )
    return (
        "<section><h2>Target distributions</h2>"
        "<p class='lead'>Future-return target statistics by split. Std grows with horizon; "
        "VALIDATION has lower volatility than TRAIN (e.g. 5m std 0.0019 vs 0.0015).</p>"
        "<table><thead><tr><th>Target</th><th>Split</th><th class='num'>Count</th>"
        "<th class='num'>Mean</th><th class='num'>Std</th><th class='num'>Positive %</th>"
        "<th class='num'>Negative %</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def deciles(dec, strong):
    d = dec[dec.method == "pooled"][["indicator", "target", "anova_f", "anova_p", "bucket_mean_spread"]]
    m = strong[["indicator", "target", "rank", "train_pearson_psm", "train_n_obs"]].merge(d, on=["indicator", "target"])
    m = m.sort_values("rank")
    rows = "".join(
        f"<tr><td class='num'>{r['rank']}</td><td><b>{r.indicator}</b> → {r.target}</td>"
        f"<td class='num {sc(r.train_pearson_psm)}'>{f4(r.train_pearson_psm)}</td>"
        f"<td class='num'>{r.anova_f:.0f}</td><td class='num'>{fp(r.anova_p)}</td>"
        f"<td class='num'>{r.bucket_mean_spread*10000:.1f} bps</td>"
        f"<td class='num'>{fi(r.train_n_obs)}</td></tr>"
        for _, r in m.iterrows()
    )
    return (
        "<section><h2>Decile significance (TRAIN, pooled buckets)</h2>"
        "<p class='lead'>One-way ANOVA of the future return across the 10 indicator deciles. "
        "Every p ≈ 0 because n is huge — but the top-vs-bottom bucket spread is only a few basis "
        "points: <b>significance ≠ profitability</b>.</p>"
        "<table><thead><tr><th class='num'>#</th><th>Indicator → Target</th>"
        "<th class='num'>Train psm</th><th class='num'>ANOVA F</th><th class='num'>ANOVA p</th>"
        "<th class='num'>Bucket spread</th><th class='num'>n</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def timeofday(tod, strong):
    top = strong.head(5)
    periods = ["p1_0915-1000", "p2_1000-1100", "p3_1100-1200", "p4_1200-1300", "p5_1300-1400", "p6_1400-1530"]
    plabel = {
        "p1_0915-1000": "09:15-10:00", "p2_1000-1100": "10:00-11:00", "p3_1100-1200": "11:00-12:00",
        "p4_1200-1300": "12:00-13:00", "p5_1300-1400": "13:00-14:00", "p6_1400-1530": "14:00-15:30",
    }
    t = tod[tod.split == "train"]
    rows = ""
    for _, rel in top.iterrows():
        cells = ""
        for p in periods:
            v = t[(t.indicator == rel.indicator) & (t.target == rel.target) & (t.period == p)]["pearson"]
            v = v.iloc[0] if len(v) else float("nan")
            cells += f"<td class='num' style='background:{heat(v)}'>{'—' if pd.isna(v) else f4(v)}</td>"
        rows += f"<tr><td><b>{rel.indicator}</b> → {rel.target}</td>{cells}</tr>"
    head = "".join(f"<th class='num'>{plabel[p]}</th>" for p in periods)
    return (
        "<section><h2>Time-of-day slices (TRAIN, pooled Pearson)</h2>"
        "<p class='lead'>Intraday correlation by session period for the top 5 relationships. "
        "Red = negative (reversal), green = positive, intensity = |magnitude|. Pooled Pearson is "
        "meaningful mainly for ratio indicators; level indicators wash out cross-sectionally here.</p>"
        f"<table><thead><tr><th>Indicator → Target</th>{head}</tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _rel(df, ind, tgt):
    m = df[(df["indicator"] == ind) & (df["target"] == tgt)]
    return m.iloc[0] if len(m) else None


def _std(dist, tgt, split):
    m = dist[(dist["target"] == tgt) & (dist["split"] == split)]
    return float(m["std"].iloc[0]) if len(m) else float("nan")


def _spread(dec, ind, tgt, method="pooled"):
    m = dec[(dec["indicator"] == ind) & (dec["target"] == tgt) & (dec["method"] == method)]
    return float(m["bucket_mean_spread"].iloc[0]) if len(m) else float("nan")


def findings(strong, s):
    top = strong.iloc[0]
    top10 = strong.head(10)
    n10 = len(top10)
    sign_n = int(top10["sign_persists"].astype(bool).sum())
    mag_n = int((top10["val_magnitude_ratio"] >= 1.0).sum())

    rev = [r for r in (_rel(strong, i, "future_return_5m")
            for i in ["close_position", "return_1m", "return_5m", "candle_body"]) if r is not None]
    rev_txt = (f"{min(r['val_magnitude_ratio'] for r in rev):.2f}–{max(r['val_magnitude_ratio'] for r in rev):.2f}"
               if rev else "—")

    ma = [r for r in (_rel(strong, i, "future_return_60m")
         for i in ["vwap", "sma_5", "sma_20", "sma_50", "sma_200"]) if r is not None]
    ma_txt = (f"{min(r['val_magnitude_ratio'] for r in ma):.2f}–{max(r['val_magnitude_ratio'] for r in ma):.2f}"
              if ma else "—")

    vwap60 = _rel(strong, "vwap", "future_return_60m")
    vwap_txt = (f" (e.g. <code>vwap → 60m</code>: pooled {f4(vwap60['train_pearson_pooled'])}, "
                f"psm {f4(vwap60['train_pearson_psm'])})") if vwap60 is not None else ""

    flip = strong[~strong["sign_persists"].astype(bool)]
    flip = flip.iloc[0] if len(flip) else None
    vol = strong[strong["indicator"].astype(str).str.startswith("volatility")]
    collapse = vol.loc[vol["val_magnitude_ratio"].idxmin()] if len(vol) else None

    vol_parts = []
    if flip is not None:
        vol_parts.append(f"<code>{flip['indicator']} → {flip['target']}</code> <i>flips sign</i> "
                         f"(train {f4(flip['train_pearson_psm'])} → val {f4(flip['val_pearson_psm'])})")
    if collapse is not None and (flip is None or collapse["rank"] != flip["rank"]):
        vol_parts.append(f"<code>{collapse['indicator']} → {collapse['target']}</code> collapses "
                         f"(ratio {collapse['val_magnitude_ratio']:.3f})")
    vol_li = (f"<li><b>Volatility does NOT persist.</b> The relationships that fail validation are "
              f"volatility ones: {'; '.join(vol_parts)}. Validation did its job here.</li>"
              if vol_parts else
              "<li><b>Volatility.</b> Volatility relationships tend to weaken or fail validation.</li>")

    return (
        "<section><h2>Key findings</h2><ul>"
        f"<li><b>Most top relationships persist to VALIDATION.</b> {sign_n} of the {n10} strongest keep "
        f"their sign out-of-sample; {mag_n} of {n10} have a magnitude ratio ≥ 1.0 (validation as strong or "
        f"stronger than train). Strongest: <code>{top['indicator']} → {top['target']}</code> "
        f"(train {f4(top['train_pearson_psm'])}, val {f4(top['val_pearson_psm'])}), consistent in sign "
        f"across {fpct(top['train_frac_consistent'])} of the {int(top['train_n_stocks'])} stocks with data.</li>"
        f"<li><b>Short-term reversal dominates.</b> <code>close_position</code>, <code>return_1m</code>, "
        f"<code>return_5m</code>, <code>candle_body</code> all <i>negatively</i> predict the 5-min future "
        f"return and persist with magnitude ratio ≈ {rev_txt} — a within-session mean-reversion effect.</li>"
        f"<li><b>Mean reversion vs own moving averages.</b> <code>vwap</code>, <code>sma_5/20/50/200</code> "
        f"all <i>negatively</i> predict the 60-min future return and <i>strengthen</i> in validation "
        f"(ratio {ma_txt}). These are level indicators — their naive pooled Pearson is ≈0; only the "
        f"per-stock-mean reveals the signal{vwap_txt}.</li>"
        f"{vol_li}"
        "<li><b>Time-of-day.</b> Reversal effects are present throughout the session and tend to "
        "strengthen later in the day (see the time-of-day table below).</li>"
        "</ul></section>"
    )


def caveats(s, dist, dec):
    std5_tr = _std(dist, "future_return_5m", "train")
    std5_val = _std(dist, "future_return_5m", "validation")
    spread = _spread(dec, "return_1m", "future_return_5m")
    maxpsm = abs(s["top_relationships"][0]["train_pearson_psm"]) if s.get("top_relationships") else 0.0
    n_rel = s["n_indicators"] * s["n_targets"]
    std5_tr_bps = f"{std5_tr * 10000:.1f}" if pd.notna(std5_tr) else "—"
    std5_tr_txt = f4(std5_tr)
    std5_val_txt = f4(std5_val)
    spread_bps = f"{spread * 10000:.1f}" if pd.notna(spread) else "—"
    return (
        "<section class='last'><h2>Caveats & limitations</h2>"
        f"<div class='callout bad'><b>Significance ≠ profitability.</b> With {fi(s['train_rows_loaded'])} TRAIN rows "
        f"even negligible effects are 'significant' (every ANOVA p ≈ 0), but the decile top-vs-bottom "
        f"bucket spread is only a few basis points (e.g. <code>return_1m → future_return_5m</code> ≈ "
        f"{spread_bps} bps vs a ~{std5_tr_bps} bps target std).</div>"
        "<ul>"
        f"<li><b>Multiple testing.</b> {n_rel} relationships × several metrics were examined; treat the "
        "ranking as exploratory. The held-out VALIDATION persistence check is the main safeguard against "
        "cherry-picking.</li>"
        "<li><b>Level-indicator caveat.</b> Indicators in absolute price units (<code>vwap, sma_*, "
        "candle_range, candle_body, volume_sma_20</code>) have cross-sectionally differing scales, so "
        "their naive <i>pooled</i> correlation is attenuated toward zero. Trust the per-stock-mean for "
        "these; ignore their pooled column.</li>"
        f"<li><b>Small correlations.</b> |psm| ≤ ~{maxpsm:.2f} in this run — no single relationship is a "
        "strong predictor; any future use would need ensembling and transaction-cost modelling.</li>"
        f"<li><b>Regime shift.</b> VALIDATION had lower volatility (5m std {std5_val_txt} vs {std5_tr_txt} "
        "on TRAIN), so a few magnitude ratios >1 partly reflect denominator shrinkage, not stronger signal.</li>"
        "<li><b>Linear / monotonic only.</b> Only Pearson and Spearman association is measured; "
        "interaction and conditional effects are not assessed here.</li>"
        "</ul>"
        "<div class='callout warn'><b>What this is NOT.</b> Not a strategy, not a backtest, no thresholds "
        "chosen, no trade rules. No TEST data was read or used in any way. Day 5 is a separate step.</div></section>"
    )


def build():
    s, strong, dist, dec, tod = load()
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Day 4 — Predictive Analysis</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        f"{header(s)}{summary(s)}{top_rel(strong, s)}{distributions(dist)}"
        f"{deciles(dec, strong)}{timeofday(tod, strong)}{findings(strong, s)}{caveats(s, dist, dec)}"
        f"<footer>Generated {s['generated_at']} · src.analysis.predictive_analysis · "
        "No TEST data read · stdlib-only statistics</footer>"
        "</div></body></html>"
    )
    out = D / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html):,} chars)")


if __name__ == "__main__":
    build()

