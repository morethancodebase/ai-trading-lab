"""Day 5: signal discovery from Day 4 predictive relationships.

Day 4 found weak but persistent associations between several bar indicators and
the 5-minute future return (notably ``close_position``, ``return_1m``,
``return_5m``, ``candle_body``; all negative, i.e. short-term mean reversion).
Day 5 asks whether those weak relationships can be converted into simple,
interpretable *directional signals* (LONG / SHORT / NO SIGNAL) that show useful
behaviour on unseen VALIDATION data. This is exploratory signal discovery, NOT a
strategy or backtest.

Research / leakage rules (same guardrails as Day 4)
---------------------------------------------------
* No look-ahead: every indicator is backward-looking (Day 2, within-session);
  ``future_return_*`` targets are never used as inputs -- they are only used to
  *score* a signal after it fires.
* TEST is never read or used: every load filters
  ``date BETWEEN TRAIN_START AND VALIDATION_END`` and a hard guard asserts no row
  has ``date >= TEST_START`` (2026-03-04).
* Every threshold is a per-stock percentile learned from that stock's TRAIN data
  only, then frozen and applied unchanged to that stock's VALIDATION rows. No
  threshold is ever recomputed on VALIDATION, and nothing is optimised against
  VALIDATION. (Per-stock thresholds are required because level indicators such as
  ``candle_body`` and ``distance_from_vwap`` are in price units and are not
  cross-sectionally comparable -- the same reason Day 4 used per-stock deciles.)
* Signal directions are taken from Day 4's per-stock-mean Pearson sign vs
  ``future_return_5m`` (negative association -> high value = SHORT, low value =
  LONG, i.e. reversal). Nothing here is re-fit.
* Signals are declared a priori (a small, fixed set) -- NOT selected by
  validation performance. We deliberately avoid brute-forcing many thresholds or
  combinations; that would be data mining.
* All returns are reported RAW in basis points, before brokerage, taxes, fees,
  slippage, bid/ask spread, and market impact. Statistical / directional interest
  is explicitly NOT a profitability claim.

Run from the project root::

    python -m src.analysis.signal_discovery
    python -m src.analysis.signal_discovery --limit-stocks 5   # quick smoke test

Only pandas / numpy / duckdb are used (already in the project stack).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.data.build_bar_indicators import INDICATOR_COLUMNS
from src.data.date_splits import (
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from src.data.nifty100 import NIFTY100_FALLBACK

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "day5_signal_discovery"

# We test signals on the 5-minute horizon -- Day 4's clearest persistent finding.
# Other horizons are intentionally NOT scanned, to avoid "which horizon looks
# best" data mining.
TARGET = "future_return_5m"

# Candidate indicators and their Day-4 per-stock-mean Pearson sign vs
# future_return_5m (train). The sign defines the signal direction: a negative
# association means "high value -> expect reversal down (SHORT), low value ->
# expect reversal up (LONG)"; a positive association means the opposite.
INDICATOR_DAY4_PSM = {
    "close_position": -0.041493,
    "return_1m": -0.028806,
    "return_5m": -0.024313,
    "candle_body": -0.026973,
    "rsi_14": -0.014444,
    "distance_from_vwap": -0.003751,
    "volume_ratio_20": 0.006099,
}
CANDIDATE_INDICATORS = list(INDICATOR_DAY4_PSM.keys())
# Signal specifications -- declared a priori, NOT selected by validation perf.
# kind: "single" (one indicator), "combo" (2 indicators must agree on direction),
# "vote" (>=2 of 3 indicators agree). threshold: which percentile bands to use.
#   extreme  -> low p10 / high p90
#   moderate -> low p25 / high p75  (threshold-sensitivity check only)
SPECS = [
    # --- single-condition, extreme (10/90) ---
    {"id": "S1", "name": "close_position extreme", "kind": "single", "threshold": "extreme", "inds": ["close_position"]},
    {"id": "S2", "name": "return_1m extreme", "kind": "single", "threshold": "extreme", "inds": ["return_1m"]},
    {"id": "S3", "name": "return_5m extreme", "kind": "single", "threshold": "extreme", "inds": ["return_5m"]},
    {"id": "S4", "name": "candle_body extreme", "kind": "single", "threshold": "extreme", "inds": ["candle_body"]},
    {"id": "S5", "name": "rsi_14 extreme", "kind": "single", "threshold": "extreme", "inds": ["rsi_14"]},
    {"id": "S6", "name": "distance_from_vwap extreme", "kind": "single", "threshold": "extreme", "inds": ["distance_from_vwap"]},
    {"id": "S7", "name": "volume_ratio_20 extreme", "kind": "single", "threshold": "extreme", "inds": ["volume_ratio_20"]},
    # --- single-condition, moderate (25/75) -- sensitivity for the 2 strongest ---
    {"id": "S1b", "name": "close_position moderate", "kind": "single", "threshold": "moderate", "inds": ["close_position"]},
    {"id": "S2b", "name": "return_1m moderate", "kind": "single", "threshold": "moderate", "inds": ["return_1m"]},
    # --- combinations, extreme (10/90) -- indicators must agree on direction ---
    {"id": "C1", "name": "close_position & return_1m agree", "kind": "combo", "threshold": "extreme", "inds": ["close_position", "return_1m"]},
    {"id": "C2", "name": "close_position & return_5m agree", "kind": "combo", "threshold": "extreme", "inds": ["close_position", "return_5m"]},
    {"id": "C3", "name": "close_position & candle_body agree", "kind": "combo", "threshold": "extreme", "inds": ["close_position", "candle_body"]},
    {"id": "C4", "name": ">=2 of {close_position,return_1m,return_5m} agree", "kind": "vote", "threshold": "extreme", "inds": ["close_position", "return_1m", "return_5m"]},
]

NEED_COLS = ["timestamp"] + CANDIDATE_INDICATORS + [TARGET]

# Guard: the target must never appear among the indicator inputs.
assert not any(c.startswith("future_return") for c in INDICATOR_COLUMNS), (
    "future_return_* must never be used as an indicator input"
)

# Intraday time-of-day buckets (same boundaries as Day 4) as minutes-of-day.
_PERIOD_BOUNDS = [
    (555, 600, "p1_0915-1000"),
    (600, 660, "p2_1000-1100"),
    (660, 720, "p3_1100-1200"),
    (720, 780, "p4_1200-1300"),
    (780, 840, "p5_1300-1400"),
    (840, 930, "p6_1400-1530"),
]


# ---------------------------------------------------------------------------
# Data loading (reuses the exact Day 4 pattern: TRAIN+VALIDATION only, TEST guard).
# ---------------------------------------------------------------------------
def discover_target_files() -> list[tuple[str, Path]]:
    """Same discovery order as Day 4: prefer canonical NIFTY100 symbols first."""
    by_lower: dict[str, Path] = {}
    for p in PROCESSED_DIR.glob("*_1min_targets.parquet"):
        stem = p.name[: -len("_1min_targets.parquet")]
        by_lower[stem.lower()] = p
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for sym in sorted({s.upper() for s in NIFTY100_FALLBACK}):
        low = sym.lower()
        if low in by_lower:
            found.append((sym, by_lower[low]))
            seen.add(low)
    for low, p in sorted(by_lower.items()):
        if low not in seen:
            found.append((low.upper(), p))
    return found


def load_stock(symbol: str, path: Path) -> pd.DataFrame | None:
    """Load one stock's TRAIN + VALIDATION rows (never TEST), needed cols only."""
    cols = ", ".join(NEED_COLS)
    q = f"""
        SELECT {cols}
        FROM read_parquet('{path.as_posix()}')
        WHERE CAST(timestamp AS DATE) BETWEEN DATE '{TRAIN_START.isoformat()}'
                                          AND DATE '{VALIDATION_END.isoformat()}'
        ORDER BY timestamp
    """
    df = duckdb.execute(q).df()
    if df.empty:
        return None
    dates = pd.to_datetime(df["timestamp"]).dt.date
    if (dates >= TEST_START).any():  # hard guard against TEST leakage
        raise RuntimeError(f"{symbol}: TEST data leaked into analysis load")
    df["_split"] = np.where(dates <= TRAIN_END, "train", "validation")
    return df


# ---------------------------------------------------------------------------
# Frozen per-stock thresholds (TRAIN-only percentiles) and signal construction.
# ---------------------------------------------------------------------------
def compute_thresholds(train_df: pd.DataFrame) -> dict[str, dict] | None:
    """Per-stock frozen thresholds: TRAIN percentiles {10,25,75,90} per indicator.

    Returns {indicator: {"p10","p25","p75","p90","n"}} for indicators with >= 20
    finite TRAIN values; the indicator is skipped (absent from the dict)
    otherwise. These thresholds are applied unchanged to the same stock's
    VALIDATION rows -- nothing is recomputed on VALIDATION.
    """
    if len(train_df) == 0:
        return None
    out: dict[str, dict] = {}
    for ind in CANDIDATE_INDICATORS:
        v = train_df[ind].to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if len(v) < 20:
            continue
        q = np.nanpercentile(v, [10, 25, 75, 90])
        out[ind] = {
            "p10": float(q[0]), "p25": float(q[1]),
            "p75": float(q[2]), "p90": float(q[3]), "n": int(len(v)),
        }
    return out


def _mask_block(values: np.ndarray, thr: dict | None) -> dict[str, np.ndarray]:
    """Boolean masks for one indicator on a split's rows, using frozen thresholds.

    NaN comparisons are False, so missing indicator values produce no signal.
    If the indicator had no usable TRAIN thresholds (``thr is None``) all masks
    are all-False for that stock.
    """
    n = len(values)
    if thr is None:
        f = np.zeros(n, dtype=bool)
        return {"high_ext": f, "low_ext": f, "high_mod": f, "low_mod": f}
    return {
        "high_ext": values >= thr["p90"],
        "low_ext": values <= thr["p10"],
        "high_mod": values >= thr["p75"],
        "low_mod": values <= thr["p25"],
    }


def signal_direction(
    spec: dict, masks: dict[str, dict[str, np.ndarray]]
) -> np.ndarray:
    """Return +1 (LONG) / -1 (SHORT) / 0 (NO SIGNAL) per row for a spec.

    Per-indicator signed direction uses the Day-4 psm sign: high value ->
    ``high_dir`` (+1 LONG if psm >= 0, else -1 SHORT), low value -> the opposite.
    Combos require the two indicators to agree; a vote needs >= 2 of 3 to agree.
    """
    inds = spec["inds"]
    band = spec["threshold"]
    hk = "high_ext" if band == "extreme" else "high_mod"
    lk = "low_ext" if band == "extreme" else "low_mod"
    n = len(masks[inds[0]][hk])
    dirs: list[np.ndarray] = []
    for ind in inds:
        hd = 1 if INDICATOR_DAY4_PSM[ind] >= 0 else -1  # high value direction
        d = np.zeros(n, dtype="int8")
        d[masks[ind][hk]] = hd    # high value
        d[masks[ind][lk]] = -hd   # low value (opposite)
        dirs.append(d)
    if spec["kind"] == "single":
        return dirs[0]
    if spec["kind"] == "combo":
        agree = (dirs[0] == dirs[1]) & (dirs[0] != 0)
        out = np.zeros(n, dtype="int8")
        out[agree] = dirs[0][agree]
        return out
    if spec["kind"] == "vote":  # >= 2 of 3 agree
        pos = sum((d == 1).astype(np.int8) for d in dirs)
        neg = sum((d == -1).astype(np.int8) for d in dirs)
        out = np.zeros(n, dtype="int8")
        out[pos >= 2] = 1
        out[neg >= 2] = -1
        return out
    raise ValueError(f"unknown signal kind: {spec['kind']}")


def period_of_day(timestamps: pd.Series) -> np.ndarray:
    """Map each timestamp to a 1..6 time-of-day bucket (0 = outside session)."""
    t = pd.to_datetime(timestamps).dt
    m = t.hour.to_numpy() * 60 + t.minute.to_numpy()
    out = np.zeros(len(timestamps), dtype="int8")
    for i, (lo, hi, _) in enumerate(_PERIOD_BOUNDS, start=1):
        out[(m >= lo) & (m < hi)] = i
    return out


def _stock_stats(long_fr: np.ndarray, short_fr: np.ndarray, n_total: int) -> dict | None:
    """Per-stock summary for one (signal, split). Returns None if no signal fired."""
    n_long = int(len(long_fr))
    n_short = int(len(short_fr))
    n_sig = n_long + n_short
    if n_sig == 0:
        return None
    exp = np.concatenate([long_fr, -short_fr])           # expected return per signal
    move = np.concatenate([long_fr, short_fr])           # underlying |move| source
    hit = int((long_fr > 0).sum() + (short_fr < 0).sum())
    return {
        "n_total": int(n_total),
        "n_signals": n_sig,
        "n_long": n_long,
        "n_short": n_short,
        "hit_rate": hit / n_sig,
        "avg_exp_ret_bps": float(exp.mean()) * 1e4,
        "median_exp_ret_bps": float(np.median(exp)) * 1e4,
        "pct_positive": float((exp > 0).mean()) * 100,
        "avg_move_bps": float(np.abs(move).mean()) * 1e4,
    }


# ---------------------------------------------------------------------------
# Streaming run: one stock at a time (memory-bounded), TRAIN thresholds frozen
# before any VALIDATION row is scored.
# ---------------------------------------------------------------------------
def run(limit_stocks: int | None = None) -> dict:
    t0 = time.time()
    files = discover_target_files()
    if limit_stocks:
        files = files[:limit_stocks]
    print(f"Discovered {len(files)} stock target files.")
    print(f"Target: {TARGET}  |  {len(SPECS)} signals declared a priori")

    # Pooled arrays per (signal_id, split, dir) for exact pooled mean/median/etc.
    arr_acc: dict[tuple[str, str, int], list[np.ndarray]] = {}
    # Per-stock summary rows (signal_by_stock.csv)
    ps_rows: list[dict] = []
    # Time-of-day streaming accumulators (signal_id, split, period) -> stats
    tod_acc: dict[tuple[str, str, int], dict] = {}
    # Frozen-threshold audit rows
    thr_audit: list[dict] = []
    n_total_split = {"train": 0, "validation": 0}
    rows_train = rows_val = 0
    stocks_ok = stocks_fail = 0

    for i, (sym, path) in enumerate(files, 1):
        try:
            df = load_stock(sym, path)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:>3}/{len(files)}] {sym}: LOAD FAILED ({e})")
            stocks_fail += 1
            continue
        if df is None:
            print(f"  [{i:>3}/{len(files)}] {sym}: no rows")
            stocks_fail += 1
            continue
        stocks_ok += 1
        train = df[df["_split"] == "train"]
        val = df[df["_split"] == "validation"]
        rows_train += int(len(train))
        rows_val += int(len(val))

        # ---- freeze per-stock TRAIN thresholds BEFORE scoring validation ----
        thr = compute_thresholds(train)
        for ind in CANDIDATE_INDICATORS:
            t = thr.get(ind) if thr else None
            v = train[ind].to_numpy(dtype="float64")
            ntr = int(np.isfinite(v).sum())
            if t is None:
                thr_audit.append({"symbol": sym, "indicator": ind, "p10": None,
                                  "p25": None, "p75": None, "p90": None,
                                  "n_train": ntr})
            else:
                thr_audit.append({"symbol": sym, "indicator": ind, "p10": t["p10"],
                                  "p25": t["p25"], "p75": t["p75"], "p90": t["p90"],
                                  "n_train": t["n"]})

        for split, sdf in (("train", train), ("validation", val)):
            if sdf.empty:
                continue
            masks = {ind: _mask_block(sdf[ind].to_numpy(dtype="float64"),
                                     thr.get(ind) if thr else None)
                     for ind in CANDIDATE_INDICATORS}
            fr = sdf[TARGET].to_numpy(dtype="float64")
            scoreable = np.isfinite(fr)
            n_total_split[split] += int(scoreable.sum())
            periods = period_of_day(sdf["timestamp"])
            for spec in SPECS:
                d = signal_direction(spec, masks)
                d[~scoreable] = 0  # never score a row without a valid target
                long_fr = fr[d == 1]
                short_fr = fr[d == -1]
                # per-stock summary
                st = _stock_stats(long_fr, short_fr, int(scoreable.sum()))
                if st is not None:
                    ps_rows.append({"signal_id": spec["id"], "signal_name": spec["name"],
                                    "split": split, "symbol": sym, **st})
                # pooled arrays (exact pooled median/percentiles later)
                arr_acc.setdefault((spec["id"], split, 1), []).append(long_fr)
                arr_acc.setdefault((spec["id"], split, -1), []).append(short_fr)
                # time-of-day (overall long+short, streaming)
                fired = d != 0
                for pidx in range(1, 7):
                    sel = fired & (periods == pidx)
                    if not sel.any():
                        continue
                    frr = fr[sel]
                    exp = d[sel].astype("float64") * frr  # +1/-1 times return
                    a = tod_acc.setdefault((spec["id"], split, pidx),
                                           {"n": 0, "sum_exp": 0.0, "n_hit": 0})
                    a["n"] += int(sel.sum())
                    a["sum_exp"] += float(exp.sum())
                    a["n_hit"] += int((exp > 0).sum())
        del df, train, val
        if i % 10 == 0 or i == len(files):
            print(f"  [{i:>3}/{len(files)}] {sym} done")

    print("Aggregating ...")
    summary_df = build_pooled_summary(arr_acc, ps_rows, n_total_split)
    cmp_df = build_train_vs_val(summary_df)
    tod_df = build_time_of_day(tod_acc)
    write_outputs(summary_df, pd.DataFrame(ps_rows), tod_df, cmp_df,
                  pd.DataFrame(thr_audit), rows_train, rows_val,
                  n_total_split, stocks_ok, stocks_fail, time.time() - t0)
    print("\n=== Day 5 signal discovery summary ===")
    print(f"stocks processed : {stocks_ok}  (failed: {stocks_fail})")
    print(f"TRAIN rows       : {rows_train:,}  (scorable: {n_total_split['train']:,})")
    print(f"VALIDATION rows  : {rows_val:,}  (scorable: {n_total_split['validation']:,})")
    print(f"TEST rows read   : 0  (never read)")
    print("\nTRAIN vs VALIDATION (avg expected return, bps):")
    for _, r in cmp_df.iterrows():
        print(f"  {r['signal_id']:<4} {r['signal_name']:<46} "
              f"train={r['train_avg_ret_bps']:+7.3f}  val={r['val_avg_ret_bps']:+7.3f}  "
              f"ratio={r['val_train_ratio']:.2f}  "
              f"{'persist' if r['persistence_flag'] else '       '}")
    return {"signals_tested": len(SPECS)}
# ---------------------------------------------------------------------------
# Aggregation / output.
# ---------------------------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: Path, float_fmt: str = "%.10g") -> None:
    df.to_csv(path, index=False, float_format=float_fmt)
    print(f"  wrote {path.name}  ({len(df)} rows)")


def _pooled_from_arrays(arr_acc, spec_id, split):
    """Concatenate per-stock arrays -> pooled long/short future-return arrays."""
    longs = arr_acc.get((spec_id, split, 1), [])
    shorts = arr_acc.get((spec_id, split, -1), [])
    lf = np.concatenate(longs) if longs else np.array([])
    sf = np.concatenate(shorts) if shorts else np.array([])
    return lf, sf


def build_pooled_summary(arr_acc, ps_rows, n_total_split) -> pd.DataFrame:
    """One row per (signal, split): pooled metrics + cross-stock consistency."""
    rows = []
    for spec in SPECS:
        for split in ("train", "validation"):
            lf, sf = _pooled_from_arrays(arr_acc, spec["id"], split)
            n_long, n_short = int(len(lf)), int(len(sf))
            n_sig = n_long + n_short
            if n_sig > 0:
                exp = np.concatenate([lf, -sf])
                move = np.concatenate([lf, sf])
                hit = int((lf > 0).sum() + (sf < 0).sum())
                hr = hit / n_sig
                hr_l = float((lf > 0).mean()) if n_long else float("nan")
                hr_s = float((sf < 0).mean()) if n_short else float("nan")
                avg_exp = float(exp.mean()) * 1e4
                avg_l = float(lf.mean()) * 1e4 if n_long else float("nan")
                avg_s = float(-sf.mean()) * 1e4 if n_short else float("nan")
                med = float(np.median(exp)) * 1e4
                pct_pos = float((exp > 0).mean()) * 100
                avg_move = float(np.abs(move).mean()) * 1e4
            else:
                hr = hr_l = hr_s = avg_exp = avg_l = avg_s = med = pct_pos = avg_move = float("nan")
            n_total = n_total_split[split]
            stk = [r for r in ps_rows
                   if r["signal_id"] == spec["id"] and r["split"] == split]
            n_stk = len(stk)
            if n_stk:
                rets = np.array([r["avg_exp_ret_bps"] for r in stk], dtype="float64")
                n_pos = int(np.sum(rets > 0))
                med_stk = float(np.median(rets))
                mean_stk = float(rets.mean())
                wi, bi = int(np.argmin(rets)), int(np.argmax(rets))
                worst, best = stk[wi]["symbol"], stk[bi]["symbol"]
                worst_ret, best_ret = float(rets[wi]), float(rets[bi])
            else:
                n_pos = 0
                med_stk = mean_stk = worst_ret = best_ret = float("nan")
                worst = best = ""
            rows.append({
                "signal_id": spec["id"], "signal_name": spec["name"],
                "kind": spec["kind"], "threshold": spec["threshold"], "split": split,
                "n_total": n_total, "n_signals": n_sig, "n_long": n_long, "n_short": n_short,
                "signal_freq": (n_sig / n_total) if n_total else float("nan"),
                "hit_rate": hr, "hit_rate_long": hr_l, "hit_rate_short": hr_s,
                "avg_exp_ret_bps": avg_exp, "avg_exp_ret_long_bps": avg_l,
                "avg_exp_ret_short_bps": avg_s, "median_exp_ret_bps": med,
                "pct_positive": pct_pos, "avg_move_bps": avg_move,
                "n_stocks_signals": n_stk, "n_stocks_positive": n_pos,
                "median_stock_ret_bps": med_stk, "mean_stock_ret_bps": mean_stk,
                "worst_stock": worst, "worst_stock_ret_bps": worst_ret,
                "best_stock": best, "best_stock_ret_bps": best_ret,
            })
    return pd.DataFrame(rows)


def build_train_vs_val(summary_df: pd.DataFrame) -> pd.DataFrame:
    """One row per signal: TRAIN vs VALIDATION comparison + persistence flag."""
    rows = []
    for spec in SPECS:
        tr = summary_df[(summary_df["signal_id"] == spec["id"]) & (summary_df["split"] == "train")]
        va = summary_df[(summary_df["signal_id"] == spec["id"]) & (summary_df["split"] == "validation")]
        if tr.empty or va.empty:
            continue
        tr, va = tr.iloc[0], va.iloc[0]
        tr_ret = float(tr["avg_exp_ret_bps"])
        va_ret = float(va["avg_exp_ret_bps"])
        ratio = (va_ret / tr_ret) if tr_ret and not np.isnan(tr_ret) else float("nan")
        both_pos = (tr_ret > 0) and (va_ret > 0)
        persist = bool(both_pos and not np.isnan(ratio) and ratio >= 0.5
                       and int(va["n_signals"]) >= 1000)
        rows.append({
            "signal_id": spec["id"], "signal_name": spec["name"],
            "kind": spec["kind"], "threshold": spec["threshold"],
            "train_n_signals": int(tr["n_signals"]), "val_n_signals": int(va["n_signals"]),
            "train_hit_rate": tr["hit_rate"], "val_hit_rate": va["hit_rate"],
            "hit_rate_delta": float(va["hit_rate"]) - float(tr["hit_rate"]),
            "train_avg_ret_bps": tr_ret, "val_avg_ret_bps": va_ret,
            "avg_ret_delta_bps": va_ret - tr_ret, "val_train_ratio": ratio,
            "train_pct_positive": tr["pct_positive"], "val_pct_positive": va["pct_positive"],
            "train_n_stocks_positive": tr["n_stocks_positive"],
            "val_n_stocks_positive": va["n_stocks_positive"],
            "train_median_stock_ret_bps": tr["median_stock_ret_bps"],
            "val_median_stock_ret_bps": va["median_stock_ret_bps"],
            "both_positive": bool(both_pos), "persistence_flag": persist,
        })
    return pd.DataFrame(rows)



def build_time_of_day(tod_acc) -> pd.DataFrame:
    rows = []
    name_by_id = {s["id"]: s["name"] for s in SPECS}
    for (spec_id, split, pidx), a in tod_acc.items():
        if a["n"] == 0:
            continue
        rows.append({
            "signal_id": spec_id, "signal_name": name_by_id[spec_id], "split": split,
            "period": _PERIOD_BOUNDS[pidx - 1][2], "n_signals": a["n"],
            "hit_rate": a["n_hit"] / a["n"],
            "avg_exp_ret_bps": (a["sum_exp"] / a["n"]) * 1e4,
        })
    return pd.DataFrame(rows)


def write_outputs(summary_df, by_stock_df, tod_df, cmp_df, thr_df,
                  rows_train, rows_val, n_total_split,
                  stocks_ok, stocks_fail, elapsed) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_df, REPORTS_DIR / "signal_summary.csv")
    _write_csv(by_stock_df, REPORTS_DIR / "signal_by_stock.csv")
    _write_csv(tod_df, REPORTS_DIR / "signal_by_time.csv")
    _write_csv(cmp_df, REPORTS_DIR / "signal_train_vs_val.csv")
    _write_csv(thr_df, REPORTS_DIR / "frozen_thresholds.csv")

    promising = cmp_df[cmp_df["persistence_flag"]].sort_values(
        "val_avg_ret_bps", ascending=False)
    rejected = cmp_df[~(cmp_df["val_avg_ret_bps"] > 0)]
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stocks_processed": stocks_ok,
        "stocks_failed": stocks_fail,
        "target": TARGET,
        "n_signals_tested": len(SPECS),
        "signal_ids": [s["id"] for s in SPECS],
        "splits_used": ["train", "validation"],
        "test_ever_read": False,
        "max_allowed_date": str(VALIDATION_END),
        "threshold_source": ("per-stock TRAIN percentiles (10/25/75/90), frozen, "
                             "applied unchanged to validation"),
        "candidate_indicators": INDICATOR_DAY4_PSM,
        "train_rows_loaded": rows_train,
        "validation_rows_loaded": rows_val,
        "train_scorable_rows": n_total_split["train"],
        "validation_scorable_rows": n_total_split["validation"],
        "elapsed_seconds": round(elapsed, 1),
        "promising_signals": [
            {"signal_id": r["signal_id"], "signal_name": r["signal_name"],
             "train_avg_ret_bps": r["train_avg_ret_bps"],
             "val_avg_ret_bps": r["val_avg_ret_bps"],
             "val_train_ratio": r["val_train_ratio"],
             "val_hit_rate": r["val_hit_rate"], "val_n_signals": r["val_n_signals"]}
            for _, r in promising.iterrows()
        ],
        "rejected_signals": [r["signal_id"] for _, r in rejected.iterrows()],
    }
    with (REPORTS_DIR / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  wrote run_summary.json")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--limit-stocks", type=int, default=None,
                   help="Process only the first N stocks (smoke test).")
    args = p.parse_args(argv)
    run(limit_stocks=args.limit_stocks)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())




