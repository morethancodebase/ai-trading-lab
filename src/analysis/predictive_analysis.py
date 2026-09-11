"""Day 4: predictive analysis of bar indicators vs future-return targets.

Research-only. Discovers indicator -> target relationships on TRAIN, checks
whether they persist on VALIDATION, and NEVER touches TEST. No ML model, no
strategy, no backtest, no threshold optimization.

Design / research rules
-----------------------
* No look-ahead: every indicator is backward-looking (Day 2, within-session).
  ``future_return_*`` targets are never used as inputs.
* TEST is never read or used: every load/query filters
  ``date BETWEEN TRAIN_START AND VALIDATION_END`` and a hard guard asserts no
  row has ``date >= TEST_START``.
* Decile bucket boundaries are learned on TRAIN only and reused verbatim on
  VALIDATION (pooled edges reused across all stocks; per-stock edges reused for
  that same stock on validation).
* Discovery and ranking happen on TRAIN only. VALIDATION is used solely to
  check persistence, never to select indicators.
* Cross-sectional awareness: pooled (naive, concatenate-all-stocks) results are
  reported, but the primary correlation metric is the mean of per-stock
  correlations, which is robust to the differing price scales of the stocks.
  Level indicators (vwap, sma_*, candle_range/body, volume_sma_20) are flagged so
  naive-pooled correlations for them are read with their cross-sectional caveat.
* Statistical significance is reported (correlation p-value, decile-bucket
  ANOVA F/p) but is explicitly NOT evidence of profitability; with ~18M TRAIN
  observations even negligible effects can be "significant", so effect size and
  out-of-sample persistence matter far more than p-values.

Run from the project root::

    python -m src.analysis.predictive_analysis
    python -m src.analysis.predictive_analysis --limit-stocks 5   # quick smoke test

No third-party stats dependency is required (scipy is not installed); p-values
use the stdlib ``math.erfc`` (normal) and a self-contained incomplete-gamma
routine (chi-square). Only pandas / numpy / duckdb are used (already in the
project stack).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from src.data.build_bar_indicators import INDICATOR_COLUMNS
from src.data.build_targets import TARGET_COLUMNS
from src.data.date_splits import (
    SPLIT_RANGES,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from src.data.nifty100 import NIFTY100_FALLBACK

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "day4_predictive_analysis"

# Indicators stored in absolute price / volume units. Their naive-pooled
# correlation across the stock universe is partly driven by cross-sectional price-scale
# differences and is therefore not directly comparable to per-stock correlations.
LEVEL_INDICATORS = {
    "vwap",
    "sma_5",
    "sma_20",
    "sma_50",
    "sma_200",
    "candle_range",
    "candle_body",
    "volume_sma_20",
}

# Inclusive SQL date literals. TRAIN and VALIDATION only -- TEST is excluded.
SPLIT_SQL = {
    "train": (f"DATE '{TRAIN_START.isoformat()}'", f"DATE '{TRAIN_END.isoformat()}'"),
    "validation": (
        f"DATE '{VALIDATION_START.isoformat()}'",
        f"DATE '{VALIDATION_END.isoformat()}'",
    ),
}

# Latest date this pipeline is ever allowed to see (hard guard vs TEST leakage).
MAX_ALLOWED_DATE = VALIDATION_END

TIME_OF_DAY_PERIODS = [
    ("p1_0915-1000", "TIME '09:15:00'", "TIME '10:00:00'"),
    ("p2_1000-1100", "TIME '10:00:00'", "TIME '11:00:00'"),
    ("p3_1100-1200", "TIME '11:00:00'", "TIME '12:00:00'"),
    ("p4_1200-1300", "TIME '12:00:00'", "TIME '13:00:00'"),
    ("p5_1300-1400", "TIME '13:00:00'", "TIME '14:00:00'"),
    ("p6_1400-1530", "TIME '14:00:00'", "TIME '15:30:00'"),
]

NEED_COLS = ["timestamp"] + INDICATOR_COLUMNS + TARGET_COLUMNS
_SQRT2 = math.sqrt(2.0)

# Guard: targets must never appear among the indicator inputs.
assert not any(c.startswith("future_return") for c in INDICATOR_COLUMNS), (
    "future_return_* must never be used as an indicator input"
)

# ---------------------------------------------------------------------------
# Small statistics helpers (no scipy; stdlib only).
# ---------------------------------------------------------------------------
def scale_type(indicator: str) -> str:
    return "level" if indicator in LEVEL_INDICATORS else "ratio"


def _normal_two_sided_p(z: float) -> float:
    """2 * P(Z > |z|) for a standard normal, via stdlib erfc."""
    if not np.isfinite(z):
        return float("nan")
    return float(math.erfc(abs(z) / _SQRT2))


def pearson_pvalue(r: float, n: int) -> float:
    """Approx two-sided p-value for Pearson r under H0: rho = 0.

    t = r * sqrt((n-2)/(1-r^2)); for n in the millions df ~ infty so t ~ Normal.
    """
    if n <= 2 or not np.isfinite(r):
        return float("nan")
    denom = 1.0 - r * r
    if denom <= 0.0:
        return 0.0 if abs(r) >= 1.0 else float("nan")
    t = abs(r) * math.sqrt((n - 2) / denom)
    return _normal_two_sided_p(t)


def _gammp_gammq(a: float, x: float) -> tuple[float, float]:
    """Regularized lower (P) and upper (Q) incomplete gamma (NR 6.2)."""
    if x < 0.0 or a <= 0.0:
        return float("nan"), float("nan")
    if x == 0.0:
        return 0.0, 1.0
    lg = math.lgamma(a)
    if x < a + 1.0:  # series expansion
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(2000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        p = s * math.exp(-x + a * math.log(x) - lg)
        return p, 1.0 - p
    tiny = 1e-300  # continued fraction (Lentz)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = h * math.exp(-x + a * math.log(x) - lg)
    return 1.0 - q, q


def chi2_sf(x: float, df: int) -> float:
    """Survival P(chi^2_df > x) = Q(df/2, x/2)."""
    if not np.isfinite(x):
        return float("nan")
    if x < 0.0:
        return 1.0
    _, q = _gammp_gammq(df / 2.0, x / 2.0)
    return q


def anova_f_p(bucket_n, bucket_mean, bucket_sumsq) -> tuple[float, float]:
    """One-way ANOVA F across buckets; p via chi^2 approx (k-1)F ~ chi^2(k-1)."""
    k = len(bucket_n)
    n = sum(bucket_n)
    if k < 2 or n - k <= 0:
        return float("nan"), float("nan")
    total = sum(ni * mi for ni, mi in zip(bucket_n, bucket_mean))
    grand = total / n
    ssb = sum(ni * (mi - grand) ** 2 for ni, mi in zip(bucket_n, bucket_mean))
    ssw = sum(
        sq - ni * mi * mi
        for sq, ni, mi in zip(bucket_sumsq, bucket_n, bucket_mean)
        if np.isfinite(sq) and np.isfinite(mi)
    )
    if ssw <= 0 or not np.isfinite(ssw):
        return float("nan"), float("nan")
    f = (ssb / (k - 1)) / (ssw / (n - k))
    return float(f), float(chi2_sf((k - 1) * f, k - 1))


# ---------------------------------------------------------------------------
# File discovery + per-stock loading (TRAIN+VALIDATION only; never TEST).
# ---------------------------------------------------------------------------
def discover_target_files() -> list[tuple[str, Path]]:
    """Return (symbol, path) for every processed target parquet, sorted.

    Prefer the NIFTY100 fallback canonical symbols; keep any extras.
    """
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


def files_sql(files: list[tuple[str, Path]]) -> str:
    """A DuckDB read_parquet list-literal of the discovered file paths."""
    inner = ", ".join("'" + str(p).replace("'", "''") + "'" for _, p in files)
    return "[" + inner + "]"


def load_stock(symbol: str, path: Path) -> pd.DataFrame | None:
    """Load one stock's TRAIN + VALIDATION rows (never TEST) with all needed cols."""
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


def _assign_buckets(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Map values into decile buckets 1..10 using TRAIN-fitted edges (0 = missing).

    ``searchsorted(side='right')`` matches the ``CASE WHEN x < edge`` ladder used
    for pooled deciles (a value equal to an edge goes to the higher bucket).
    """
    out = np.zeros(len(values), dtype="int16")
    v = values.to_numpy(dtype="float64")
    m = ~np.isnan(v)
    out[m] = np.searchsorted(edges, v[m], side="right") + 1
    return pd.Series(out, index=values.index)

# ---------------------------------------------------------------------------
# Phase A: per-stock (streaming; robust to cross-sectional price scales).
# ---------------------------------------------------------------------------
def per_stock_correlations(symbol: str, df: pd.DataFrame) -> list[dict]:
    """Pearson + Spearman per (indicator, target) within a single stock."""
    rows: list[dict] = []
    for split in ("train", "validation"):
        sub = df[df["_split"] == split]
        if sub.empty:
            continue
        data = sub[INDICATOR_COLUMNS + TARGET_COLUMNS].astype("float64")
        pcorr = data.corr(method="pearson")
        scorr = data.corr(method="spearman")
        for ind in INDICATOR_COLUMNS:
            for tgt in TARGET_COLUMNS:
                n = int(sub[[ind, tgt]].dropna().shape[0])
                rows.append(
                    {
                        "symbol": symbol,
                        "indicator": ind,
                        "target": tgt,
                        "split": split,
                        "n_obs": n,
                        "pearson": float(pcorr.loc[ind, tgt]),
                        "spearman": float(scorr.loc[ind, tgt]),
                    }
                )
    return rows


def per_stock_deciles(
    symbol: str, df: pd.DataFrame, acc: dict
) -> dict[str, np.ndarray]:
    """Per-stock decile buckets; each stock's TRAIN edges reused on its val rows.

    Accumulates cross-stock weighted statistics into ``acc`` keyed by
    (indicator, target, split, bucket) -> {n, sum_wmean, sum_pos, n_stocks}.
    Returns the stock's TRAIN edges (for reproducibility/inspection).
    """
    edges_map: dict[str, np.ndarray] = {}
    train = df[df["_split"] == "train"]
    val = df[df["_split"] == "validation"]
    for ind in INDICATOR_COLUMNS:
        tvals = train[ind].dropna()
        if len(tvals) < 20:
            continue
        edges = np.asarray(
            tvals.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            dtype="float64",
        )
        edges_map[ind] = edges
        for split, sdf in (("train", train), ("validation", val)):
            buckets = _assign_buckets(sdf[ind], edges)
            gd = pd.concat(
                [buckets.rename("b"), sdf[TARGET_COLUMNS].astype("float64")], axis=1
            )
            gd = gd[gd["b"] > 0]
            if gd.empty:
                continue
            for tgt in TARGET_COLUMNS:
                sub = gd[["b", tgt]].dropna(subset=[tgt])
                if sub.empty:
                    continue
                g_n = sub.groupby("b")[tgt].count()
                g_mean = sub.groupby("b")[tgt].mean()
                g_pos = (sub[tgt] > 0).groupby(sub["b"]).sum()
                for k in range(1, 11):
                    n = int(g_n.get(k, 0))
                    if n == 0:
                        continue
                    m = float(g_mean.get(k, float("nan")))
                    pos = int(g_pos.get(k, 0))
                    key = (ind, tgt, split, k)
                    a = acc.setdefault(
                        key, {"n": 0, "sum_wmean": 0.0, "sum_pos": 0, "n_stocks": 0}
                    )
                    a["n"] += n
                    if not np.isnan(m):
                        a["sum_wmean"] += m * n
                    a["sum_pos"] += pos
                    a["n_stocks"] += 1
    return edges_map


def finalize_per_stock_deciles(acc: dict) -> pd.DataFrame:
    rows = []
    for (ind, tgt, split, bk), a in acc.items():
        n = a["n"]
        wmean = a["sum_wmean"] / n if n else float("nan")
        pos = a["sum_pos"]
        rows.append(
            {
                "indicator": ind,
                "target": tgt,
                "split": split,
                "method": "per_stock",
                "bucket": bk,
                "n": n,
                "mean": wmean,
                "median": float("nan"),
                "pos_count": pos,
                "pos_pct": pos / n if n else float("nan"),
                "n_stocks": a["n_stocks"],
            }
        )
    cols = ["indicator", "target", "split", "method", "bucket", "n",
            "mean", "median", "pos_count", "pos_pct", "n_stocks"]
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["indicator", "target", "split", "bucket"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase B: pooled (naive, concatenate-all-stocks) via DuckDB on the parquet glob.
# Reported alongside Phase A; the cross-sectionally-robust per-stock mean is the
# primary correlation metric.
# ---------------------------------------------------------------------------
def _safe_float(x: Any) -> float:
    return float(x) if x is not None else float("nan")


def pooled_target_distributions(files_sql_str: str) -> list[dict]:
    """Exact TRAIN/VALIDATION target distribution (incl. median, pos/neg counts)."""
    rows: list[dict] = []
    for split in ("train", "validation"):
        s0, s1 = SPLIT_SQL[split]
        for tgt in TARGET_COLUMNS:
            q = f"""
                SELECT count({tgt}) AS c, avg({tgt}) AS m, median({tgt}) AS md,
                       stddev_samp({tgt}) AS sd, min({tgt}) AS mn, max({tgt}) AS mx,
                       count(*) FILTER (WHERE {tgt} > 0) AS pos,
                       count(*) FILTER (WHERE {tgt} < 0) AS neg
                FROM read_parquet({files_sql_str})
                WHERE CAST(timestamp AS DATE) BETWEEN {s0} AND {s1}
            """
            r = duckdb.execute(q).fetchone()
            cnt = int(r[0])
            rows.append(
                {
                    "target": tgt,
                    "split": split,
                    "count": cnt,
                    "mean": _safe_float(r[1]),
                    "median": _safe_float(r[2]),
                    "std": _safe_float(r[3]),
                    "min": _safe_float(r[4]),
                    "max": _safe_float(r[5]),
                    "positive_count": int(r[6]),
                    "negative_count": int(r[7]),
                    "positive_pct": int(r[6]) / cnt if cnt else float("nan"),
                    "negative_pct": int(r[7]) / cnt if cnt else float("nan"),
                }
            )
    return rows


def pooled_correlations(files_sql_str: str) -> list[dict]:
    """Naive pooled Pearson + Spearman per (indicator, target) for each split.

    Spearman needs ranks materialized in a subquery before corr() (DuckDB does
    not allow rank() inside an aggregate). Null pairs are dropped per target.
    """
    rows: list[dict] = []
    for split in ("train", "validation"):
        s0, s1 = SPLIT_SQL[split]
        for ind in INDICATOR_COLUMNS:
            q = f"""
            WITH b  AS (
                SELECT {ind} AS x, future_return_5m AS t5, future_return_15m AS t15,
                              future_return_30m AS t30, future_return_60m AS t60
                FROM read_parquet({files_sql_str})
                WHERE CAST(timestamp AS DATE) BETWEEN {s0} AND {s1}
            ),
            s5  AS (SELECT x, t5  FROM b WHERE x IS NOT NULL AND t5  IS NOT NULL),
            s15 AS (SELECT x, t15 FROM b WHERE x IS NOT NULL AND t15 IS NOT NULL),
            s30 AS (SELECT x, t30 FROM b WHERE x IS NOT NULL AND t30 IS NOT NULL),
            s60 AS (SELECT x, t60 FROM b WHERE x IS NOT NULL AND t60 IS NOT NULL),
            r5  AS (SELECT rank() OVER (ORDER BY x) AS rx, rank() OVER (ORDER BY t5)  AS rt FROM s5),
            r15 AS (SELECT rank() OVER (ORDER BY x) AS rx, rank() OVER (ORDER BY t15) AS rt FROM s15),
            r30 AS (SELECT rank() OVER (ORDER BY x) AS rx, rank() OVER (ORDER BY t30) AS rt FROM s30),
            r60 AS (SELECT rank() OVER (ORDER BY x) AS rx, rank() OVER (ORDER BY t60) AS rt FROM s60)
            SELECT
              (SELECT count(*) FROM s5)  AS n5,
              (SELECT corr(x, t5)  FROM s5)  AS p5,  (SELECT corr(rx, rt) FROM r5)  AS sp5,
              (SELECT count(*) FROM s15) AS n15,
              (SELECT corr(x, t15) FROM s15) AS p15, (SELECT corr(rx, rt) FROM r15) AS sp15,
              (SELECT count(*) FROM s30) AS n30,
              (SELECT corr(x, t30) FROM s30) AS p30, (SELECT corr(rx, rt) FROM r30) AS sp30,
              (SELECT count(*) FROM s60) AS n60,
              (SELECT corr(x, t60) FROM s60) AS p60, (SELECT corr(rx, rt) FROM r60) AS sp60
            """
            r = duckdb.execute(q).fetchone()
            # SELECT order per target: count, pearson, spearman -> [n,p,sp]*4
            for i, tgt in enumerate(TARGET_COLUMNS):
                base = i * 3
                rows.append(
                    {
                        "indicator": ind,
                        "target": tgt,
                        "split": split,
                        "method": "pooled_naive",
                        "scale_type": scale_type(ind),
                        "n_obs": int(r[base]),
                        "pearson": _safe_float(r[base + 1]),
                        "spearman": _safe_float(r[base + 2]),
                    }
                )
    return rows


def pooled_edges(files_sql_str: str) -> dict[str, np.ndarray]:
    """TRAIN decile edges (9 quantiles) per indicator, reused on VALIDATION."""
    s0, s1 = SPLIT_SQL["train"]
    edges: dict[str, np.ndarray] = {}
    for ind in INDICATOR_COLUMNS:
        q = f"""
            SELECT quantile_cont({ind}, 0.1), quantile_cont({ind}, 0.2),
                   quantile_cont({ind}, 0.3), quantile_cont({ind}, 0.4),
                   quantile_cont({ind}, 0.5), quantile_cont({ind}, 0.6),
                   quantile_cont({ind}, 0.7), quantile_cont({ind}, 0.8),
                   quantile_cont({ind}, 0.9)
            FROM read_parquet({files_sql_str})
            WHERE CAST(timestamp AS DATE) BETWEEN {s0} AND {s1} AND {ind} IS NOT NULL
        """
        r = duckdb.execute(q).fetchone()
        edges[ind] = np.array([_safe_float(x) for x in r], dtype="float64")
    return edges


def _build_bucket_case(ind: str, edges: np.ndarray) -> str:
    """SQL CASE mapping an indicator to 1..10 using TRAIN edges; NULL -> NULL."""
    parts = [f"WHEN {ind} IS NULL THEN NULL"]
    for i, e in enumerate(edges, start=1):
        parts.append(f"WHEN {ind} < {float(e):.17g} THEN {i}")
    parts.append("ELSE 10")
    return "CASE " + " ".join(parts) + " END"


def pooled_deciles(
    files_sql_str: str, edges: dict[str, np.ndarray]
) -> tuple[list[dict], list[dict]]:
    """Pooled decile buckets per (indicator, target) + ANOVA F/p significance.

    Bucket boundaries are the TRAIN edges reused verbatim on VALIDATION.
    """
    rows: list[dict] = []
    sig: list[dict] = []
    short = {"future_return_5m": "5", "future_return_15m": "15",
             "future_return_30m": "30", "future_return_60m": "60"}
    for split in ("train", "validation"):
        s0, s1 = SPLIT_SQL[split]
        for ind in INDICATOR_COLUMNS:
            case = _build_bucket_case(ind, edges[ind])
            q = f"""
            WITH b AS (
                SELECT {case} AS bucket,
                       future_return_5m AS t5,  future_return_15m AS t15,
                       future_return_30m AS t30, future_return_60m AS t60
                FROM read_parquet({files_sql_str})
                WHERE CAST(timestamp AS DATE) BETWEEN {s0} AND {s1}
            )
            SELECT bucket,
              count(t5)  AS n5,  avg(t5)  AS mean5,  median(t5)  AS med5,
              sum(t5)    AS sum5,  sum(t5 * t5)  AS ss5,  count(*) FILTER (WHERE t5  > 0) AS pos5,
              count(t15) AS n15, avg(t15) AS mean15, median(t15) AS med15,
              sum(t15)   AS sum15, sum(t15 * t15) AS ss15, count(*) FILTER (WHERE t15 > 0) AS pos15,
              count(t30) AS n30, avg(t30) AS mean30, median(t30) AS med30,
              sum(t30)   AS sum30, sum(t30 * t30) AS ss30, count(*) FILTER (WHERE t30 > 0) AS pos30,
              count(t60) AS n60, avg(t60) AS mean60, median(t60) AS med60,
              sum(t60)   AS sum60, sum(t60 * t60) AS ss60, count(*) FILTER (WHERE t60 > 0) AS pos60
            FROM b WHERE bucket IS NOT NULL GROUP BY bucket ORDER BY bucket
            """
            res = duckdb.execute(q).fetchdf()
            for tgt in TARGET_COLUMNS:
                sfx = short[tgt]
                dec_rows, bn, bm, bss = [], [], [], []
                for _, rr in res.iterrows():
                    n = int(rr[f"n{sfx}"])
                    mean = _safe_float(rr[f"mean{sfx}"])
                    pos = int(rr[f"pos{sfx}"])
                    bk = int(rr["bucket"])
                    dec_rows.append(
                        {
                            "indicator": ind, "target": tgt, "split": split,
                            "method": "pooled", "bucket": bk, "n": n,
                            "mean": mean, "median": _safe_float(rr[f"med{sfx}"]),
                            "pos_count": pos,
                            "pos_pct": pos / n if n else float("nan"),
                        }
                    )
                    if n > 0:
                        bn.append(n)
                        bm.append(mean if np.isfinite(mean) else 0.0)
                        bss.append(_safe_float(rr[f"ss{sfx}"]))
                rows.extend(dec_rows)
                f, p = anova_f_p(bn, bm, bss)
                finite_means = [x for x in bm if np.isfinite(x)]
                bmin = min(finite_means) if finite_means else float("nan")
                bmax = max(finite_means) if finite_means else float("nan")
                sig.append(
                    {
                        "indicator": ind, "target": tgt, "split": split,
                        "method": "pooled", "k": len(bn), "n_total": int(sum(bn)),
                        "anova_f": f, "anova_p": p,
                        "bucket_mean_min": bmin, "bucket_mean_max": bmax,
                        "bucket_mean_spread": (
                            bmax - bmin if np.isfinite(bmin) and np.isfinite(bmax)
                            else float("nan")
                        ),
                    }
                )
    return rows, sig


# ---------------------------------------------------------------------------
# Phase C: aggregate, rank (TRAIN only), check VALIDATION persistence.
# ---------------------------------------------------------------------------
def aggregate_per_stock_corr(per_stock_rows: list[dict]) -> pd.DataFrame:
    """Cross-stock mean/median + sign-consistency of per-stock correlations."""
    df = pd.DataFrame(per_stock_rows)
    out: list[dict] = []
    for (ind, tgt, split), g in df.groupby(["indicator", "target", "split"]):
        p = g["pearson"].dropna()
        s = g["spearman"].dropna()
        pmean = float(p.mean()) if len(p) else float("nan")
        pmed = float(p.median()) if len(p) else float("nan")
        smean = float(s.mean()) if len(s) else float("nan")
        smed = float(s.median()) if len(s) else float("nan")
        frac_p = (
            float((np.sign(p) == np.sign(pmean)).mean())
            if len(p) and pmean != 0 else float("nan")
        )
        frac_s = (
            float((np.sign(s) == np.sign(smean)).mean())
            if len(s) and smean != 0 else float("nan")
        )
        out.append(
            {
                "indicator": ind, "target": tgt, "split": split,
                "scale_type": scale_type(ind),
                "pearson_per_stock_mean": pmean,
                "pearson_per_stock_median": pmed,
                "spearman_per_stock_mean": smean,
                "spearman_per_stock_median": smed,
                "frac_consistent_pearson": frac_p,
                "frac_consistent_spearman": frac_s,
                "n_stocks": int(len(g)),
                "n_obs": int(g["n_obs"].sum()),
            }
        )
    return pd.DataFrame(out)


def build_corr_table(
    pooled_rows: list[dict], per_stock_agg: pd.DataFrame
) -> pd.DataFrame:
    """Wide per (indicator, target, split) table: naive pooled + per-stock stats."""
    p = pd.DataFrame(pooled_rows)
    p = p.rename(
        columns={
            "pearson": "pearson_pooled_naive",
            "spearman": "spearman_pooled_naive",
            "n_obs": "n_obs_pooled",
        }
    )
    m = p.merge(
        per_stock_agg.drop(columns=["scale_type"]),
        on=["indicator", "target", "split"], how="left",
    )
    m["pearson_pooled_pvalue"] = m.apply(
        lambda r: pearson_pvalue(r["pearson_pooled_naive"], r["n_obs_pooled"]),
        axis=1,
    )
    m["abs_train"] = m["pearson_per_stock_mean"].abs()
    front = [
        "indicator", "target", "split", "scale_type",
        "n_obs_pooled", "n_stocks",
        "pearson_pooled_naive", "spearman_pooled_naive", "pearson_pooled_pvalue",
        "pearson_per_stock_mean", "pearson_per_stock_median",
        "spearman_per_stock_mean", "spearman_per_stock_median",
        "frac_consistent_pearson", "frac_consistent_spearman", "n_obs",
    ]
    return m[front + ["abs_train"]]


def rank_strongest(
    corr: pd.DataFrame, sig: pd.DataFrame, top: int = 20
) -> pd.DataFrame:
    """Rank TRAIN relationships by |per-stock-mean Pearson|; attach val persistence."""
    sig = sig[["indicator", "target", "split", "anova_f", "anova_p",
               "bucket_mean_spread"]].rename(
        columns={"bucket_mean_spread": "decile_mean_spread"}
    )
    tr = corr[corr["split"] == "train"].set_index(["indicator", "target"])
    va = corr[corr["split"] == "validation"].set_index(["indicator", "target"])
    st = sig[sig["split"] == "train"].set_index(["indicator", "target"])
    sv = sig[sig["split"] == "validation"].set_index(["indicator", "target"])
    rows: list[dict] = []
    for (ind, tgt), t in tr.iterrows():
        v = va.loc[(ind, tgt)] if (ind, tgt) in va.index else None
        tp = t["pearson_per_stock_mean"]
        vp = v["pearson_per_stock_mean"] if v is not None else float("nan")
        rows.append(
            {
                "indicator": ind, "target": tgt, "scale_type": t["scale_type"],
                "train_pearson_pooled": t["pearson_pooled_naive"],
                "train_spearman_pooled": t["spearman_pooled_naive"],
                "train_pearson_psm": tp,
                "train_spearman_psm": t["spearman_per_stock_mean"],
                "train_frac_consistent": t["frac_consistent_pearson"],
                "train_n_obs": t["n_obs_pooled"],
                "train_n_stocks": t["n_stocks"],
                "val_pearson_pooled": (
                    v["pearson_pooled_naive"] if v is not None else float("nan")
                ),
                "val_spearman_pooled": (
                    v["spearman_pooled_naive"] if v is not None else float("nan")
                ),
                "val_pearson_psm": vp,
                "val_spearman_psm": (
                    v["spearman_per_stock_mean"] if v is not None else float("nan")
                ),
                "val_frac_consistent": (
                    v["frac_consistent_pearson"] if v is not None else float("nan")
                ),
                "val_n_obs": (
                    v["n_obs_pooled"] if v is not None else float("nan")
                ),
                "sign_persists": (
                    bool(np.sign(tp) == np.sign(vp))
                    if np.isfinite(tp) and np.isfinite(vp)
                    and tp != 0 and vp != 0 else False
                ),
                "val_magnitude_ratio": (
                    abs(vp) / abs(tp) if tp else float("nan")
                ),
                "train_anova_f": st.loc[(ind, tgt), "anova_f"]
                if (ind, tgt) in st.index else float("nan"),
                "train_anova_p": st.loc[(ind, tgt), "anova_p"]
                if (ind, tgt) in st.index else float("nan"),
                "val_anova_f": sv.loc[(ind, tgt), "anova_f"]
                if (ind, tgt) in sv.index else float("nan"),
                "val_anova_p": sv.loc[(ind, tgt), "anova_p"]
                if (ind, tgt) in sv.index else float("nan"),
            }
        )
    df = pd.DataFrame(rows)
    df["abs_train_psm"] = df["train_pearson_psm"].abs()
    df = df.sort_values("abs_train_psm", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.head(top)


def strongest_per_stock(
    per_stock_rows: list[dict], top_pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Per-stock correlation detail (train + val) for the top relationships."""
    df = pd.DataFrame(per_stock_rows)
    df = df[df[["indicator", "target"]].apply(tuple, axis=1).isin(top_pairs)]
    piv = df.pivot_table(
        index=["symbol", "indicator", "target"], columns="split",
        values=["pearson", "spearman", "n_obs"], aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return piv.reset_index().sort_values(
        ["indicator", "target", "symbol"]
    ).reset_index(drop=True)


def _tod_case() -> str:
    parts = []
    for label, lo, hi in TIME_OF_DAY_PERIODS:
        parts.append(
            f"WHEN CAST(timestamp AS TIME) >= {lo} "
            f"AND CAST(timestamp AS TIME) < {hi} THEN '{label}'"
        )
    return "CASE " + " ".join(parts) + " ELSE 'other' END"


def time_of_day_analysis(
    files_sql_str: str, top_pairs: list[tuple[str, str]], max_pairs: int = 8
) -> pd.DataFrame:
    """Pooled Pearson of top relationships per intraday period (TRAIN + VAL)."""
    rows: list[dict] = []
    case = _tod_case()
    for (ind, tgt) in top_pairs[:max_pairs]:
        for split in ("train", "validation"):
            s0, s1 = SPLIT_SQL[split]
            q = f"""
            WITH b AS (
                SELECT {ind} AS x, {tgt} AS y, {case} AS period
                FROM read_parquet({files_sql_str})
                WHERE CAST(timestamp AS DATE) BETWEEN {s0} AND {s1}
            )
            SELECT period, count(*) AS n, corr(x, y) AS pearson
            FROM b WHERE x IS NOT NULL AND y IS NOT NULL
            GROUP BY period ORDER BY period
            """
            res = duckdb.execute(q).fetchdf()
            for _, rr in res.iterrows():
                rows.append(
                    {
                        "indicator": ind, "target": tgt, "split": split,
                        "period": rr["period"], "n_obs": int(rr["n"]),
                        "pearson": _safe_float(rr["pearson"]),
                    }
                )
    cols = ["indicator", "target", "split", "period", "n_obs", "pearson"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Output writing.
# ---------------------------------------------------------------------------
def _write_csv(df: pd.DataFrame, path: Path, float_fmt: str = "%.10g") -> None:
    df.to_csv(path, index=False, float_format=float_fmt)
    print(f"  wrote {path.name}  ({len(df)} rows)")


def write_outputs(
    out_dir: Path,
    target_dist: list[dict],
    corr_table: pd.DataFrame,
    per_stock_corr_df: pd.DataFrame,
    per_stock_dec_df: pd.DataFrame,
    pooled_dec_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    strongest: pd.DataFrame,
    strongest_ps: pd.DataFrame,
    tod: pd.DataFrame,
    summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        _write_csv(corr_table[corr_table["split"] == split].drop(
            columns=["abs_train"]), out_dir / f"indicator_correlations_{split}.csv")
        _write_csv(per_stock_corr_df[per_stock_corr_df["split"] == split],
                   out_dir / f"indicator_correlations_per_stock_{split}.csv")
        _write_csv(
            pd.concat(
                [
                    pooled_dec_df[pooled_dec_df["split"] == split],
                    per_stock_dec_df[per_stock_dec_df["split"] == split],
                ], ignore_index=True
            ),
            out_dir / f"decile_analysis_{split}.csv")
        _write_csv(sig_df[sig_df["split"] == split],
                   out_dir / f"decile_significance_{split}.csv")
    _write_csv(pd.DataFrame(target_dist), out_dir / "target_distributions.csv")
    _write_csv(strongest.drop(columns=["abs_train_psm"]),
               out_dir / "strongest_relationships.csv")
    _write_csv(strongest_ps, out_dir / "strongest_per_stock.csv")
    _write_csv(tod, out_dir / "time_of_day_analysis.csv")
    with (out_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  wrote run_summary.json")


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def run(limit_stocks: int | None = None, top: int = 20) -> dict:
    t0 = time.time()
    files = discover_target_files()
    if limit_stocks:
        files = files[:limit_stocks]
    print(f"Discovered {len(files)} stock target files.")
    files_sql_str = files_sql(files)

    # ---- Phase A: per-stock streaming -------------------------------------
    print("Phase A: per-stock correlations + deciles ...")
    per_stock_corr_rows: list[dict] = []
    decile_acc: dict = {}
    stocks_ok, stocks_fail = 0, 0
    rows_train, rows_val = 0, 0
    a0 = time.time()
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
        rows_train += int((df["_split"] == "train").sum())
        rows_val += int((df["_split"] == "validation").sum())
        per_stock_corr_rows.extend(per_stock_correlations(sym, df))
        per_stock_deciles(sym, df, decile_acc)
        stocks_ok += 1
        if i % 10 == 0 or i == len(files):
            print(f"  [{i:>3}/{len(files)}] {sym} done "
                  f"({time.time() - a0:.0f}s elapsed)")
        del df
        if i % 20 == 0:
            gc.collect()
    per_stock_corr_df = pd.DataFrame(per_stock_corr_rows)
    per_stock_dec_df = finalize_per_stock_deciles(decile_acc)

    # ---- Phase B: pooled (DuckDB) -----------------------------------------
    print("Phase B: pooled target distributions ...")
    target_dist = pooled_target_distributions(files_sql_str)
    print("Phase B: pooled correlations ...")
    pooled_corr_rows = pooled_correlations(files_sql_str)
    print("Phase B: pooled decile edges (TRAIN) + buckets + significance ...")
    edges = pooled_edges(files_sql_str)
    pooled_dec_rows, sig_rows = pooled_deciles(files_sql_str, edges)

    # ---- Phase C: aggregate / rank ---------------------------------------
    per_stock_agg = aggregate_per_stock_corr(per_stock_corr_rows)
    corr_table = build_corr_table(pooled_corr_rows, per_stock_agg)
    sig_df = pd.DataFrame(sig_rows)
    strongest = rank_strongest(corr_table, sig_df, top=top)
    top_pairs = [
        (r["indicator"], r["target"]) for _, r in strongest.iterrows()
    ]
    strongest_ps = strongest_per_stock(per_stock_corr_rows, top_pairs)
    print("Phase C: time-of-day slices for top relationships ...")
    tod = time_of_day_analysis(files_sql_str, top_pairs)
    pooled_dec_df = pd.DataFrame(pooled_dec_rows)

    # ---- Summary ---------------------------------------------------------
    n_total = rows_train + rows_val
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stocks_processed": stocks_ok,
        "stocks_failed": stocks_fail,
        "n_indicators": len(INDICATOR_COLUMNS),
        "n_targets": len(TARGET_COLUMNS),
        "splits_used": ["train", "validation"],
        "test_ever_read": False,
        "max_allowed_date": str(MAX_ALLOWED_DATE),
        "train_rows_loaded": rows_train,
        "validation_rows_loaded": rows_val,
        "total_rows_loaded": n_total,
        "elapsed_seconds": round(time.time() - t0, 1),
        "top_relationships": [
            {
                "rank": int(r["rank"]),
                "indicator": r["indicator"],
                "target": r["target"],
                "train_pearson_psm": r["train_pearson_psm"],
                "val_pearson_psm": r["val_pearson_psm"],
                "sign_persists": bool(r["sign_persists"]),
                "val_magnitude_ratio": r["val_magnitude_ratio"],
            }
            for _, r in strongest.iterrows()
        ],
    }

    out_dir = REPORTS_DIR
    print("Writing outputs ...")
    write_outputs(
        out_dir, target_dist, corr_table, per_stock_corr_df,
        per_stock_dec_df, pooled_dec_df, sig_df,
        strongest, strongest_ps, tod, summary,
    )

    print("\n=== Day 4 predictive analysis summary ===")
    print(f"stocks processed : {stocks_ok}  (failed: {stocks_fail})")
    print(f"TRAIN rows       : {rows_train:,}")
    print(f"VALIDATION rows  : {rows_val:,}")
    print(f"TEST rows read   : 0  (never read)")
    print(f"elapsed          : {summary['elapsed_seconds']}s")
    print("\nTop 10 TRAIN relationships (|per-stock-mean Pearson|):")
    for _, r in strongest.head(10).iterrows():
        print(
            f"  {int(r['rank']):>2}. {r['indicator']:<18} -> {r['target']:<16} "
            f"train_psm={r['train_pearson_psm']:+.4f}  "
            f"val_psm={r['val_pearson_psm']:+.4f}  "
            f"persist={'Y' if r['sign_persists'] else 'N'}  "
            f"mag_ratio={r['val_magnitude_ratio']:.2f}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--limit-stocks", type=int, default=None,
        help="Process only the first N stocks (smoke test). Pooled DuckDB "
             "queries still read only these N files.",
    )
    p.add_argument("--top", type=int, default=20,
                   help="Number of strongest relationships to report (default 20).")
    args = p.parse_args(argv)
    run(limit_stocks=args.limit_stocks, top=args.top)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
