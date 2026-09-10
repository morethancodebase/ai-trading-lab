"""Day 6: cost-aware backtesting engine for Day 5 directional signals.

Day 5 found weak but persistent directional signals (LONG / SHORT / NO SIGNAL)
from bar indicators and scored them against the 5-minute future return *without*
trading costs. Day 6 asks: do those signals survive realistic Indian equity
intraday trading costs when executed as actual 5-minute round-trip trades?

Execution model (no look-ahead, no overnight)
---------------------------------------------
* Signal computed at close of candle T using backward-looking indicators -- the
  *same* ``signal_direction`` from Day 5, reused verbatim by importing it.
* Entry is at the open of the next candle, open[T+1] (the signal is only knowable
  after close[T], so you cannot trade at close[T]).
* Held 5 trading minutes; exit at close[T+5] (candles T+1..T+5).
* One position per stock: while a 5-minute position is open, later signals on that
  stock are ignored. A new signal is eligible once the previous position has
  exited (signal bar >= last exit bar), so back-to-back trades are allowed.
* No overnight / cross-session holds: a signal is only traded if candles T+1..T+5
  are contiguous 1-min bars (exit exactly 5 min after the signal), so a hold never
  crosses an overnight gap, the Diwali Muhurat evening session, a circuit halt, or
  a missing bar. Last-5-bar signals are skipped (matches Day 5, where
  ``future_return_5m`` is NaN there).

Costs / capital: fixed Rs notional per trade (default Rs 1,00,000); per-trade
P&L = notional * realised return - round-trip cost. This is a per-signal-edge
evaluation, NOT a capital-constrained portfolio (concurrent positions across the
100 stocks can exceed initial capital; reported as ``max_concurrent_positions``).
Two cost modes: ``zero`` (gross-edge baseline) and ``model`` (DEFAULT_COST).

Guardrails (same as Days 4-5): no look-ahead; TEST is never read (load filters
TRAIN..VALIDATION_END and raises if any row >= TEST_START 2026-03-04; summary
reports test_rows_read=0); per-stock thresholds frozen on TRAIN; a built-in
cross-check reproduces Day 5's pooled all-signals metric and compares it to Day
5's ``signal_train_vs_val.csv``.

Run::

    python -m src.backtesting.run_backtest
    python -m src.backtesting.run_backtest --limit-stocks 5 --signals C1   # smoke
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.analysis.signal_discovery import (
    CANDIDATE_INDICATORS,
    SPECS,
    _PERIOD_BOUNDS,
    _mask_block,
    compute_thresholds,
    discover_target_files,
    period_of_day,
    signal_direction,
)
from src.backtesting.costs import DEFAULT_COST, ZERO_COST, CostModel
from src.data.date_splits import (
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
)

TARGET = "future_return_5m"
HOLD_BARS = 5                       # exit after 5 trading minutes
DEFAULT_NOTIONAL = 100_000.0        # Rs 1,00,000 fixed notional per trade
DEFAULT_INITIAL_CAPITAL = 1_000_000.0  # Rs 10,00,000 equity-curve baseline
LOAD_COLS = ["timestamp", "open", "close"] + CANDIDATE_INDICATORS + [TARGET]

COST_MODELS: dict[str, CostModel] = {"zero": ZERO_COST, "model": DEFAULT_COST}

GROSS_TRADE_COLUMNS = [
    "symbol", "signal_id", "signal_name", "split", "direction",
    "signal_time", "entry_time", "exit_time", "entry_price", "exit_price",
    "hold_bars", "gross_ret", "sig_exp_ret", "notional", "gross_pnl",
]
TRADE_COLUMNS = GROSS_TRADE_COLUMNS + [
    "cost_mode", "cost_rs", "cost_bps", "net_pnl", "net_ret", "net_ret_bps",
]

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports" / "day6_backtest"
DAY5_TRAIN_VAL_CSV = ROOT / "reports" / "day5_signal_discovery" / "signal_train_vs_val.csv"


# ---------------------------------------------------------------------------
# Data loading (TRAIN + VALIDATION only, hard TEST guard -- same pattern as Day 5).
# ---------------------------------------------------------------------------
def load_stock(symbol: str, path: Path) -> pd.DataFrame | None:
    """Load one stock's TRAIN + VALIDATION rows (never TEST), needed cols only."""
    cols = ", ".join(LOAD_COLS)
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
        raise RuntimeError(f"{symbol}: TEST data leaked into backtest load")
    df["_split"] = np.where(dates <= TRAIN_END, "train", "validation")
    df["_date"] = dates
    return df
# ---------------------------------------------------------------------------
# Signal scoring (reproduces Day 5's pooled all-signals metric for cross-check).
# ---------------------------------------------------------------------------
def _all_signals_contribution(d_score: np.ndarray, fr: np.ndarray) -> dict:
    """Per-stock Day-5-style contribution over ALL scoreable signal bars.

    Matches Day 5: expected return = direction * future_return (sign included),
    accumulated as a streaming sum so the pooled mean reproduces Day 5's reported
    ``avg_ret_bps`` exactly. Day 5 set ``d[~scoreable] = 0`` before scoring.
    """
    mask = d_score != 0
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "n_long": 0, "n_short": 0, "sum_exp": 0.0, "n_hit": 0}
    exp = d_score[mask].astype("float64") * fr[mask]
    return {
        "n": n,
        "n_long": int((d_score[mask] == 1).sum()),
        "n_short": int((d_score[mask] == -1).sum()),
        "sum_exp": float(exp.sum()),
        "n_hit": int((exp > 0).sum()),
    }


def _pooled_all_signals(contribs: list[dict]) -> dict:
    n = sum(c["n"] for c in contribs)
    if n == 0:
        return {"n_signals": 0, "n_long": 0, "n_short": 0,
                "avg_exp_ret_bps": np.nan, "hit_rate": np.nan}
    return {
        "n_signals": n,
        "n_long": sum(c["n_long"] for c in contribs),
        "n_short": sum(c["n_short"] for c in contribs),
        "avg_exp_ret_bps": (sum(c["sum_exp"] for c in contribs) / n) * 1e4,
        "hit_rate": sum(c["n_hit"] for c in contribs) / n,
    }


# ---------------------------------------------------------------------------
# Trade simulation: non-overlapping, no-overnight 5-min trades per stock.
# ---------------------------------------------------------------------------
def simulate_stock(symbol: str, df: pd.DataFrame, specs: list[dict],
                   splits: list[str], notional: float):
    """Simulate gross trades for one stock across the requested specs/splits.

    Returns (trades, allsig, trading_days, rows):
      trades        : {(signal_id, split): list[tuple]}  gross trade tuples
      allsig        : {(signal_id, split): dict}          per-stock all-signal contrib
      trading_days  : {split: set[date]}
      rows          : {split: int}
    """
    trades: dict[tuple[str, str], list] = {}
    allsig: dict[tuple[str, str], dict] = {}
    trading_days: dict[str, set] = {s: set() for s in splits}
    rows: dict[str, int] = {s: 0 for s in splits}

    train = df[df["_split"] == "train"]
    thr = compute_thresholds(train)  # frozen on TRAIN (per stock)

    for split in splits:
        sdf = df[df["_split"] == split]
        if sdf.empty:
            continue
        rows[split] += int(len(sdf))
        for d in sdf["_date"].unique():
            trading_days[split].add(d)
        n = len(sdf)
        if n <= HOLD_BARS:
            zero = {"n": 0, "n_long": 0, "n_short": 0, "sum_exp": 0.0, "n_hit": 0}
            for spec in specs:
                allsig[(spec["id"], split)] = zero
            continue
        op = sdf["open"].to_numpy(dtype="float64")
        cl = sdf["close"].to_numpy(dtype="float64")
        fr = sdf[TARGET].to_numpy(dtype="float64")
        ts = pd.to_datetime(sdf["timestamp"]).to_numpy()
        scoreable = np.isfinite(fr)
        for spec in specs:
            masks = {ind: _mask_block(sdf[ind].to_numpy(dtype="float64"),
                                     thr.get(ind) if thr else None)
                     for ind in spec["inds"]}
            d = signal_direction(spec, masks)
            d_score = d.copy()
            d_score[~scoreable] = 0  # Day 5 rule: never score without a valid target
            allsig[(spec["id"], split)] = _all_signals_contribution(d_score, fr)
            sig_bars = np.flatnonzero(d != 0)
            last_exit = -1
            tk = trades.setdefault((spec["id"], split), [])
            for i in sig_bars:
                if i < last_exit:
                    continue  # position held -> skip (one position per stock)
                exit_bar = i + HOLD_BARS
                if exit_bar >= n:
                    continue
                entry_bar = i + 1
                # Contiguous 5-minute hold: entry is the very next 1-min bar and
                # exit is exactly 5 min after the signal. This blocks any hold
                # that crosses a session break (overnight, Diwali Muhurat evening
                # session, circuit halt) or spans a missing bar.
                entry_gap = (ts[entry_bar] - ts[i]).astype("timedelta64[s]").astype(np.int64)
                hold_span = (ts[exit_bar] - ts[i]).astype("timedelta64[s]").astype(np.int64)
                if entry_gap != 60 or hold_span != HOLD_BARS * 60:
                    continue
                ep = op[entry_bar]
                xp = cl[exit_bar]
                if not (np.isfinite(ep) and np.isfinite(xp) and ep > 0 and xp > 0):
                    continue
                direction = int(d[i])
                gross_ret = (xp / ep - 1.0) if direction == 1 else (ep / xp - 1.0)
                sig_exp = direction * fr[i] if np.isfinite(fr[i]) else np.nan
                tk.append((
                    symbol, spec["id"], spec["name"], split, direction,
                    ts[i], ts[entry_bar], ts[exit_bar] + np.timedelta64(60, "s"),
                    float(ep), float(xp), HOLD_BARS,
                    float(gross_ret), float(sig_exp), float(notional),
                    float(notional * gross_ret),
                ))
                last_exit = exit_bar
    return trades, allsig, trading_days, rows
# ---------------------------------------------------------------------------
# Costing and metrics.
# ---------------------------------------------------------------------------
def apply_costs(gross_df: pd.DataFrame, cost_model: CostModel,
                cost_mode_name: str, notional: float) -> pd.DataFrame:
    """Add per-trade cost / net-PnL columns to a gross-trade ledger."""
    df = gross_df.copy()
    if df.empty:
        for c in ("cost_mode", "cost_rs", "cost_bps", "net_pnl", "net_ret", "net_ret_bps"):
            df[c] = pd.Series(dtype="float64")
        df["cost_mode"] = cost_mode_name
        return df
    ratio = df["exit_price"].to_numpy(dtype="float64") / df["entry_price"].to_numpy(dtype="float64")
    entry_value = np.full(len(df), notional, dtype="float64")
    exit_value = notional * ratio
    direction = df["direction"].to_numpy(dtype="float64")
    cost = np.asarray(cost_model.round_trip_cost(direction, entry_value, exit_value),
                      dtype="float64")
    df["cost_mode"] = cost_mode_name
    df["cost_rs"] = cost
    df["cost_bps"] = cost / notional * 1e4
    df["net_pnl"] = df["gross_pnl"].to_numpy(dtype="float64") - cost
    df["net_ret"] = df["net_pnl"] / notional
    df["net_ret_bps"] = df["net_ret"] * 1e4
    return df


def _max_consecutive_true(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _max_concurrent_positions(entry_times: np.ndarray, exit_times: np.ndarray) -> int:
    """Max simultaneously-open trades (half-open [entry, exit); exits are
    processed before entries at a common instant, so touching trades are not
    counted as overlapping)."""
    events = []
    for t in entry_times:
        events.append((t, 1))   # entry
    for t in exit_times:
        events.append((t, 0))   # exit (sorts first at the same instant)
    events.sort(key=lambda x: (x[0], x[1]))
    cur = mx = 0
    for _, kind in events:
        cur += 1 if kind == 1 else -1
        if cur > mx:
            mx = cur
    return int(mx)


def compute_metrics(trades_df: pd.DataFrame, initial_capital: float,
                    n_trading_days: int) -> dict:
    empty = {
        "n_trades": 0, "n_long": 0, "n_short": 0, "gross_pnl": 0.0,
        "total_cost": 0.0, "net_pnl": 0.0, "avg_gross_ret_bps": np.nan,
        "avg_cost_bps": np.nan, "avg_net_ret_bps": np.nan, "hit_rate": np.nan,
        "hit_rate_long": np.nan, "hit_rate_short": np.nan, "profit_factor": np.nan,
        "max_dd_rs": 0.0, "max_dd_pct": 0.0, "losing_streak_max": 0,
        "trades_per_day": 0.0, "max_concurrent_positions": 0, "win_rate": np.nan,
        "avg_win_rs": np.nan, "avg_loss_rs": np.nan, "expectancy_rs": np.nan,
    }
    n = len(trades_df)
    if n == 0:
        return empty
    d = trades_df["direction"].to_numpy()
    net = trades_df["net_pnl"].to_numpy(dtype="float64")
    gross = trades_df["gross_pnl"].to_numpy(dtype="float64")
    cost = trades_df["cost_rs"].to_numpy(dtype="float64")
    win = net > 0
    n_long = int((d == 1).sum())
    n_short = int((d == -1).sum())
    long_net = net[d == 1]
    short_net = net[d == -1]
    pos = float(net[net > 0].sum())
    neg = float(net[net <= 0].sum())
    profit_factor = pos / abs(neg) if neg != 0 else np.inf
    order = np.argsort(trades_df["exit_time"].to_numpy(), kind="stable")
    net_sorted = net[order]
    cum = np.cumsum(net_sorted)
    equity = initial_capital + cum
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd_rs = float(dd.min())
    with np.errstate(divide="ignore", invalid="ignore"):
        max_dd_pct = float((dd / peak).min() * 100.0) if np.all(peak > 0) else np.nan
    losing_streak_max = _max_consecutive_true(net_sorted <= 0)
    max_concurrent = _max_concurrent_positions(
        trades_df["entry_time"].to_numpy(), trades_df["exit_time"].to_numpy())
    return {
        "n_trades": n, "n_long": n_long, "n_short": n_short,
        "gross_pnl": float(gross.sum()), "total_cost": float(cost.sum()),
        "net_pnl": float(net.sum()),
        "avg_gross_ret_bps": float(trades_df["gross_ret"].mean() * 1e4),
        "avg_cost_bps": float(trades_df["cost_bps"].mean()),
        "avg_net_ret_bps": float(trades_df["net_ret"].mean() * 1e4),
        "hit_rate": float(win.mean()),
        "hit_rate_long": float((long_net > 0).mean()) if n_long else np.nan,
        "hit_rate_short": float((short_net > 0).mean()) if n_short else np.nan,
        "profit_factor": float(profit_factor),
        "max_dd_rs": max_dd_rs, "max_dd_pct": max_dd_pct,
        "losing_streak_max": int(losing_streak_max),
        "trades_per_day": float(n / n_trading_days) if n_trading_days else np.nan,
        "max_concurrent_positions": int(max_concurrent),
        "win_rate": float(win.mean()),
        "avg_win_rs": float(net[win].mean()) if win.any() else np.nan,
        "avg_loss_rs": float(net[~win].mean()) if (~win).any() else np.nan,
        "expectancy_rs": float(net.mean()),
    }
def aggregate_by_stock(trades_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "n_trades", "n_long", "n_short", "gross_pnl", "net_pnl",
            "total_cost", "avg_gross_ret_bps", "avg_net_ret_bps", "avg_cost_bps",
            "hit_rate"]
    if trades_df.empty:
        return pd.DataFrame(columns=cols)
    df = trades_df.copy()
    df["_win"] = (df["net_pnl"] > 0).astype("int8")
    df["_long"] = (df["direction"] == 1).astype("int8")
    df["_short"] = (df["direction"] == -1).astype("int8")
    out = df.groupby("symbol", sort=True).agg(
        n_trades=("net_pnl", "count"),
        n_long=("_long", "sum"), n_short=("_short", "sum"),
        gross_pnl=("gross_pnl", "sum"), net_pnl=("net_pnl", "sum"),
        total_cost=("cost_rs", "sum"),
        avg_gross_ret_bps=("gross_ret", "mean"),
        avg_net_ret_bps=("net_ret", "mean"),
        avg_cost_bps=("cost_bps", "mean"),
        hit_rate=("_win", "mean"),
    ).reset_index()
    out["avg_gross_ret_bps"] = out["avg_gross_ret_bps"] * 1e4
    out["avg_net_ret_bps"] = out["avg_net_ret_bps"] * 1e4
    return out[cols]


def aggregate_by_time(trades_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["period", "period_name", "n_trades", "n_long", "n_short",
            "gross_pnl", "net_pnl", "avg_net_ret_bps", "hit_rate"]
    if trades_df.empty:
        return pd.DataFrame(columns=cols)
    df = trades_df.copy()
    df["_period"] = period_of_day(df["signal_time"])
    df["_win"] = (df["net_pnl"] > 0).astype("int8")
    df["_long"] = (df["direction"] == 1).astype("int8")
    df["_short"] = (df["direction"] == -1).astype("int8")
    out = df.groupby("_period", sort=True).agg(
        n_trades=("net_pnl", "count"),
        n_long=("_long", "sum"), n_short=("_short", "sum"),
        gross_pnl=("gross_pnl", "sum"), net_pnl=("net_pnl", "sum"),
        avg_net_ret_bps=("net_ret", "mean"), hit_rate=("_win", "mean"),
    ).reset_index().rename(columns={"_period": "period"})
    out["avg_net_ret_bps"] = out["avg_net_ret_bps"] * 1e4
    name_by_p = {0: "p0_offhours", **{i: _PERIOD_BOUNDS[i - 1][2] for i in range(1, 7)}}
    out["period_name"] = out["period"].map(name_by_p)
    return out[cols]


def equity_curve(trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    cols = ["exit_time", "trade_idx", "net_pnl", "cum_net_pnl", "equity",
            "peak", "drawdown", "drawdown_pct"]
    if trades_df.empty:
        return pd.DataFrame(columns=cols)
    df = trades_df.sort_values("exit_time").reset_index(drop=True)
    cum = df["net_pnl"].cumsum().to_numpy(dtype="float64")
    equity = initial_capital + cum
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = (dd / peak) * 100.0
    return pd.DataFrame({
        "exit_time": df["exit_time"].to_numpy(),
        "trade_idx": np.arange(1, len(df) + 1),
        "net_pnl": df["net_pnl"].to_numpy(dtype="float64"),
        "cum_net_pnl": cum, "equity": equity, "peak": peak,
        "drawdown": dd, "drawdown_pct": dd_pct,
    })


def _downsample(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    n = len(df)
    if n <= max_points:
        return df
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return df.iloc[idx].reset_index(drop=True)


def plot_equity(equity_df: pd.DataFrame, signal_id: str, split: str,
                cost_mode_name: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    x = equity_df["trade_idx"].to_numpy()
    base = float(equity_df["equity"].iloc[0]) if len(equity_df) else 0.0
    ax1.plot(x, equity_df["equity"].to_numpy(), color="#1f77b4", lw=1.0, label="equity")
    ax1.axhline(base, color="grey", ls="--", lw=0.8, alpha=0.6, label="initial capital")
    ax1.set_ylabel("Equity (Rs)")
    ax1.set_title(f"{signal_id} | {split} | cost={cost_mode_name}")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best")
    ax2.fill_between(x, equity_df["drawdown"].to_numpy(), 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown (Rs)")
    ax2.set_xlabel("Trade # (ordered by exit time)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    signal_ids: list[str]
    splits: list[str] = field(default_factory=lambda: ["train", "validation"])
    cost_modes: list[str] = field(default_factory=lambda: ["zero", "model"])
    notional: float = DEFAULT_NOTIONAL
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    limit_stocks: int | None = None
    write_trades: bool = True
    max_trades_rows: int = 50_000
    plot: bool = True


def _write_csv(df: pd.DataFrame, path: Path, float_fmt: str = "%.10g") -> None:
    df.to_csv(path, index=False, float_format=float_fmt)
    print(f"  wrote {path.name}  ({len(df)} rows)")


def _compare_to_day5(allsig_df: pd.DataFrame, day5_csv_path: Path | None) -> dict:
    """Cross-check the backtest's pooled all-signals metric against Day 5.

    Day 5 reported per-signal n_signals and avg_ret_bps in
    ``signal_train_vs_val.csv``. We reproduce the same metric from the loaded
    data (the backtest imports Day 5's signal logic verbatim) and compare. A
    match confirms the backtest is scoring the *exact* same signals as Day 5.
    """
    if day5_csv_path is None or not Path(day5_csv_path).exists():
        return {"available": False,
                "note": "Day 5 signal_train_vs_val.csv not found; cross-check skipped"}
    d5 = pd.read_csv(day5_csv_path)
    rows = []
    for _, r in allsig_df.iterrows():
        m = d5[d5["signal_id"] == r["signal_id"]]
        if m.empty:
            rows.append({"signal_id": r["signal_id"], "split": r["split"],
                         "backtest_n_signals": int(r["n_signals"]),
                         "day5_n_signals": None, "n_match": False,
                         "backtest_avg_ret_bps": float(r["avg_exp_ret_bps"]),
                         "day5_avg_ret_bps": None, "bps_delta": np.nan,
                         "day5_present": False, "match": False})
            continue
        m = m.iloc[0]
        split = r["split"]
        day5_n = int(m["train_n_signals"] if split == "train" else m["val_n_signals"])
        day5_bps = float(m["train_avg_ret_bps"] if split == "train" else m["val_avg_ret_bps"])
        bt_n = int(r["n_signals"])
        bt_bps = float(r["avg_exp_ret_bps"])
        bps_delta = bt_bps - day5_bps if np.isfinite(bt_bps) else np.nan
        n_match = bt_n == day5_n
        rows.append({
            "signal_id": r["signal_id"], "split": split,
            "backtest_n_signals": bt_n, "day5_n_signals": day5_n, "n_match": n_match,
            "backtest_avg_ret_bps": bt_bps, "day5_avg_ret_bps": day5_bps,
            "bps_delta": bps_delta, "day5_present": True,
            "match": bool(n_match and (np.isnan(bps_delta) or abs(bps_delta) < 1e-6)),
        })
    all_match = bool(rows) and all(r["match"] for r in rows)
    return {"available": True, "comparisons": rows, "all_match": all_match}
def run_backtest(config: BacktestConfig, reports_dir: Path,
                 day5_csv_path: Path | None = None) -> dict:
    """Run the full backtest and write all outputs to ``reports_dir``."""
    t0 = time.time()
    reports_dir.mkdir(parents=True, exist_ok=True)
    spec_by_id = {s["id"]: s for s in SPECS}
    unknown = [sid for sid in config.signal_ids if sid not in spec_by_id]
    if unknown:
        raise ValueError(f"unknown signal ids: {unknown}")
    specs = [spec_by_id[sid] for sid in config.signal_ids]
    bad_cm = [c for c in config.cost_modes if c not in COST_MODELS]
    if bad_cm:
        raise ValueError(f"unknown cost modes: {bad_cm}; known: {sorted(COST_MODELS)}")
    bad_sp = [s for s in config.splits if s not in ("train", "validation")]
    if bad_sp:
        raise ValueError(f"unknown splits: {bad_sp}; use train/validation (test is never read)")

    files = discover_target_files()
    if config.limit_stocks:
        files = files[: config.limit_stocks]
    print(f"Discovered {len(files)} stock files. Backtesting {len(specs)} signals "
          f"x {config.splits} splits x {config.cost_modes} cost modes | notional "
          f"Rs {config.notional:,.0f} | hold {HOLD_BARS}m | entry next-open, exit same-session")

    gross_trades = {(s["id"], sp): [] for s in specs for sp in config.splits}
    allsig_acc = {(s["id"], sp): [] for s in specs for sp in config.splits}
    trading_days = {sp: set() for sp in config.splits}
    rows_loaded = {sp: 0 for sp in config.splits}
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
        tk, al, td, rw = simulate_stock(sym, df, specs, config.splits, config.notional)
        for k, v in tk.items():
            gross_trades[k].extend(v)
        for k, v in al.items():
            allsig_acc[k].append(v)
        for sp, ds in td.items():
            trading_days[sp].update(ds)
        for sp, c in rw.items():
            rows_loaded[sp] += c
        del df
        if i % 10 == 0 or i == len(files):
            print(f"  [{i:>3}/{len(files)}] {sym} done")

    # TEST guard: load_stock raises if any TEST row appears, so test_rows_read == 0.
    return _finalize(config, reports_dir, gross_trades, allsig_acc, trading_days,
                     rows_loaded, stocks_ok, stocks_fail, day5_csv_path, t0)
def _finalize(config, reports_dir, gross_trades, allsig_acc, trading_days,
              rows_loaded, stocks_ok, stocks_fail, day5_csv_path, t0) -> dict:
    """Build per-signal outputs, the run summary, and print the headline."""
    spec_by_id = {s["id"]: s for s in SPECS}
    plot_cm = "model" if "model" in config.cost_modes else config.cost_modes[-1]
    metrics_rows, all_signals_rows = [], []
    for sid in config.signal_ids:
        spec = spec_by_id[sid]
        for split in config.splits:
            key = (spec["id"], split)
            gdf = pd.DataFrame(gross_trades[key], columns=GROSS_TRADE_COLUMNS)
            pooled = _pooled_all_signals(allsig_acc[key])
            all_signals_rows.append({"signal_id": spec["id"], "signal_name": spec["name"],
                                     "split": split, **pooled})
            ndays = len(trading_days[split])
            for cm_name in config.cost_modes:
                cm = COST_MODELS[cm_name]
                net_df = apply_costs(gdf, cm, cm_name, config.notional)
                m = compute_metrics(net_df, config.initial_capital, ndays)
                metrics_rows.append({"signal_id": spec["id"], "signal_name": spec["name"],
                                     "split": split, "cost_mode": cm_name, **m})
                tag = f"{spec['id']}_{split}_{cm_name}"
                if config.write_trades and len(net_df):
                    _write_csv(net_df.head(config.max_trades_rows),
                               reports_dir / f"trades_{tag}.csv")
                _write_csv(aggregate_by_stock(net_df), reports_dir / f"by_stock_{tag}.csv")
                _write_csv(aggregate_by_time(net_df), reports_dir / f"by_time_{tag}.csv")
                eq = equity_curve(net_df, config.initial_capital)
                _write_csv(_downsample(eq), reports_dir / f"equity_{tag}.csv")
                if config.plot and cm_name == plot_cm and len(eq):
                    try:
                        plot_equity(_downsample(eq), spec["id"], split, cm_name,
                                    reports_dir / f"equity_{tag}.png")
                    except Exception as e:  # noqa: BLE001
                        print(f"    plot skip {tag}: {e}")

    metrics_df = pd.DataFrame(metrics_rows)
    _write_csv(metrics_df, reports_dir / "metrics.csv")
    allsig_df = pd.DataFrame(all_signals_rows)
    _write_csv(allsig_df, reports_dir / "all_signals_check.csv")
    day5_cmp = _compare_to_day5(allsig_df, day5_csv_path)

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": "cost-aware backtest of Day 5 directional signals",
        "execution_model": {
            "entry": "open of candle T+1 (signal known after close of T)",
            "exit": f"close of candle T+{HOLD_BARS} (5-minute hold)",
            "one_position_per_stock": True, "no_overnight": True,
            "notional_model": "fixed per-trade notional (not capital-constrained)"},
        "config": {"signals": config.signal_ids, "splits": config.splits,
                   "cost_modes": config.cost_modes, "notional": config.notional,
                   "initial_capital": config.initial_capital, "hold_bars": HOLD_BARS,
                   "limit_stocks": config.limit_stocks, "write_trades": config.write_trades,
                   "max_trades_rows": config.max_trades_rows},
        "cost_model_params": {n: COST_MODELS[n].params() for n in config.cost_modes},
        "guards": {"test_ever_read": False, "test_rows_read": 0,
                   "max_allowed_date": str(VALIDATION_END),
                   "test_leak_check": "passed (load_stock raises if any row date >= TEST_START)"},
        "data": {"stocks_processed": stocks_ok, "stocks_failed": stocks_fail,
                 "rows_loaded": rows_loaded,
                 "trading_days": {sp: len(trading_days[sp]) for sp in config.splits}},
        "day5_cross_check": day5_cmp,
        "metrics": metrics_df.to_dict(orient="records"),
        "all_signals": allsig_df.to_dict(orient="records"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with (reports_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("  wrote run_summary.json")
    _print_headline(metrics_df, day5_cmp, config, summary)
    return summary
def _print_headline(metrics_df: pd.DataFrame, day5_cmp: dict,
                    config: BacktestConfig, summary: dict) -> None:
    print("\n=== Day 6 backtest summary ===")
    print(f"stocks processed : {summary['data']['stocks_processed']}  "
          f"(failed: {summary['data']['stocks_failed']})")
    for sp in config.splits:
        print(f"{sp.upper():<11} rows: {summary['data']['rows_loaded'].get(sp, 0):,}  "
              f"trading days: {summary['data']['trading_days'].get(sp, 0)}")
    print(f"TEST rows read    : 0  (never read; max allowed date "
          f"{summary['guards']['max_allowed_date']})")
    if day5_cmp.get("available"):
        status = "ALL MATCH" if day5_cmp["all_match"] else "MISMATCH (see all_signals_check.csv)"
        print(f"Day 5 cross-check : {status}  (backtest reproduces Day 5 all-signals metric)")
    print("\nNet avg return per trade (bps) -- does the signal survive costs?")
    print(f"{'signal':<6}{'split':<12}{'cost':<7}{'n_tr':>9}{'gross_bps':>11}"
          f"{'cost_bps':>10}{'net_bps':>10}{'hit%':>7}{'PF':>7}{'net_pnl_rs':>16}")
    for _, r in metrics_df.iterrows():
        print(f"{r['signal_id']:<6}{r['split']:<12}{r['cost_mode']:<7}"
              f"{int(r['n_trades']):>9}{r['avg_gross_ret_bps']:>11.3f}"
              f"{r['avg_cost_bps']:>10.3f}{r['avg_net_ret_bps']:>10.3f}"
              f"{r['hit_rate'] * 100:>7.1f}{r['profit_factor']:>7.2f}"
              f"{r['net_pnl']:>16,.0f}")

