#!/usr/bin/env python3
"""Trend-Continuation VWAP-bounce backtest (5-min) -> self-contained HTML journal.

STRATEGY (followed exactly, no deviations):
  * Trend-continuation setup: stock in a one-sided move -- higher highs & higher
    lows (long) / lower highs & lower lows (short).
  * Timeframe: 5-minute.  Indicator: session VWAP.
  * Watchlist = stocks showing a clear one-directional move (filtered at signal time).
  * Wait for price to bounce on VWAP: the 5-min candle wicks INTO VWAP and closes
    back on the trend side; it must NOT close on the wrong side of VWAP.
  * Entry: on the high (long) / low (short) of the bounce candle +/- a 0.05 buffer,
    executed as a stop order (buy-stop above / sell-stop below).
  * Stop-loss: low (long) / high (short) of the bounce candle -/+ a 0.05 buffer.
  * Target: manually exit near 15:10-15:15 -- modelled as exiting at the CLOSE of
    the 15:10 5-min candle (i.e. the price at ~15:15).
  * Risk management: once price reaches 1:2 (2R), trail the stop to entry (breakeven).
  * Never take a trade before 09:45 or after 13:00 (1 pm).
  * Avoid trading on big candles: skip if the bounce candle range > 2 x ATR(14).
  * At least 4-5 candles away from VWAP making higher highs (long) / lower lows
    (short) before the bounce, otherwise avoid the trade.
  * Pin bar / hammer bouncing off VWAP is a good sign (recorded as a quality flag).
  * Bounce candle touching VWAP must be GREEN for long / RED for short.

BACKTEST PARAMETERS:
  * Period: 2026-01-01 .. 2026-08-31 (NSE trading days only).
  * Risk per trade: Rs 1000  ->  quantity = floor(1000 / risk_per_share).
  * Max 5 trades per calendar day (the first 5 entries by entry time; later
    setups on the same day are skipped, exactly as a trader would stop trading).

OUTPUTS (all under reports/day9_trend_continuation/ -- untracked, NOT committed):
  * trend_continuation_report.html  -- self-contained HTML journal (the report)
  * trade_journal.csv                -- one row per trade
  * daily_summary.csv                -- one row per trading day
  * run_summary.json                 -- config + overall metrics

Run from the project root:
    .venv/bin/python reports/day9_trend_continuation/run_backtest.py
"""
from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path
from datetime import time as dtime

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.backtesting.costs import CostModel, DEFAULT_COST  # noqa: E402

# ----------------------------- configuration --------------------------------
# This is the *v2* script. It runs TWO variants on AUGUST 2026 only, for a fast
# before/after comparison:
#   * BASELINE = original rules (09:45 start, NO notional cap, trail-to-BE at 2R)
#   * V2       = improved (10:50 start, qty capped so notional<=Rs 2.5L AND risk
#                <=Rs 1000 [BOTH must hold], ratchet trail instead of BE).
# Globals below are defaults; main() reassigns them per variant via run_variant().
START_DATE = "2026-08-01"        # August 2026 only (fast)
END_DATE = "2026-08-31"
RISK_PER_TRADE = 1000.0          # Rs risk per trade (hard cap)
MAX_NOTIONAL = 500000.0          # cap on buy value per trade (None = uncapped)
BUFFER = 0.05                    # rs buffer on entry & exit
BIG_CANDLE_MULT = 2.0           # bounce candle range > 2x ATR14 -> skip
ATR_WINDOW = 14
NOT_BEFORE = dtime(9, 45)        # never trade before 09:45 (baseline)
NOT_AFTER = dtime(13, 0)        # never enter a trade after 13:00 (1 pm)
# Optional avoidable-pattern filters (Day-11 loss analysis). Defaults keep the
# original strategy behaviour; the custom runner flips them from the CLI.
ALLOW_LONG = True               # if False, long setups are skipped (shorts only)
ALLOW_SHORT = True              # if False, short setups are skipped (longs only)
SKIP_LUNCH_START = None         # if set, no entries in [START, END) lunch window
SKIP_LUNCH_END = None           #   (e.g. 11:30-12:30 -- low follow-through volume)
EXIT_BAR_OPEN = dtime(15, 10)    # exit at close of the 15:10 5-min candle (~15:15)
LOOKBACK = 5                     # "4-5 candles far away" run before the bounce
ABOVE_VWAP_MIN = 4              # at least 4 of the LOOKBACK candles on trend side
EXTEND_MIN_DIST = 0.0010        # 0.10% -- price must have genuinely moved away
GAP_MIN_ATR = 0.20              # min near-edge-to-VWAP gap in ATR(14) units for a
                                # lookback candle to count as "far away" (not touching)
REQUIRE_GREEN_BOUNCE = True      # bounce candle must be GREEN (long) / RED (short):
                                # the candle touching VWAP must be in the trade direction
MAX_TRADES_PER_DAY = 5
TRAIL_MODE = "be"               # "be"=trail to breakeven at 2R (baseline)
                                # "ratchet"=lock 1R at 2R, lock 2R at 3R (v2)
OUT_SUFFIX = "_aug_base"        # output filename suffix per variant
VARIANT_LABEL = "BASELINE (original rules)"
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "reports" / "day9_trend_continuation"
COST: CostModel = DEFAULT_COST


# ----------------------------- data loading ---------------------------------
def discover_stocks() -> list[tuple[str, Path]]:
    files = sorted(DATA_DIR.glob("*_1min_indicators.parquet"))
    return [(p.name[: -len("_1min_indicators.parquet")], p) for p in files]


def load_5min(symbol: str, path: Path) -> pd.DataFrame | None:
    """Load 1-min OHLCV+VWAP for the backtest window and resample to 5-min candles."""
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
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            vwap=("vwap", "last"),
        )
        .dropna(subset=["open", "close"])
        .reset_index()
    )
    agg = agg[(agg["timestamp"].dt.time >= dtime(9, 15)) & (agg["timestamp"].dt.time <= dtime(15, 30))]
    if agg.empty:
        return None
    agg["date"] = agg["timestamp"].dt.date
    agg["range"] = (agg["high"] - agg["low"]).astype(float)
    agg["body"] = (agg["close"] - agg["open"]).astype(float)
    agg["is_green"] = agg["close"] > agg["open"]
    low_wick = agg[["open", "close"]].min(axis=1) - agg["low"]
    up_wick = agg["high"] - agg[["open", "close"]].max(axis=1)
    body_abs = agg["body"].abs()
    agg["is_hammer"] = (low_wick >= 2.0 * body_abs) & (up_wick <= body_abs.clip(lower=0.01))
    agg["dist_vwap"] = (agg["close"] - agg["vwap"]) / agg["vwap"]
    agg["atr14"] = (
        agg.groupby("date")["range"]
        .transform(lambda s: s.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean())
        .shift(1)
    )
    agg["day_open"] = agg.groupby("date")["open"].transform("first")
    agg["is_exit_bar"] = False
    for _, g in agg.groupby("date"):
        t = g["timestamp"].dt.time
        idx = g.index[t == EXIT_BAR_OPEN]
        target = idx[0] if len(idx) else g.index[-1]
        agg.loc[target, "is_exit_bar"] = True
    return agg.reset_index(drop=True)
# ----------------------------- setup filters --------------------------------
def check_filters(s, i, cl, hi, lo, vwap, dist, day_open, atr, rng, times, is_green):
    """Return True if candle i is a valid VWAP-bounce setup in direction s (+1/-1)."""
    if pd.isna(atr[i]) or i < ATR_WINDOW or times[i] < NOT_BEFORE or times[i] >= NOT_AFTER:
        return False
    # 1) VWAP bounce: wick into VWAP, close back on the trend side (not beyond VWAP)
    if s == 1:
        if not (lo[i] <= vwap[i] and cl[i] > vwap[i]):
            return False
    else:
        if not (hi[i] >= vwap[i] and cl[i] < vwap[i]):
            return False
    # 1b) bounce candle must be in the trade direction: GREEN for long / RED for short
    #     (the candle that touches VWAP must close on its open in the trade direction)
    if REQUIRE_GREEN_BOUNCE:
        if s == 1 and not is_green[i]:
            return False
        if s == -1 and is_green[i]:
            return False
    # 2) avoid big candles
    if rng[i] > BIG_CANDLE_MULT * atr[i]:
        return False
    a = i - LOOKBACK
    if a < 0:
        return False
    seg_cl, seg_vw = cl[a:i], vwap[a:i]
    seg_hi, seg_lo = hi[a:i], lo[a:i]
    seg_dist = dist[a:i]
    if len(seg_cl) < LOOKBACK:
        return False
    if s == 1:
        # "4-5 candles far away from VWAP": the NEAR EDGE (low) of the lookback
        # candles must sit a clear GAP above VWAP -- not merely *close* above VWAP
        # while its low still touches/pierces VWAP.  >= ABOVE_VWAP_MIN of the
        # LOOKBACK candles must have (low - vwap)/ATR >= GAP_MIN_ATR.
        near = seg_lo
        gap_atr = (near - seg_vw) / atr[i]
        if int((gap_atr >= GAP_MIN_ATR).sum()) < ABOVE_VWAP_MIN:
            return False
        if not (seg_hi[-1] >= seg_hi[0]):                         # higher high
            return False
        if not (seg_lo[-1] >= seg_lo[0]):                         # higher low
            return False
        if float(seg_dist.max()) < EXTEND_MIN_DIST:             # genuinely extended above
            return False
        if not (cl[i] > day_open):                              # one-directional up day
            return False
    else:
        near = seg_hi
        gap_atr = (seg_vw - near) / atr[i]
        if int((gap_atr >= GAP_MIN_ATR).sum()) < ABOVE_VWAP_MIN:
            return False
        if not (seg_hi[-1] <= seg_hi[0]):                         # lower high
            return False
        if not (seg_lo[-1] <= seg_lo[0]):                         # lower low
            return False
        if float((-seg_dist).max()) < EXTEND_MIN_DIST:         # genuinely extended below
            return False
        if not (cl[i] < day_open):                              # one-directional down day
            return False
    return True


# ----------------------------- trade row builder ----------------------------
def _trade_dict(sym, s, date, i, j_entry, j_exit, entry_px, exit_px, sl_init,
                qty, risk, reason, mfe_r, trailed, arrs):
    """Build the journal row for one completed round-trip."""
    op, hi, lo, cl, vwap, is_green, is_hammer, rng, atr, dist, is_exit, ts = arrs
    if s == 1:
        buy_price, sell_price = entry_px, exit_px
        bd = COST.round_trip_breakdown(1, qty * buy_price, qty * sell_price)
    else:
        buy_price, sell_price = exit_px, entry_px
        bd = COST.round_trip_breakdown(-1, qty * sell_price, qty * buy_price)
    buy_value = qty * buy_price
    sell_value = qty * sell_price
    gross = sell_value - buy_value
    charges = bd["total"]
    net = gross - charges
    risk_money = qty * risk
    r_mult = gross / risk_money if risk_money else 0.0
    return {
        "symbol": sym, "direction": "LONG" if s == 1 else "SHORT", "date": date,
        "setup_time": ts[i], "entry_time": ts[j_entry], "exit_time": ts[j_exit],
        "bounce_high": float(hi[i]), "bounce_low": float(lo[i]),
        "vwap_at_bounce": float(vwap[i]),
        "entry_price": float(entry_px), "exit_price": float(exit_px),
        "sl_initial": float(sl_init), "qty": int(qty),
        "risk_per_share": float(risk), "risk_money": float(risk_money),
        "buy_price": float(buy_price), "sell_price": float(sell_price),
        "buy_value": float(buy_value), "sell_value": float(sell_value),
        "gross_pnl": float(gross),
        "stt": float(bd["stt"]), "brokerage": float(bd["brokerage"]),
        "exchange": float(bd["exchange"]), "sebi": float(bd["sebi"]),
        "stamp": float(bd["stamp"]), "gst": float(bd["gst"]),
        "slippage": float(bd["slippage"]), "charges_total": float(charges),
        "net_pnl": float(net), "r_multiple": float(r_mult), "mfe_r": float(mfe_r),
        "exit_reason": reason, "trailed_to_be": bool(trailed),
        "green_setup": bool(is_green[i]), "hammer_setup": bool(is_hammer[i]),
    }
# ----------------------------- trade simulation -----------------------------
def _trail_stop(s, entry_px, risk, fav_r, mode):
    """Protective stop for current favourable-R, or None (no trail yet).

    "be"      : trail to breakeven (entry) once price reaches 2R  (baseline)
    "ratchet" : lock 1R at 2R, lock 2R at 3R  (v2 -- lets winners breathe)
    """
    if mode == "ratchet":
        if fav_r >= 3.0:
            return entry_px + s * 2.0 * risk
        if fav_r >= 2.0:
            return entry_px + s * 1.0 * risk
        return None
    if fav_r >= 2.0:                       # "be"
        return entry_px
    return None


def simulate_setup(sym, s, date, i, arrs, exit_idx):
    """Forward-scan candles after the bounce to fill the stop entry, manage SL /
    trail-to-breakeven, and exit by SL or the 15:15 time-stop. No look-ahead."""
    op, hi, lo, cl, vwap, is_green, is_hammer, rng, atr, dist, is_exit, ts = arrs
    if s == 1:
        entry = hi[i] + BUFFER
        sl = lo[i] - BUFFER
        risk = entry - sl
    else:
        entry = lo[i] - BUFFER
        sl = hi[i] + BUFFER
        risk = sl - entry
    if risk <= 0:
        return None
    # Sizing: BOTH conditions must hold -- Rs 1000 max risk AND max notional.
    qty = int(RISK_PER_TRADE // risk)
    if MAX_NOTIONAL is not None and entry > 0:
        qty = min(qty, int(MAX_NOTIONAL // entry))
    if qty < 1:
        return None
    entered = False
    entry_px = None
    sl_now = sl
    max_fav = -np.inf if s == 1 else np.inf
    trailed = False
    mfe_r = 0.0
    j_entry = None
    for j in range(i + 1, exit_idx + 1):
        if not entered:
            trig = (hi[j] >= entry) if s == 1 else (lo[j] <= entry)
            slhit = (lo[j] <= sl) if s == 1 else (hi[j] >= sl)        # uses initial sl
            if slhit and not trig:
                return None                                          # cancelled before entry
            if trig:
                entered = True
                entry_px = max(op[j], entry) if s == 1 else min(op[j], entry)
                j_entry = j
                max_fav = hi[j] if s == 1 else lo[j]
                mf = s * (max_fav - entry_px) / risk
                new_sl = _trail_stop(s, entry_px, risk, mf, TRAIL_MODE)
                if new_sl is not None:
                    if (not trailed) or (s == 1 and new_sl > sl_now) or (s == -1 and new_sl < sl_now):
                        sl_now = new_sl
                        trailed = True
                if slhit:                                            # enter then SL same candle
                    exit_px = min(op[j], sl) if s == 1 else max(op[j], sl)
                    mfe_r = s * (max_fav - entry_px) / risk
                    return _trade_dict(sym, s, date, i, j_entry, j, entry_px, exit_px,
                                       sl, qty, risk, "SL", mfe_r, trailed, arrs)
                if is_exit[j]:                                       # enter & time-exit same candle
                    exit_px = cl[j]
                    mfe_r = s * (max_fav - entry_px) / risk
                    return _trade_dict(sym, s, date, i, j_entry, j, entry_px, exit_px,
                                       sl, qty, risk, "Time(15:15)", mfe_r, trailed, arrs)
                continue
        else:
            fav = hi[j] if s == 1 else lo[j]
            max_fav = max(max_fav, fav) if s == 1 else min(max_fav, fav)
            mfe_r = max(mfe_r, s * (max_fav - entry_px) / risk)
            new_sl = _trail_stop(s, entry_px, risk, mfe_r, TRAIL_MODE)
            if new_sl is not None:
                if (not trailed) or (s == 1 and new_sl > sl_now) or (s == -1 and new_sl < sl_now):
                    sl_now = new_sl
                    trailed = True
            slhit = (lo[j] <= sl_now) if s == 1 else (hi[j] >= sl_now)
            if slhit:
                exit_px = min(op[j], sl_now) if s == 1 else max(op[j], sl_now)
                if trailed:
                    if TRAIL_MODE == "ratchet":
                        locked = s * (sl_now - entry_px) / risk
                        reason = f"Trail-{locked:.1f}R"
                    else:
                        reason = "Trail-BE"
                else:
                    reason = "SL"
                return _trade_dict(sym, s, date, i, j_entry, j, entry_px, exit_px,
                                   sl, qty, risk, reason, mfe_r, trailed, arrs)
            if is_exit[j]:
                exit_px = cl[j]
                return _trade_dict(sym, s, date, i, j_entry, j, entry_px, exit_px,
                                   sl, qty, risk, "Time(15:15)", mfe_r, trailed, arrs)
    if entered:                                                     # safety: force time exit
        j = exit_idx
        return _trade_dict(sym, s, date, i, j_entry, j, entry_px, cl[j],
                           sl, qty, risk, "Time(15:15)", mfe_r, trailed, arrs)
    return None


def simulate_stock(sym: str, df: pd.DataFrame) -> list[dict]:
    """One trade per stock per day (first valid setup that triggers an entry)."""
    trades: list[dict] = []
    for date, g in df.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        n = len(g)
        op = g["open"].values.astype(float)
        hi = g["high"].values.astype(float)
        lo = g["low"].values.astype(float)
        cl = g["close"].values.astype(float)
        vwap = g["vwap"].values.astype(float)
        rng = g["range"].values.astype(float)
        atr = g["atr14"].values.astype(float)
        dist = g["dist_vwap"].values.astype(float)
        is_green = g["is_green"].values
        is_hammer = g["is_hammer"].values
        is_exit = g["is_exit_bar"].values
        ts = g["timestamp"].values
        times = pd.to_datetime(g["timestamp"]).dt.time.values
        day_open = float(g["day_open"].iloc[0])
        exit_idx = int(np.where(is_exit)[0][0])
        arrs = (op, hi, lo, cl, vwap, is_green, is_hammer, rng, atr, dist, is_exit, ts)
        done = False
        for i in range(n):
            if done:
                break
            if lo[i] <= vwap[i] and cl[i] > vwap[i]:
                s = 1
            elif hi[i] >= vwap[i] and cl[i] < vwap[i]:
                s = -1
            else:
                continue
            # Optional direction gate (avoidable-pattern filter).
            if s == 1 and not ALLOW_LONG:
                continue
            if s == -1 and not ALLOW_SHORT:
                continue
            if not check_filters(s, i, cl, hi, lo, vwap, dist, day_open, atr, rng, times, is_green):
                continue
            if i >= exit_idx:            # no room to enter before the time-stop
                continue
            tr = simulate_setup(sym, s, date, i, arrs, exit_idx)
            if tr is not None:
                # Optional lunch-lull skip on the actual ENTRY fill time
                # (matches the loss analysis, which bucketed by entry_time).
                if (SKIP_LUNCH_START is not None and SKIP_LUNCH_END is not None
                        and SKIP_LUNCH_START <= pd.Timestamp(tr["entry_time"]).time() < SKIP_LUNCH_END):
                    continue
                trades.append(tr)
                done = True
    return trades
# ----------------------------- aggregation / metrics -------------------------
def apply_daily_cap(trades: list[dict]) -> list[dict]:
    """Keep the first MAX_TRADES_PER_DAY entries per day, by entry time."""
    by_date: dict = {}
    for t in trades:
        by_date.setdefault(t["date"], []).append(t)
    kept: list[dict] = []
    for d, lst in by_date.items():
        lst.sort(key=lambda x: (x["entry_time"], x["symbol"]))
        kept.extend(lst[:MAX_TRADES_PER_DAY])
    kept.sort(key=lambda x: (x["date"], x["entry_time"], x["symbol"]))
    return kept


def build_daily(tdf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cum = 0.0
    for d, g in tdf.groupby("date", sort=True):
        gross = g["gross_pnl"].sum()
        ch = g["charges_total"].sum()
        net = g["net_pnl"].sum()
        cum += net
        n = len(g)
        wins = int((g["net_pnl"] > 0).sum())
        rows.append({
            "date": d, "n_trades": n,
            "n_long": int((g["direction"] == "LONG").sum()),
            "n_short": int((g["direction"] == "SHORT").sum()),
            "gross_pnl": float(gross), "charges": float(ch), "net_pnl": float(net),
            "cum_net": float(cum), "win_rate": wins / n if n else 0.0,
            "best_trade": float(g["net_pnl"].max()), "worst_trade": float(g["net_pnl"].min()),
        })
    return pd.DataFrame(rows)


def build_monthly(tdf: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar month (YYYY-MM): trades, gross/charges/net, cum net."""
    cols = ["month", "n_trades", "n_long", "n_short", "gross_pnl", "charges",
            "net_pnl", "cum_net", "win_rate", "best_trade", "worst_trade"]
    if tdf.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    cum = 0.0
    month_key = tdf["date"].astype(str).str[:7]
    for month, g in tdf.groupby(month_key, sort=True):
        gross = float(g["gross_pnl"].sum())
        ch = float(g["charges_total"].sum())
        net = float(g["net_pnl"].sum())
        cum += net
        n = len(g)
        wins = int((g["net_pnl"] > 0).sum())
        rows.append({
            "month": month, "n_trades": n,
            "n_long": int((g["direction"] == "LONG").sum()),
            "n_short": int((g["direction"] == "SHORT").sum()),
            "gross_pnl": gross, "charges": ch, "net_pnl": net,
            "cum_net": float(cum), "win_rate": wins / n if n else 0.0,
            "best_trade": float(g["net_pnl"].max()), "worst_trade": float(g["net_pnl"].min()),
        })
    return pd.DataFrame(rows, columns=cols)


def build_overall(tdf: pd.DataFrame, daily: pd.DataFrame, trading_days: set) -> dict:
    net = tdf["net_pnl"]
    wins = tdf[net > 0]
    losses = tdf[net <= 0]
    if not daily.empty:
        cum = daily["cum_net"]
        peak = cum.cummax()
        max_dd = float((cum - peak).min())
        final_net = float(cum.iloc[-1])
    else:
        max_dd = 0.0
        final_net = 0.0
    loss_sum = float(losses["net_pnl"].sum())
    pf = float(wins["net_pnl"].sum() / abs(loss_sum)) if loss_sum != 0 else float("inf")
    return {
        "total_trades": int(len(tdf)),
        "winners": int((net > 0).sum()), "losers": int((net <= 0).sum()),
        "win_rate": float((net > 0).mean()) if len(tdf) else 0.0,
        "total_gross": float(tdf["gross_pnl"].sum()),
        "total_charges": float(tdf["charges_total"].sum()),
        "total_net": float(net.sum()),
        "final_net": final_net,
        "avg_r": float(tdf["r_multiple"].mean()) if len(tdf) else 0.0,
        "profit_factor": pf,
        "max_win": float(net.max()) if len(tdf) else 0.0,
        "max_loss": float(net.min()) if len(tdf) else 0.0,
        "avg_win": float(wins["net_pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["net_pnl"].mean()) if len(losses) else 0.0,
        "max_drawdown": max_dd,
        "days_traded": int((daily["n_trades"] > 0).sum()) if not daily.empty else 0,
        "total_trading_days": int(len(trading_days)),
        "avg_trades_per_day": len(tdf) / max(len(trading_days), 1),
        "long_trades": int((tdf["direction"] == "LONG").sum()),
        "short_trades": int((tdf["direction"] == "SHORT").sum()),
        "long_net": float(tdf[tdf["direction"] == "LONG"]["net_pnl"].sum()),
        "short_net": float(tdf[tdf["direction"] == "SHORT"]["net_pnl"].sum()),
        "exit_reasons": tdf["exit_reason"].value_counts().to_dict(),
        "total_turnover": float((tdf["buy_value"] + tdf["sell_value"]).sum()),
        "max_qty": int(tdf["qty"].max()) if len(tdf) else 0,
        "max_notional": float((tdf["buy_value"] + tdf["sell_value"]).max()) if len(tdf) else 0.0,
    }


def write_artifacts(tdf: pd.DataFrame, daily: pd.DataFrame, monthly: pd.DataFrame,
                    overall: dict, trading_days: int):
    tdf.to_csv(OUT_DIR / f"trade_journal{OUT_SUFFIX}.csv", index=False)
    daily.to_csv(OUT_DIR / f"daily_summary{OUT_SUFFIX}.csv", index=False)
    monthly.to_csv(OUT_DIR / f"monthly_summary{OUT_SUFFIX}.csv", index=False)
    summary = {
        "config": {
            "strategy": "VWAP Trend Continuation (5-min)",
            "variant": VARIANT_LABEL,
            "start_date": START_DATE, "end_date": END_DATE,
            "risk_per_trade": RISK_PER_TRADE, "max_trades_per_day": MAX_TRADES_PER_DAY,
            "max_notional": (None if MAX_NOTIONAL is None else float(MAX_NOTIONAL)),
            "buffer": BUFFER, "big_candle_mult": BIG_CANDLE_MULT, "atr_window": ATR_WINDOW,
            "not_before": str(NOT_BEFORE), "not_after": str(NOT_AFTER), "exit_bar_open": str(EXIT_BAR_OPEN),
            "lookback": LOOKBACK, "above_vwap_min": ABOVE_VWAP_MIN,
            "extend_min_dist": EXTEND_MIN_DIST, "gap_min_atr": GAP_MIN_ATR,
            "require_green_bounce": REQUIRE_GREEN_BOUNCE,
            "trail_mode": TRAIL_MODE,
            "allow_long": ALLOW_LONG, "allow_short": ALLOW_SHORT,
            "skip_lunch_start": (None if SKIP_LUNCH_START is None else str(SKIP_LUNCH_START)),
            "skip_lunch_end": (None if SKIP_LUNCH_END is None else str(SKIP_LUNCH_END)),
        },
        "overall": overall,
        "monthly": monthly.to_dict("records"),
    }
    with (OUT_DIR / f"run_summary{OUT_SUFFIX}.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
# ----------------------------- HTML report ----------------------------------
CSS = """
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  margin:0;color:#1f2933;background:#f4f6f8;line-height:1.45}
.wrap{max-width:1280px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:22px;margin:0 0 4px;color:#102a43}
h2{font-size:16px;margin:26px 0 10px;color:#243b53;border-bottom:2px solid #d9e2ec;padding-bottom:6px}
.sub{color:#486581;font-size:13px;margin-bottom:6px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0}
.card{background:#fff;border:1px solid #e0e6eb;border-radius:10px;padding:14px 16px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#829ab1}
.card .v{font-size:20px;font-weight:700;margin-top:4px}
.pos{color:#0a7c2a}.neg{color:#c02a2a}.muted{color:#627d98}
table{border-collapse:collapse;width:100%;font-size:12px;background:#fff}
th,td{padding:6px 8px;border-bottom:1px solid #eef1f4;text-align:right;white-space:nowrap}
th{background:#f0f4f8;color:#334e68;font-weight:600;position:sticky;top:0}
td.l,th.l{text-align:left}
tr.dayhead td{background:#d9e2ec;color:#102a43;font-weight:700;font-size:12px;text-align:left;
  padding:7px 10px;border-top:2px solid #bcccdc}
tr:hover td{background:#fbfcfd}
.nav{position:sticky;top:0;background:#f4f6f8;padding:8px 0;border-bottom:1px solid #d9e2ec;z-index:5}
.nav a{color:#0f4c81;text-decoration:none;margin-right:16px;font-size:13px;font-weight:600}
.leg{font-size:11px;color:#627d98;margin-top:6px}
code{background:#eef1f4;padding:1px 5px;border-radius:4px;font-size:11px}
.small{font-size:12px;color:#486581}
.tag{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:700}
.tag-l{background:#dcf7e3;color:#0a7c2a}.tag-s{background:#fde2e2;color:#c02a2a}
"""


def _esc(x) -> str:
    amp = chr(38) + "amp;"
    lt = chr(38) + "lt;"
    gt = chr(38) + "gt;"
    return str(x).replace("&", amp).replace("<", lt).replace(">", gt)


def _inr(v, dp=2) -> str:
    return "\u20b9" + f"{v:,.{dp}f}"


def _inr_signed(v, dp=2) -> str:
    sign = "-" if v < 0 else ""
    return f"{sign}\u20b9{abs(v):,.{dp}f}"


def equity_svg(daily: pd.DataFrame) -> str:
    if daily.empty:
        return "<p class='small'>No equity data.</p>"
    ys = daily["cum_net"].tolist()
    n = len(ys)
    w, h, pad = 1000, 260, 46
    mn, mx = min(ys), max(ys)
    if mx == mn:
        mx = mn + 1.0
    span = mx - mn
    mn -= span * 0.08
    mx += span * 0.08
    span = mx - mn

    def X(i):
        return pad + (w - 2 * pad) * i / (n - 1 or 1)

    def Y(v):
        return h - pad - (h - 2 * pad) * (v - mn) / span

    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(ys))
    zy = Y(0)
    last = ys[-1]
    grid = ""
    for f in (0.25, 0.5, 0.75):
        gy = h - pad - (h - 2 * pad) * f
        grid += f'<line x1="{pad}" y1="{gy:.1f}" x2="{w-pad}" y2="{gy:.1f}" stroke="#eef1f4"/>'
    color = "#0a7c2a" if last >= 0 else "#c02a2a"
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="xMidYMid meet">'
        f'<rect width="{w}" height="{h}" fill="#fff" rx="8"/>'
        f'{grid}'
        f'<line x1="{pad}" y1="{zy:.1f}" x2="{w-pad}" y2="{zy:.1f}" stroke="#9fb3c8" stroke-dasharray="4 4"/>'
        f'<text x="{pad}" y="22" font-size="12" fill="#486581">Cumulative net P&L by day (\u20b9)</text>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
        f'<circle cx="{X(n-1):.1f}" cy="{Y(last):.1f}" r="4" fill="{color}"/>'
        f'<text x="{X(n-1)-6:.1f}" y="{Y(last)-8:.1f}" font-size="11" fill="{color}" text-anchor="end">'
        f'{_inr_signed(last)}</text>'
        f'</svg>'
    )
# ----------------------------- report sections ------------------------------
def _card(k: str, v: str, cls: str = "") -> str:
    return f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'


def _mini_table(headers: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table style="margin-top:6px"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def _section_summary(overall: dict, tdf: pd.DataFrame) -> str:
    pf = overall["profit_factor"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    cls_net = "pos" if overall["final_net"] >= 0 else "neg"
    cards = "".join([
        _card("Net P&L", _inr_signed(overall["final_net"]), cls_net),
        _card("Total Gross", _inr(overall["total_gross"])),
        _card("Total Charges", _inr(overall["total_charges"]), "muted"),
        _card("Trades", str(overall["total_trades"])),
        _card("Win Rate", f'{overall["win_rate"] * 100:.1f}%'),
        _card("Profit Factor", pf_s),
        _card("Avg R", f'{overall["avg_r"]:.2f}'),
        _card("Max Drawdown", _inr_signed(overall["max_drawdown"]), "neg"),
        _card("Days Traded", f'{overall["days_traded"]} / {overall["total_trading_days"]}'),
        _card("Avg Trades/Day", f'{overall["avg_trades_per_day"]:.2f}'),
        _card("Max Win", _inr_signed(overall["max_win"]), "pos"),
        _card("Max Loss", _inr_signed(overall["max_loss"]), "neg"),
        _card("Max Qty", str(overall["max_qty"])),
        _card("Total Turnover", _inr(overall["total_turnover"]), "muted"),
    ])
    dir_rows = [
        ["LONG", str(overall["long_trades"]), _inr_signed(overall["long_net"])],
        ["SHORT", str(overall["short_trades"]), _inr_signed(overall["short_net"])],
    ]
    reason_rows = []
    for reason, cnt in overall["exit_reasons"].items():
        net = float(tdf[tdf["exit_reason"] == reason]["net_pnl"].sum())
        reason_rows.append([_esc(reason), str(int(cnt)), _inr_signed(net)])
    mini = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:8px">'
        f'<div><div class="small" style="font-weight:700">By Direction</div>{_mini_table(["Side", "Trades", "Net P&L"], dir_rows)}</div>'
        f'<div><div class="small" style="font-weight:700">By Exit Reason</div>{_mini_table(["Reason", "Trades", "Net P&L"], reason_rows)}</div>'
        "</div>"
    )
    return f'<section id="summary"><h2>Overall Summary</h2><div class="cards">{cards}</div>{mini}</section>'


def _section_equity(daily: pd.DataFrame) -> str:
    return f'<section id="equity"><h2>Equity Curve (cumulative net P&L by trading day)</h2>{equity_svg(daily)}</section>'


def _section_daily(daily: pd.DataFrame) -> str:
    rows = ""
    for _, r in daily.iterrows():
        cls = "pos" if r["net_pnl"] >= 0 else "neg"
        ccls = "pos" if r["cum_net"] >= 0 else "neg"
        rows += (
            f'<tr><td class="l">{r["date"]}</td><td>{int(r["n_trades"])}</td>'
            f'<td>{int(r["n_long"])}</td><td>{int(r["n_short"])}</td>'
            f'<td>{_inr(r["gross_pnl"])}</td><td class="muted">{_inr(r["charges"])}</td>'
            f'<td class="{cls}">{_inr_signed(r["net_pnl"])}</td>'
            f'<td class="{ccls}">{_inr_signed(r["cum_net"])}</td><td>{r["win_rate"] * 100:.0f}%</td></tr>'
        )
    head = ("<tr><th class='l'>Date</th><th>Trades</th><th>Long</th><th>Short</th>"
            "<th>Gross P&L</th><th>Charges</th><th>Net P&L</th><th>Cum. Net</th><th>Win%</th></tr>")
    return f'<section id="daily"><h2>Daily Summary</h2><table><thead>{head}</thead><tbody>{rows}</tbody></table></section>'


def _section_monthly(monthly: pd.DataFrame) -> str:
    if monthly.empty:
        return '<section id="monthly"><h2>Monthly Summary</h2><p class="small">No trades.</p></section>'
    rows = ""
    for _, r in monthly.iterrows():
        cls = "pos" if r["net_pnl"] >= 0 else "neg"
        ccls = "pos" if r["cum_net"] >= 0 else "neg"
        rows += (
            f'<tr><td class="l">{r["month"]}</td><td>{int(r["n_trades"])}</td>'
            f'<td>{int(r["n_long"])}</td><td>{int(r["n_short"])}</td>'
            f'<td>{_inr(r["gross_pnl"])}</td><td class="muted">{_inr(r["charges"])}</td>'
            f'<td class="{cls}"><b>{_inr_signed(r["net_pnl"])}</b></td>'
            f'<td class="{ccls}">{_inr_signed(r["cum_net"])}</td><td>{r["win_rate"] * 100:.0f}%</td></tr>'
        )
    head = ("<tr><th class='l'>Month</th><th>Trades</th><th>Long</th><th>Short</th>"
            "<th>Gross P&L</th><th>Charges</th><th>Net P&L</th><th>Cum. Net</th><th>Win%</th></tr>")
    return f'<section id="monthly"><h2>Monthly Summary (net profit / loss by month)</h2><table><thead>{head}</thead><tbody>{rows}</tbody></table></section>'


def _section_costs() -> str:
    """Cost-model table reconciled to Zerodha's current equity-intraday (NSE) charges.

    Worked example: Rs 10L buy + Rs 10L sell (turnover Rs 20L), 1 buy + 1 sell order --
    matches Zerodha's published NSE equity-intraday fee table exactly (slippage excluded).
    """
    buy, sell = 1000000.0, 1000000.0          # Rs 10L buy + Rs 10L sell
    bd = COST.round_trip_breakdown(1, buy, sell)
    comp = [
        ("Brokerage", "\u20b920 buy + \u20b920 sell", float(bd["brokerage"])),
        ("STT", "0.025% \u00d7 \u20b910L sell", float(bd["stt"])),
        ("NSE transaction charges", "0.00307% \u00d7 \u20b920L", float(bd["exchange"])),
        ("SEBI charges", "\u20b910/crore \u00d7 \u20b920L", float(bd["sebi"])),
        ("Stamp duty", "0.003% \u00d7 \u20b910L buy", float(bd["stamp"])),
        ("GST", f"18% \u00d7 (\u20b9{bd['brokerage']:.0f} + \u20b9{bd['exchange']:.2f} + \u20b9{bd['sebi']:.0f})", float(bd["gst"])),
        ("Slippage", "excluded -- Zerodha statutory charges only (no market-impact assumption)", float(bd["slippage"])),
    ]
    body = "".join(
        f'<tr><td class="l">{name}</td><td class="l">{rate}</td><td>{_inr(amt)}</td></tr>'
        for name, rate, amt in comp
    )
    body += (
        f'<tr style="font-weight:700;border-top:2px solid #bcccdc"><td class="l">Total charges</td>'
        f'<td class="l">Zerodha equity-intraday round-trip (no slippage assumption)</td><td>{_inr(bd["total"])}</td></tr>'
    )
    head = ('<tr><th class="l">Charge</th><th class="l">Calculation (Zerodha NSE equity intraday)</th>'
            '<th>Amount on \u20b910L buy + \u20b910L sell</th></tr>')
    return (
        '<section id="costs"><h2>Cost Model &mdash; Zerodha equity-intraday (NSE)</h2>'
        '<p class="small">Worked example: \u20b910,00,000 buy + \u20b910,00,000 sell (turnover \u20b920L), '
        "1 buy + 1 sell order. Statutory components match Zerodha&#39;s current charges "
        "(verified against zerodha.com/charges). Slippage is excluded (set to 0 in "
        "src/backtesting/costs.py) so the model matches Zerodha&#39;s fee table exactly -- "
        "statutory charges only, no market-impact assumption.</p>"
        f'<table><thead>{head}</thead><tbody>{body}</tbody></table></section>'
    )


def _section_journal(tdf: pd.DataFrame) -> str:
    df = tdf.copy()
    df["et"] = pd.to_datetime(df["entry_time"]).dt.strftime("%H:%M")
    df["xt"] = pd.to_datetime(df["exit_time"]).dt.strftime("%H:%M")
    ncols = 24
    head = (
        "<tr><th>#</th><th class='l'>Entry</th><th class='l'>Exit</th><th class='l'>Symbol</th>"
        "<th>Side</th><th>Qty</th><th>Buy \u20b9</th><th>Sell \u20b9</th>"
        "<th>Buy Value</th><th>Sell Value</th><th>R</th><th>MFE R</th>"
        "<th>Gross P&L</th><th>STT</th><th>Brkrg</th><th>Exch</th><th>SEBI</th>"
        "<th>Stamp</th><th>GST</th><th>Slip</th><th>Charges</th><th>Net P&L</th>"
        "<th class='l'>Reason</th><th class='l'>Flags</th></tr>"
    )
    body = ""
    for d, g in df.groupby("date", sort=True):
        dg = g["gross_pnl"].sum()
        dch = g["charges_total"].sum()
        dn = g["net_pnl"].sum()
        dcls = "pos" if dn >= 0 else "neg"
        body += (
            f'<tr class="dayhead"><td colspan="{ncols}">{d} &nbsp;|&nbsp; {len(g)} trade(s)'
            f' &nbsp;|&nbsp; Gross {_inr(dg)} &nbsp;|&nbsp; Charges {_inr(dch)}'
            f' &nbsp;|&nbsp; <span class="{dcls}">Net {_inr_signed(dn)}</span></td></tr>'
        )
        for _, r in g.iterrows():
            ncls = "pos" if r["net_pnl"] >= 0 else "neg"
            gcls = "pos" if r["gross_pnl"] >= 0 else "neg"
            tag = "tag-l" if r["direction"] == "LONG" else "tag-s"
            flags = []
            if r["green_setup"]:
                flags.append("green")
            if r["hammer_setup"]:
                flags.append("hammer")
            if r["trailed_to_be"]:
                flags.append("2R-BE")
            fstr = " ".join(flags) or "-"
            body += (
                f'<tr><td>{int(r["trade_no"])}</td><td class="l">{r["et"]}</td><td class="l">{r["xt"]}</td>'
                f'<td class="l">{_esc(r["symbol"])}</td><td><span class="tag {tag}">{r["direction"]}</span></td>'
                f'<td>{int(r["qty"])}</td><td>{r["buy_price"]:.2f}</td><td>{r["sell_price"]:.2f}</td>'
                f'<td>{_inr(r["buy_value"])}</td><td>{_inr(r["sell_value"])}</td>'
                f'<td>{r["r_multiple"]:.2f}</td><td>{r["mfe_r"]:.2f}</td>'
                f'<td class="{gcls}">{_inr_signed(r["gross_pnl"])}</td>'
                f'<td class="muted">{_inr(r["stt"])}</td><td class="muted">{_inr(r["brokerage"])}</td>'
                f'<td class="muted">{_inr(r["exchange"])}</td><td class="muted">{_inr(r["sebi"])}</td>'
                f'<td class="muted">{_inr(r["stamp"])}</td><td class="muted">{_inr(r["gst"])}</td>'
                f'<td class="muted">{_inr(r["slippage"])}</td><td class="muted">{_inr(r["charges_total"])}</td>'
                f'<td class="{ncls}"><b>{_inr_signed(r["net_pnl"])}</b></td>'
                f'<td class="l">{_esc(r["exit_reason"])}</td><td class="l">{fstr}</td></tr>'
            )
    return (
        '<section id="journal"><h2>Trade Journal (1 row = 1 trade, grouped by day)</h2>'
        f'<div style="overflow-x:auto"><table><thead>{head}</thead><tbody>{body}</tbody></table></div></section>'
    )
def _section_notes(cfg: dict) -> str:
    if cfg.get("trail_mode") == "ratchet":
        rmgmt = ("Risk mgmt: ratchet trail -- once price travels 2R in favour the stop "
                 "locks in +1R; once it travels 3R the stop locks in +2R (lets winners "
                 "breathe instead of flattening to breakeven).")
    else:
        rmgmt = "Risk mgmt: once price travels 2R (1:2) in favour, stop trails to entry (breakeven)."
    if cfg.get("max_notional") is None:
        sizing = (f"Sizing: risk per trade = Rs {cfg['risk_per_trade']:.0f}; quantity = "
                  f"floor(risk / (entry - SL)). One trade per stock per day. (NO notional cap "
                  f"-- baseline; positions can be very large when the stop is tight.)")
    else:
        sizing = (f"Sizing: BOTH conditions must hold -- max risk Rs {cfg['risk_per_trade']:.0f} AND "
                  f"max notional Rs {cfg['max_notional']:,.0f} per trade; quantity = "
                  f"min(floor(1000 / (entry - SL)), floor({cfg['max_notional']:,.0f} / entry)). "
                  f"One trade per stock per day.")
    bullets = [
        f"VARIANT: {cfg.get('variant', '')}",
        "Universe: all NIFTY-100 stocks with processed 1-min data in data/processed/.",
        f"Period: {cfg['start_date']} to {cfg['end_date']} (NSE trading days only).",
        "Data: 1-min OHLCV + session VWAP resampled to 5-min candles. VWAP is the cumulative intraday VWAP sampled at each 5-min candle's close.",
        "Trend / watchlist filter: the prior 5 candles must be FAR from VWAP -- their near edge (low for long / high for short) must sit a clear GAP beyond VWAP, >=0.20 x ATR(14) away, for >=4 of 5 candles (not merely closing on the trend side while the candle still touches VWAP); they must make higher highs AND higher lows (long) / lower highs AND lower lows (short), price extended >=0.10% from VWAP, and the day one-directional (close vs day open).",
        "VWAP bounce: candle wicks INTO VWAP (low<=VWAP long / high>=VWAP short) and closes back on the trend side (close>VWAP long / close<VWAP short) -- must NOT close on the wrong side. The bounce candle must be GREEN for long / RED for short (close vs open, i.e. the candle touching VWAP must be in the trade direction).",
        f"Entry: stop order at bounce-candle high+Rs{cfg['buffer']} (long) / low-Rs{cfg['buffer']} (short); filled at the stop or the gap-open if price gaps through.",
        f"Stop-loss: bounce-candle low-Rs{cfg['buffer']} (long) / high+Rs{cfg['buffer']} (short).",
        rmgmt,
        "Time-stop: exit at the CLOSE of the 15:10 5-min candle (price at ~15:15).",
        f"No trade before {cfg['not_before']} or after {cfg.get('not_after', '13:00:00')}; warm-up needs ATR({cfg['atr_window']}).",
        f"Avoid big candles: skip if bounce-candle range > {cfg['big_candle_mult']} x ATR({cfg['atr_window']}).",
        "Pin bar / hammer and green bounce candle are recorded as quality flags (not mandatory -- 'preferable' / 'good sign').",
        sizing,
        f"Max {cfg['max_trades_per_day']} trades/calendar day -- first entries by entry time; later setups skipped (as a trader stops trading).",
        "Charges: Zerodha equity-intraday round-trip costs (brokerage, STT, exchange, SEBI, stamp, GST) via src/backtesting/costs.py. Slippage / market-impact assumption is excluded (slippage_bps = 0).",
    ]
    # Avoidable-pattern filters applied to this run (Day-11 loss analysis).
    if not cfg.get("allow_long", True):
        bullets.append("DIRECTION FILTER: LONG setups DISABLED (shorts only) -- the long side showed no gross edge in the in-sample loss analysis.")
    if not cfg.get("allow_short", True):
        bullets.append("DIRECTION FILTER: SHORT setups DISABLED (longs only).")
    if cfg.get("skip_lunch_start") and cfg.get("skip_lunch_end"):
        bullets.append(f"TIME FILTER: no new entries during the lunch lull {cfg['skip_lunch_start']}-{cfg['skip_lunch_end']} -- Indian lunch-hour volume dries up and VWAP bounces lack follow-through.")
    lim = [
        "Candle-based execution with no intrabar look-ahead; if a stop and opposite stop can both trigger in one candle, the conservative fill (SL hit) is assumed.",
        "VWAP for the bounce test is sampled at the 5-min candle close (a small approximation vs per-minute VWAP).",
        "No lot-size/margin constraints beyond the configured notional cap; quantity scales to keep within both the risk and notional limits (actual risk can fall below Rs 1000 on cheap, tight-stop stocks).",
        "Slippage/charges use the project's default cost model; results are net of all costs.",
    ]
    bl = "".join(f"<li>{html.escape(b, quote=False)}</li>" for b in bullets)
    ll = "".join(f"<li>{html.escape(b, quote=False)}</li>" for b in lim)
    return (
        '<section id="notes"><h2>Methodology, Rules & Assumptions</h2>'
        '<h2 style="margin-top:14px">Rules implemented (exactly as specified)</h2><ul style="font-size:13px">' + bl + "</ul>"
        '<h2 style="margin-top:14px">Assumptions & limitations</h2><ul style="font-size:13px">' + ll + "</ul></section>"
    )
def build_html(tdf, daily, monthly, overall, summary):
    cfg = summary["config"]
    nav = ('<nav class="nav"><a href="#summary">Summary</a><a href="#costs">Costs</a>'
           '<a href="#equity">Equity</a><a href="#monthly">Monthly</a>'
           '<a href="#daily">Daily</a><a href="#journal">Journal</a><a href="#notes">Methodology</a></nav>')
    notional_txt = "no notional cap" if cfg.get("max_notional") is None else f"max notional Rs {cfg['max_notional']:,.0f}"
    header = (
        f'<h1>VWAP Trend-Continuation Backtest</h1>'
        f'<div class="sub"><b>{VARIANT_LABEL}</b> &bull; 5-min &bull; {START_DATE} to {END_DATE}'
        f' &bull; Rs {RISK_PER_TRADE:.0f} risk/trade &bull; {notional_txt}'
        f' &bull; trail={TRAIL_MODE} &bull; max {MAX_TRADES_PER_DAY} trades/day &bull; {len(tdf)} trades'
        f' over {overall["days_traded"]} days (of {overall["total_trading_days"]} trading days in window)</div>'
    )
    parts = [nav, header, _section_summary(overall, tdf), _section_costs(),
             _section_equity(daily), _section_monthly(monthly),
             _section_daily(daily), _section_journal(tdf), _section_notes(cfg)]
    foot = ('<p class="small" style="margin-top:30px">Generated by reports/day9_trend_continuation/run_backtest_v2.py. '
            'Report and data artifacts live under reports/day9_trend_continuation/ which is untracked (not committed).</p>')
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>VWAP Trend-Continuation Backtest Journal</title><style>' + CSS + '</style></head>'
        '<body><div class="wrap">' + "".join(parts) + foot + '</div></body></html>'
    )


def run_variant(label, suffix, notional, trail_mode, not_before, cached):
    """Run one variant: set the global knobs, simulate, write suffixed artifacts +
    HTML, and return the overall metrics dict for the comparison table."""
    global VARIANT_LABEL, OUT_SUFFIX, MAX_NOTIONAL, TRAIL_MODE, NOT_BEFORE
    VARIANT_LABEL = label
    OUT_SUFFIX = suffix
    MAX_NOTIONAL = notional
    TRAIL_MODE = trail_mode
    NOT_BEFORE = not_before
    notional_txt = "no cap" if notional is None else f"cap Rs {notional:,.0f}"
    print("\n" + "=" * 72)
    print(f"VARIANT: {label}")
    print(f"  start>={not_before}  |  trail={trail_mode}  |  notional={notional_txt}")
    print("-" * 72)
    t0 = time.time()
    all_trades: list[dict] = []
    trading_days: set = set()
    for sym, df in cached:
        trading_days.update(df["date"].unique())
        trs = simulate_stock(sym, df)
        if trs:
            all_trades.extend(trs)
    print(f"  Simulation done in {time.time()-t0:.1f}s. Entered (pre-cap): {len(all_trades)}")
    capped = apply_daily_cap(all_trades)
    print(f"  After {MAX_TRADES_PER_DAY}/day cap: {len(capped)} trades kept.")
    if not capped:
        print("  No trades -- skipping this variant.")
        return None
    tdf = pd.DataFrame(capped)
    tdf = tdf.sort_values(["date", "entry_time", "symbol"]).reset_index(drop=True)
    tdf["trade_no"] = range(1, len(tdf) + 1)
    daily = build_daily(tdf)
    monthly = build_monthly(tdf)
    overall = build_overall(tdf, daily, trading_days)
    cfg = {
        "strategy": "VWAP Trend Continuation (5-min)", "variant": label,
        "start_date": START_DATE, "end_date": END_DATE,
        "risk_per_trade": RISK_PER_TRADE, "max_trades_per_day": MAX_TRADES_PER_DAY,
        "max_notional": (None if MAX_NOTIONAL is None else float(MAX_NOTIONAL)),
        "buffer": BUFFER, "big_candle_mult": BIG_CANDLE_MULT, "atr_window": ATR_WINDOW,
        "not_before": str(NOT_BEFORE), "not_after": str(NOT_AFTER), "exit_bar_open": str(EXIT_BAR_OPEN),
        "lookback": LOOKBACK, "above_vwap_min": ABOVE_VWAP_MIN,
        "extend_min_dist": EXTEND_MIN_DIST, "gap_min_atr": GAP_MIN_ATR,
        "require_green_bounce": REQUIRE_GREEN_BOUNCE,
        "trail_mode": TRAIL_MODE,
        "allow_long": ALLOW_LONG, "allow_short": ALLOW_SHORT,
        "skip_lunch_start": (None if SKIP_LUNCH_START is None else str(SKIP_LUNCH_START)),
        "skip_lunch_end": (None if SKIP_LUNCH_END is None else str(SKIP_LUNCH_END)),
    }
    summary = {"config": cfg, "overall": overall}
    write_artifacts(tdf, daily, monthly, overall, len(trading_days))
    html = build_html(tdf, daily, monthly, overall, summary)
    (OUT_DIR / f"trend_continuation_report{OUT_SUFFIX}.html").write_text(html, encoding="utf-8")
    pf = "inf" if overall["profit_factor"] == float("inf") else f"{overall['profit_factor']:.2f}"
    print(f"  Trades: {overall['total_trades']}  |  Win: {overall['win_rate']*100:.1f}%  |  Net: {_inr_signed(overall['final_net'])}")
    print(f"  Gross: {_inr(overall['total_gross'])}  |  Charges: {_inr(overall['total_charges'])}  |  PF: {pf}  |  Avg R: {overall['avg_r']:.2f}")
    print(f"  REPORT: {OUT_DIR / f'trend_continuation_report{OUT_SUFFIX}.html'}")
    return overall


def _cmp_row(label, o):
    pf = "inf" if o["profit_factor"] == float("inf") else f"{o['profit_factor']:.2f}"
    return (label, str(o["total_trades"]), f"{o['win_rate']*100:.1f}%",
            _inr_signed(o["final_net"]), _inr(o["total_gross"]),
            _inr(o["total_charges"]), pf, f"{o['avg_r']:.2f}",
            _inr_signed(o["max_drawdown"]))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stocks = discover_stocks()
    print("=" * 72)
    print("VWAP TREND-CONTINUATION BACKTEST (5-min) -- v2 BEFORE/AFTER (AUGUST 2026)")
    print(f"Window: {START_DATE} .. {END_DATE}  |  Stocks: {len(stocks)}")
    print(f"Risk/trade: Rs {RISK_PER_TRADE:.0f}  |  Buffer Rs {BUFFER}")
    print("=" * 72)
    # Load 5-min data ONCE (both variants share the same August window).
    t0 = time.time()
    cached: list[tuple[str, pd.DataFrame]] = []
    for k, (sym, path) in enumerate(stocks, 1):
        df = load_5min(sym, path)
        if df is not None:
            cached.append((sym, df))
        if k % 20 == 0 or k == len(stocks):
            print(f"  loaded {k}/{len(stocks)} stocks  ({time.time()-t0:.1f}s)")
    print(f"Data loaded: {len(cached)} stocks in {time.time()-t0:.1f}s\n")

    base = run_variant(
        "BASELINE (original rules: 09:45 start, no notional cap, trail-to-BE at 2R)",
        "_aug_base", None, "be", dtime(9, 45), cached)
    cap_only = run_variant(
        "CAP-ONLY (Rs 2.5L notional cap, else original: 09:45, trail-to-BE)",
        "_aug_cap", 250000.0, "be", dtime(9, 45), cached)
    ratchet_only = run_variant(
        "RATCHET-ONLY (ratchet trail, else original: 09:45, no notional cap)",
        "_aug_ratchet", None, "ratchet", dtime(9, 45), cached)
    v2 = run_variant(
        "V2 (Rs 2.5L notional cap + ratchet trail + 10:50 start)",
        "_aug_v2", 250000.0, "ratchet", dtime(10, 50), cached)

    print("\n" + "=" * 72)
    print("BEFORE / AFTER COMPARISON (August 2026)")
    print("=" * 72)
    rows = []
    for label, o in [("BASELINE", base), ("CAP-ONLY", cap_only),
                     ("RATCHET-ONLY", ratchet_only), ("V2 (all 3)", v2)]:
        if o:
            rows.append(_cmp_row(label, o))
    hdr = ("Variant", "Trades", "Win", "Net P&L", "Gross", "Charges", "PF", "Avg R", "Max DD")
    widths = [14, 7, 6, 14, 14, 14, 5, 6, 14]
    def fmt_row(r):
        return "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    print(fmt_row(hdr))
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))
    print("=" * 72)
    print("Files (untracked, not committed) under reports/day9_trend_continuation/:")
    print("  BASELINE    : trend_continuation_report_aug_base.html")
    print("  CAP-ONLY    : trend_continuation_report_aug_cap.html")
    print("  RATCHET-ONLY: trend_continuation_report_aug_ratchet.html")
    print("  V2 (all 3)  : trend_continuation_report_aug_v2.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
