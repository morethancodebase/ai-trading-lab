"""Day 2: build bar indicators from raw 1-minute OHLCV.

Process each stock independently (never load all ~27M rows at once).
Run from the project root:

    python -m src.data.build_bar_indicators
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .kite_ohlcv import OUTPUT_COLUMNS, load_parquet, save_parquet
from .nifty100 import NIFTY100_FALLBACK

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# NSE-style tradingsymbol after case normalization (handles macOS case-insensitive FS).
SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-]*$")

INDICATOR_COLUMNS = [
    "return_1m",
    "return_5m",
    "return_15m",
    "return_30m",
    "return_60m",
    "candle_range",
    "candle_body",
    "body_ratio",
    "close_position",
    "volume_change_1m",
    "volume_sma_20",
    "volume_ratio_20",
    "sma_5",
    "sma_20",
    "sma_50",
    "sma_200",
    "vwap",
    "distance_from_vwap",
    "volatility_20",
    "volatility_60",
    "rsi_14",
]

EXPECTED_COLUMNS = OUTPUT_COLUMNS + INDICATOR_COLUMNS


def discover_raw_files(raw_dir: Path = RAW_DIR) -> list[tuple[str, Path]]:
    """Find NIFTY 100 `{SYMBOL}_1min.parquet` files (case-insensitive).

    Canonicalizes symbols to uppercase so legacy names like ``reliance_1min.parquet``
    still map to ``RELIANCE`` on case-insensitive filesystems, while unrelated
    non-symbol files are ignored.
    """
    by_lower: dict[str, Path] = {}
    for path in raw_dir.glob("*_1min.parquet"):
        stem = path.name[: -len("_1min.parquet")]
        symbol = stem.upper()
        if not SYMBOL_RE.match(symbol):
            continue
        by_lower[symbol.lower()] = path

    known = {symbol.upper() for symbol in NIFTY100_FALLBACK}
    found: list[tuple[str, Path]] = []
    for symbol in sorted(known):
        path = by_lower.get(symbol.lower())
        if path is not None:
            found.append((symbol, path))

    # Also include any other uppercase-normalized 1-min files not in the snapshot.
    known_lower = {symbol.lower() for symbol, _ in found}
    for lower, path in sorted(by_lower.items()):
        if lower in known_lower:
            continue
        found.append((lower.upper(), path))
    return found


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN when the denominator is zero."""
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _session_day(df: pd.DataFrame) -> pd.Series:
    """Calendar trading session key from timestamp (one session per date)."""
    return pd.to_datetime(df["timestamp"]).dt.floor("D")


def _group_pct_change(series: pd.Series, day: pd.Series, periods: int) -> pd.Series:
    """Pct-change within each trading session; no cross-day lookback."""
    return series.groupby(day, sort=False).pct_change(periods)


def _group_rolling_mean(series: pd.Series, day: pd.Series, window: int) -> pd.Series:
    """Rolling mean within each trading session."""
    return series.groupby(day, sort=False).transform(
        lambda values: values.rolling(window=window, min_periods=window).mean()
    )


def _group_rolling_std(series: pd.Series, day: pd.Series, window: int) -> pd.Series:
    """Rolling std within each trading session."""
    return series.groupby(day, sort=False).transform(
        lambda values: values.rolling(window=window, min_periods=window).std()
    )


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI using EWM alpha=1/period (no look-ahead)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = _safe_div(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When average loss is zero and average gain > 0, RSI is conventionally 100.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), np.nan)
    return rsi


def compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
    """Cumulative intraday VWAP from typical price; resets each calendar day."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    day = pd.to_datetime(df["timestamp"]).dt.floor("D")
    tp_volume = typical_price * df["volume"].astype("float64")
    cum_tp_volume = tp_volume.groupby(day, sort=False).cumsum()
    cum_volume = df["volume"].astype("float64").groupby(day, sort=False).cumsum()
    return _safe_div(cum_tp_volume, cum_volume)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add Day-2 bar indicator columns. Preserves OHLCV; warm-up rows stay NaN."""
    if df.empty:
        return df.copy()

    out = df[OUTPUT_COLUMNS].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    close = out["close"].astype("float64")
    open_ = out["open"].astype("float64")
    high = out["high"].astype("float64")
    low = out["low"].astype("float64")
    volume = out["volume"].astype("float64")
    candle_range = high - low
    day = _session_day(out)

    for minutes in (1, 5, 15, 30, 60):
        out[f"return_{minutes}m"] = _group_pct_change(close, day, minutes)

    out["candle_range"] = candle_range
    out["candle_body"] = close - open_
    out["body_ratio"] = _safe_div((close - open_).abs(), candle_range)
    out["close_position"] = _safe_div(close - low, candle_range)

    out["volume_change_1m"] = _group_pct_change(volume, day, 1)
    out["volume_sma_20"] = _group_rolling_mean(volume, day, window=20)
    out["volume_ratio_20"] = _safe_div(volume, out["volume_sma_20"])

    for window in (5, 20, 50, 200):
        out[f"sma_{window}"] = _group_rolling_mean(close, day, window=window)

    out["vwap"] = compute_intraday_vwap(out)
    out["distance_from_vwap"] = _safe_div(close - out["vwap"], out["vwap"])

    out["volatility_20"] = _group_rolling_std(out["return_1m"], day, window=20)
    out["volatility_60"] = _group_rolling_std(out["return_1m"], day, window=60)
    out["rsi_14"] = close.groupby(day, sort=False).transform(
        lambda values: compute_rsi(values.astype("float64"), period=14)
    )

    # pct_change / ratios can produce ±inf when a prior value is zero; treat as missing.
    for col in INDICATOR_COLUMNS:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out[EXPECTED_COLUMNS]


def _session_boundary_checks(processed: pd.DataFrame) -> list[str]:
    """Ensure shift/rolling indicators do not use the previous trading session."""
    errors: list[str] = []
    day = _session_day(processed)
    session_start = day != day.shift(1)

    boundary_cols = [
        "return_1m",
        "return_5m",
        "return_15m",
        "return_30m",
        "return_60m",
        "volume_change_1m",
        "volume_sma_20",
        "sma_5",
        "sma_20",
        "sma_50",
        "sma_200",
        "volatility_20",
        "volatility_60",
        "rsi_14",
    ]
    for col in boundary_cols:
        if col not in processed.columns:
            continue
        bad = int(processed.loc[session_start, col].notna().sum())
        if bad:
            errors.append(f"session boundary leak in {col}: {bad} non-NaN at session starts")

    return errors


def validate_indicators(raw: pd.DataFrame, processed: pd.DataFrame, symbol: str) -> dict[str, Any]:
    """Validate one processed frame against its raw source."""
    errors: list[str] = []

    if len(processed) != len(raw):
        errors.append(f"row count mismatch: raw={len(raw)} processed={len(processed)}")

    if list(processed.columns) != EXPECTED_COLUMNS:
        unexpected = [c for c in processed.columns if c not in EXPECTED_COLUMNS]
        missing = [c for c in EXPECTED_COLUMNS if c not in processed.columns]
        if unexpected:
            errors.append(f"unexpected columns: {unexpected}")
        if missing:
            errors.append(f"missing columns: {missing}")

    raw_sorted = raw[OUTPUT_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    processed_ohlcv = processed[OUTPUT_COLUMNS].reset_index(drop=True)

    if not processed["timestamp"].equals(raw_sorted["timestamp"]):
        errors.append("timestamps do not match raw file")

    dupes = int(processed["timestamp"].duplicated().sum())
    if dupes:
        errors.append(f"duplicate timestamps: {dupes}")

    for col in OUTPUT_COLUMNS:
        if col == "timestamp":
            continue
        left = pd.to_numeric(processed_ohlcv[col], errors="coerce")
        right = pd.to_numeric(raw_sorted[col], errors="coerce")
        if not np.allclose(left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64"), equal_nan=True):
            errors.append(f"OHLCV column changed: {col}")

    indicator_vals = processed[INDICATOR_COLUMNS]
    inf_count = int(np.isinf(indicator_vals.to_numpy(dtype="float64")).sum())
    if inf_count:
        errors.append(f"infinite indicator values: {inf_count}")

    errors.extend(_session_boundary_checks(processed))

    nan_counts = {col: int(processed[col].isna().sum()) for col in INDICATOR_COLUMNS}

    return {
        "symbol": symbol,
        "rows": int(len(processed)),
        "ok": not errors,
        "errors": errors,
        "nan_counts": nan_counts,
        "duplicate_timestamps": dupes,
        "infinite_values": inf_count,
    }


def duckdb_spot_check(path: Path) -> dict[str, Any]:
    """Light DuckDB verification on one processed Parquet file."""
    row = duckdb.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT timestamp) AS distinct_ts,
            MIN(timestamp) AS min_ts,
            MAX(timestamp) AS max_ts,
            SUM(CASE WHEN isinf(return_1m) OR isinf(vwap) OR isinf(rsi_14) THEN 1 ELSE 0 END) AS inf_sample
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()
    cols = duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()["column_name"].tolist()
    return {
        "path": str(path),
        "rows": int(row[0]),
        "distinct_timestamps": int(row[1]),
        "min_timestamp": str(row[2]),
        "max_timestamp": str(row[3]),
        "inf_sample": int(row[4] or 0),
        "columns": cols,
        "ok": int(row[0]) == int(row[1]) and int(row[4] or 0) == 0 and cols == EXPECTED_COLUMNS,
    }


def process_symbol(symbol: str, raw_path: Path, processed_dir: Path = PROCESSED_DIR) -> dict[str, Any]:
    """Load one raw file, compute bar indicators, validate, and write processed Parquet."""
    raw = load_parquet(raw_path)
    if raw is None or raw.empty:
        raise RuntimeError(f"{symbol}: raw file missing or empty ({raw_path})")

    processed = add_indicators(raw)
    validation = validate_indicators(raw, processed, symbol)
    if not validation["ok"]:
        raise RuntimeError(f"{symbol}: validation failed: {validation['errors']}")

    out_path = processed_dir / f"{symbol}_1min_indicators.parquet"
    save_parquet(processed, out_path)
    validation["path"] = str(out_path)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build bar-indicator Parquet files from raw 1-minute OHLCV."
    )
    parser.add_argument("--symbol", action="append", default=None, help="Limit to one or more symbols.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory with raw *_1min.parquet files.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for *_1min_indicators.parquet files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir
    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = discover_raw_files(raw_dir)
    if args.symbol:
        wanted = {item.upper() for item in args.symbol}
        files = [(symbol, path) for symbol, path in files if symbol.upper() in wanted]

    if not files:
        print(f"No matching raw *_1min.parquet files found in {raw_dir}", file=sys.stderr)
        return 1

    print(f"Building bar indicators: {len(files)} stock(s)")
    print(f"Raw dir: {raw_dir}")
    print(f"Processed dir: {processed_dir}")

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, (symbol, raw_path) in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {symbol}")
        try:
            t0 = time.perf_counter()
            result = process_symbol(symbol, raw_path, processed_dir)
            elapsed = time.perf_counter() - t0
            successes.append(result)
            nan_preview = {k: result["nan_counts"][k] for k in ("return_1m", "sma_200", "rsi_14", "vwap")}
            print(
                f"  OK rows={result['rows']} elapsed={elapsed:.1f}s "
                f"nan_preview={nan_preview}"
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            print(f"  FAIL: {exc}")

    total_elapsed = time.perf_counter() - started
    total_rows = int(sum(item["rows"] for item in successes))

    print("\n=== Day 2 bar-indicator summary ===")
    print(f"Stocks processed: {len(successes)}")
    print(f"Stocks failed: {len(failures)}")
    print(f"Total rows processed: {total_rows}")
    print(f"Total processing time: {total_elapsed:.1f}s")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure['symbol']}: {failure['error']}")

    if successes:
        sample = successes[0]
        sample_path = Path(sample["path"])
        check = duckdb_spot_check(sample_path)
        print("\n=== DuckDB spot check ===")
        print(f"File: {check['path']}")
        print(f"Rows: {check['rows']} (distinct timestamps: {check['distinct_timestamps']})")
        print(f"Range: {check['min_timestamp']} -> {check['max_timestamp']}")
        print(f"Inf sample count: {check['inf_sample']}")
        print(f"Columns ({len(check['columns'])}): {check['columns']}")
        print(f"DuckDB OK: {check['ok']}")

        schema = duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(sample_path)]).fetchdf()
        print("\n=== Example schema ===")
        print(schema.to_string(index=False))
        preview = duckdb.execute(
            "SELECT * FROM read_parquet(?) ORDER BY timestamp LIMIT 3",
            [str(sample_path)],
        ).fetchdf()
        print("\n=== Example rows (first 3) ===")
        print(preview.to_string(index=False))

    return 0 if successes and not failures else (0 if successes else 1)


if __name__ == "__main__":
    sys.exit(main())
