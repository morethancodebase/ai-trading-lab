"""Day 3: build future-return target columns from processed bar indicators.

Process each stock independently (never load all ~27M rows at once).
Run from the project root:

    python -m src.data.build_targets
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

from .build_bar_indicators import EXPECTED_COLUMNS as INDICATOR_COLUMNS
from .date_splits import SPLIT_RANGES, classify_timestamp, split_bounds
from .kite_ohlcv import load_parquet, save_parquet
from .nifty100 import NIFTY100_FALLBACK

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9&\-]*$")

TARGET_COLUMNS = [
    "future_return_5m",
    "future_return_15m",
    "future_return_30m",
    "future_return_60m",
]

FUTURE_HORIZONS = (5, 15, 30, 60)

EXPECTED_COLUMNS = INDICATOR_COLUMNS + TARGET_COLUMNS


def discover_indicator_files(processed_dir: Path = PROCESSED_DIR) -> list[tuple[str, Path]]:
    """Find `{SYMBOL}_1min_indicators.parquet` files (case-insensitive)."""
    by_lower: dict[str, Path] = {}
    for path in processed_dir.glob("*_1min_indicators.parquet"):
        stem = path.name[: -len("_1min_indicators.parquet")]
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

    known_lower = {symbol.lower() for symbol, _ in found}
    for lower, path in sorted(by_lower.items()):
        if lower in known_lower:
            continue
        found.append((lower.upper(), path))
    return found


def _session_day(df: pd.DataFrame) -> pd.Series:
    """Calendar trading session key from timestamp (one session per date)."""
    return pd.to_datetime(df["timestamp"]).dt.floor("D")


def add_future_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Append future-return targets within each trading session (no cross-session look-ahead)."""
    if df.empty:
        out = df.copy()
        for col in TARGET_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out[EXPECTED_COLUMNS]

    out = df[INDICATOR_COLUMNS].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    close = out["close"].astype("float64")
    day = _session_day(out)
    for minutes in FUTURE_HORIZONS:
        # Shift within the session only: the last `minutes` bars of a session
        # have no future bar in the same session, so the target is NaN.
        future_close = close.groupby(day, sort=False).shift(-minutes)
        out[f"future_return_{minutes}m"] = (future_close / close) - 1.0

    for col in TARGET_COLUMNS:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out[EXPECTED_COLUMNS]


def _manual_return_checks(processed: pd.DataFrame, sample_count: int = 5) -> list[str]:
    """Verify future-return math on random rows whose furthest horizon stays in-session."""
    errors: list[str] = []
    if len(processed) <= max(FUTURE_HORIZONS):
        return errors

    day = pd.to_datetime(processed["timestamp"]).dt.floor("D").to_numpy()
    close = processed["close"].astype("float64").to_numpy()
    n = len(processed)
    furthest = max(FUTURE_HORIZONS)
    # Candidate rows: the bar `furthest` ahead exists AND is in the same session,
    # so every horizon (<= furthest) is also within the same session.
    candidates = [i for i in range(n - furthest) if day[i] == day[i + furthest]]
    if not candidates:
        return errors

    rng = np.random.default_rng(42)
    picks = rng.choice(len(candidates), size=min(sample_count, len(candidates)), replace=False)
    for ci in picks:
        idx = candidates[ci]
        for minutes in FUTURE_HORIZONS:
            col = f"future_return_{minutes}m"
            expected = (close[idx + minutes] / close[idx]) - 1.0
            actual = processed.iloc[idx][col]
            if not np.isclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=False):
                errors.append(
                    f"manual check failed at row {idx} {col}: expected={expected}, actual={actual}"
                )
    return errors


def _tail_nan_checks(processed: pd.DataFrame) -> list[str]:
    """Last N rows must be NaN for future_return_Nm."""
    errors: list[str] = []
    for minutes in FUTURE_HORIZONS:
        col = f"future_return_{minutes}m"
        tail = processed[col].iloc[-minutes:]
        if not tail.isna().all():
            bad = int(tail.notna().sum())
            errors.append(f"tail NaN check failed for {col}: {bad} non-NaN in last {minutes} rows")
    return errors


def _session_boundary_target_checks(processed: pd.DataFrame) -> list[str]:
    """Future-return targets must never cross a trading-session boundary."""
    errors: list[str] = []
    if processed.empty:
        return errors
    day = pd.to_datetime(processed["timestamp"]).dt.floor("D")
    for minutes in FUTURE_HORIZONS:
        col = f"future_return_{minutes}m"
        vals = processed[col]
        # Day of the bar `minutes` positions ahead (NaT when it does not exist).
        future_day = day.shift(-minutes)
        # A row crosses if that future bar is missing or belongs to another session.
        crosses = ~((future_day == day).fillna(False))
        bad = int((crosses & vals.notna()).sum())
        if bad:
            errors.append(
                f"session-boundary leak in {col}: {bad} targets cross/missing session end but are not NaN"
            )
    return errors


def validate_targets(
    indicators: pd.DataFrame,
    processed: pd.DataFrame,
    symbol: str,
) -> dict[str, Any]:
    """Validate one targets frame against its indicators source."""
    errors: list[str] = []

    if len(processed) != len(indicators):
        errors.append(f"row count mismatch: input={len(indicators)} output={len(processed)}")

    if list(processed.columns) != EXPECTED_COLUMNS:
        unexpected = [c for c in processed.columns if c not in EXPECTED_COLUMNS]
        missing = [c for c in EXPECTED_COLUMNS if c not in processed.columns]
        if unexpected:
            errors.append(f"unexpected columns: {unexpected}")
        if missing:
            errors.append(f"missing columns: {missing}")

    indicators_sorted = indicators[INDICATOR_COLUMNS].sort_values("timestamp").reset_index(drop=True)
    processed_indicators = processed[INDICATOR_COLUMNS].reset_index(drop=True)

    if not processed["timestamp"].equals(indicators_sorted["timestamp"]):
        errors.append("timestamps do not match input file")

    dupes = int(processed["timestamp"].duplicated().sum())
    if dupes:
        errors.append(f"duplicate timestamps: {dupes}")

    for col in INDICATOR_COLUMNS:
        if col == "timestamp":
            continue
        left = pd.to_numeric(processed_indicators[col], errors="coerce")
        right = pd.to_numeric(indicators_sorted[col], errors="coerce")
        if not np.allclose(left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64"), equal_nan=True):
            errors.append(f"existing column changed: {col}")

    for col in TARGET_COLUMNS:
        if col not in processed.columns:
            errors.append(f"missing target column: {col}")

    target_vals = processed[TARGET_COLUMNS]
    inf_count = int(np.isinf(target_vals.to_numpy(dtype="float64")).sum())
    if inf_count:
        errors.append(f"infinite target values: {inf_count}")

    errors.extend(_manual_return_checks(processed))
    errors.extend(_tail_nan_checks(processed))
    errors.extend(_session_boundary_target_checks(processed))

    split_counts = {"train": 0, "validation": 0, "test": 0, "outside": 0}
    for ts in processed["timestamp"]:
        split = classify_timestamp(ts)
        if split is None:
            split_counts["outside"] += 1
        else:
            split_counts[split] += 1

    nan_counts = {col: int(processed[col].isna().sum()) for col in TARGET_COLUMNS}

    return {
        "symbol": symbol,
        "rows": int(len(processed)),
        "ok": not errors,
        "errors": errors,
        "nan_counts": nan_counts,
        "duplicate_timestamps": dupes,
        "infinite_values": inf_count,
        "split_counts": split_counts,
    }


def duckdb_spot_check(path: Path) -> dict[str, Any]:
    """Light DuckDB verification on one targets Parquet file."""
    row = duckdb.execute(
        """
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT timestamp) AS distinct_ts,
            MIN(timestamp) AS min_ts,
            MAX(timestamp) AS max_ts,
            SUM(
                CASE
                    WHEN isinf(future_return_5m)
                      OR isinf(future_return_15m)
                      OR isinf(future_return_30m)
                      OR isinf(future_return_60m)
                    THEN 1 ELSE 0
                END
            ) AS inf_targets
        FROM read_parquet(?)
        """,
        [str(path)],
    ).fetchone()
    cols = duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()["column_name"].tolist()

    split_rows = duckdb.execute(
        """
        SELECT
            SUM(CASE WHEN CAST(timestamp AS DATE) BETWEEN ? AND ? THEN 1 ELSE 0 END) AS train_rows,
            SUM(CASE WHEN CAST(timestamp AS DATE) BETWEEN ? AND ? THEN 1 ELSE 0 END) AS validation_rows,
            SUM(CASE WHEN CAST(timestamp AS DATE) BETWEEN ? AND ? THEN 1 ELSE 0 END) AS test_rows
        FROM read_parquet(?)
        """,
        [
            str(SPLIT_RANGES["train"][0]),
            str(SPLIT_RANGES["train"][1]),
            str(SPLIT_RANGES["validation"][0]),
            str(SPLIT_RANGES["validation"][1]),
            str(SPLIT_RANGES["test"][0]),
            str(SPLIT_RANGES["test"][1]),
            str(path),
        ],
    ).fetchone()

    return {
        "path": str(path),
        "rows": int(row[0]),
        "distinct_timestamps": int(row[1]),
        "min_timestamp": str(row[2]),
        "max_timestamp": str(row[3]),
        "inf_targets": int(row[4] or 0),
        "columns": cols,
        "train_rows": int(split_rows[0] or 0),
        "validation_rows": int(split_rows[1] or 0),
        "test_rows": int(split_rows[2] or 0),
        "ok": int(row[0]) == int(row[1]) and int(row[4] or 0) == 0 and cols == EXPECTED_COLUMNS,
    }


def process_symbol(
    symbol: str,
    indicators_path: Path,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, Any]:
    """Load one indicators file, add targets, validate, and write targets Parquet."""
    indicators = load_parquet(indicators_path)
    if indicators is None or indicators.empty:
        raise RuntimeError(f"{symbol}: indicators file missing or empty ({indicators_path})")

    if list(indicators.columns) != INDICATOR_COLUMNS:
        raise RuntimeError(
            f"{symbol}: unexpected indicator schema (expected {len(INDICATOR_COLUMNS)} columns)"
        )

    processed = add_future_returns(indicators)
    validation = validate_targets(indicators, processed, symbol)
    if not validation["ok"]:
        raise RuntimeError(f"{symbol}: validation failed: {validation['errors']}")

    out_path = processed_dir / f"{symbol}_1min_targets.parquet"
    save_parquet(processed, out_path)
    validation["path"] = str(out_path)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build future-return target Parquet files from processed bar indicators."
    )
    parser.add_argument("--symbol", action="append", default=None, help="Limit to one or more symbols.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory with *_1min_indicators.parquet and output *_1min_targets.parquet files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    processed_dir = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = discover_indicator_files(processed_dir)
    if args.symbol:
        wanted = {item.upper() for item in args.symbol}
        files = [(symbol, path) for symbol, path in files if symbol.upper() in wanted]

    if not files:
        print(f"No matching *_1min_indicators.parquet files found in {processed_dir}", file=sys.stderr)
        return 1

    bounds = split_bounds()
    print(f"Building future-return targets: {len(files)} stock(s)")
    print(f"Processed dir: {processed_dir}")
    print("Date splits (inclusive):")
    for name in ("train", "validation", "test"):
        start = bounds[name]["start"]
        end = bounds[name]["end"]
        print(f"  {name}: {start} -> {end}")

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, (symbol, indicators_path) in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {symbol}")
        try:
            t0 = time.perf_counter()
            result = process_symbol(symbol, indicators_path, processed_dir)
            elapsed = time.perf_counter() - t0
            successes.append(result)
            nan_preview = {k: result["nan_counts"][k] for k in TARGET_COLUMNS}
            print(
                f"  OK rows={result['rows']} elapsed={elapsed:.1f}s "
                f"nan_preview={nan_preview} splits={result['split_counts']}"
            )
        except Exception as exc:
            failures.append({"symbol": symbol, "error": str(exc)})
            print(f"  FAIL: {exc}")

    total_elapsed = time.perf_counter() - started
    total_rows = int(sum(item["rows"] for item in successes))

    print("\n=== Day 3 future-return target summary ===")
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
        print(f"Inf target count: {check['inf_targets']}")
        print(
            f"Split rows: train={check['train_rows']} "
            f"validation={check['validation_rows']} test={check['test_rows']}"
        )
        print(f"Columns ({len(check['columns'])}): {check['columns']}")
        print(f"DuckDB OK: {check['ok']}")

        schema = duckdb.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(sample_path)]).fetchdf()
        print("\n=== Example schema ===")
        print(schema.to_string(index=False))

        preview = duckdb.execute(
            """
            SELECT
                timestamp, close,
                future_return_5m, future_return_15m,
                future_return_30m, future_return_60m
            FROM read_parquet(?)
            ORDER BY timestamp
            LIMIT 5
            """,
            [str(sample_path)],
        ).fetchdf()
        print("\n=== Example rows (first 5, target columns) ===")
        print(preview.to_string(index=False))

        tail_preview = duckdb.execute(
            """
            SELECT
                timestamp, close,
                future_return_5m, future_return_15m,
                future_return_30m, future_return_60m
            FROM read_parquet(?)
            ORDER BY timestamp DESC
            LIMIT 5
            """,
            [str(sample_path)],
        ).fetchdf()
        print("\n=== Example rows (last 5, should be NaN targets) ===")
        print(tail_preview.to_string(index=False))

    return 0 if successes and not failures else (0 if successes else 1)


if __name__ == "__main__":
    sys.exit(main())
