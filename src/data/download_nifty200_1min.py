"""Download 3 years of 1-minute OHLCV for the NIFTY 200 "next 100" via Kite Connect.

These are the NIFTY 200 constituents that are NOT in NIFTY 100 (ranks 101-200),
complementing ``download_nifty100_1min.py`` so ``data/raw/`` holds the full
NIFTY 200 universe. Already-downloaded symbols (covering the full date range)
are skipped automatically (resume support).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.kite_ohlcv import (  # noqa: E402
    CHUNK_DAYS,
    MAX_MINUTE_DAYS,
    download_range,
    equity_token_map,
    load_parquet,
    merge_frames,
    missing_ranges,
    plan_chunks,
    save_parquet,
    validate_frame,
)
from data.nifty200 import load_next100_symbols  # noqa: E402
from kite_client import authenticate  # noqa: E402

START = datetime(2023, 9, 4, 9, 15, 0)
END = datetime(2026, 9, 4, 15, 30, 0)
RAW_DIR = ROOT / "data" / "raw"
SUMMARY_PATH = RAW_DIR / "nifty200_1min_summary.json"
FAILURES_PATH = RAW_DIR / "nifty200_1min_failures.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NIFTY 200 (next 100) 1-minute candles from Kite.")
    parser.add_argument("--request-token", default=None, help="Kite login request token.")
    parser.add_argument("--plan-only", action="store_true", help="Print date/chunk plan and exit.")
    parser.add_argument("--symbol", action="append", default=None, help="Limit to one or more symbols.")
    return parser.parse_args()


def parquet_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol}_1min.parquet"


def legacy_paths(symbol: str) -> list[Path]:
    return [RAW_DIR / f"{symbol.lower()}_1min.parquet"]


def print_plan() -> dict:
    plan = plan_chunks(START, END, CHUNK_DAYS)
    print("NIFTY 200 (next 100) 1-minute download plan")
    print(f"START: {START.date()} ({START})")
    print(f"END:   {END.date()} ({END})")
    print(f"Interval: 1 minute")
    print(f"Kite max days per 1-minute request: {MAX_MINUTE_DAYS}")
    print(f"Chunk size: {plan['chunk_days']} calendar days")
    print(f"Chunk count: {plan['chunk_count']}")
    print(f"Max chunk span: {plan['max_span_days']} days")
    print(f"All chunks within 60-day limit: {plan['within_limit']}")
    print(f"First chunk: {plan['chunks'][0][0]} -> {plan['chunks'][0][1]}")
    print(f"Last chunk:  {plan['chunks'][-1][0]} -> {plan['chunks'][-1][1]}")
    if START.date().isoformat() != "2023-09-04" or END.date().isoformat() != "2026-09-04":
        raise RuntimeError("Refusing to download: date range is not 2023-09-04 → 2026-09-04.")
    if not plan["within_limit"]:
        raise RuntimeError("Refusing to download: a chunk exceeds the 60-day Kite limit.")
    return plan


def load_existing(symbol: str) -> pd.DataFrame | None:
    frames = []
    path = parquet_path(symbol)
    loaded = load_parquet(path)
    if loaded is not None:
        frames.append(loaded)
    for legacy in legacy_paths(symbol):
        if legacy != path:
            loaded = load_parquet(legacy)
            if loaded is not None:
                frames.append(loaded)
    if not frames:
        return None
    return merge_frames(*frames)


def download_symbol(kite, token: int, symbol: str) -> dict:
    existing = load_existing(symbol)
    ranges = missing_ranges(START, END, existing)
    if not ranges:
        print(f"  skip {symbol}: existing file already covers {START.date()} -> {END.date()}")
        df = existing
    else:
        parts = [existing] if existing is not None else []
        for range_start, range_end in ranges:
            print(f"  fetch {symbol}: {range_start} -> {range_end}")
            parts.append(download_range(kite, token, range_start, range_end, CHUNK_DAYS))
        df = merge_frames(*parts)

    stats = validate_frame(df, symbol)
    out_path = parquet_path(symbol)
    save_parquet(df, out_path)
    checked = duckdb.execute(
        """
        SELECT COUNT(*) AS rows,
               MIN(timestamp) AS min_ts,
               MAX(timestamp) AS max_ts
        FROM read_parquet(?)
        """,
        [str(out_path)],
    ).fetchone()
    if int(checked[0]) != stats["rows"]:
        raise RuntimeError(f"{symbol}: DuckDB row count mismatch.")
    stats["path"] = str(out_path)
    stats["duckdb_ok"] = True
    stats["min_timestamp"] = str(stats["min_timestamp"])
    stats["max_timestamp"] = str(stats["max_timestamp"])
    return stats


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> int:
    args = parse_args()
    try:
        plan = print_plan()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.plan_only:
        print(f"Would download {plan['chunk_count']} chunks per resolved stock.")
        return 0

    symbols, source = load_next100_symbols()
    if args.symbol:
        wanted = {item.upper() for item in args.symbol}
        symbols = [symbol for symbol in symbols if symbol.upper() in wanted]

    print(f"Constituents: {len(symbols)} from {source}")

    try:
        kite = authenticate(request_token=args.request_token)
        tokens = equity_token_map(kite)
        print(f"Loaded {len(tokens)} NSE EQ instruments")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    successes: list[dict] = []
    failures: list[dict] = []

    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] {symbol}")
        token = tokens.get(symbol)
        if token is None:
            failure = {"symbol": symbol, "error": "instrument token not found for NSE EQ"}
            failures.append(failure)
            print(f"  FAIL: {failure['error']}")
            write_json(FAILURES_PATH, failures)
            continue
        try:
            stats = download_symbol(kite, token, symbol)
            successes.append(stats)
            print(
                f"  OK rows={stats['rows']} min={stats['min_timestamp']} max={stats['max_timestamp']}"
            )
        except Exception as exc:
            failure = {"symbol": symbol, "error": str(exc)}
            failures.append(failure)
            print(f"  FAIL: {exc}")
            write_json(FAILURES_PATH, failures)
            continue

    summary = {
        "total_stocks": len(symbols),
        "successful": len(successes),
        "failed": len(failures),
        "total_rows": int(sum(item["rows"] for item in successes)),
        "earliest_timestamp": min((item["min_timestamp"] for item in successes), default=None),
        "latest_timestamp": max((item["max_timestamp"] for item in successes), default=None),
        "failures": failures,
        "successes": successes,
        "start": str(START),
        "end": str(END),
        "constituent_source": source,
    }
    write_json(SUMMARY_PATH, summary)

    print("\n=== NIFTY 200 (next 100) 1-minute summary ===")
    print(f"Total stocks: {summary['total_stocks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total rows: {summary['total_rows']}")
    print(f"Earliest timestamp: {summary['earliest_timestamp']}")
    print(f"Latest timestamp: {summary['latest_timestamp']}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure['symbol']}: {failure['error']}")
    return 0 if successes else 1


if __name__ == "__main__":
    sys.exit(main())
