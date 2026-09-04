"""Download one month of NSE RELIANCE 1-minute OHLCV from Kite Connect."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import (
    DataException,
    KiteException,
    NetworkException,
    TokenException,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kite_client import authenticate  # noqa: E402

TICKER = "RELIANCE"
EXCHANGE = "NSE"
INTERVAL = "minute"
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
OUTPUT_PATH = ROOT / "data" / "raw" / "reliance_1min.parquet"
CHUNK_DAYS = 7
MAX_MINUTE_DAYS = 60
REQUEST_PAUSE_SECONDS = 1.0
MAX_RETRIES = 5
VALID_OHLC_MIN_ROWS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download RELIANCE 1-minute candles from Kite.")
    parser.add_argument("--request-token", default=None, help="Kite login request token.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Calendar days of history to download after the small probe (default: 30).",
    )
    return parser.parse_args()


def resolve_instrument_token(kite: KiteConnect) -> int:
    try:
        instruments = kite.instruments(EXCHANGE)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {EXCHANGE} instruments: {exc}") from exc

    matches = [
        row
        for row in instruments
        if row.get("tradingsymbol") == TICKER
        and row.get("instrument_type") == "EQ"
        and row.get("exchange") == EXCHANGE
    ]
    if not matches:
        raise RuntimeError(f"Could not find equity instrument {EXCHANGE}:{TICKER}.")
    token = matches[0].get("instrument_token")
    if not token:
        raise RuntimeError(f"Instrument {EXCHANGE}:{TICKER} is missing instrument_token.")
    return int(token)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (NetworkException, DataException)):
        return True
    if isinstance(exc, TokenException):
        return False
    if isinstance(exc, KiteException) and exc.code in {429, 500, 502, 503, 504}:
        return True
    return False


def fetch_historical_chunk(
    kite: KiteConnect,
    instrument_token: int,
    start: datetime,
    end: datetime,
) -> list[dict]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return kite.historical_data(
                instrument_token,
                start,
                end,
                INTERVAL,
                continuous=False,
                oi=False,
            )
        except Exception as exc:
            last_error = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Historical API failed for {start} to {end}: {exc}"
                ) from exc
            sleep_for = REQUEST_PAUSE_SECONDS * (2 ** (attempt - 1))
            print(f"Retry {attempt}/{MAX_RETRIES} after error ({exc}); sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"Historical API failed for {start} to {end}: {last_error}")


def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(candles)
    timestamps = pd.to_datetime(df["date"])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce").astype("Int64"),
        }
    )
    out = out.dropna(subset=OUTPUT_COLUMNS)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[OUTPUT_COLUMNS]


def validate_candles(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise RuntimeError(f"{label}: no candles returned.")
    if len(df) < VALID_OHLC_MIN_ROWS:
        raise RuntimeError(f"{label}: expected at least {VALID_OHLC_MIN_ROWS} candles, got {len(df)}.")
    if list(df.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(f"{label}: unexpected columns {list(df.columns)}.")
    invalid = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["volume"] < 0)
    )
    if invalid.any():
        raise RuntimeError(f"{label}: found {int(invalid.sum())} candles with invalid OHLC/volume.")


def iter_chunks(start: datetime, end: datetime, chunk_days: int) -> list[tuple[datetime, datetime]]:
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    delta = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def download_range(
    kite: KiteConnect,
    instrument_token: int,
    start: datetime,
    end: datetime,
    chunk_days: int,
) -> pd.DataFrame:
    span_days = (end - start).days
    if span_days > MAX_MINUTE_DAYS and chunk_days > MAX_MINUTE_DAYS:
        raise RuntimeError(
            f"1-minute requests cannot exceed {MAX_MINUTE_DAYS} days per call; use smaller chunks."
        )

    frames: list[pd.DataFrame] = []
    chunks = iter_chunks(start, end, min(chunk_days, MAX_MINUTE_DAYS))
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"Fetching chunk {index}/{len(chunks)}: {chunk_start} -> {chunk_end}")
        candles = fetch_historical_chunk(kite, instrument_token, chunk_start, chunk_end)
        frames.append(candles_to_frame(candles))
        if index < len(chunks):
            time.sleep(REQUEST_PAUSE_SECONDS)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def last_completed_weekday(now: datetime) -> datetime:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if now.hour < 16:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        duckdb.execute(
            "COPY (SELECT * FROM df) TO ? (FORMAT PARQUET, OVERWRITE TRUE)",
            [str(path)],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to write Parquet file {path}: {exc}") from exc


def verify_parquet(path: Path, expected_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, object, object]:
    if not path.exists():
        raise RuntimeError(f"Parquet file was not created: {path}")
    try:
        count = duckdb.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()[0]
        head = duckdb.execute(
            "SELECT * FROM read_parquet(?) ORDER BY timestamp LIMIT 5",
            [str(path)],
        ).df()
        tail = duckdb.execute(
            """
            SELECT * FROM (
                SELECT * FROM read_parquet(?) ORDER BY timestamp DESC LIMIT 5
            ) ORDER BY timestamp
            """,
            [str(path)],
        ).df()
        bounds = duckdb.execute(
            "SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
    except Exception as exc:
        raise RuntimeError(f"DuckDB could not query {path}: {exc}") from exc

    if int(count) != expected_rows:
        raise RuntimeError(f"Parquet row count mismatch: expected {expected_rows}, DuckDB found {count}.")
    return head, tail, bounds[0], bounds[1]


def main() -> int:
    args = parse_args()
    now = datetime.now()
    probe_day = last_completed_weekday(now)
    probe_start = probe_day.replace(hour=9, minute=15, second=0)
    probe_end = probe_day.replace(hour=15, minute=30, second=0)
    month_end = now.replace(second=0, microsecond=0)
    month_start = (month_end - timedelta(days=args.days)).replace(hour=9, minute=15, second=0)

    try:
        kite = authenticate(request_token=args.request_token)
        token = resolve_instrument_token(kite)
        print(f"Resolved {EXCHANGE}:{TICKER} instrument_token={token}")

        print(f"Probe download: {probe_start} -> {probe_end}")
        probe = download_range(kite, token, probe_start, probe_end, chunk_days=1)
        validate_candles(probe, "probe")
        print(f"Probe OK: {len(probe)} candles")

        print(f"Month download: {month_start} -> {month_end}")
        df = download_range(kite, token, month_start, month_end, chunk_days=CHUNK_DAYS)
        validate_candles(df, "month download")
        save_parquet(df, OUTPUT_PATH)
        head, tail, min_ts, max_ts = verify_parquet(OUTPUT_PATH, len(df))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Min timestamp: {min_ts}")
    print(f"Max timestamp: {max_ts}")
    print("First 5 rows:")
    print(head.to_string(index=False))
    print("Last 5 rows:")
    print(tail.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
