"""Shared Kite 1-minute OHLCV download, validation, and Parquet helpers."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
from kiteconnect import KiteConnect
from kiteconnect.exceptions import DataException, KiteException, NetworkException, TokenException

EXCHANGE = "NSE"
INTERVAL = "minute"
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
MAX_MINUTE_DAYS = 60
# Inclusive from/to windows of 60 days can be rejected; stay strictly under the cap.
CHUNK_DAYS = 59
REQUEST_PAUSE_SECONDS = 0.4
MAX_RETRIES = 5


def iter_chunks(start: datetime, end: datetime, chunk_days: int = CHUNK_DAYS) -> list[tuple[datetime, datetime]]:
    if chunk_days > MAX_MINUTE_DAYS:
        raise ValueError(f"chunk_days={chunk_days} exceeds Kite 1-minute limit of {MAX_MINUTE_DAYS} days")
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    delta = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        span = (chunk_end.date() - cursor.date()).days
        if span > MAX_MINUTE_DAYS:
            raise ValueError(f"Chunk {cursor} -> {chunk_end} spans {span} days (> {MAX_MINUTE_DAYS})")
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def plan_chunks(start: datetime, end: datetime, chunk_days: int = CHUNK_DAYS) -> dict:
    chunks = iter_chunks(start, end, chunk_days)
    spans = [(chunk_end.date() - chunk_start.date()).days for chunk_start, chunk_end in chunks]
    return {
        "start": start,
        "end": end,
        "chunk_days": chunk_days,
        "chunk_count": len(chunks),
        "max_span_days": max(spans) if spans else 0,
        "within_limit": all(span <= MAX_MINUTE_DAYS for span in spans),
        "chunks": chunks,
    }


def equity_token_map(kite: KiteConnect) -> dict[str, int]:
    instruments = kite.instruments(EXCHANGE)
    mapping: dict[str, int] = {}
    for row in instruments:
        if row.get("instrument_type") != "EQ" or row.get("exchange") != EXCHANGE:
            continue
        symbol = row.get("tradingsymbol")
        token = row.get("instrument_token")
        if symbol and token:
            mapping[str(symbol)] = int(token)
    return mapping


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
                raise RuntimeError(f"Historical API failed for {start} -> {end}: {exc}") from exc
            sleep_for = REQUEST_PAUSE_SECONDS * (2 ** (attempt - 1))
            print(f"    retry {attempt}/{MAX_RETRIES} ({exc}); sleep {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"Historical API failed for {start} -> {end}: {last_error}")


def candles_to_frame(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(candles)
    timestamps = pd.to_datetime(df["date"])
    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    out = out.dropna(subset=OUTPUT_COLUMNS)
    out["volume"] = out["volume"].round().astype("int64")
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out[OUTPUT_COLUMNS]


def missing_ranges(
    start: datetime,
    end: datetime,
    existing: pd.DataFrame | None,
) -> list[tuple[datetime, datetime]]:
    if existing is None or existing.empty:
        return [(start, end)]

    have_min = pd.Timestamp(existing["timestamp"].min()).to_pydatetime()
    have_max = pd.Timestamp(existing["timestamp"].max()).to_pydatetime()
    ranges: list[tuple[datetime, datetime]] = []
    if have_min > start:
        ranges.append((start, have_min))
    if have_max < end:
        ranges.append((have_max + timedelta(minutes=1), end))
    return [(a, b) for a, b in ranges if a < b]


def download_range(
    kite: KiteConnect,
    instrument_token: int,
    start: datetime,
    end: datetime,
    chunk_days: int = CHUNK_DAYS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    chunks = iter_chunks(start, end, chunk_days)
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"    chunk {index}/{len(chunks)}: {chunk_start} -> {chunk_end}")
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


def merge_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return (
        pd.concat(nonempty, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    duckdb.execute(
        "COPY (SELECT * FROM df) TO ? (FORMAT PARQUET, OVERWRITE TRUE)",
        [str(path)],
    )


def load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return duckdb.execute("SELECT * FROM read_parquet(?) ORDER BY timestamp", [str(path)]).df()


def validate_frame(df: pd.DataFrame, symbol: str) -> dict:
    if df.empty:
        raise RuntimeError(f"{symbol}: no candles returned.")
    nulls = int(df[OUTPUT_COLUMNS].isna().sum().sum())
    dupes = int(df["timestamp"].duplicated().sum())
    invalid = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["volume"] < 0)
    )
    invalid_count = int(invalid.sum())
    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "min_timestamp": df["timestamp"].min(),
        "max_timestamp": df["timestamp"].max(),
        "nulls": nulls,
        "duplicate_timestamps": dupes,
        "invalid_ohlc": invalid_count,
        "negative_volume": int((df["volume"] < 0).sum()),
    }
