"""Download historical daily OHLCV for RELIANCE.NS and save it as Parquet."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import yfinance as yf

TICKER = "RELIANCE.NS"
START_DATE = "2023-01-01"
OUTPUT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reliance_daily.parquet"


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        history = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to download OHLCV for {ticker}: {exc}") from exc

    if history is None or history.empty:
        raise RuntimeError(f"No OHLCV data returned for {ticker} from {start} to {end}.")

    df = history.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    if date_col not in df.columns:
        raise RuntimeError(f"Download for {ticker} did not include a date column. Columns: {list(df.columns)}")

    missing = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col not in df.columns]
    if missing:
        raise RuntimeError(f"Download for {ticker} is missing required columns: {missing}")

    timestamps = pd.to_datetime(df[date_col])
    if timestamps.dt.tz is not None:
        timestamps = timestamps.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    timestamps = timestamps.dt.normalize()

    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": pd.to_numeric(df["Open"], errors="coerce"),
            "high": pd.to_numeric(df["High"], errors="coerce"),
            "low": pd.to_numeric(df["Low"], errors="coerce"),
            "close": pd.to_numeric(df["Close"], errors="coerce"),
            "volume": pd.to_numeric(df["Volume"], errors="coerce").astype("Int64"),
        }
    )
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    if out.empty:
        raise RuntimeError(f"OHLCV data for {ticker} was empty after cleaning.")

    return out[OUTPUT_COLUMNS]


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        duckdb.execute(
            "COPY (SELECT * FROM df) TO ? (FORMAT PARQUET, OVERWRITE TRUE)",
            [str(path)],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to write Parquet file {path}: {exc}") from exc


def verify_parquet(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Parquet file was not created: {path}")

    try:
        preview = duckdb.execute(
            "SELECT * FROM read_parquet(?) ORDER BY timestamp LIMIT 5",
            [str(path)],
        ).df()
        count_row = duckdb.execute(
            "SELECT COUNT(*) AS n FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
    except Exception as exc:
        raise RuntimeError(f"DuckDB could not query {path}: {exc}") from exc

    parquet_rows = int(count_row[0]) if count_row is not None else 0
    if parquet_rows != expected_rows:
        raise RuntimeError(
            f"Parquet row count mismatch: expected {expected_rows}, DuckDB found {parquet_rows}."
        )
    if list(preview.columns) != OUTPUT_COLUMNS:
        raise RuntimeError(f"Unexpected Parquet columns: {list(preview.columns)}")

    return preview


def main() -> int:
    end = date.today().isoformat()
    print(f"Downloading {TICKER} daily OHLCV from {START_DATE} to {end}...")

    try:
        df = download_ohlcv(TICKER, START_DATE, end)
        save_parquet(df, OUTPUT_PATH)
        preview = verify_parquet(OUTPUT_PATH, len(df))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print("First 5 rows:")
    print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
