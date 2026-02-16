import time

import os
import ccxt
import pandas as pd
import argparse

from pathlib import Path
from typing import Optional, List

from helpers.fs import to_project_root

def _to_ms_utc(dt_str: str) -> int:
    ts = pd.Timestamp(dt_str)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def fetch_ohlcv(
    symbol: str,
    start: str,
    timeframe: str = "1m",
    exchange_id: str = "binance",
    limit: int = 1000,
    max_retries: int = 8,
    end: Optional[str] = None,
) -> pd.DataFrame:
    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})

    start_ms = _to_ms_utc(start)
    end_ms = _to_ms_utc(end) if end is not None else int(pd.Timestamp.utcnow().timestamp() * 1000)
    step_ms = int(ex.parse_timeframe(timeframe) * 1000)

    try:
        ex.load_markets()
    except Exception:
        pass

    rows: List[List] = []
    since = start_ms
    last_ts = -1
    iteration = 0
    total_rows = 0

    while since < end_ms:
        iteration += 1
        batch = None
        for attempt in range(max_retries):
            try:
                print(f"\rFetching: iteration={iteration}, rows={total_rows}, current_time=Retrying... (attempt {attempt + 1}/{max_retries})", end="", flush=True)
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except Exception as e:
                print(f"\rFetching: iteration={iteration}, rows={total_rows}, current_time=Retrying... (attempt {attempt + 1}/{max_retries}) - Error: {e}", end="", flush=True)
                time.sleep(min(2 ** attempt, 30))

        if not batch:
            break

        batch = [r for r in batch if r and r[0] > last_ts and r[0] <= end_ms]
        if not batch:
            break

        rows.extend(batch)
        total_rows = len(rows)
        last_ts = batch[-1][0]
        current_dt = pd.to_datetime(last_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")

        # In-place progress update
        print(f"\rFetching: iteration={iteration}, rows={total_rows}, current_time={current_dt}", end="", flush=True)

        since = last_ts + step_ms

        if len(batch) < limit:
            break

    # Print newline after loop completes
    if rows:
        print()  # Move to next line after final update

    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df.index = pd.DatetimeIndex([], name="Datetime", tz="UTC")
    else:
        df["Datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"]).set_index("Datetime").sort_index()
    return df


def write_output_file(df: pd.DataFrame, out_path: Optional[str], output_format: str, symbol: str, timeframe: str):
    if out_path is None:
        safe = symbol.replace("/", "-").replace(":", "-")
        ext = ".parquet" if output_format == "parquet" else ".csv"
        out_path = f"data/{safe}_{timeframe}{ext}"

    out_path = Path(to_project_root(out_path))
    print(f"Writing output to {out_path}...")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if output_format == "parquet":
        df.to_parquet(tmp_path)
        os.replace(tmp_path, out_path)
    elif output_format == "csv":
        df.to_csv(tmp_path)
        os.replace(tmp_path, out_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download data for a given symbol and start time.")
    parser.add_argument('symbol', type=str, help='Symbol to download data for')
    parser.add_argument('--start_time', type=str, default=None, help='Start time for data download (format: YYYY-MM-DD, default: 30 days ago)')
    parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe for OHLCV data')
    parser.add_argument('--out_path', type=str, default=None, help='Output path for the output file')
    parser.add_argument('--exchange_id', type=str, default='binance', help='Exchange ID')
    parser.add_argument('--limit', type=int, default=1000, help='Limit per fetch')
    parser.add_argument('--max_retries', type=int, default=8, help='Max retries per fetch')
    parser.add_argument('--end_time', type=str, default=None, help='End time for data download')
    parser.add_argument('--output_format', type=str, default='csv', choices=['parquet', 'csv'], help='Output format: parquet or csv')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    symbol = args.symbol
    if args.start_time is not None:
        start_time = args.start_time
    else:
        # Default: 30 days ago from now (UTC)
        start_time = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    print(f"Downloading data for {symbol} starting from {start_time}...")
    df = fetch_ohlcv(
        symbol,
        start_time,
        timeframe=args.timeframe,
        exchange_id=args.exchange_id,
        limit=args.limit,
        max_retries=args.max_retries,
        end=args.end_time
    )
    write_output_file(df, args.out_path, args.output_format, symbol, args.timeframe)
    print(df.head())
