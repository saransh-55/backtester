from pathlib import Path
import time
from typing import Optional, List

import os
import ccxt
import pandas as pd


def _to_ms_utc(dt_str: str) -> int:
    ts = pd.Timestamp(dt_str)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def fetch_ohlcv_parquet(
    symbol: str,
    start: str,
    timeframe: str = "1m",
    out_path: Optional[str] = None,
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

    while since < end_ms:
        batch = None
        for attempt in range(max_retries):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
                break
            except Exception:
                time.sleep(min(2 ** attempt, 30))

        if not batch:
            break

        batch = [r for r in batch if r and r[0] > last_ts and r[0] <= end_ms]
        if not batch:
            break

        rows.extend(batch)
        last_ts = batch[-1][0]
        since = last_ts + step_ms

        if len(batch) < limit:
            break

    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df.index = pd.DatetimeIndex([], name="Datetime", tz="UTC")
    else:
        df["Datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"]).set_index("Datetime").sort_index()

    if out_path is None:
        safe = symbol.replace("/", "-").replace(":", "-")
        out_path = f"data/{safe}_{timeframe}.parquet"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp_path)
    os.replace(tmp_path, out_path)

    return df



# example:
# df = fetch_ohlcv_parquet("BTC/USDT", "2026-01-12", timeframe="1m")
# print(df.head())