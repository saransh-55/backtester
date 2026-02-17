import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.model.instruments import Instrument

ONE_MINUTE_NS = 60_000_000_000

def load_ohlcv_data(path: str, parquet: bool = False) -> pd.DataFrame:
    if parquet:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, header=0)

    # Try both "Datetime" and "datetime" column names
    datetime_col = None
    if "Datetime" in df.columns:
        datetime_col = "Datetime"
    elif "datetime" in df.columns:
        datetime_col = "datetime"
    else:
        raise ValueError(f"No datetime column found in data. Columns: {df.columns.tolist()}")

    return (
        df.reindex(columns=[datetime_col, "Open", "High", "Low", "Close", "Volume"])
        .assign(datetime=lambda d: pd.to_datetime(d[datetime_col], utc=True))
        .rename(
            columns={
                "datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .drop(columns=[datetime_col] if datetime_col != "datetime" else [])
        .set_index("timestamp")
        .sort_index()
    )


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.reindex(columns=["datetime", "Open", "High", "Low", "Close", "Volume"])
        .assign(datetime=lambda d: pd.to_datetime(d["datetime"], format="%Y-%m-%d %H:%M:%S"))
        .rename(
            columns={
                "datetime": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        .set_index("timestamp")
        .sort_index()
    )


def wrangle_bars(df: pd.DataFrame, bar_type: BarType, instrument: Instrument, timeframe_ns: int = ONE_MINUTE_NS) -> list[Bar]:
    return BarDataWrangler(bar_type, instrument).process(df, ts_init_delta=ONE_MINUTE_NS)

