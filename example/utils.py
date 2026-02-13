from pathlib import Path
from typing import Any
import random
import pandas as pd
import json
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.core.data import Data
from nautilus_trader.model.data import Bar, BarType, CustomData, DataType
from nautilus_trader.model.identifiers import ClientId, InstrumentId, TraderId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.trading.strategy import Strategy
from strategy import ApiFeatures


def _safe_to_csv(df: pd.DataFrame | None, path: Path) -> None:
    if df is None or df.empty:
        return
    df.to_csv(path, index=False)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=0)
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


def wrangle_1m_bars(df: pd.DataFrame, bar_type: BarType, instrument: Instrument) -> list[Bar]:
    ONE_MINUTE_NS = 60_000_000_000
    return BarDataWrangler(bar_type, instrument).process(df, ts_init_delta=ONE_MINUTE_NS)


def build_api_custom_data_from_btc_bars(
    btc_bars_1m: list[Bar],
    btc_instrument_id: InstrumentId,
    seed: int = 11,
) -> list[CustomData]:
    rng = random.Random(seed)
    dt = DataType(ApiFeatures)

    out: list[CustomData] = []
    for b in btc_bars_1m:
        ts_event = int(b.ts_event)
        ts_init = int(b.ts_init) - 1

        feat = ApiFeatures(
            instrument_id=btc_instrument_id,
            v1=rng.uniform(-1.0, 1.0),
            v2=rng.uniform(-1.0, 1.0),
            ts_event=ts_event,
            ts_init=ts_init,
        )
        out.append(CustomData(dt, feat))
    return out
