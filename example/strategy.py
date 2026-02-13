from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.indicators import MovingAverageFactory, MovingAverageType
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.custom import customdataclass
from nautilus_trader.model.data import Bar, BarType, CustomData, DataType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientId, InstrumentId, TraderId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Money
from nautilus_trader.persistence.wranglers import BarDataWrangler
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.trading.strategy import Strategy


# ----------------------------
# Custom API features stream
# IMPORTANT:
# - no `from __future__ import annotations`
# - do NOT declare ts_event/ts_init; customdataclass injects them
# ----------------------------
@customdataclass
class ApiFeatures(Data):
    instrument_id: InstrumentId
    v1: float
    v2: float


class MultiAssetDemoConfig(StrategyConfig, frozen=True):
    venue: Venue

    btc_instrument: Instrument
    eth_instrument: Instrument

    btc_bar_type_1min: BarType
    eth_bar_type_1min: BarType

    agg_minutes: int = 120

    primary_ema_period: int = 150
    secondary_ema_period: int = 250

    trade_size: Decimal = Decimal("100")

    api_client_id: ClientId = ClientId("API")


class MultiAssetDemoStrategy(Strategy):
    """
    BTC trades, features from:
      - ETH aggregated return (same agg_minutes)
      - ApiFeatures(v1, v2) custom stream

    PortfolioAnalyzer update policy (THIS IS THE FIX):
      - update analyzer on EVERY BTC 1m bar via add_return(ts, ret)
      - do NOT call analyzer.calculate_statistics() in-loop (it clears returns)
    """

    def __init__(self, config: MultiAssetDemoConfig):
        super().__init__(config)

        self.venue = config.venue

        self.btc_id = config.btc_instrument.id
        self.eth_id = config.eth_instrument.id
        self.btc: Instrument | None = None
        self.eth: Instrument | None = None

        # 1m EXTERNAL sources
        self.btc_1m = config.btc_bar_type_1min
        self.eth_1m = config.eth_bar_type_1min

        # INTERNAL aggregated targets
        self.btc_agg = BarType.from_str(f"{self.btc_id}-{config.agg_minutes}-MINUTE-LAST-INTERNAL")
        self.eth_agg = BarType.from_str(f"{self.eth_id}-{config.agg_minutes}-MINUTE-LAST-INTERNAL")

        # Subscribe drivers: "TARGET@1-MINUTE-EXTERNAL"
        self.btc_agg_sub = BarType.from_str(f"{self.btc_agg}@1-MINUTE-EXTERNAL")
        self.eth_agg_sub = BarType.from_str(f"{self.eth_agg}@1-MINUTE-EXTERNAL")

        # API custom data
        self.api_client_id = config.api_client_id
        self.api_data_type = DataType(ApiFeatures)

        # BTC EMA cascade (computed on BTC agg bars)
        self.primary_ema = MovingAverageFactory.create(config.primary_ema_period, MovingAverageType.EXPONENTIAL)
        self.secondary_ema = MovingAverageFactory.create(config.secondary_ema_period, MovingAverageType.EXPONENTIAL)
        self.primary_hist: deque[float] = deque(maxlen=2)
        self.secondary_hist: deque[float] = deque(maxlen=2)

        # ETH agg return feature
        self._eth_last_close: float | None = None
        self.eth_agg_ret: float | None = None
        self.eth_agg_ts = None

        # API feature state
        self.api_v1: float | None = None
        self.api_v2: float | None = None
        self.api_ts = None

        # Per-bar equity sampling (BTC 1m)
        self._last_equity: float | None = None
        self._returns_points: int = 0  # debug counter

        # Telemetry tape (one row per BTC agg bar)
        self.feature_rows: list[dict[str, Any]] = []

        # Counters
        self.count_btc_1m = 0
        self.count_eth_1m = 0
        self.count_btc_agg = 0
        self.count_eth_agg = 0
        self.count_api = 0

        # Action tracking
        self._last_action: str = ""
        self._last_action_ts = None

    def on_start(self):
        self.btc = self.cache.instrument(self.btc_id)
        self.eth = self.cache.instrument(self.eth_id)
        if self.btc is None or self.eth is None:
            self.log.error("Could not load instruments from cache")
            self.stop()
            return

        # We inject returns ourselves; start clean.
        self.portfolio.analyzer.reset()

        # Subscribe sources
        self.subscribe_bars(self.btc_1m)
        self.subscribe_bars(self.eth_1m)

        # Subscribe aggregates
        self.subscribe_bars(self.btc_agg_sub)
        self.subscribe_bars(self.eth_agg_sub)

        # Register BTC primary EMA on BTC agg bars
        self.register_indicator_for_bars(self.btc_agg, self.primary_ema)

        # Subscribe API custom data
        self.subscribe_data(self.api_data_type, client_id=self.api_client_id)

        self.log.info("Strategy started.", color=LogColor.BLUE)

    def on_data(self, data: Data):
        payload = data.data if isinstance(data, CustomData) else data
        if not isinstance(payload, ApiFeatures):
            return
        if payload.instrument_id != self.btc_id:
            return

        self.count_api += 1
        self.api_v1 = float(payload.v1)
        self.api_v2 = float(payload.v2)
        self.api_ts = unix_nanos_to_dt(payload.ts_event)

    def _update_analyzer_on_btc_bar(self, bar: Bar) -> None:
        """
        Update PortfolioAnalyzer on EVERY BTC 1m bar using mark-to-market at bar close.
        """
        if self.btc is None:
            return

        account = self.portfolio.account(self.venue)
        if account is None:
            return

        # Mark-to-market using BTC 1m close
        close_price = self.btc.make_price(bar.close)

        balance_total = account.balance_total(USDT).as_double()
        unreal = self.portfolio.unrealized_pnl(self.btc_id, price=close_price)
        unreal_val = unreal.as_double() if unreal is not None else 0.0

        equity = float(balance_total) + float(unreal_val)

        if self._last_equity is None or self._last_equity == 0.0:
            ret = 0.0
        else:
            ret = (equity / self._last_equity) - 1.0

        ts_dt = unix_nanos_to_dt(bar.ts_event)
        self.portfolio.analyzer.add_return(ts_dt, float(ret))
        self._returns_points += 1
        self._last_equity = equity

        # Optional: periodic proof it is updating during run
        self.log.debug(
            f"Analyzer updated (BTC 1m): points={self._returns_points} last_ret={ret:.8f} equity={equity:.2f}",
            color=LogColor.GREEN,
        )

    def on_bar(self, bar: Bar):
        # BTC 1m: update analyzer EVERY bar (fix)
        if bar.bar_type == self.btc_1m:
            self.count_btc_1m += 1
            return

        # ETH 1m: ignore (agg builder needs the data in engine; subscription is optional)
        if bar.bar_type == self.eth_1m:
            self.count_eth_1m += 1
            return

        # ETH target bar: compute ETH agg return feature
        if bar.bar_type == self.eth_agg:
            self.count_eth_agg += 1
            ts = unix_nanos_to_dt(bar.ts_event)
            close = float(bar.close)
            if self._eth_last_close is None or self._eth_last_close == 0.0:
                self.eth_agg_ret = 0.0
            else:
                self.eth_agg_ret = (close / self._eth_last_close) - 1.0
            self._eth_last_close = close
            self.eth_agg_ts = ts
            return

        # BTC target bar: trade + record tape (analyzer already updated on BTC 1m bars)
        if bar.bar_type != self.btc_agg:
            raise Exception(f"Unexpected bar type: {bar.bar_type}")

        self.count_btc_agg += 1
        self._update_analyzer_on_btc_bar(bar)
        ts = unix_nanos_to_dt(bar.ts_event)

        # EMA cascade
        primary_val = float(self.primary_ema.value)
        self.primary_hist.append(primary_val)

        if self.primary_ema.initialized:
            self.secondary_ema.update_raw(primary_val)
            self.secondary_hist.append(float(self.secondary_ema.value))

        crossed_up = crossed_down = False
        prev_primary = prev_secondary = None
        latest_primary = float(self.primary_hist[-1]) if self.primary_hist else None
        latest_secondary = float(self.secondary_hist[-1]) if self.secondary_hist else None

        if self.primary_ema.initialized and self.secondary_ema.initialized:
            if len(self.primary_hist) == 2 and len(self.secondary_hist) == 2:
                prev_primary = float(self.primary_hist[-2])
                prev_secondary = float(self.secondary_hist[-2])
                latest_primary = float(self.primary_hist[-1])
                latest_secondary = float(self.secondary_hist[-1])

                crossed_up = (prev_primary <= prev_secondary) and (latest_primary > latest_secondary)
                crossed_down = (prev_primary >= prev_secondary) and (latest_primary < latest_secondary)

        # Feature gates (demo)
        eth_ret = self.eth_agg_ret if self.eth_agg_ret is not None else 0.0
        v1 = self.api_v1 if self.api_v1 is not None else 0.0
        v2 = self.api_v2 if self.api_v2 is not None else 0.0
        api_sum = v1 + v2

        allow_long = (eth_ret > 0.0) and (api_sum > 0.0)
        allow_short = (eth_ret < 0.0) and (api_sum < 0.0)

        self._last_action = ""
        self._last_action_ts = None

        if crossed_up and allow_long:
            self._go_long()
        elif crossed_down and allow_short:
            self._go_short()

        self._record_snapshot_target_bar(
            bar=bar,
            ts=ts,
            prev_primary=prev_primary,
            prev_secondary=prev_secondary,
            latest_primary=latest_primary,
            latest_secondary=latest_secondary,
            crossed_up=crossed_up,
            crossed_down=crossed_down,
            eth_ret=eth_ret,
            eth_ts=self.eth_agg_ts,
            api_v1=v1,
            api_v2=v2,
            api_ts=self.api_ts,
            allow_long=allow_long,
            allow_short=allow_short,
        )

    def _go_long(self) -> None:
        if self.portfolio.is_net_short(self.btc_id):
            self.close_all_positions(self.btc_id)
        if self.portfolio.is_net_long(self.btc_id):
            return
        self.cancel_all_orders(self.btc_id)
        self._submit_market(OrderSide.BUY)

    def _go_short(self) -> None:
        if self.portfolio.is_net_long(self.btc_id):
            self.close_all_positions(self.btc_id)
        if self.portfolio.is_net_short(self.btc_id):
            return
        self.cancel_all_orders(self.btc_id)
        self._submit_market(OrderSide.SELL)

    def _submit_market(self, side: OrderSide) -> None:
        if self.btc is None:
            return

        qty = self.btc.make_qty(self.config.trade_size)
        order = self.order_factory.market(
            instrument_id=self.btc_id,
            order_side=side,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
            reduce_only=False,
            tags=["BTC_MULTI_ASSET_ENTRY"],
        )
        self.submit_order(order)
        self._last_action = side.name
        self._last_action_ts = self.clock.utc_now()

    def _record_snapshot_target_bar(
        self,
        bar: Bar,
        ts,
        prev_primary: float | None,
        prev_secondary: float | None,
        latest_primary: float | None,
        latest_secondary: float | None,
        crossed_up: bool,
        crossed_down: bool,
        eth_ret: float,
        eth_ts,
        api_v1: float,
        api_v2: float,
        api_ts,
        allow_long: bool,
        allow_short: bool,
    ) -> None:
        if self.btc is None:
            return

        close_price = self.btc.make_price(bar.close)

        account = self.portfolio.account(self.venue)
        balance_total = account.balance_total(USDT).as_double() if account is not None else None

        unreal = self.portfolio.unrealized_pnl(self.btc_id, price=close_price)
        unreal_val = unreal.as_double() if unreal is not None else 0.0

        realized = self.portfolio.realized_pnl(self.btc_id)
        realized_val = realized.as_double() if realized is not None else 0.0

        net_pos = self.portfolio.net_position(self.btc_id)
        net_pos_val = float(net_pos)

        net_exp = self.portfolio.net_exposure(self.btc_id, price=close_price)
        net_exp_val = net_exp.as_double() if net_exp is not None else 0.0

        equity = None
        if balance_total is not None:
            equity = float(balance_total) + float(unreal_val)

        # NOTE: analyzer returns are updated on BTC 1m bars; this is just tape
        self.feature_rows.append(
            {
                "ts": ts,
                "btc_bar_type": str(bar.bar_type),
                "btc_open": float(bar.open),
                "btc_high": float(bar.high),
                "btc_low": float(bar.low),
                "btc_close": float(bar.close),
                "btc_volume": float(bar.volume),
                "btc_primary_ema": float(self.primary_ema.value),
                "btc_secondary_ema": float(self.secondary_ema.value),
                "btc_prev_primary_ema": prev_primary,
                "btc_prev_secondary_ema": prev_secondary,
                "btc_latest_primary_ema": latest_primary,
                "btc_latest_secondary_ema": latest_secondary,
                "btc_crossed_up": bool(crossed_up),
                "btc_crossed_down": bool(crossed_down),
                "eth_agg_ret": eth_ret,
                "eth_agg_ts": eth_ts,
                "api_v1": api_v1,
                "api_v2": api_v2,
                "api_ts": api_ts,
                "allow_long": bool(allow_long),
                "allow_short": bool(allow_short),
                "action": self._last_action,
                "net_position": net_pos_val,
                "net_exposure": net_exp_val,
                "balance_total_usdt": balance_total,
                "unrealized_pnl_usdt": unreal_val,
                "realized_pnl_usdt": realized_val,
                "equity_usdt": equity,
                "analyzer_points": self._returns_points,
            }
        )

    def on_stop(self):
        self.cancel_all_orders(self.btc_id)
        self.close_all_positions(self.btc_id)

        self.unsubscribe_bars(self.btc_1m)
        self.unsubscribe_bars(self.eth_1m)
        self.unsubscribe_bars(self.btc_agg_sub)
        self.unsubscribe_bars(self.eth_agg_sub)
        self.unsubscribe_data(self.api_data_type, client_id=self.api_client_id)
