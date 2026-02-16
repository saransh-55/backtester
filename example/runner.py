from decimal import Decimal
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import ClientId, TraderId, Venue
from nautilus_trader.model.objects import Money
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from strategy import MultiAssetDemoStrategy, MultiAssetDemoConfig
from utils import _safe_to_csv, _write_json, load_ohlcv_csv, wrangle_1m_bars, build_api_custom_data_from_btc_bars

if __name__ == "__main__":
    btc_csv = "example/btc_ohlcv.csv"
    eth_csv = "example/eth_ohlcv.csv"

    out_dir = Path("backtest_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    btc_df = load_ohlcv_csv(btc_csv)
    eth_df = load_ohlcv_csv(eth_csv)

    BINANCE = Venue("BINANCE")

    try:
        BTC = TestInstrumentProvider.btcusdt_perp_binance()
    except AttributeError:
        BTC = TestInstrumentProvider.btcusdt_binance()

    try:
        ETH = TestInstrumentProvider.ethusdt_perp_binance()
    except AttributeError:
        ETH = TestInstrumentProvider.ethusdt_binance()

    BTC_1M = BarType.from_str(f"{BTC.id}-1-MINUTE-LAST-EXTERNAL")
    ETH_1M = BarType.from_str(f"{ETH.id}-1-MINUTE-LAST-EXTERNAL")

    btc_bars = wrangle_1m_bars(btc_df, BTC_1M, BTC)
    eth_bars = wrangle_1m_bars(eth_df, ETH_1M, ETH)

    api_client_id = ClientId("API")
    api_stream = build_api_custom_data_from_btc_bars(btc_bars, BTC.id, seed=11)

    engine = BacktestEngine(
        config=BacktestEngineConfig(
            trader_id=TraderId("BACKTEST-BTC-ETH-API-001"),
            logging=LoggingConfig(log_level="INFO"),
        )
    )

    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        starting_balances=[Money(1_000_000, USDT)],
        base_currency=USDT,
        default_leverage=Decimal(1),
    )

    engine.add_instrument(BTC)
    engine.add_instrument(ETH)

    engine.add_data(btc_bars)
    engine.add_data(eth_bars)
    engine.add_data(api_stream, client_id=api_client_id)

    strategy = MultiAssetDemoStrategy(
        MultiAssetDemoConfig(
            venue=BINANCE,
            btc_instrument=BTC,
            eth_instrument=ETH,
            btc_bar_type_1min=BTC_1M,
            eth_bar_type_1min=ETH_1M,
            agg_minutes=120,
            primary_ema_period=150,
            secondary_ema_period=250,
            trade_size=Decimal("10"),
            api_client_id=api_client_id,
        )
    )

    strategy2 = MultiAssetDemoStrategy(
        MultiAssetDemoConfig(
            venue=BINANCE,
            btc_instrument=BTC,
            eth_instrument=ETH,
            btc_bar_type_1min=BTC_1M,
            eth_bar_type_1min=ETH_1M,
            agg_minutes=60,
            primary_ema_period=50,
            secondary_ema_period=100,
            trade_size=Decimal("100"),
            api_client_id=api_client_id,
        )
    )

    engine.add_strategy(strategy)
    engine.add_strategy(strategy2)

    engine.run()

    # ----------------------------
    # Exports (same as your original telemetry)
    # ----------------------------
    # Feature tape (every BTC target bar)
    features_df = pd.DataFrame(strategy.feature_rows)
    if not features_df.empty:
        features_path = out_dir / "feature_tape_btc_target.csv"
        features_df.to_csv(features_path, index=False)
        print(f"Wrote: {features_path}")

    # Analyzer returns (NOW per BTC 1m bar)
    returns = engine.portfolio.analyzer.returns()
    if returns is not None and len(returns) > 0:
        returns_path = out_dir / "returns.csv"
        returns.rename("return").to_csv(returns_path, index=True)
        print(f"Wrote: {returns_path}")

    # Performance stats
    analyzer = engine.portfolio.analyzer
    last_close = float(btc_df["close"].iloc[-1])
    last_unreal = engine.portfolio.unrealized_pnl(BTC.id, price=BTC.make_price(last_close))

    stats = {
        "returns_stats": analyzer.get_performance_stats_returns(),
        "general_stats": analyzer.get_performance_stats_general(),
        "pnls_stats_usdt": analyzer.get_performance_stats_pnls(currency=USDT, unrealized_pnl=last_unreal),
        "formatted": {
            "returns": analyzer.get_stats_returns_formatted(),
            "general": analyzer.get_stats_general_formatted(),
            "pnls_usdt": analyzer.get_stats_pnls_formatted(currency=USDT, unrealized_pnl=last_unreal),
        },
    }
    stats_path = out_dir / "performance_stats.json"
    _write_json(stats_path, stats)
    print(f"Wrote: {stats_path}")

    # Reports
    try:
        account_report = engine.trader.generate_account_report(BINANCE)
        _safe_to_csv(account_report, out_dir / "account_report.csv")
        print(f"Wrote: {out_dir / 'account_report.csv'}")
    except Exception as e:
        print(f"[WARN] account_report failed: {e}")

    try:
        orders_report = engine.trader.generate_orders_report()
        _safe_to_csv(orders_report, out_dir / "orders_report.csv")
        print(f"Wrote: {out_dir / 'orders_report.csv'}")
    except Exception as e:
        print(f"[WARN] orders_report failed: {e}")

    try:
        order_fills_report = engine.trader.generate_order_fills_report()
        _safe_to_csv(order_fills_report, out_dir / "order_fills_report.csv")
        print(f"Wrote: {out_dir / 'order_fills_report.csv'}")
    except Exception as e:
        print(f"[WARN] order_fills_report failed: {e}")

    try:
        fills_report = engine.trader.generate_fills_report()
        _safe_to_csv(fills_report, out_dir / "fills_report.csv")
        print(f"Wrote: {out_dir / 'fills_report.csv'}")
    except Exception as e:
        print(f"[WARN] fills_report failed: {e}")

    try:
        positions_report = engine.trader.generate_positions_report()
        _safe_to_csv(positions_report, out_dir / "positions_report.csv")
        print(f"Wrote: {out_dir / 'positions_report.csv'}")
    except Exception as e:
        print(f"[WARN] positions_report failed: {e}")

    # Visualization (HTML)
    try:
        from nautilus_trader.analysis import TearsheetConfig
        from nautilus_trader.analysis.tearsheet import create_bars_with_fills, create_tearsheet

        tearsheet_path = out_dir / "tearsheet.html"
        config = TearsheetConfig(
            charts=[
                "stats_table",
                "equity",
                "drawdown",
                "monthly_returns",
                "distribution",
                "rolling_sharpe",
                "yearly_returns",
                "bars_with_fills"
            ],
            # chart_args={"bars_with_fills": {"bar_type": str(BTC_1M)}},
            theme="nautilus_dark",
            include_benchmark=True,
            height=2200,
        )
        create_tearsheet(engine=engine, output_path=str(tearsheet_path), currency=USDT, config=config)
        print(f"Wrote: {tearsheet_path}")

        bars_with_fills_path = out_dir / "bars_with_fills_1m.html"
        fig = create_bars_with_fills(engine=engine, bar_type=BTC_1M, title="BTC 1m - Fills Overlay")
        fig.write_html(str(bars_with_fills_path))
        print(f"Wrote: {bars_with_fills_path}")

    except ImportError as e:
        print(f"[WARN] Visualization skipped (plotly missing): {e}")
    except Exception as e:
        print(f"[WARN] Visualization failed: {e}")

    engine.dispose()
