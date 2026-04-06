"""End-to-end integration tests using synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path


def _create_synthetic_universe(
    tmp_path: Path,
    n_stocks: int = 5,
    n_days: int = 500,
    seed: int = 42,
) -> None:
    """Create synthetic parquet files in tmp_path."""
    rng = np.random.default_rng(seed)
    for i in range(n_stocks):
        symbol = f"STOCK_{i}"
        dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
        close = 100 + np.cumsum(rng.standard_normal(n_days) * 0.5)
        close = np.maximum(close, 1.0)
        noise = rng.standard_normal(n_days)
        df = pd.DataFrame(
            {
                "date": dates,
                "open": close + noise * 0.2,
                "high": close + abs(noise * 0.5),
                "low": close - abs(noise * 0.5),
                "close": close,
                "volume": rng.integers(50_000, 500_000, n_days).astype(float),
            }
        )
        # Enforce OHLC validity
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
        df.to_parquet(tmp_path / f"{symbol}.parquet", index=False)


@pytest.fixture
def universe_dir(tmp_path: Path) -> Path:
    _create_synthetic_universe(tmp_path)
    return tmp_path


@pytest.fixture
def small_universe_dir(tmp_path: Path) -> Path:
    _create_synthetic_universe(tmp_path, n_stocks=3, n_days=300)
    return tmp_path


# ---------------------------------------------------------------------------
# Core strategy end-to-end tests
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_ma_crossover_runs(self, universe_dir: Path) -> None:
        from vbacktest import BacktestConfig, BacktestEngine, BacktestResult
        from vbacktest.config import DataConfig
        from vbacktest.strategies import MACrossoverStrategy

        config = BacktestConfig(data=DataConfig(validated_dir=str(universe_dir), min_history_days=30))
        engine = BacktestEngine(config, MACrossoverStrategy(fast_period=5, slow_period=10, trend_period=20))
        result = engine.run()
        assert isinstance(result, BacktestResult)
        assert result.stats.total_trades >= 0
        assert len(result.equity_curve) > 0

    def test_momentum_runs(self, universe_dir: Path) -> None:
        from vbacktest import BacktestConfig, BacktestEngine
        from vbacktest.config import DataConfig
        from vbacktest.strategies import MomentumStrategy

        config = BacktestConfig(data=DataConfig(validated_dir=str(universe_dir), min_history_days=30))
        engine = BacktestEngine(
            config,
            MomentumStrategy(roc_periods=(5, 10), trend_period=20, high_period=30),
        )
        result = engine.run()
        assert result.stats.total_trades >= 0

    def test_turtle_trading_runs(self, universe_dir: Path) -> None:
        from vbacktest import BacktestConfig, BacktestEngine
        from vbacktest.config import DataConfig
        from vbacktest.strategies import TurtleTradingStrategy

        config = BacktestConfig(data=DataConfig(validated_dir=str(universe_dir), min_history_days=30))
        engine = BacktestEngine(
            config, TurtleTradingStrategy(entry_breakout_days=10, exit_breakdown_days=5)
        )
        result = engine.run()
        assert result.stats.total_trades >= 0

    def test_result_json_roundtrip(self, universe_dir: Path, tmp_path: Path) -> None:
        from vbacktest import BacktestConfig, BacktestEngine, BacktestResult
        from vbacktest.config import DataConfig
        from vbacktest.strategies import MACrossoverStrategy

        config = BacktestConfig(data=DataConfig(validated_dir=str(universe_dir), min_history_days=30))
        engine = BacktestEngine(config, MACrossoverStrategy(fast_period=5, slow_period=10, trend_period=20))
        result = engine.run()
        json_path = tmp_path / "result.json"
        json_path.write_text(result.to_json())
        loaded = BacktestResult.from_json(json_path.read_text())
        assert loaded.schema_version == "1.0"
        assert loaded.stats.total_trades == result.stats.total_trades

    def test_equity_series_is_date_indexed(self, universe_dir: Path) -> None:
        from vbacktest import BacktestConfig, BacktestEngine
        from vbacktest.config import DataConfig
        from vbacktest.strategies import MACrossoverStrategy

        config = BacktestConfig(data=DataConfig(validated_dir=str(universe_dir), min_history_days=30))
        engine = BacktestEngine(config, MACrossoverStrategy(fast_period=5, slow_period=10, trend_period=20))
        result = engine.run()
        eq = result.equity_series()
        assert hasattr(eq.index, "year"), "equity_series() should have DatetimeIndex"

    def test_go_no_go_on_real_result(self, universe_dir: Path) -> None:
        from vbacktest import BacktestConfig, BacktestEngine
        from vbacktest.analysis import GoNoGo
        from vbacktest.config import DataConfig
        from vbacktest.strategies import MACrossoverStrategy

        config = BacktestConfig(data=DataConfig(validated_dir=str(universe_dir), min_history_days=30))
        engine = BacktestEngine(config, MACrossoverStrategy(fast_period=5, slow_period=10, trend_period=20))
        result = engine.run()
        if result.stats.total_trades > 0:
            report = GoNoGo(
                result.trades,
                result.equity_series(),
                initial_capital=100_000,
                mc_sims=50,
            ).run()
            assert report.overall in ("PASS", "WARN", "FAIL")


# ---------------------------------------------------------------------------
# Custom strategy extensibility
# ---------------------------------------------------------------------------

class TestCustomStrategy:
    """Verify downstream projects can extend vbacktest."""

    def test_custom_strategy_runs(self, small_universe_dir: Path) -> None:
        from vbacktest import BarContext, BacktestConfig, BacktestEngine, BacktestResult, Strategy
        from vbacktest.config import DataConfig
        from vbacktest.indicators import IndicatorSpec

        class AlwaysPassStrategy(Strategy):
            def indicators(self) -> list[IndicatorSpec]:
                return [IndicatorSpec("sma", {"period": 10})]

            def exit_rules(self) -> list:
                return []

            def on_bar(self, ctx: BarContext) -> list:
                return []  # never trades — proves extensibility

        config = BacktestConfig(data=DataConfig(validated_dir=str(small_universe_dir), min_history_days=30))
        engine = BacktestEngine(config, AlwaysPassStrategy())
        result = engine.run()
        assert isinstance(result, BacktestResult)
        assert result.stats.total_trades == 0

    def test_custom_strategy_with_signals(self, small_universe_dir: Path) -> None:
        from vbacktest import BarContext, BacktestConfig, BacktestEngine, BacktestResult, Signal, SignalAction, Strategy
        from vbacktest.config import DataConfig
        from vbacktest.exit_rules import StopLossRule, TimeStopRule
        from vbacktest.indicators import IndicatorSpec

        class SimpleBuyStrategy(Strategy):
            def indicators(self) -> list[IndicatorSpec]:
                return [IndicatorSpec("sma", {"period": 5}), IndicatorSpec("atr", {})]

            def exit_rules(self) -> list:
                return [StopLossRule(), TimeStopRule(max_holding_days=10)]

            def on_bar(self, ctx: BarContext) -> list[Signal]:
                signals = []
                for symbol, df in ctx.universe.items():
                    if ctx.portfolio and ctx.portfolio.has_position(symbol):
                        continue
                    if symbol not in ctx.universe_idx:
                        continue
                    idx = ctx.universe_idx[symbol]
                    if idx < 10:
                        continue
                    bar = df.iloc[idx]
                    if "sma_5" not in df.columns or "atr" not in df.columns:
                        continue
                    import pandas as pd
                    if pd.isna(bar["sma_5"]) or pd.isna(bar["atr"]):
                        continue
                    close = float(bar["close"])
                    atr = float(bar["atr"])
                    stop = close - 2 * atr
                    signals.append(
                        Signal(
                            symbol=symbol,
                            action=SignalAction.BUY,
                            date=ctx.date,
                            stop_price=stop,
                            score=1.0,
                        )
                    )
                return signals

        config = BacktestConfig(data=DataConfig(validated_dir=str(small_universe_dir), min_history_days=20))
        engine = BacktestEngine(config, SimpleBuyStrategy())
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_register_and_run_custom_strategy(self, small_universe_dir: Path) -> None:
        from vbacktest import BarContext, BacktestConfig, BacktestEngine, Strategy
        from vbacktest.config import DataConfig
        from vbacktest.indicators import IndicatorSpec
        from vbacktest.registry import strategy_registry

        @strategy_registry.register("_test_noop_strategy", override=True)
        class NoOpStrategy(Strategy):
            def indicators(self) -> list[IndicatorSpec]:
                return []

            def exit_rules(self) -> list:
                return []

            def on_bar(self, ctx: BarContext) -> list:
                return []

        assert "_test_noop_strategy" in strategy_registry

        config = BacktestConfig(data=DataConfig(validated_dir=str(small_universe_dir), min_history_days=20))
        cls = strategy_registry.get("_test_noop_strategy")
        engine = BacktestEngine(config, cls())
        result = engine.run()
        assert result.stats.total_trades == 0
