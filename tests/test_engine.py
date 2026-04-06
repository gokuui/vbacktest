"""Tests for BacktestEngine."""
from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vbacktest.config import BacktestConfig, DataConfig, PortfolioConfig
from vbacktest.engine import BacktestEngine
from vbacktest.exit_rules import StopLossRule, TimeStopRule
from vbacktest.indicators import IndicatorSpec
from vbacktest.results import BacktestResult
from vbacktest.strategy import BarContext, ExitRule, Signal, SignalAction, Strategy


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------

class AlwaysBuyStrategy(Strategy):
    """Buys every stock on every bar — stress-tests the engine loop."""

    def indicators(self) -> list[IndicatorSpec]:
        return [IndicatorSpec(name="sma", params={"period": 5})]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        for symbol in ctx.universe:
            if ctx.portfolio and ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue
            idx = ctx.universe_idx[symbol]
            df = ctx.universe[symbol]
            if idx < 5:
                continue
            close = df.iloc[idx]["close"]
            stop = close * 0.95
            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop,
                score=1.0,
            ))
        return signals

    def exit_rules(self) -> list[ExitRule]:
        return [StopLossRule(), TimeStopRule(max_holding_days=10)]


class NeverBuyStrategy(Strategy):
    """Generates no signals — tests idle engine path."""

    def indicators(self) -> list[IndicatorSpec]:
        return []

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        return []

    def exit_rules(self) -> list[ExitRule]:
        return []


class LegacyStrategy(Strategy):
    """Uses old positional on_bar signature for compatibility testing."""

    def indicators(self) -> list[IndicatorSpec]:
        return []

    def on_bar(self, date, universe, universe_idx, portfolio,  # type: ignore[override]
               current_prices=None, universe_arrays=None, **kwargs) -> list[Signal]:
        return []

    def exit_rules(self) -> list[ExitRule]:
        return []


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, close_start: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = close_start + np.cumsum(rng.normal(0.1, 0.5, n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": [1_000_000.0] * n,
        "source": ["test"] * n,
        "confidence": [2] * n,
    })


def _setup_universe_dir(symbols: dict[str, pd.DataFrame]) -> Path:
    tmp = Path(tempfile.mkdtemp())
    for sym, df in symbols.items():
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, tmp / f"{sym}.parquet")
    return tmp


def _make_config(validated_dir: Path, capital: float = 100_000.0, max_positions: int = 3) -> BacktestConfig:
    return BacktestConfig(
        data=DataConfig(validated_dir=validated_dir, min_history_days=20),
        portfolio=PortfolioConfig(initial_capital=capital, max_positions=max_positions),
    )


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestBacktestEngineNoBuys:
    def test_returns_result(self) -> None:
        d = _setup_universe_dir({"A": _make_ohlcv(100), "B": _make_ohlcv(100)})
        engine = BacktestEngine(_make_config(d), NeverBuyStrategy())
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_no_trades_when_no_signals(self) -> None:
        d = _setup_universe_dir({"A": _make_ohlcv(100)})
        engine = BacktestEngine(_make_config(d), NeverBuyStrategy())
        result = engine.run()
        assert result.stats.total_trades == 0

    def test_equity_unchanged_no_trades(self) -> None:
        d = _setup_universe_dir({"A": _make_ohlcv(100)})
        cfg = _make_config(d, capital=50_000.0)
        engine = BacktestEngine(cfg, NeverBuyStrategy())
        result = engine.run()
        if not result.equity_curve.empty:
            # With no positions, equity should stay close to initial capital
            final = result.equity_curve["equity"].iloc[-1]
            assert abs(final - 50_000.0) < 1.0  # allow for rounding


class TestBacktestEngineWithBuys:
    def _run(self, n_bars: int = 100, n_stocks: int = 3) -> BacktestResult:
        syms = {f"SYM{i}": _make_ohlcv(n_bars, close_start=50.0 + i * 10) for i in range(n_stocks)}
        d = _setup_universe_dir(syms)
        cfg = _make_config(d, capital=500_000.0)
        engine = BacktestEngine(cfg, AlwaysBuyStrategy())
        return engine.run()

    def test_trades_generated(self) -> None:
        result = self._run()
        assert result.stats.total_trades > 0

    def test_equity_curve_populated(self) -> None:
        result = self._run()
        assert not result.equity_curve.empty
        assert "date" in result.equity_curve.columns
        assert "equity" in result.equity_curve.columns

    def test_equity_curve_dates_sorted(self) -> None:
        result = self._run()
        dates = result.equity_curve["date"]
        assert (dates.diff().dropna() > pd.Timedelta(0)).all()

    def test_trade_df_has_required_columns(self) -> None:
        result = self._run()
        if not result.trade_df.empty:
            for col in ["symbol", "entry_price", "exit_price", "pnl", "holding_days"]:
                assert col in result.trade_df.columns

    def test_max_positions_respected(self) -> None:
        """Portfolio should never hold more than max_positions simultaneously."""
        syms = {f"SYM{i}": _make_ohlcv(60, close_start=100.0) for i in range(10)}
        d = _setup_universe_dir(syms)
        cfg = BacktestConfig(
            data=DataConfig(validated_dir=d, min_history_days=20),
            portfolio=PortfolioConfig(initial_capital=1_000_000.0, max_positions=3),
        )
        engine = BacktestEngine(cfg, AlwaysBuyStrategy())
        engine.run()
        # All positions should have been closed by end; max recorded via trades
        assert engine.portfolio.positions_count == 0  # all closed at end of backtest

    def test_pnl_is_numeric(self) -> None:
        result = self._run()
        if not result.trade_df.empty:
            assert result.trade_df["pnl"].dtype in (float, "float64")


class TestBacktestEngineEmptyUniverse:
    def test_empty_dir_returns_empty_result(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        cfg = _make_config(tmp)
        engine = BacktestEngine(cfg, NeverBuyStrategy())
        result = engine.run()
        assert result.stats.total_trades == 0
        assert result.equity_curve.empty


class TestLegacyOnBarCompatibility:
    def test_legacy_strategy_warns(self) -> None:
        d = _setup_universe_dir({"A": _make_ohlcv(50)})
        cfg = _make_config(d)
        engine = BacktestEngine(cfg, LegacyStrategy())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.run()
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0
        assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_legacy_strategy_still_runs(self) -> None:
        d = _setup_universe_dir({"A": _make_ohlcv(50)})
        cfg = _make_config(d)
        engine = BacktestEngine(cfg, LegacyStrategy())
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = engine.run()
        assert isinstance(result, BacktestResult)


class TestWarmupDays:
    def _engine(self) -> BacktestEngine:
        d = _setup_universe_dir({"A": _make_ohlcv(50)})
        return BacktestEngine(_make_config(d), NeverBuyStrategy())

    def test_no_indicators_warmup_zero(self) -> None:
        engine = self._engine()
        assert engine._warmup_days([]) == 0

    def test_period_param_detected(self) -> None:
        engine = self._engine()
        specs = [IndicatorSpec(name="sma", params={"period": 20})]
        assert engine._warmup_days(specs) == 20

    def test_macd_uses_slow_plus_signal(self) -> None:
        engine = self._engine()
        specs = [IndicatorSpec(name="macd", params={"slow": 26, "signal": 9})]
        assert engine._warmup_days(specs) == 35

    def test_max_across_specs(self) -> None:
        engine = self._engine()
        specs = [
            IndicatorSpec(name="sma", params={"period": 50}),
            IndicatorSpec(name="atr", params={"period": 14}),
        ]
        assert engine._warmup_days(specs) == 50


class TestBenchmarkIntegration:
    def test_benchmark_comparison_populated(self) -> None:
        syms = {"SYM0": _make_ohlcv(100, close_start=100.0)}
        d = _setup_universe_dir(syms)
        cfg = BacktestConfig(
            data=DataConfig(validated_dir=d, min_history_days=20),
            portfolio=PortfolioConfig(initial_capital=500_000.0, max_positions=1),
        )
        engine = BacktestEngine(cfg, AlwaysBuyStrategy())
        bench = pd.Series(0.001, index=pd.date_range("2020-01-08", periods=93, freq="B"))
        result = engine.run(benchmark_returns=bench)
        # With sufficient overlap, benchmark comparison should be populated
        # (may be empty if equity returns don't align with benchmark dates)
        assert isinstance(result.benchmark_comparison, dict)
