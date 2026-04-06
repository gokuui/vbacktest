"""Tests for built-in strategies."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbacktest.strategy import BarContext, Signal, SignalAction
from vbacktest.strategies import (
    BollingerBreakoutStrategy,
    MACrossoverStrategy,
    MomentumStrategy,
    RSIMeanReversionStrategy,
    TurtleTradingStrategy,
    VolumeBreakoutStrategy,
)
from vbacktest.indicators import IndicatorSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 300, close: float = 100.0, trend: str = "up") -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    if trend == "up":
        prices = close + np.arange(n) * 0.1
    elif trend == "flat":
        prices = np.full(n, close)
    else:
        prices = close - np.arange(n) * 0.1
    prices = np.maximum(prices, 1.0)
    return pd.DataFrame({
        "date": dates,
        "open": prices - 0.5,
        "high": prices + 1.0,
        "low": prices - 1.0,
        "close": prices,
        "volume": [2_000_000.0] * n,
    })


def _apply_indicators(df: pd.DataFrame, specs: list[IndicatorSpec]) -> pd.DataFrame:
    from vbacktest.indicators import INDICATOR_REGISTRY
    for spec in specs:
        if spec.name in INDICATOR_REGISTRY:
            INDICATOR_REGISTRY[spec.name](df, **spec.params)
    return df


def _make_ctx(
    strategy,
    symbol: str = "SYM",
    df: pd.DataFrame | None = None,
    idx: int | None = None,
) -> BarContext:
    if df is None:
        df = _make_df()
        specs = strategy.indicators()
        _apply_indicators(df, specs)

    if idx is None:
        idx = len(df) - 1

    date = df["date"].iloc[idx]

    class _DummyPortfolio:
        def has_position(self, s: str) -> bool:
            return False

    bar_dict = {col: df[col].iloc[idx] for col in df.columns if col != "date"}

    return BarContext(
        date=date,
        universe={symbol: df},
        universe_idx={symbol: idx},
        portfolio=_DummyPortfolio(),  # type: ignore[arg-type]
        current_prices={symbol: {**bar_dict, "date": date}},
        universe_arrays=None,  # test slow path
    )


# ---------------------------------------------------------------------------
# MACrossoverStrategy
# ---------------------------------------------------------------------------

class TestMACrossoverStrategy:
    def test_indicators_returned(self) -> None:
        s = MACrossoverStrategy()
        specs = s.indicators()
        assert len(specs) > 0
        names = [sp.name for sp in specs]
        assert "sma" in names
        assert "atr" in names

    def test_exit_rules_returned(self) -> None:
        s = MACrossoverStrategy()
        rules = s.exit_rules()
        assert len(rules) > 0

    def test_exit_rules_fresh_each_call(self) -> None:
        s = MACrossoverStrategy()
        r1 = s.exit_rules()
        r2 = s.exit_rules()
        assert r1 is not r2

    def test_on_bar_returns_list(self) -> None:
        s = MACrossoverStrategy(fast_period=5, slow_period=10, trend_period=20)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        result = s.on_bar(ctx)
        assert isinstance(result, list)

    def test_signals_have_correct_action(self) -> None:
        s = MACrossoverStrategy(fast_period=5, slow_period=10, trend_period=20)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        for sig in s.on_bar(ctx):
            assert sig.action == SignalAction.BUY
            assert sig.stop_price < float(df["close"].iloc[-1])

    def test_partial_exit_rule_added_when_configured(self) -> None:
        s = MACrossoverStrategy(partial_exit_r=2.0)
        rules = s.exit_rules()
        names = [type(r).__name__ for r in rules]
        assert "TakeProfitPartialRule" in names

    def test_no_partial_exit_by_default(self) -> None:
        s = MACrossoverStrategy()
        names = [type(r).__name__ for r in s.exit_rules()]
        assert "TakeProfitPartialRule" not in names


# ---------------------------------------------------------------------------
# RSIMeanReversionStrategy
# ---------------------------------------------------------------------------

class TestRSIMeanReversionStrategy:
    def test_indicators_include_rsi(self) -> None:
        s = RSIMeanReversionStrategy()
        names = [sp.name for sp in s.indicators()]
        assert "rsi" in names

    def test_exit_rules_include_stop_and_time(self) -> None:
        s = RSIMeanReversionStrategy()
        names = [type(r).__name__ for r in s.exit_rules()]
        assert "StopLossRule" in names
        assert "TimeStopRule" in names

    def test_on_bar_returns_list(self) -> None:
        s = RSIMeanReversionStrategy(trend_period=None, volume_period=None)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        assert isinstance(s.on_bar(ctx), list)

    def test_signals_stop_below_close(self) -> None:
        s = RSIMeanReversionStrategy(trend_period=None, volume_period=None)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        for sig in s.on_bar(ctx):
            assert sig.stop_price < float(df["close"].iloc[-1])


# ---------------------------------------------------------------------------
# BollingerBreakoutStrategy
# ---------------------------------------------------------------------------

class TestBollingerBreakoutStrategy:
    def test_indicators_include_bollinger(self) -> None:
        s = BollingerBreakoutStrategy()
        names = [sp.name for sp in s.indicators()]
        assert "bollinger_bands" in names

    def test_exit_rules_not_empty(self) -> None:
        s = BollingerBreakoutStrategy()
        assert len(s.exit_rules()) > 0

    def test_on_bar_returns_list(self) -> None:
        s = BollingerBreakoutStrategy(trend_period=20)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        assert isinstance(s.on_bar(ctx), list)


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_indicators_include_roc(self) -> None:
        s = MomentumStrategy()
        names = [sp.name for sp in s.indicators()]
        assert "roc" in names

    def test_exit_rules_include_time_stop(self) -> None:
        s = MomentumStrategy()
        names = [type(r).__name__ for r in s.exit_rules()]
        assert "TimeStopRule" in names

    def test_on_bar_returns_list(self) -> None:
        s = MomentumStrategy(roc_periods=(5, 10), trend_period=20, high_period=30)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        assert isinstance(s.on_bar(ctx), list)


# ---------------------------------------------------------------------------
# VolumeBreakoutStrategy
# ---------------------------------------------------------------------------

class TestVolumeBreakoutStrategy:
    def test_indicators_include_volume_sma(self) -> None:
        s = VolumeBreakoutStrategy()
        names = [sp.name for sp in s.indicators()]
        assert "volume_sma" in names

    def test_exit_rules_not_empty(self) -> None:
        s = VolumeBreakoutStrategy()
        assert len(s.exit_rules()) > 0

    def test_on_bar_returns_list(self) -> None:
        s = VolumeBreakoutStrategy(breakout_period=20, volume_period=10)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        assert isinstance(s.on_bar(ctx), list)


# ---------------------------------------------------------------------------
# TurtleTradingStrategy
# ---------------------------------------------------------------------------

class TestTurtleTradingStrategy:
    def test_indicators_include_rolling_high(self) -> None:
        s = TurtleTradingStrategy()
        names = [sp.name for sp in s.indicators()]
        assert "rolling_high" in names
        assert "rolling_low" in names

    def test_exit_rules_include_trailing_low(self) -> None:
        s = TurtleTradingStrategy()
        names = [type(r).__name__ for r in s.exit_rules()]
        assert "TrailingLowRule" in names

    def test_on_bar_returns_list(self) -> None:
        s = TurtleTradingStrategy(entry_breakout_days=10, exit_breakdown_days=5)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        assert isinstance(s.on_bar(ctx), list)


# ---------------------------------------------------------------------------
# All strategies — common contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy_cls,kwargs", [
    (MACrossoverStrategy, {"fast_period": 5, "slow_period": 10, "trend_period": 20}),
    (RSIMeanReversionStrategy, {"trend_period": None, "volume_period": None}),
    (BollingerBreakoutStrategy, {"trend_period": 20}),
    (MomentumStrategy, {"roc_periods": (5, 10), "trend_period": 20, "high_period": 30}),
    (VolumeBreakoutStrategy, {"breakout_period": 20, "volume_period": 10}),
    (TurtleTradingStrategy, {"entry_breakout_days": 10, "exit_breakdown_days": 5}),
])
class TestStrategyContract:
    def test_indicators_returns_list(self, strategy_cls, kwargs) -> None:
        s = strategy_cls(**kwargs)
        specs = s.indicators()
        assert isinstance(specs, list)
        for sp in specs:
            assert isinstance(sp, IndicatorSpec)

    def test_exit_rules_returns_list(self, strategy_cls, kwargs) -> None:
        s = strategy_cls(**kwargs)
        rules = s.exit_rules()
        assert isinstance(rules, list)

    def test_exit_rules_fresh_instances(self, strategy_cls, kwargs) -> None:
        s = strategy_cls(**kwargs)
        r1 = s.exit_rules()
        r2 = s.exit_rules()
        assert r1 is not r2

    def test_on_bar_returns_list_of_signals(self, strategy_cls, kwargs) -> None:
        s = strategy_cls(**kwargs)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())
        ctx = _make_ctx(s, df=df)
        result = s.on_bar(ctx)
        assert isinstance(result, list)
        for sig in result:
            assert isinstance(sig, Signal)

    def test_no_signal_for_held_symbol(self, strategy_cls, kwargs) -> None:
        """Strategy must not signal a symbol it already holds."""
        s = strategy_cls(**kwargs)
        df = _make_df(100)
        _apply_indicators(df, s.indicators())

        class _HoldsAll:
            def has_position(self, sym: str) -> bool:
                return True

        ctx = _make_ctx(s, df=df)
        ctx = BarContext(
            date=ctx.date,
            universe=ctx.universe,
            universe_idx=ctx.universe_idx,
            portfolio=_HoldsAll(),  # type: ignore[arg-type]
            current_prices=ctx.current_prices,
        )
        assert s.on_bar(ctx) == []
