import pytest
import pandas as pd
import numpy as np
from vbacktest.strategy import (
    Signal, SignalAction, ExitSignal, ExitCondition,
    ExitRule, Strategy, BarContext, IndicatorSpec,
)


class TestSignal:
    def test_creation(self) -> None:
        s = Signal(symbol="AAPL", action=SignalAction.BUY,
                   date=pd.Timestamp("2020-01-01"), stop_price=90.0)
        assert s.symbol == "AAPL"
        assert s.score == 0.0

    def test_metadata_default(self) -> None:
        s = Signal(symbol="X", action=SignalAction.BUY,
                   date=pd.Timestamp("2020-01-01"), stop_price=10.0)
        assert s.metadata == {}


class TestExitSignal:
    def test_fraction_clamped_high(self) -> None:
        es = ExitSignal(condition=ExitCondition.STOP_LOSS, fraction=1.5, reason="test")
        assert es.fraction == 1.0

    def test_fraction_clamped_low(self) -> None:
        es = ExitSignal(condition=ExitCondition.STOP_LOSS, fraction=-0.5, reason="test")
        assert es.fraction == 0.0


class TestBarContext:
    def test_creation(self) -> None:
        ctx = BarContext(
            date=pd.Timestamp("2020-01-01"),
            universe={}, universe_idx={}, portfolio=None,
        )
        assert ctx.date == pd.Timestamp("2020-01-01")
        assert ctx.current_prices is None
        assert ctx.universe_arrays is None


class TestExitRuleABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            ExitRule()  # type: ignore[abstract]


class TestStrategyABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_concrete_strategy(self) -> None:
        class MyStrategy(Strategy):
            def indicators(self) -> list[IndicatorSpec]:
                return []
            def on_bar(self, ctx: BarContext) -> list[Signal]:
                return []
            def exit_rules(self) -> list[ExitRule]:
                return []

        s = MyStrategy()
        ctx = BarContext(date=pd.Timestamp("2020-01-01"),
                        universe={}, universe_idx={}, portfolio=None)
        assert s.on_bar(ctx) == []


class TestIndicatorSpec:
    def test_creation(self) -> None:
        spec = IndicatorSpec(name="sma", params={"period": 10})
        assert spec.name == "sma"
        assert spec.params == {"period": 10}

    def test_default_params(self) -> None:
        spec = IndicatorSpec(name="atr")
        assert spec.params == {}
