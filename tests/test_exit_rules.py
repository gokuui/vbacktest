"""Tests for built-in exit rules."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vbacktest.strategy import ExitCondition
from vbacktest.exit_rules import (
    StopLossRule,
    TrailingATRStopRule,
    TrailingMARule,
    TakeProfitPartialRule,
    TimeStopRule,
    MA10ExitRule,
    TrailingLowRule,
    MaxHoldingBarsRule,
    TargetProfitRule,
    MACrossbackRule,
)


# ---------------------------------------------------------------------------
# Minimal Position stub
# ---------------------------------------------------------------------------

@dataclass
class _Position:
    symbol: str = "AAPL"
    entry_price: float = 100.0
    stop_price: float = 90.0
    entry_date: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2020-01-01"))
    entry_idx: int = 0
    shares: int = 10


def _bar(close: float = 100.0, date: str = "2020-01-15", **extra: Any) -> pd.Series:
    data: dict[str, Any] = {
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "date": pd.Timestamp(date),
    }
    data.update(extra)
    return pd.Series(data)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# StopLossRule
# ---------------------------------------------------------------------------

class TestStopLossRule:
    def test_exit_at_stop(self) -> None:
        rule = StopLossRule()
        pos = _Position(stop_price=90.0)
        bar = _bar(close=90.0)
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.STOP_LOSS
        assert signal.fraction == 1.0

    def test_exit_below_stop(self) -> None:
        rule = StopLossRule()
        pos = _Position(stop_price=90.0)
        bar = _bar(close=85.0)
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None

    def test_no_exit_above_stop(self) -> None:
        rule = StopLossRule()
        pos = _Position(stop_price=90.0)
        bar = _bar(close=95.0)
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_update_is_noop(self) -> None:
        rule = StopLossRule()
        pos = _Position()
        rule.update(pos, _bar(), 0, _empty_df())  # should not raise


# ---------------------------------------------------------------------------
# TrailingATRStopRule
# ---------------------------------------------------------------------------

class TestTrailingATRStopRule:
    def test_no_exit_before_update(self) -> None:
        rule = TrailingATRStopRule(multiplier=2.0)
        pos = _Position(entry_price=100.0, stop_price=90.0)
        bar = _bar(close=80.0)
        assert rule.check(pos, bar, 1, _empty_df()) is None

    def test_trailing_stop_ratchets_up(self) -> None:
        rule = TrailingATRStopRule(atr_column="atr", multiplier=2.0)
        pos = _Position(entry_price=100.0, stop_price=90.0)

        # Update at close=110, atr=2 → stop = 110 - 4 = 106
        rule.update(pos, _bar(close=110.0, atr=2.0), 1, _empty_df())
        assert rule._trailing_stop is not None
        assert abs(rule._trailing_stop - 106.0) < 0.01

    def test_stop_never_decreases(self) -> None:
        rule = TrailingATRStopRule(atr_column="atr", multiplier=2.0)
        pos = _Position(entry_price=100.0, stop_price=90.0)

        rule.update(pos, _bar(close=110.0, atr=2.0), 1, _empty_df())
        stop_after_high = rule._trailing_stop
        # Price drops, would suggest lower stop — must not decrease
        rule.update(pos, _bar(close=95.0, atr=2.0), 2, _empty_df())
        assert rule._trailing_stop == stop_after_high

    def test_exit_when_stop_hit(self) -> None:
        rule = TrailingATRStopRule(atr_column="atr", multiplier=2.0)
        pos = _Position(entry_price=100.0, stop_price=90.0)
        rule.update(pos, _bar(close=110.0, atr=2.0), 1, _empty_df())

        # Close at 105 (above stop ~106): no exit
        assert rule.check(pos, _bar(close=107.0), 2, _empty_df()) is None

        # Close at 100 (below stop ~106): exit
        signal = rule.check(pos, _bar(close=100.0), 2, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.TRAILING_STOP

    def test_nan_atr_falls_back_to_initial_stop(self) -> None:
        rule = TrailingATRStopRule(atr_column="atr", multiplier=2.0)
        pos = _Position(stop_price=90.0)
        # NaN ATR: should fall back to position stop price
        rule.update(pos, _bar(close=100.0, atr=float("nan")), 1, _empty_df())
        assert rule._trailing_stop == 90.0

    def test_deepcopy_resets_state(self) -> None:
        rule = TrailingATRStopRule()
        pos = _Position()
        rule.update(pos, _bar(close=110.0, atr=2.0), 1, _empty_df())
        assert rule._trailing_stop is not None

        cloned = copy.deepcopy(rule)
        assert cloned._trailing_stop is None  # reset
        assert cloned.multiplier == rule.multiplier


# ---------------------------------------------------------------------------
# TrailingMARule
# ---------------------------------------------------------------------------

class TestTrailingMARule:
    def test_exit_below_ma(self) -> None:
        rule = TrailingMARule(ma_column="sma_10")
        pos = _Position()
        bar = _bar(close=95.0, sma_10=100.0)
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.TRAILING_MA

    def test_no_exit_above_ma(self) -> None:
        rule = TrailingMARule(ma_column="sma_10")
        pos = _Position()
        bar = _bar(close=105.0, sma_10=100.0)
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_no_exit_at_ma(self) -> None:
        """Strict less-than: equal to MA should not trigger exit."""
        rule = TrailingMARule(ma_column="sma_10")
        pos = _Position()
        bar = _bar(close=100.0, sma_10=100.0)
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_no_exit_nan_ma(self) -> None:
        rule = TrailingMARule(ma_column="sma_10")
        pos = _Position()
        bar = _bar(close=90.0, sma_10=float("nan"))
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_no_exit_missing_ma_column(self) -> None:
        rule = TrailingMARule(ma_column="sma_10")
        pos = _Position()
        bar = _bar(close=90.0)  # no sma_10 key
        assert rule.check(pos, bar, 5, _empty_df()) is None


# ---------------------------------------------------------------------------
# TakeProfitPartialRule
# ---------------------------------------------------------------------------

class TestTakeProfitPartialRule:
    def test_fires_at_r_multiple(self) -> None:
        rule = TakeProfitPartialRule(r_multiple=2.0, fraction=0.5)
        pos = _Position(entry_price=100.0, stop_price=90.0)  # R = 10
        # Profit = 20 = 2R → should fire
        bar = _bar(close=120.0)
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None
        assert signal.fraction == 0.5
        assert signal.condition == ExitCondition.TAKE_PROFIT

    def test_does_not_fire_below_target(self) -> None:
        rule = TakeProfitPartialRule(r_multiple=2.0, fraction=0.5)
        pos = _Position(entry_price=100.0, stop_price=90.0)
        bar = _bar(close=115.0)  # profit 15 < 2R (20)
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_fires_only_once(self) -> None:
        rule = TakeProfitPartialRule(r_multiple=2.0, fraction=0.5)
        pos = _Position(entry_price=100.0, stop_price=90.0)
        bar = _bar(close=125.0)
        rule.check(pos, bar, 5, _empty_df())  # fire
        assert rule.check(pos, bar, 6, _empty_df()) is None  # no second fire

    def test_invalid_risk_no_exit(self) -> None:
        """If stop >= entry, risk is invalid — no exit."""
        rule = TakeProfitPartialRule()
        pos = _Position(entry_price=100.0, stop_price=100.0)  # R = 0
        bar = _bar(close=150.0)
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_deepcopy_resets_fired(self) -> None:
        rule = TakeProfitPartialRule()
        pos = _Position(entry_price=100.0, stop_price=90.0)
        rule.check(pos, _bar(close=125.0), 1, _empty_df())
        assert rule._fired is True

        cloned = copy.deepcopy(rule)
        assert cloned._fired is False


# ---------------------------------------------------------------------------
# TimeStopRule
# ---------------------------------------------------------------------------

class TestTimeStopRule:
    def test_exits_after_max_days(self) -> None:
        rule = TimeStopRule(max_holding_days=10)
        pos = _Position(entry_date=pd.Timestamp("2020-01-01"))
        bar = _bar(close=100.0, date="2020-01-15")  # 14 days later
        signal = rule.check(pos, bar, 10, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.TIME_STOP

    def test_no_exit_before_max_days(self) -> None:
        rule = TimeStopRule(max_holding_days=30)
        pos = _Position(entry_date=pd.Timestamp("2020-01-01"))
        bar = _bar(close=100.0, date="2020-01-10")  # 9 days
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_exits_exactly_at_max_days(self) -> None:
        rule = TimeStopRule(max_holding_days=10)
        pos = _Position(entry_date=pd.Timestamp("2020-01-01"))
        bar = _bar(close=100.0, date="2020-01-11")  # exactly 10 days
        signal = rule.check(pos, bar, 10, _empty_df())
        assert signal is not None


# ---------------------------------------------------------------------------
# MA10ExitRule
# ---------------------------------------------------------------------------

class TestMA10ExitRule:
    def test_exit_below_ma(self) -> None:
        rule = MA10ExitRule()
        pos = _Position()
        bar = _bar(close=95.0, sma_10=100.0)
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.TRAILING_MA

    def test_no_exit_above_ma(self) -> None:
        rule = MA10ExitRule()
        pos = _Position()
        bar = _bar(close=105.0, sma_10=100.0)
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_custom_ma_column(self) -> None:
        rule = MA10ExitRule(ma_column="ema_10")
        pos = _Position()
        bar = _bar(close=90.0, ema_10=100.0)
        assert rule.check(pos, bar, 5, _empty_df()) is not None


# ---------------------------------------------------------------------------
# TrailingLowRule
# ---------------------------------------------------------------------------

class TestTrailingLowRule:
    def test_exit_below_low(self) -> None:
        rule = TrailingLowRule(lookback_days=20)
        pos = _Position()
        bar = _bar(close=95.0, **{"low_20d": 97.0})
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.TRAILING_STOP

    def test_no_exit_above_low(self) -> None:
        rule = TrailingLowRule(lookback_days=20)
        pos = _Position()
        bar = _bar(close=100.0, **{"low_20d": 95.0})
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_custom_column_name(self) -> None:
        rule = TrailingLowRule(lookback_days=20, low_column="my_low")
        pos = _Position()
        bar = _bar(close=90.0, my_low=95.0)
        assert rule.check(pos, bar, 5, _empty_df()) is not None

    def test_auto_column_name(self) -> None:
        rule = TrailingLowRule(lookback_days=10)
        assert rule.low_column == "low_10d"


# ---------------------------------------------------------------------------
# MaxHoldingBarsRule
# ---------------------------------------------------------------------------

class TestMaxHoldingBarsRule:
    def test_exits_after_max_bars(self) -> None:
        rule = MaxHoldingBarsRule(max_bars=5)
        pos = _Position(entry_idx=0)
        signal = rule.check(pos, _bar(), 5, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.TIME_STOP

    def test_no_exit_before_max_bars(self) -> None:
        rule = MaxHoldingBarsRule(max_bars=5)
        pos = _Position(entry_idx=0)
        assert rule.check(pos, _bar(), 3, _empty_df()) is None

    def test_no_entry_idx_attribute(self) -> None:
        """Positions without entry_idx default to 0 holding bars."""
        rule = MaxHoldingBarsRule(max_bars=5)

        class NoIdx:
            entry_price = 100.0
            stop_price = 90.0

        pos = NoIdx()
        assert rule.check(pos, _bar(), 10, _empty_df()) is None  # 0 bars held


# ---------------------------------------------------------------------------
# TargetProfitRule
# ---------------------------------------------------------------------------

class TestTargetProfitRule:
    def test_exit_at_target(self) -> None:
        rule = TargetProfitRule(target_pct=10.0)
        pos = _Position(entry_price=100.0)
        bar = _bar(close=111.0)  # above 10% target
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None
        assert signal.fraction == 1.0
        assert signal.condition == ExitCondition.TAKE_PROFIT

    def test_no_exit_below_target(self) -> None:
        rule = TargetProfitRule(target_pct=10.0)
        pos = _Position(entry_price=100.0)
        bar = _bar(close=109.0)
        assert rule.check(pos, bar, 5, _empty_df()) is None


# ---------------------------------------------------------------------------
# MACrossbackRule
# ---------------------------------------------------------------------------

class TestMACrossbackRule:
    def test_exit_above_ma_with_buffer(self) -> None:
        rule = MACrossbackRule(ma_column="sma_50")
        pos = _Position()
        bar = _bar(close=104.0, sma_50=100.0)  # 4% above → > 1.02x
        signal = rule.check(pos, bar, 5, _empty_df())
        assert signal is not None
        assert signal.condition == ExitCondition.STRATEGY_EXIT

    def test_no_exit_just_above_ma(self) -> None:
        """Within 2% buffer: should not trigger."""
        rule = MACrossbackRule(ma_column="sma_50")
        pos = _Position()
        bar = _bar(close=101.0, sma_50=100.0)  # 1% above — within buffer
        assert rule.check(pos, bar, 5, _empty_df()) is None

    def test_no_exit_nan_ma(self) -> None:
        rule = MACrossbackRule()
        pos = _Position()
        bar = _bar(close=110.0, sma_50=float("nan"))
        assert rule.check(pos, bar, 5, _empty_df()) is None
