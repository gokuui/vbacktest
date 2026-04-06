"""Built-in exit rules for backtesting."""
from __future__ import annotations

import pandas as pd

from vbacktest.strategy import ExitRule, ExitSignal, ExitCondition


class StopLossRule(ExitRule):
    """Simple stop loss exit rule.

    Exits when close price drops to or below the stop price.
    """

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if stop loss is hit."""
        if bar['close'] <= position.stop_price:  # type: ignore[union-attr]
            return ExitSignal(
                condition=ExitCondition.STOP_LOSS,
                fraction=1.0,
                reason=f"Stop loss hit: close {bar['close']:.2f} <= stop {position.stop_price:.2f}",  # type: ignore[union-attr]
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update for simple stop loss."""


class TrailingATRStopRule(ExitRule):
    """Trailing ATR stop loss rule.

    Stop ratchets up as price increases: stop = close - (multiplier * ATR).
    Stop never decreases (trailing).
    """

    def __init__(self, atr_column: str = "atr", multiplier: float = 2.0) -> None:
        self.atr_column = atr_column
        self.multiplier = multiplier
        self._trailing_stop: float | None = None

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if trailing stop is hit."""
        if self._trailing_stop is None:
            return None

        if bar['close'] <= self._trailing_stop:
            return ExitSignal(
                condition=ExitCondition.TRAILING_STOP,
                fraction=1.0,
                reason=f"Trailing ATR stop hit: close {bar['close']:.2f} <= stop {self._trailing_stop:.2f}",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """Update trailing stop (ratchet up only)."""
        if self.atr_column not in bar or pd.isna(bar[self.atr_column]):
            if self._trailing_stop is None:
                self._trailing_stop = position.stop_price  # type: ignore[union-attr]
            return

        new_stop = bar['close'] - (self.multiplier * bar[self.atr_column])

        if self._trailing_stop is None:
            self._trailing_stop = max(position.stop_price, new_stop)  # type: ignore[union-attr]
        else:
            self._trailing_stop = max(self._trailing_stop, new_stop)

    def __deepcopy__(self, memo: dict[int, object]) -> TrailingATRStopRule:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.atr_column = self.atr_column
        result.multiplier = self.multiplier
        result._trailing_stop = None  # reset for new position
        return result


class TrailingMARule(ExitRule):
    """Trailing moving average exit rule.

    Exits when close drops below the specified moving average.
    """

    def __init__(self, ma_column: str = "sma_10") -> None:
        self.ma_column = ma_column

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if price is below MA."""
        if self.ma_column not in bar or pd.isna(bar[self.ma_column]):
            return None

        if bar['close'] < bar[self.ma_column]:
            return ExitSignal(
                condition=ExitCondition.TRAILING_MA,
                fraction=1.0,
                reason=f"Close below {self.ma_column}: {bar['close']:.2f} < {bar[self.ma_column]:.2f}",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update for MA rule."""


class TakeProfitPartialRule(ExitRule):
    """Partial take-profit exit rule.

    Exits a fraction of position when profit reaches N*R (R-multiple).
    Fires only once per position.
    """

    def __init__(self, r_multiple: float = 2.0, fraction: float = 0.5) -> None:
        self.r_multiple = r_multiple
        self.fraction = fraction
        self._fired = False

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if take-profit target is reached."""
        if self._fired:
            return None

        r = position.entry_price - position.stop_price  # type: ignore[union-attr]
        if r <= 0:
            return None

        profit_per_share = bar['close'] - position.entry_price  # type: ignore[union-attr]

        if profit_per_share >= (self.r_multiple * r):
            self._fired = True
            return ExitSignal(
                condition=ExitCondition.TAKE_PROFIT,
                fraction=self.fraction,
                reason=f"Take profit at {self.r_multiple}R: profit {profit_per_share:.2f} >= {self.r_multiple * r:.2f}",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state update needed (check handles firing logic)."""

    def __deepcopy__(self, memo: dict[int, object]) -> TakeProfitPartialRule:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.r_multiple = self.r_multiple
        result.fraction = self.fraction
        result._fired = False  # reset for new position
        return result


class TimeStopRule(ExitRule):
    """Time-based exit rule.

    Exits after holding for a maximum number of calendar days.
    """

    def __init__(self, max_holding_days: int = 30) -> None:
        self.max_holding_days = max_holding_days

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if maximum holding period reached."""
        holding_days = (bar['date'] - position.entry_date).days  # type: ignore[union-attr]

        if holding_days >= self.max_holding_days:
            return ExitSignal(
                condition=ExitCondition.TIME_STOP,
                fraction=1.0,
                reason=f"Time stop: held {holding_days} days >= {self.max_holding_days} max",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update for time stop."""


class MA10ExitRule(ExitRule):
    """Moving Average (10) exit rule.

    Exits when price closes below 10-period SMA (exit next day at open).
    Acts as a trailing exit that follows price momentum.
    """

    def __init__(self, ma_column: str = "sma_10") -> None:
        self.ma_column = ma_column

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if price closed below MA(10)."""
        if self.ma_column not in bar or pd.isna(bar[self.ma_column]):
            return None

        ma_value = bar[self.ma_column]

        if bar['close'] < ma_value:
            return ExitSignal(
                condition=ExitCondition.TRAILING_MA,
                fraction=1.0,
                reason=f"Close below MA(10): {bar['close']:.2f} < {ma_value:.2f}",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update for MA exit."""


class TrailingLowRule(ExitRule):
    """Trailing N-day rolling low exit rule.

    Exits when close price drops below the N-day rolling low.
    Used by Turtle Trading strategy.
    """

    def __init__(self, lookback_days: int = 20, low_column: str | None = None) -> None:
        self.lookback_days = lookback_days
        self.low_column = low_column if low_column is not None else f"low_{lookback_days}d"

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if close is below N-day rolling low."""
        if self.low_column not in bar or pd.isna(bar[self.low_column]):
            return None

        if bar['close'] < bar[self.low_column]:
            return ExitSignal(
                condition=ExitCondition.TRAILING_STOP,
                fraction=1.0,
                reason=f"Close below {self.lookback_days}d low: {bar['close']:.2f} < {bar[self.low_column]:.2f}",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update for trailing low rule."""


class MaxHoldingBarsRule(ExitRule):
    """Bar-count-based exit rule.

    Exits after N trading bars (not calendar days).
    Uses ``position.entry_idx`` and ``bar_idx`` to count bars held.
    """

    def __init__(self, max_bars: int = 2) -> None:
        self.max_bars = max_bars

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        """Check if maximum holding bars reached."""
        holding_bars = (bar_idx - position.entry_idx) if hasattr(position, 'entry_idx') else 0  # type: ignore[union-attr]

        if holding_bars >= self.max_bars:
            return ExitSignal(
                condition=ExitCondition.TIME_STOP,
                fraction=1.0,
                reason=f"Max holding bars: held {holding_bars} bars >= {self.max_bars} max",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update for max holding bars rule."""


class TargetProfitRule(ExitRule):
    """Fixed percentage profit target exit.

    Exits full position when price reaches the target percentage above entry.
    Designed for mean reversion strategies expecting a specific snap-back.
    """

    def __init__(self, target_pct: float = 5.0) -> None:
        self.target_pct = target_pct

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        target_price = position.entry_price * (1 + self.target_pct / 100)  # type: ignore[union-attr]
        if bar['close'] >= target_price:
            return ExitSignal(
                condition=ExitCondition.TAKE_PROFIT,
                fraction=1.0,
                reason=f"Target profit {self.target_pct}%: {bar['close']:.2f} >= {target_price:.2f}",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update."""


class MACrossbackRule(ExitRule):
    """Exit when price crosses back above a moving average.

    For mean reversion: enter on pullback below MA, exit when price recovers.
    """

    def __init__(self, ma_column: str = "sma_50") -> None:
        self.ma_column = ma_column

    def check(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> ExitSignal | None:
        if self.ma_column not in bar or pd.isna(bar[self.ma_column]):
            return None
        if bar['close'] > bar[self.ma_column] * 1.02:
            return ExitSignal(
                condition=ExitCondition.STRATEGY_EXIT,
                fraction=1.0,
                reason=f"MA crossback: {bar['close']:.2f} > {bar[self.ma_column]:.2f}",
            )
        return None

    def update(self, position: object, bar: pd.Series, bar_idx: int, stock_data: pd.DataFrame) -> None:
        """No state to update."""
