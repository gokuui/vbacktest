"""Strategy and ExitRule abstractions for backtesting."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vbacktest.portfolio import Portfolio, Position


@dataclass
class IndicatorSpec:
    """Specification for a technical indicator."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class SignalAction(Enum):
    BUY = "buy"


class ExitCondition(Enum):
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TRAILING_MA = "trailing_ma"
    TIME_STOP = "time_stop"
    TAKE_PROFIT = "take_profit"
    STRATEGY_EXIT = "strategy_exit"


@dataclass
class Signal:
    """Entry signal from strategy."""
    symbol: str
    action: SignalAction
    date: pd.Timestamp
    stop_price: float
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitSignal:
    """Exit signal from exit rule."""
    condition: ExitCondition
    fraction: float  # Fraction of position to exit (0-1)
    reason: str

    def __post_init__(self) -> None:
        self.fraction = max(0.0, min(1.0, self.fraction))


@dataclass
class BarContext:
    """Context passed to Strategy.on_bar() each trading day.

    Attributes:
        date: Current trading date.
        universe: Dict mapping symbol -> DataFrame with indicators.
        universe_idx: Dict mapping symbol -> current bar index (anti-lookahead).
        portfolio: Current portfolio state.
        current_prices: Pre-fetched bar data {symbol: {col: value}} for O(1) access.
        universe_arrays: ADVANCED/UNSTABLE. Engine internals for performance.
            Pre-computed numpy arrays {symbol: {col: ndarray}}.
            May change between minor versions — do not rely on its structure
            for stable downstream code.
    """
    date: pd.Timestamp
    universe: dict[str, pd.DataFrame]
    universe_idx: dict[str, int]
    portfolio: Portfolio | None
    current_prices: dict[str, dict[str, float]] | None = None
    universe_arrays: dict[str, dict[str, np.ndarray]] | None = None


class ExitRule(ABC):
    """Abstract base class for exit rules."""

    @abstractmethod
    def check(
        self,
        position: Position,
        bar: pd.Series,
        bar_idx: int,
        stock_data: pd.DataFrame,
    ) -> ExitSignal | None:
        """Check if exit condition is met. Returns ExitSignal or None."""
        pass

    @abstractmethod
    def update(
        self,
        position: Position,
        bar: pd.Series,
        bar_idx: int,
        stock_data: pd.DataFrame,
    ) -> None:
        """Update rule state (e.g., trailing stops). Called after check()."""
        pass

    def __deepcopy__(self, memo: dict[int, Any]) -> ExitRule:
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def indicators(self) -> list[IndicatorSpec]:
        """Return list of indicators required by this strategy."""
        pass

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate entry signals for the current bar.

        Args:
            ctx: BarContext with date, universe, portfolio, and optional
                 performance-oriented fields (current_prices, universe_arrays).

        Returns:
            List of entry signals.
        """
        pass

    @abstractmethod
    def exit_rules(self) -> list[ExitRule]:
        """Return a fresh list of exit rule instances."""
        pass

    def on_fill(
        self, signal: Signal, fill_price: float, shares: int, date: pd.Timestamp
    ) -> None:
        """Optional callback when entry order is filled."""
        pass

    def on_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        shares: int,
        pnl: float,
        holding_days: int,
        exit_reason: str,
        date: pd.Timestamp,
    ) -> None:
        """Optional callback when position is exited."""
        pass
