"""vbacktest — Production-grade, market-agnostic backtesting framework."""
from __future__ import annotations

__version__ = "0.1.0"

from vbacktest.config import BacktestConfig, DataConfig, ExecutionConfig, PortfolioConfig
from vbacktest.engine import BacktestEngine
from vbacktest.indicators import IndicatorSpec
from vbacktest.portfolio import Portfolio, Position, Trade
from vbacktest.registry import (
    get_indicator,
    get_strategy,
    list_indicators,
    list_strategies,
    register_indicator,
    register_strategy,
)
from vbacktest.results import BacktestResult, BacktestStats
from vbacktest.strategy import (
    BarContext,
    ExitCondition,
    ExitRule,
    ExitSignal,
    Signal,
    SignalAction,
    Strategy,
)

__all__ = [
    "__version__",
    # Engine + config
    "BacktestEngine",
    "BacktestConfig",
    "DataConfig",
    "ExecutionConfig",
    "PortfolioConfig",
    # Strategy abstractions
    "Strategy",
    "Signal",
    "SignalAction",
    "ExitRule",
    "ExitSignal",
    "ExitCondition",
    "BarContext",
    # Results
    "BacktestResult",
    "BacktestStats",
    # Indicators
    "IndicatorSpec",
    # Portfolio
    "Portfolio",
    "Trade",
    "Position",
    # Registry
    "register_strategy",
    "register_indicator",
    "get_strategy",
    "get_indicator",
    "list_strategies",
    "list_indicators",
]
