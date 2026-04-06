"""Public API surface tests — verify stable top-level imports."""
from __future__ import annotations


def test_top_level_imports() -> None:
    from vbacktest import (
        BacktestConfig,
        BacktestEngine,
        BacktestResult,
        BacktestStats,
        BarContext,
        DataConfig,
        ExecutionConfig,
        ExitCondition,
        ExitRule,
        ExitSignal,
        IndicatorSpec,
        Portfolio,
        PortfolioConfig,
        Position,
        Signal,
        SignalAction,
        Strategy,
        Trade,
        __version__,
    )
    assert __version__ == "0.1.0"


def test_strategy_imports() -> None:
    from vbacktest.strategies import (
        BollingerBreakoutStrategy,
        MACrossoverStrategy,
        MomentumStrategy,
        RSIMeanReversionStrategy,
        TurtleTradingStrategy,
        VolumeBreakoutStrategy,
    )
    # All are Strategy subclasses
    from vbacktest import Strategy
    for cls in (
        MACrossoverStrategy,
        RSIMeanReversionStrategy,
        BollingerBreakoutStrategy,
        MomentumStrategy,
        VolumeBreakoutStrategy,
        TurtleTradingStrategy,
    ):
        assert issubclass(cls, Strategy)


def test_analysis_imports() -> None:
    from vbacktest.analysis import (
        GoNoGo,
        GoNoGoThresholds,
        TestResult,
        ValidationReport,
    )


def test_exit_rule_imports() -> None:
    from vbacktest.exit_rules import (
        StopLossRule,
        TakeProfitPartialRule,
        TimeStopRule,
        TrailingATRStopRule,
        TrailingMARule,
    )


def test_registry_convenience_functions() -> None:
    from vbacktest.registry import (
        list_indicators,
        list_strategies,
        register_indicator,
        register_strategy,
        strategy_registry,
    )
    # Ensure 6 core strategies are available after importing strategies module
    import vbacktest.strategies  # noqa: F401

    keys = list_strategies()
    assert "ma_crossover" in keys
    assert "rsi_mean_reversion" in keys


def test_indicator_registry_populated() -> None:
    from vbacktest.indicators import INDICATOR_REGISTRY
    from vbacktest.registry import list_indicators

    # At minimum the standard indicators are registered
    assert "sma" in INDICATOR_REGISTRY
    assert "atr" in INDICATOR_REGISTRY
    assert "rsi" in INDICATOR_REGISTRY


def test_config_simple_factory() -> None:
    import tempfile
    from vbacktest import BacktestConfig

    with tempfile.TemporaryDirectory() as d:
        cfg = BacktestConfig.simple(d, capital=50_000)
    assert cfg.portfolio.initial_capital == 50_000


def test_version_string_format() -> None:
    from vbacktest import __version__

    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
