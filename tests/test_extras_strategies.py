"""Smoke tests for the extras strategies package."""
from __future__ import annotations

import vbacktest.strategies  # noqa: F401 — registers core 6
import vbacktest.strategies.extras  # noqa: F401 — registers extras


def test_extras_importable() -> None:
    """All extras strategies are importable without error."""
    from vbacktest.strategies.extras import (
        AccelBreakoutStrategy,
        DarvasBoxStrategy,
        DonchianTrendStrategy,
        MinerviniSEPAStrategy,
        RJGrowthMomentumStrategy,
        VCPBreakoutStrategy,
    )
    # Spot-check a few
    for cls in (
        AccelBreakoutStrategy,
        DarvasBoxStrategy,
        DonchianTrendStrategy,
        MinerviniSEPAStrategy,
        RJGrowthMomentumStrategy,
        VCPBreakoutStrategy,
    ):
        s = cls()
        assert hasattr(s, "indicators")
        assert hasattr(s, "on_bar")
        assert hasattr(s, "exit_rules")


def test_extras_registered_in_strategy_registry() -> None:
    """Importing extras registers all strategies."""
    from vbacktest.registry import strategy_registry

    keys = strategy_registry.keys()
    # 6 core + 47 extras = 53 total
    assert len(keys) >= 53, f"Expected ≥53 registered strategies, got {len(keys)}: {keys}"

    assert "rj_growth_momentum" in keys
    assert "vcp_breakout" in keys
    assert "darvas_box" in keys
    assert "minervini_sepa" in keys


def test_extras_indicators_return_list() -> None:
    """Every extras strategy returns a list of IndicatorSpec from indicators()."""
    from vbacktest.strategies import extras
    from vbacktest.indicators import IndicatorSpec

    modules = [
        extras.AccelBreakoutStrategy,
        extras.DonchianTrendStrategy,
        extras.MomentumMasterStrategy,
        extras.WeinsteinStage2Strategy,
    ]
    for cls in modules:
        s = cls()
        specs = s.indicators()
        assert isinstance(specs, list), f"{cls.__name__}.indicators() not a list"
        for sp in specs:
            assert isinstance(sp, IndicatorSpec), f"{cls.__name__} returned non-IndicatorSpec"


def test_extras_exit_rules_fresh_per_call() -> None:
    """exit_rules() returns a new list on each call."""
    from vbacktest.strategies.extras import RJGrowthMomentumStrategy

    s = RJGrowthMomentumStrategy()
    r1 = s.exit_rules()
    r2 = s.exit_rules()
    assert r1 is not r2
