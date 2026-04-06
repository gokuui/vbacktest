"""Pinned threshold defaults for go/no-go validation.

Threshold changes require a major version bump.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StatisticalThresholds:
    min_trades_pass: int = 200
    min_trades_warn: int = 150
    min_cagr_pass: float = 5.0
    min_cagr_warn: float = 0.0
    min_profit_factor_pass: float = 1.3
    min_profit_factor_warn: float = 1.1
    min_expectancy_pass: float = 0.5
    min_expectancy_warn: float = 0.0
    max_pvalue_pass: float = 0.10
    max_pvalue_warn: float = 0.20
    min_win_rate_pass: float = 40.0
    min_win_rate_warn: float = 35.0


@dataclass
class MonteCarloThresholds:
    max_reshuffle_dd_pass: float = 30.0
    max_reshuffle_dd_warn: float = 40.0
    min_bootstrap_cagr_p5: float = 0.0
    min_slip_inject_cagr_p5: float = 0.0


@dataclass
class RiskThresholds:
    max_drawdown_pass: float = 25.0
    max_drawdown_warn: float = 35.0
    max_drawdown_hard_nogo: float = 50.0
    max_dd_duration_pass_months: int = 12
    max_dd_duration_warn_months: int = 18
    max_kelly_pass: float = 0.25
    max_risk_of_ruin_pass: float = 5.0
    max_risk_of_ruin_warn: float = 15.0
    min_calmar_pass: float = 1.0
    min_calmar_warn: float = 0.5


@dataclass
class GoNoGoThresholds:
    statistical: StatisticalThresholds = field(default_factory=StatisticalThresholds)
    monte_carlo: MonteCarloThresholds = field(default_factory=MonteCarloThresholds)
    risk: RiskThresholds = field(default_factory=RiskThresholds)
