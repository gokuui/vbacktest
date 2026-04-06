"""Go/no-go validation analysis for backtested strategies."""
from __future__ import annotations

from vbacktest.analysis.annual_breakdown import run_annual_breakdown
from vbacktest.analysis.defaults import GoNoGoThresholds
from vbacktest.analysis.execution_realism import run_execution_realism
from vbacktest.analysis.go_no_go import GoNoGo
from vbacktest.analysis.monte_carlo import run_monte_carlo
from vbacktest.analysis.regime_rolling import run_regime_rolling
from vbacktest.analysis.report import TestResult, ValidationReport
from vbacktest.analysis.risk_metrics import run_advanced_risk, run_risk_metrics

__all__ = [
    "GoNoGo",
    "GoNoGoThresholds",
    "ValidationReport",
    "TestResult",
    "run_monte_carlo",
    "run_risk_metrics",
    "run_advanced_risk",
    "run_execution_realism",
    "run_regime_rolling",
    "run_annual_breakdown",
]
