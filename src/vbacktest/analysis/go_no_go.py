"""GoNoGo — orchestrates the full pre-live validation pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from vbacktest.analysis.annual_breakdown import run_annual_breakdown
from vbacktest.analysis.defaults import GoNoGoThresholds
from vbacktest.analysis.execution_realism import run_execution_realism
from vbacktest.analysis.monte_carlo import run_monte_carlo
from vbacktest.analysis.regime_rolling import run_regime_rolling
from vbacktest.analysis.report import TestResult, ValidationReport
from vbacktest.analysis.risk_metrics import run_advanced_risk, run_risk_metrics


class GoNoGo:
    """Pre-live go/no-go validation suite.

    Runs six categories of statistical tests on completed backtest results
    and produces a :class:`~vbacktest.analysis.report.ValidationReport`.

    Example::

        from vbacktest.analysis import GoNoGo

        report = GoNoGo(trades=result.trades, equity=result.equity_series()).run()
        report.print_terminal()
        print(report.verdict())  # "PASS", "WARN", or "FAIL"

    Args:
        trades: List of :class:`~vbacktest.portfolio.Trade` objects.
        equity: Date-indexed equity Series (one value per bar).
        initial_capital: Starting capital in currency units. Default ``100_000``.
        mc_sims: Number of Monte Carlo simulations. Default ``5000``.
        seed: Random seed for reproducibility. Default ``42``.
        benchmark_returns: Optional date-indexed daily benchmark returns (e.g. index).
        thresholds: Override default pass/warn thresholds.
    """

    def __init__(
        self,
        trades: list[Any],
        equity: pd.Series,
        initial_capital: float = 100_000,
        mc_sims: int = 5000,
        seed: int = 42,
        benchmark_returns: pd.Series | None = None,
        thresholds: GoNoGoThresholds | None = None,
        n_strategies_tested: int = 1,
    ) -> None:
        self.trades = trades
        self.equity = equity
        self.initial_capital = initial_capital
        self.mc_sims = mc_sims
        self.seed = seed
        self.benchmark_returns = benchmark_returns
        self.thresholds = thresholds or GoNoGoThresholds()
        self.n_strategies_tested = n_strategies_tested

    def run(self, name: str = "strategy") -> ValidationReport:
        """Run all test categories and return a :class:`ValidationReport`.

        Args:
            name: Label for the strategy in the report.
        """
        report = ValidationReport(name=name)
        calendar_days = max(
            (self.equity.index[-1] - self.equity.index[0]).days, 1
        )

        report.results["Statistical Foundation"] = self._run_statistical()

        mc_results, shuffled = run_monte_carlo(
            self.trades,
            self.equity,
            self.initial_capital,
            calendar_days,
            self.mc_sims,
            self.seed,
            self.thresholds.monte_carlo,
        )
        report.results["Monte Carlo"] = mc_results

        report.results["Execution Realism"] = run_execution_realism(
            self.trades, self.equity, self.initial_capital, calendar_days
        )

        report.results["Risk Metrics"] = run_risk_metrics(
            self.trades,
            self.equity,
            shuffled,
            self.initial_capital,
            self.thresholds.risk,
        )

        report.results["Advanced Risk"] = run_advanced_risk(
            self.trades,
            self.equity,
            self.initial_capital,
            self.benchmark_returns,
            self.n_strategies_tested,
        )

        report.results["Regime & Rolling"] = run_regime_rolling(
            self.equity, self.benchmark_returns
        )

        report.results["Annual Breakdown"] = run_annual_breakdown(
            self.equity, self.benchmark_returns
        )

        # Aggregate verdict
        all_tests = [t for cat in report.results.values() for t in cat]
        if any(t.status == "FAIL" for t in all_tests):
            report.overall = "FAIL"
            report.hard_nogo_triggered = any(
                t.hard_nogo and t.status == "FAIL" for t in all_tests
            )
        elif any(t.status == "WARN" for t in all_tests):
            report.overall = "WARN"
        else:
            report.overall = "PASS"

        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_statistical(self) -> list[TestResult]:
        """Six statistical foundation tests."""
        results: list[TestResult] = []
        thresholds = self.thresholds.statistical

        if not self.trades:
            return [
                TestResult(
                    "Trade count",
                    "0",
                    "FAIL",
                    hard_nogo=True,
                    detail="No trades generated",
                )
            ]

        pnl_pcts = np.array([t.pnl_pct for t in self.trades])
        n = len(pnl_pcts)
        winners = pnl_pcts[pnl_pcts > 0]
        losers = pnl_pcts[pnl_pcts < 0]

        # Trade count
        if n >= thresholds.min_trades_pass:
            status = "PASS"
        elif n >= thresholds.min_trades_warn:
            status = "WARN"
        else:
            status = "FAIL"
        results.append(
            TestResult("Trade count", str(n), status, hard_nogo=(n < thresholds.min_trades_warn))
        )

        # CAGR
        if len(self.equity) >= 2:
            total_ret = float(self.equity.iloc[-1]) / float(self.equity.iloc[0]) - 1
            days = max((self.equity.index[-1] - self.equity.index[0]).days, 1)
            cagr_pct = ((1 + total_ret) ** (365 / days) - 1) * 100
        else:
            cagr_pct = 0.0
        if cagr_pct > thresholds.min_cagr_pass:
            status = "PASS"
        elif cagr_pct > thresholds.min_cagr_warn:
            status = "WARN"
        else:
            status = "FAIL"
        results.append(
            TestResult("CAGR", f"{cagr_pct:.1f}%", status, hard_nogo=(cagr_pct <= 0))
        )

        # Profit factor
        gross_win = float(winners.sum()) if len(winners) > 0 else 0.0
        gross_loss = float(abs(losers.sum())) if len(losers) > 0 else 1e-9
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        if pf >= thresholds.min_profit_factor_pass:
            status = "PASS"
        elif pf >= thresholds.min_profit_factor_warn:
            status = "WARN"
        else:
            status = "FAIL"
        results.append(TestResult("Profit factor", f"{pf:.2f}", status))

        # Expectancy (mean pnl_pct per trade)
        exp = float(pnl_pcts.mean())
        if exp > thresholds.min_expectancy_pass:
            status = "PASS"
        elif exp > thresholds.min_expectancy_warn:
            status = "WARN"
        else:
            status = "FAIL"
        results.append(
            TestResult(
                "Expectancy",
                f"{exp:+.2f}% per trade",
                status,
                hard_nogo=(exp <= 0),
            )
        )

        # t-test (one-tailed: is mean > 0?)
        if n >= 2:
            t_stat, p_two = stats.ttest_1samp(pnl_pcts, 0)
            p_one = p_two / 2 if t_stat > 0 else 1.0
        else:
            p_one = 1.0
        if p_one < thresholds.max_pvalue_pass:
            status = "PASS"
        elif p_one < thresholds.max_pvalue_warn:
            status = "WARN"
        else:
            status = "FAIL"
        detail = "(p<0.10 preferred, p<0.20 required)" if status == "WARN" else ""
        results.append(
            TestResult(
                "t-test p-value",
                f"{p_one:.3f}",
                status,
                hard_nogo=bool(p_one >= thresholds.max_pvalue_warn),
                detail=detail,
            )
        )

        # Win rate
        win_rate = len(winners) / n * 100
        if win_rate >= thresholds.min_win_rate_pass:
            status = "PASS"
        elif win_rate >= thresholds.min_win_rate_warn:
            status = "WARN"
        else:
            status = "FAIL"
        results.append(TestResult("Win rate", f"{win_rate:.1f}%", status))

        return results
