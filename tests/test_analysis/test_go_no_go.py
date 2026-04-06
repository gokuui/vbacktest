"""Tests for the go/no-go analysis module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbacktest.analysis import GoNoGo, GoNoGoThresholds, TestResult, ValidationReport
from vbacktest.analysis.annual_breakdown import run_annual_breakdown
from vbacktest.analysis.defaults import MonteCarloThresholds, RiskThresholds
from vbacktest.analysis.execution_realism import run_execution_realism
from vbacktest.analysis.monte_carlo import run_monte_carlo, _portfolio_pnl_pcts
from vbacktest.analysis.regime_rolling import run_regime_rolling
from vbacktest.analysis.report import ValidationReport as VR


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _fake_trade(pnl: float, symbol: str = "SYM", idx: int = 0):
    """Create a minimal fake trade object."""
    exit_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=idx * 5)
    return type(
        "Trade",
        (),
        {
            "pnl": pnl,
            "pnl_pct": pnl / 10_000,
            "symbol": symbol,
            "entry_price": 100.0,
            "shares": 100,
            "exit_date": exit_date,
            "holding_days": 5,
        },
    )()


def _fake_trades(n: int = 200, win_rate: float = 0.55, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    trades = []
    for i in range(n):
        is_win = rng.random() < win_rate
        pnl = (
            abs(rng.normal(500, 200)) if is_win else -abs(rng.normal(300, 150))
        )
        trades.append(_fake_trade(pnl, symbol=f"STOCK_{i % 20}", idx=i))
    return trades


def _fake_equity(n: int = 500, initial: float = 100_000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = 1 + rng.normal(0.0005, 0.01, n)
    equity = initial * np.cumprod(returns)
    return pd.Series(equity, index=dates)


# ---------------------------------------------------------------------------
# GoNoGo — top-level tests
# ---------------------------------------------------------------------------

class TestGoNoGo:
    def test_run_returns_validation_report(self) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50, seed=42).run()
        assert isinstance(report, ValidationReport)

    def test_overall_is_valid_verdict(self) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50).run()
        assert report.overall in ("PASS", "WARN", "FAIL")

    def test_verdict_method_matches_overall(self) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50).run()
        assert report.verdict() == report.overall

    def test_report_has_all_expected_categories(self) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50).run()
        expected = {
            "Statistical Foundation",
            "Monte Carlo",
            "Execution Realism",
            "Risk Metrics",
            "Advanced Risk",
            "Regime & Rolling",
            "Annual Breakdown",
        }
        assert expected.issubset(set(report.results.keys()))

    def test_all_tests_have_valid_status(self) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50).run()
        for cat, tests in report.results.items():
            for t in tests:
                assert t.status in ("PASS", "WARN", "FAIL"), \
                    f"{cat}/{t.name} has invalid status: {t.status!r}"

    def test_no_trades_yields_fail(self) -> None:
        equity = _fake_equity()
        report = GoNoGo([], equity, mc_sims=50).run()
        assert report.overall == "FAIL"

    def test_custom_thresholds_respected(self) -> None:
        thresholds = GoNoGoThresholds()
        thresholds.statistical.min_trades_pass = 50
        thresholds.statistical.min_trades_warn = 20
        report = GoNoGo(
            _fake_trades(60), _fake_equity(), mc_sims=50, thresholds=thresholds
        ).run()
        # With 60 trades and min_trades_pass=50, trade count should PASS
        stat_tests = report.results["Statistical Foundation"]
        count_test = next(t for t in stat_tests if t.name == "Trade count")
        assert count_test.status == "PASS"

    def test_deterministic_with_seed(self) -> None:
        r1 = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=100, seed=42).run()
        r2 = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=100, seed=42).run()
        for cat in r1.results:
            for t1, t2 in zip(r1.results[cat], r2.results[cat]):
                assert t1.value == t2.value, f"Non-deterministic: {cat}/{t1.name}"

    def test_with_benchmark_returns(self) -> None:
        equity = _fake_equity()
        benchmark = pd.Series(
            np.random.default_rng(0).normal(0.0003, 0.01, len(equity)),
            index=equity.index,
        )
        report = GoNoGo(
            _fake_trades(), equity, mc_sims=50, benchmark_returns=benchmark
        ).run()
        assert isinstance(report, ValidationReport)

    def test_hard_nogo_flag_propagates(self) -> None:
        """Strategies with 0 trades should trigger hard_nogo."""
        equity = _fake_equity()
        report = GoNoGo([], equity, mc_sims=50).run()
        # hard_nogo_triggered should be True when a hard_nogo test FAILs
        hard_nogo_tests = [
            t
            for cat in report.results.values()
            for t in cat
            if t.hard_nogo and t.status == "FAIL"
        ]
        assert report.hard_nogo_triggered == bool(hard_nogo_tests)


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_to_dict_structure(self) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50).run()
        d = report.to_dict()
        assert "name" in d
        assert "overall" in d
        assert "hard_nogo" in d
        assert "categories" in d
        for cat, tests in d["categories"].items():
            for t in tests:
                assert "name" in t
                assert "status" in t
                assert "value" in t

    def test_print_terminal_does_not_raise(self) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50).run()
        report.print_terminal()  # should not raise

    def test_write_markdown(self, tmp_path) -> None:
        report = GoNoGo(_fake_trades(), _fake_equity(), mc_sims=50).run()
        path = str(tmp_path / "report.md")
        report.write_markdown(path)
        content = open(path).read()
        assert "Go/No-Go" in content
        assert report.name in content


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_returns_three_results(self) -> None:
        trades = _fake_trades()
        equity = _fake_equity()
        calendar_days = (equity.index[-1] - equity.index[0]).days
        results, _ = run_monte_carlo(trades, equity, 100_000, calendar_days, n_sims=50)
        assert len(results) == 3

    def test_empty_trades_returns_fail(self) -> None:
        equity = _fake_equity()
        calendar_days = (equity.index[-1] - equity.index[0]).days
        results, _ = run_monte_carlo([], equity, 100_000, calendar_days, n_sims=50)
        assert all(t.status == "FAIL" for t in results)

    def test_portfolio_pnl_pcts_length(self) -> None:
        trades = _fake_trades(50)
        equity = _fake_equity()
        pcts = _portfolio_pnl_pcts(trades, equity)
        assert len(pcts) == 50

    def test_custom_mc_thresholds(self) -> None:
        trades = _fake_trades()
        equity = _fake_equity()
        calendar_days = (equity.index[-1] - equity.index[0]).days
        t = MonteCarloThresholds(max_reshuffle_dd_pass=1.0, max_reshuffle_dd_warn=2.0)
        results, _ = run_monte_carlo(
            trades, equity, 100_000, calendar_days, n_sims=50, thresholds=t
        )
        # With very tight DD threshold, reshuffle likely FAIL
        reshuffle = next(r for r in results if "Reshuffle" in r.name)
        assert reshuffle.status in ("WARN", "FAIL")


# ---------------------------------------------------------------------------
# Execution Realism
# ---------------------------------------------------------------------------

class TestExecutionRealism:
    def test_returns_two_results(self) -> None:
        trades = _fake_trades()
        equity = _fake_equity()
        calendar_days = (equity.index[-1] - equity.index[0]).days
        results = run_execution_realism(trades, equity, 100_000, calendar_days)
        assert len(results) == 2
        names = {r.name for r in results}
        assert "Slippage ladder" in names
        assert "Commission ladder" in names

    def test_empty_trades_returns_fail(self) -> None:
        equity = _fake_equity()
        calendar_days = (equity.index[-1] - equity.index[0]).days
        results = run_execution_realism([], equity, 100_000, calendar_days)
        assert all(r.status == "FAIL" for r in results)


# ---------------------------------------------------------------------------
# Regime Rolling
# ---------------------------------------------------------------------------

class TestRegimeRolling:
    def test_short_equity_returns_warn(self) -> None:
        equity = _fake_equity(n=100)
        results = run_regime_rolling(equity)
        assert len(results) == 1
        assert results[0].status == "WARN"

    def test_full_equity_returns_multiple_tests(self) -> None:
        equity = _fake_equity(n=500)
        results = run_regime_rolling(equity)
        assert len(results) >= 3  # rolling Sharpe, annual consistency, alpha/beta, Sharpe

    def test_with_benchmark(self) -> None:
        equity = _fake_equity(n=500)
        bench = pd.Series(
            np.random.default_rng(0).normal(0.0003, 0.01, 500),
            index=equity.index,
        )
        results = run_regime_rolling(equity, benchmark_returns=bench)
        names = {r.name for r in results}
        assert "Alpha vs benchmark" in names
        assert "Beta vs benchmark" in names


# ---------------------------------------------------------------------------
# Annual Breakdown
# ---------------------------------------------------------------------------

class TestAnnualBreakdown:
    def test_returns_one_result(self) -> None:
        equity = _fake_equity()
        results = run_annual_breakdown(equity)
        assert len(results) == 1
        assert results[0].name == "Annual returns"

    def test_value_contains_year_data(self) -> None:
        equity = _fake_equity()
        result = run_annual_breakdown(equity)[0]
        assert "2020" in result.value

    def test_with_benchmark_adds_bear_flag(self) -> None:
        dates = pd.date_range("2015-01-01", "2025-12-31", freq="B")
        equity = pd.Series(
            100_000 * np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, len(dates))),
            index=dates,
        )
        # Make a year with -15% benchmark return
        bench_returns = np.random.default_rng(1).normal(0.0003, 0.01, len(dates))
        # Force 2020 to be bear
        mask_2020 = dates.year == 2020
        bench_returns[mask_2020] = -0.001
        bench = pd.Series(bench_returns, index=dates)
        result = run_annual_breakdown(equity, benchmark_returns=bench)[0]
        assert "BEAR YEAR" in result.value


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_thresholds_reasonable(self) -> None:
        t = GoNoGoThresholds()
        assert t.statistical.min_trades_pass == 200
        assert t.risk.max_drawdown_hard_nogo == 50.0
        assert t.monte_carlo.max_reshuffle_dd_pass == 30.0

    def test_thresholds_are_mutable(self) -> None:
        t = GoNoGoThresholds()
        t.statistical.min_trades_pass = 100
        assert t.statistical.min_trades_pass == 100

    def test_custom_risk_thresholds(self) -> None:
        rt = RiskThresholds(max_drawdown_pass=15.0)
        assert rt.max_drawdown_pass == 15.0
