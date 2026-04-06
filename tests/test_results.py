"""Tests for BacktestResult and compute_stats."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from vbacktest.results import (
    BacktestResult,
    BacktestStats,
    compute_stats,
    print_report,
    save_report,
)
from vbacktest.portfolio import Trade


# ---------------------------------------------------------------------------
# Config + Portfolio stubs
# ---------------------------------------------------------------------------

@dataclass
class _PosSize:
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 20.0


@dataclass
class _PortCfg:
    initial_capital: float = 100_000.0
    max_positions: int = 10


@dataclass
class _ExecCfg:
    commission_pct: float = 0.1
    slippage_pct: float = 0.05


@dataclass
class _DataCfg:
    start_date: str | None = "2020-01-01"
    end_date: str | None = "2021-12-31"


@dataclass
class _Config:
    portfolio: _PortCfg = field(default_factory=_PortCfg)
    execution: _ExecCfg = field(default_factory=_ExecCfg)
    position_sizing: _PosSize = field(default_factory=_PosSize)
    data: _DataCfg = field(default_factory=_DataCfg)


class _Portfolio:
    def __init__(
        self,
        capital: float = 100_000.0,
        trades: list[Trade] | None = None,
        equity_history: list[tuple[pd.Timestamp, float]] | None = None,
    ) -> None:
        self.trades = trades or []
        self.equity_history = equity_history or []


def _make_equity_history(
    n: int = 252,
    start: float = 100_000.0,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 42,
) -> list[tuple[pd.Timestamp, float]]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    equity = start
    history = [(dates[0], equity)]
    for i in range(1, n):
        equity *= 1 + rng.normal(drift, vol)
        history.append((dates[i], equity))
    return history


def _make_trades(n: int = 20, win_rate: float = 0.6) -> list[Trade]:
    trades = []
    dates = pd.date_range("2020-01-10", periods=n * 2, freq="5B")
    for i in range(n):
        is_win = i < int(n * win_rate)
        pnl = 200.0 if is_win else -100.0
        trades.append(Trade(
            symbol=f"SYM{i % 5}",
            entry_date=dates[i * 2],
            entry_price=100.0,
            exit_date=dates[i * 2 + 1],
            exit_price=102.0 if is_win else 98.0,
            shares=10,
            pnl=pnl,
            pnl_pct=pnl / 1000.0 * 100,
            holding_days=5,
            exit_reason="stop_loss" if not is_win else "take_profit",
            commission=5.0,
        ))
    return trades


# ---------------------------------------------------------------------------
# BacktestStats
# ---------------------------------------------------------------------------

class TestBacktestStats:
    def test_defaults_zero(self) -> None:
        s = BacktestStats()
        assert s.total_return_pct == 0.0
        assert s.total_trades == 0

    def test_fields_assignable(self) -> None:
        s = BacktestStats(total_return_pct=50.0, cagr=20.0)
        assert s.total_return_pct == 50.0


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def _make_result(self) -> BacktestResult:
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        equity = pd.DataFrame({"date": dates, "equity": 100_000 + np.arange(10) * 100})
        return BacktestResult(
            equity_curve=equity,
            trades=[],
            trade_df=pd.DataFrame(),
            stats=BacktestStats(total_return_pct=10.0),
            monthly_returns=pd.DataFrame(),
            yearly_returns=pd.DataFrame(),
            config_snapshot={"initial_capital": 100_000},
        )

    def test_schema_version(self) -> None:
        r = BacktestResult.empty()
        assert r.schema_version == "1.0"

    def test_empty(self) -> None:
        r = BacktestResult.empty()
        assert r.equity_curve.empty
        assert r.trades == []
        assert r.stats.total_trades == 0

    def test_equity_series(self) -> None:
        r = self._make_result()
        s = r.equity_series()
        assert isinstance(s, pd.Series)
        assert len(s) == 10
        assert s.index.name == "date"

    def test_equity_series_empty(self) -> None:
        r = BacktestResult.empty()
        s = r.equity_series()
        assert len(s) == 0

    def test_to_json_roundtrip(self) -> None:
        r = self._make_result()
        js = r.to_json()
        assert isinstance(js, str)
        payload = json.loads(js)
        assert payload["schema_version"] == "1.0"
        assert "equity_curve" in payload
        assert "stats" in payload

    def test_from_json_roundtrip(self) -> None:
        r = self._make_result()
        js = r.to_json()
        r2 = BacktestResult.from_json(js)
        assert r2.stats.total_return_pct == pytest.approx(10.0)
        assert not r2.equity_curve.empty

    def test_from_json_migration_old_config_key(self) -> None:
        """Pre-1.0 JSON used 'config' instead of 'config_snapshot'."""
        payload = {
            "schema_version": "0.9",
            "config": {"initial_capital": 50_000},  # old key
            "stats": {},
            "equity_curve": [],
            "trades": [],
        }
        r = BacktestResult.from_json(payload)
        assert r.config_snapshot == {"initial_capital": 50_000}

    def test_from_json_dates_parsed(self) -> None:
        r = self._make_result()
        r2 = BacktestResult.from_json(r.to_json())
        assert pd.api.types.is_datetime64_any_dtype(r2.equity_curve["date"])


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

class TestComputeStats:
    def _run(
        self,
        equity_history: list[tuple[pd.Timestamp, float]] | None = None,
        trades: list[Trade] | None = None,
    ) -> BacktestResult:
        p = _Portfolio(
            equity_history=equity_history or _make_equity_history(),
            trades=trades or _make_trades(),
        )
        return compute_stats(p, _Config())

    def test_returns_backtest_result(self) -> None:
        assert isinstance(self._run(), BacktestResult)

    def test_total_return_positive_on_growing_equity(self) -> None:
        hist = _make_equity_history(drift=0.002)
        result = self._run(equity_history=hist)
        assert result.stats.total_return_pct > 0

    def test_sharpe_nonzero(self) -> None:
        result = self._run()
        assert result.stats.sharpe_ratio != 0.0

    def test_max_drawdown_positive(self) -> None:
        result = self._run()
        assert result.stats.max_drawdown_pct >= 0

    def test_max_drawdown_bounded(self) -> None:
        result = self._run()
        assert result.stats.max_drawdown_pct <= 100.0

    def test_win_rate_bounded(self) -> None:
        result = self._run()
        assert 0.0 <= result.stats.win_rate <= 100.0

    def test_win_rate_matches_trade_data(self) -> None:
        trades = _make_trades(20, win_rate=0.6)
        result = self._run(trades=trades)
        assert result.stats.win_rate == pytest.approx(60.0)

    def test_trade_count(self) -> None:
        trades = _make_trades(15)
        result = self._run(trades=trades)
        assert result.stats.total_trades == 15

    def test_profit_factor_gt_one_for_positive_strategy(self) -> None:
        trades = _make_trades(20, win_rate=0.8)
        result = self._run(trades=trades)
        assert result.stats.profit_factor > 1.0

    def test_calmar_ratio(self) -> None:
        hist = _make_equity_history(drift=0.002, vol=0.005)
        result = self._run(equity_history=hist)
        if result.stats.max_drawdown_pct > 0:
            expected = result.stats.cagr / result.stats.max_drawdown_pct
            assert result.stats.calmar_ratio == pytest.approx(expected, rel=1e-4)

    def test_empty_portfolio(self) -> None:
        p = _Portfolio(equity_history=[], trades=[])
        result = compute_stats(p, _Config())
        assert result.stats.total_trades == 0
        assert result.stats.total_return_pct == 0.0

    def test_config_snapshot_captured(self) -> None:
        result = self._run()
        assert result.config_snapshot["initial_capital"] == 100_000.0

    def test_monthly_returns_populated(self) -> None:
        result = self._run()
        assert not result.monthly_returns.empty
        assert "month" in result.monthly_returns.columns

    def test_yearly_returns_populated(self) -> None:
        result = self._run()
        assert not result.yearly_returns.empty

    def test_equity_curve_has_date_and_equity(self) -> None:
        result = self._run()
        assert "date" in result.equity_curve.columns
        assert "equity" in result.equity_curve.columns


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------

class TestBenchmarkComparison:
    def test_benchmark_comparison_populated(self) -> None:
        hist = _make_equity_history()
        p = _Portfolio(equity_history=hist, trades=_make_trades())
        # Simple uptrend as benchmark
        n = len(hist)
        dates = pd.date_range("2020-01-03", periods=n - 1, freq="B")
        bench = pd.Series(0.001, index=dates)
        result = compute_stats(p, _Config(), benchmark_returns=bench)
        assert "alpha" in result.benchmark_comparison
        assert "beta" in result.benchmark_comparison

    def test_no_benchmark_empty_comparison(self) -> None:
        p = _Portfolio(equity_history=_make_equity_history())
        result = compute_stats(p, _Config(), benchmark_returns=None)
        assert result.benchmark_comparison == {}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

class TestPrintReport:
    def test_runs_without_error(self, capsys: pytest.CaptureFixture) -> None:
        r = BacktestResult.empty()
        print_report(r)
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS" in captured.out


class TestSaveReport:
    def test_creates_files(self) -> None:
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        equity = pd.DataFrame({"date": dates, "equity": [100_000.0] * 5})
        r = BacktestResult(
            equity_curve=equity,
            trades=[],
            trade_df=pd.DataFrame(),
            stats=BacktestStats(),
            monthly_returns=pd.DataFrame(),
            yearly_returns=pd.DataFrame(),
            config_snapshot={"initial_capital": 100_000},
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_report(r, tmp)
            assert (Path(tmp) / "equity_curve.csv").exists()
            assert (Path(tmp) / "summary.json").exists()
