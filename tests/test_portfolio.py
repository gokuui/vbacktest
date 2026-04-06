"""Tests for portfolio management."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from vbacktest.portfolio import Portfolio, Position, Order, Trade
from vbacktest.strategy import ExitCondition, ExitSignal


# ---------------------------------------------------------------------------
# Config stubs
# ---------------------------------------------------------------------------

@dataclass
class _PortfolioCfg:
    initial_capital: float = 100_000.0
    max_positions: int = 10
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 20.0


@dataclass
class _ExecCfg:
    commission_pct: float = 0.1
    slippage_pct: float = 0.05


def _portfolio(
    capital: float = 100_000.0,
    max_positions: int = 10,
    commission_pct: float = 0.1,
    slippage_pct: float = 0.05,
) -> Portfolio:
    return Portfolio(_PortfolioCfg(initial_capital=capital, max_positions=max_positions),
                     _ExecCfg(commission_pct=commission_pct, slippage_pct=slippage_pct))


def _signal(
    symbol: str = "AAPL",
    stop_price: float = 90.0,
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    sig = MagicMock()
    sig.symbol = symbol
    sig.stop_price = stop_price
    sig.metadata = metadata or {}
    return sig


def _exit_signal(fraction: float = 1.0) -> ExitSignal:
    return ExitSignal(
        condition=ExitCondition.STOP_LOSS,
        fraction=fraction,
        reason="test",
    )


def _prices(symbol: str, open_: float, close: float) -> dict[str, dict[str, float]]:
    return {symbol: {"open": open_, "close": close, "high": close * 1.01, "low": open_ * 0.99}}


def _date(s: str = "2020-01-02") -> pd.Timestamp:
    return pd.Timestamp(s)


# ---------------------------------------------------------------------------
# Position tests
# ---------------------------------------------------------------------------

class TestPosition:
    def _make(self, **kw: Any) -> Position:
        defaults: dict[str, Any] = dict(
            symbol="AAPL", entry_date=_date(), entry_price=100.0,
            shares=10, original_shares=10, stop_price=90.0,
            cost_basis=1001.0, exit_rules=[], _current_price=100.0,
        )
        defaults.update(kw)
        return Position(**defaults)

    def test_basic_creation(self) -> None:
        pos = self._make()
        assert pos.symbol == "AAPL"
        assert pos.shares == 10

    def test_invalid_entry_price(self) -> None:
        with pytest.raises(ValueError, match="entry_price"):
            self._make(entry_price=-1.0, stop_price=-5.0)

    def test_stop_gte_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_price"):
            self._make(entry_price=100.0, stop_price=100.0)

    def test_invalid_shares(self) -> None:
        with pytest.raises(ValueError, match="shares"):
            self._make(shares=0)

    def test_update_current_price_valid(self) -> None:
        pos = self._make()
        pos.update_current_price(120.0)
        assert pos._current_price == 120.0

    def test_update_current_price_invalid(self) -> None:
        pos = self._make()
        with pytest.raises(ValueError):
            pos.update_current_price(-1.0)
        with pytest.raises(ValueError):
            pos.update_current_price(float("inf"))

    def test_position_value(self) -> None:
        pos = self._make(shares=10, _current_price=110.0)
        assert pos.position_value == 1100.0

    def test_shares_for_fraction(self) -> None:
        pos = self._make(shares=10)
        assert pos.shares_for_fraction(0.5) == 5
        assert pos.shares_for_fraction(1.0) == 10
        assert pos.shares_for_fraction(0.3) == 3  # floor

    def test_unrealized_pnl(self) -> None:
        pos = self._make(
            shares=10, original_shares=10,
            entry_price=100.0, cost_basis=1001.0,
            _current_price=110.0,
        )
        assert pos.unrealized_pnl == pytest.approx(99.0)  # 1100 - 1001


# ---------------------------------------------------------------------------
# Portfolio — equity and cash
# ---------------------------------------------------------------------------

class TestPortfolioEquity:
    def test_initial_equity_equals_capital(self) -> None:
        p = _portfolio(capital=50_000.0)
        assert p.equity == 50_000.0
        assert p.cash == 50_000.0

    def test_equity_includes_positions(self) -> None:
        p = _portfolio(capital=100_000.0)
        pos = Position(
            symbol="AAPL", entry_date=_date(), entry_price=100.0,
            shares=10, original_shares=10, stop_price=90.0,
            cost_basis=1010.0, exit_rules=[], _current_price=120.0,
        )
        p.positions["AAPL"] = pos
        p.cash = 90_000.0
        assert p.equity == pytest.approx(90_000.0 + 1200.0)

    def test_record_equity(self) -> None:
        p = _portfolio()
        p.record_equity(_date())
        assert len(p.equity_history) == 1
        assert p.equity_history[0][1] == pytest.approx(100_000.0)


# ---------------------------------------------------------------------------
# Portfolio — calculate_shares
# ---------------------------------------------------------------------------

class TestCalculateShares:
    def test_basic_sizing(self) -> None:
        p = _portfolio(capital=100_000.0)
        # risk=1% → $1000 risk, stop_distance=10 → 100 shares by risk
        shares = p.calculate_shares(entry_price=100.0, stop_price=90.0,
                                     risk_pct=1.0, max_position_pct=20.0)
        assert shares == 100

    def test_capped_by_max_position(self) -> None:
        p = _portfolio(capital=100_000.0)
        # max_position=5% → $5000 → 50 shares at $100
        shares = p.calculate_shares(entry_price=100.0, stop_price=99.0,
                                     risk_pct=10.0, max_position_pct=5.0)
        assert shares == 50

    def test_capped_by_cash(self) -> None:
        p = _portfolio(capital=1_000.0)  # very small capital
        shares = p.calculate_shares(entry_price=100.0, stop_price=90.0,
                                     risk_pct=1.0, max_position_pct=20.0)
        assert shares <= 10  # can't buy more than cash allows

    def test_invalid_stop_returns_zero(self) -> None:
        p = _portfolio()
        assert p.calculate_shares(100.0, 100.0, 1.0, 20.0) == 0  # stop == entry
        assert p.calculate_shares(100.0, 110.0, 1.0, 20.0) == 0  # stop > entry

    def test_returns_int(self) -> None:
        p = _portfolio()
        result = p.calculate_shares(100.0, 90.0, 1.0, 20.0)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Portfolio — execute_entries
# ---------------------------------------------------------------------------

class TestExecuteEntries:
    def test_entry_creates_position(self) -> None:
        p = _portfolio(capital=100_000.0)
        sig = _signal("AAPL", stop_price=90.0, metadata={"exit_rules": []})
        order = Order(symbol="AAPL", shares=10, is_entry=True, signal=sig)
        p.pending_orders.append(order)

        p.execute_entries(_date(), _prices("AAPL", open_=100.0, close=100.0), {"AAPL": 5})

        assert "AAPL" in p.positions
        assert p.positions["AAPL"].shares == 10

    def test_entry_deducts_cash(self) -> None:
        p = _portfolio(capital=100_000.0)
        sig = _signal("AAPL", stop_price=90.0, metadata={"exit_rules": []})
        order = Order(symbol="AAPL", shares=10, is_entry=True, signal=sig)
        p.pending_orders.append(order)

        p.execute_entries(_date(), _prices("AAPL", open_=100.0, close=100.0), {"AAPL": 5})

        assert p.cash < 100_000.0

    def test_entry_skipped_if_no_price(self) -> None:
        p = _portfolio()
        sig = _signal("AAPL", stop_price=90.0, metadata={"exit_rules": []})
        order = Order(symbol="AAPL", shares=10, is_entry=True, signal=sig)
        p.pending_orders.append(order)

        p.execute_entries(_date(), {}, {})  # empty prices

        assert "AAPL" not in p.positions

    def test_entry_skipped_if_insufficient_cash(self) -> None:
        p = _portfolio(capital=100.0)  # only $100
        sig = _signal("AAPL", stop_price=90.0, metadata={"exit_rules": []})
        order = Order(symbol="AAPL", shares=10, is_entry=True, signal=sig)  # $1000+
        p.pending_orders.append(order)

        p.execute_entries(_date(), _prices("AAPL", open_=100.0, close=100.0), {})

        assert "AAPL" not in p.positions

    def test_slippage_applied_to_fill_price(self) -> None:
        p = _portfolio(capital=100_000.0)
        sig = _signal("AAPL", stop_price=90.0, metadata={"exit_rules": []})
        order = Order(symbol="AAPL", shares=1, is_entry=True, signal=sig)
        p.pending_orders.append(order)

        p.execute_entries(_date(), _prices("AAPL", open_=100.0, close=100.0), {})

        pos = p.positions["AAPL"]
        assert pos.entry_price > 100.0  # slippage adds to price


# ---------------------------------------------------------------------------
# Portfolio — execute_exits
# ---------------------------------------------------------------------------

class TestExecuteExits:
    def _with_position(self, entry_price: float = 100.0) -> Portfolio:
        p = _portfolio(capital=100_000.0)
        pos = Position(
            symbol="AAPL", entry_date=_date("2020-01-01"),
            entry_price=entry_price, shares=10, original_shares=10,
            stop_price=90.0, cost_basis=entry_price * 10 * 1.001,
            exit_rules=[], _current_price=entry_price,
        )
        p.positions["AAPL"] = pos
        p.cash = 100_000.0 - pos.cost_basis
        return p

    def test_exit_creates_trade(self) -> None:
        p = self._with_position(100.0)
        order = Order(symbol="AAPL", shares=10, is_entry=False, exit_signal=_exit_signal())
        p.pending_orders.append(order)

        p.execute_exits(_date(), _prices("AAPL", open_=110.0, close=110.0))

        assert len(p.trades) == 1
        assert "AAPL" not in p.positions

    def test_exit_adds_cash(self) -> None:
        p = self._with_position(100.0)
        cash_before = p.cash
        order = Order(symbol="AAPL", shares=10, is_entry=False, exit_signal=_exit_signal())
        p.pending_orders.append(order)

        p.execute_exits(_date(), _prices("AAPL", open_=110.0, close=110.0))

        assert p.cash > cash_before

    def test_partial_exit_leaves_position(self) -> None:
        p = self._with_position(100.0)
        order = Order(symbol="AAPL", shares=5, is_entry=False, exit_signal=_exit_signal(0.5))
        p.pending_orders.append(order)

        p.execute_exits(_date(), _prices("AAPL", open_=100.0, close=100.0))

        assert "AAPL" in p.positions
        assert p.positions["AAPL"].shares == 5

    def test_exit_pnl_positive_on_gain(self) -> None:
        p = self._with_position(100.0)
        order = Order(symbol="AAPL", shares=10, is_entry=False, exit_signal=_exit_signal())
        p.pending_orders.append(order)

        p.execute_exits(_date(), _prices("AAPL", open_=120.0, close=120.0))

        assert p.trades[0].pnl > 0

    def test_exit_pnl_negative_on_loss(self) -> None:
        p = self._with_position(100.0)
        order = Order(symbol="AAPL", shares=10, is_entry=False, exit_signal=_exit_signal())
        p.pending_orders.append(order)

        p.execute_exits(_date(), _prices("AAPL", open_=80.0, close=80.0))

        assert p.trades[0].pnl < 0


# ---------------------------------------------------------------------------
# Portfolio — mark_exits
# ---------------------------------------------------------------------------

class TestMarkExits:
    def _portfolio_with_position(self) -> tuple[Portfolio, pd.DataFrame]:
        p = _portfolio()
        df = pd.DataFrame({
            "close": [100.0, 95.0, 80.0],
            "atr": [2.0, 2.0, 2.0],
            "sma_10": [102.0, 100.0, 98.0],
        })
        pos = Position(
            symbol="AAPL", entry_date=_date("2020-01-01"),
            entry_price=100.0, shares=10, original_shares=10,
            stop_price=90.0, cost_basis=1001.0, exit_rules=[],
            _current_price=95.0,
        )
        p.positions["AAPL"] = pos
        return p, df

    def test_stop_loss_marked_via_current_prices(self) -> None:
        p, df = self._portfolio_with_position()
        universe = {"AAPL": df}
        universe_idx = {"AAPL": 2}
        current_prices = {"AAPL": {"close": 80.0, "open": 81.0}}  # below stop 90

        p.mark_exits(_date(), universe, universe_idx, current_prices)

        # Should have created a pending exit order
        assert any(o.symbol == "AAPL" for o in p.pending_orders)

    def test_no_exit_above_stop(self) -> None:
        p, df = self._portfolio_with_position()
        universe = {"AAPL": df}
        universe_idx = {"AAPL": 1}
        current_prices = {"AAPL": {"close": 95.0, "open": 95.0}}  # above stop 90

        p.mark_exits(_date(), universe, universe_idx, current_prices)

        assert len(p.pending_orders) == 0

    def test_symbol_not_in_universe_skipped(self) -> None:
        p, df = self._portfolio_with_position()
        p.mark_exits(_date(), {}, {})  # empty universe
        assert len(p.pending_orders) == 0


# ---------------------------------------------------------------------------
# Portfolio — available_slots, has_position
# ---------------------------------------------------------------------------

class TestPortfolioSlots:
    def test_initial_slots(self) -> None:
        p = _portfolio(max_positions=5)
        assert p.available_slots() == 5

    def test_slots_decrease_with_positions(self) -> None:
        p = _portfolio(max_positions=5)
        pos = Position(
            symbol="AAPL", entry_date=_date(), entry_price=100.0,
            shares=10, original_shares=10, stop_price=90.0,
            cost_basis=1001.0, exit_rules=[], _current_price=100.0,
        )
        p.positions["AAPL"] = pos
        assert p.available_slots() == 4

    def test_has_position_true(self) -> None:
        p = _portfolio()
        pos = Position(
            symbol="MSFT", entry_date=_date(), entry_price=200.0,
            shares=5, original_shares=5, stop_price=180.0,
            cost_basis=1001.0, exit_rules=[], _current_price=200.0,
        )
        p.positions["MSFT"] = pos
        assert p.has_position("MSFT") is True

    def test_has_position_false(self) -> None:
        p = _portfolio()
        assert p.has_position("TSLA") is False
