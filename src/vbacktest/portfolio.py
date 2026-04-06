"""Portfolio management for backtesting."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from vbacktest.strategy import ExitRule, ExitSignal, Signal
    from vbacktest.config import ExecutionConfig, PortfolioConfig


@dataclass
class Position:
    """Open position in the portfolio."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    original_shares: int
    stop_price: float
    cost_basis: float  # Total cost including commission
    exit_rules: list[ExitRule]
    entry_idx: int = 0
    marked_for_exit: ExitSignal | None = None
    _current_price: float = 0.0

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {self.entry_price}")
        if self.stop_price >= self.entry_price:
            raise ValueError(
                f"stop_price must be < entry_price, got stop={self.stop_price} >= entry={self.entry_price}"
            )
        if self.shares <= 0:
            raise ValueError(f"shares must be > 0, got {self.shares}")
        if self.original_shares <= 0:
            raise ValueError(f"original_shares must be > 0, got {self.original_shares}")

    def update_current_price(self, price: float) -> None:
        """Update current price for position value calculation.

        Raises:
            ValueError: If price <= 0 or is not finite.
        """
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        if not math.isfinite(price):
            raise ValueError(f"price must be finite, got {price}")
        self._current_price = price

    def shares_for_fraction(self, fraction: float) -> int:
        """Calculate shares for a fractional exit (rounds down)."""
        return math.floor(self.shares * fraction)

    @property
    def position_value(self) -> float:
        """Current market value of position."""
        return self.shares * self._current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss in currency units."""
        return self.position_value - self.cost_basis * (self.shares / self.original_shares)

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss as percentage of cost basis."""
        cost_basis_current = self.cost_basis * (self.shares / self.original_shares)
        if cost_basis_current == 0:
            return 0.0
        return (self.unrealized_pnl / cost_basis_current) * 100


@dataclass
class Order:
    """Pending order awaiting execution."""

    symbol: str
    shares: int
    is_entry: bool
    signal: Signal | None = None
    exit_signal: ExitSignal | None = None
    created_date: pd.Timestamp | None = None


@dataclass
class Trade:
    """Completed (closed) trade record."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    holding_days: int
    exit_reason: str
    commission: float  # Total commission paid (entry + exit)


class Portfolio:
    """Portfolio manager handling positions, orders, and cash."""

    def __init__(self, config: PortfolioConfig, exec_config: ExecutionConfig) -> None:
        self.config = config
        self.exec_config = exec_config

        self.cash: float = config.initial_capital
        self.positions: dict[str, Position] = {}
        self.pending_orders: list[Order] = []
        self.trades: list[Trade] = []
        self.equity_history: list[tuple[pd.Timestamp, float]] = []

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_shares(
        self,
        entry_price: float,
        stop_price: float,
        risk_pct: float,
        max_position_pct: float,
    ) -> int:
        """Calculate position size using risk-based sizing with safety caps.

        Returns 0 if parameters are invalid or insufficient cash.
        """
        if stop_price >= entry_price or entry_price <= 0:
            return 0

        equity = self.equity
        stop_distance = entry_price - stop_price

        shares_by_risk = math.floor(equity * (risk_pct / 100) / stop_distance)
        shares_by_max_pos = math.floor(equity * (max_position_pct / 100) / entry_price)
        shares_by_cash = math.floor(self.cash / entry_price)

        return max(0, min(shares_by_risk, shares_by_max_pos, shares_by_cash))

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_entries(
        self,
        date: pd.Timestamp,
        prices: dict[str, dict[str, float]],
        universe_idx: dict[str, int],
    ) -> None:
        """Execute pending entry orders at open price with slippage."""
        executed: list[Order] = []

        for order in self.pending_orders:
            if not order.is_entry:
                continue

            symbol = order.symbol
            if symbol not in prices or order.signal is None:
                continue

            open_price = prices[symbol]["open"]
            slipped_price = open_price * (1 + self.exec_config.slippage_pct / 100)

            cost = order.shares * slipped_price
            commission = cost * (self.exec_config.commission_pct / 100)
            total_cost = cost + commission

            if total_cost > self.cash:
                continue

            self.cash -= total_cost

            exit_rules = order.signal.metadata.get("exit_rules", [])

            # Re-anchor stop to actual fill price when ATR info is present
            atr = order.signal.metadata.get("atr")
            atr_mult = order.signal.metadata.get("atr_stop_multiplier")
            if atr and atr > 0 and atr_mult:
                adjusted_stop = slipped_price - (atr_mult * atr)
            else:
                adjusted_stop = order.signal.stop_price

            # Apply hard stop cap if configured
            max_stop_pct = order.signal.metadata.get("max_stop_pct", 0)
            if max_stop_pct > 0:
                adjusted_stop = max(adjusted_stop, slipped_price * (1.0 - max_stop_pct / 100.0))

            # Ensure stop is always < fill price
            if adjusted_stop >= slipped_price:
                adjusted_stop = slipped_price - (2.0 * atr) if (atr and atr > 0) else slipped_price * 0.95

            position = Position(
                symbol=symbol,
                entry_date=date,
                entry_price=slipped_price,
                shares=order.shares,
                original_shares=order.shares,
                stop_price=adjusted_stop,
                cost_basis=total_cost,
                exit_rules=exit_rules,
                entry_idx=universe_idx.get(symbol, 0),
                _current_price=slipped_price,
            )

            self.positions[symbol] = position
            executed.append(order)

        self.pending_orders = [o for o in self.pending_orders if o not in executed]

    def execute_exits(
        self,
        date: pd.Timestamp,
        prices: dict[str, dict[str, float]],
    ) -> None:
        """Execute pending exit orders at open price with slippage."""
        executed: list[Order] = []

        for order in self.pending_orders:
            if order.is_entry:
                continue

            symbol = order.symbol
            if symbol not in self.positions or symbol not in prices:
                continue

            position = self.positions[symbol]

            open_price = prices[symbol]["open"]
            slipped_price = open_price * (1 - self.exec_config.slippage_pct / 100)

            actual_shares = min(order.shares, position.shares)
            proceeds = actual_shares * slipped_price
            commission = proceeds * (self.exec_config.commission_pct / 100)
            net_proceeds = proceeds - commission

            self.cash += net_proceeds

            entry_commission = position.cost_basis - (position.original_shares * position.entry_price)
            exit_commission = commission
            total_commission = (
                entry_commission * actual_shares / position.original_shares
            ) + exit_commission

            cost_for_shares = (position.cost_basis / position.original_shares) * actual_shares
            pnl = net_proceeds - cost_for_shares
            pnl_pct = (pnl / cost_for_shares) * 100 if cost_for_shares > 0 else 0.0

            self.trades.append(Trade(
                symbol=symbol,
                entry_date=position.entry_date,
                entry_price=position.entry_price,
                exit_date=date,
                exit_price=slipped_price,
                shares=actual_shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                holding_days=(date - position.entry_date).days,
                exit_reason=order.exit_signal.reason if order.exit_signal else "unknown",
                commission=total_commission,
            ))

            position.shares -= actual_shares
            if position.shares <= 0:
                del self.positions[symbol]

            executed.append(order)

        self.pending_orders = [o for o in self.pending_orders if o not in executed]

    def mark_exits(
        self,
        date: pd.Timestamp,
        universe: dict[str, pd.DataFrame],
        universe_idx: dict[str, int],
        current_prices: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Evaluate exit rules and mark positions for exit.

        Optimised: inlines stop-loss check via current_prices dict to avoid
        a full iloc scan when only the stop is triggered.
        """
        from vbacktest.strategy import ExitSignal, ExitCondition

        for symbol, position in self.positions.items():
            if symbol not in universe or symbol not in universe_idx:
                continue

            # Fast path: inline stop-loss check
            if current_prices and symbol in current_prices:
                bar_close = current_prices[symbol].get("close")
                if bar_close is not None and bar_close <= position.stop_price:
                    position.marked_for_exit = ExitSignal(
                        condition=ExitCondition.STOP_LOSS,
                        fraction=1.0,
                        reason=f"Stop loss hit: close {bar_close:.2f} <= stop {position.stop_price:.2f}",
                    )
                    stock_data = universe[symbol]
                    bar_idx = universe_idx[symbol]
                    bar = current_prices[symbol]
                    for rule in position.exit_rules:
                        rule.update(position, bar, bar_idx, stock_data)
                    continue

            stock_data = universe[symbol]
            bar_idx = universe_idx[symbol]
            bar = (
                current_prices[symbol]
                if (current_prices and symbol in current_prices)
                else stock_data.iloc[bar_idx]
            )

            exit_signals = []
            for rule in position.exit_rules:
                signal = rule.check(position, bar, bar_idx, stock_data)
                if signal:
                    exit_signals.append(signal)

            for rule in position.exit_rules:
                rule.update(position, bar, bar_idx, stock_data)

            if exit_signals:
                full_exits = [s for s in exit_signals if s.fraction == 1.0]
                if full_exits:
                    position.marked_for_exit = full_exits[0]
                else:
                    position.marked_for_exit = max(exit_signals, key=lambda s: s.fraction)

        # Convert marked positions to pending exit orders
        for symbol, position in list(self.positions.items()):
            if position.marked_for_exit:
                shares_to_exit = position.shares_for_fraction(position.marked_for_exit.fraction)
                if shares_to_exit > 0:
                    self.pending_orders.append(Order(
                        symbol=symbol,
                        shares=shares_to_exit,
                        is_entry=False,
                        exit_signal=position.marked_for_exit,
                        created_date=date,
                    ))
                position.marked_for_exit = None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def record_equity(self, date: pd.Timestamp) -> None:
        """Append current equity to the equity history."""
        self.equity_history.append((date, self.equity))

    @property
    def equity(self) -> float:
        """Total equity: cash + sum of position market values."""
        return self.cash + sum(pos.position_value for pos in self.positions.values())

    @property
    def positions_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        """Return True if a position for *symbol* is open."""
        return symbol in self.positions

    def available_slots(self) -> int:
        """Number of additional position slots available under max_positions."""
        return max(0, self.config.max_positions - self.positions_count)
