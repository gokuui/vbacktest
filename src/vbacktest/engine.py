"""Backtest engine — day-by-day orchestration."""
from __future__ import annotations

import copy
import logging
import time
from typing import TYPE_CHECKING

import pandas as pd

from vbacktest.data_loader import (
    build_date_index,
    build_trading_calendar,
    convert_universe_to_dict_index,
    convert_universe_to_numpy_arrays,
    load_universe,
    precompute_indicators,
)
from vbacktest.portfolio import Order, Portfolio
from vbacktest.strategy import BarContext

if TYPE_CHECKING:
    from vbacktest.config import BacktestConfig
    from vbacktest.results import BacktestResult
    from vbacktest.strategy import Signal, Strategy

log = logging.getLogger(__name__)


class BacktestEngine:
    """Day-by-day backtest engine.

    Execution order each day:
    1. Execute pending exits at open.
    2. Execute pending entries at open.
    3. Expire unfilled entry orders (1-day signal validity).
    4. Update position prices to close.
    5. Evaluate exit rules → mark positions for exit.
    6. Call ``strategy.on_bar()`` → generate entry signals.
    7. Size + filter signals → create pending entry orders.
    8. Record equity.
    """

    def __init__(self, config: BacktestConfig, strategy: Strategy) -> None:
        self.config = config
        self.strategy = strategy
        self.portfolio = Portfolio(config.portfolio, config.execution)

    def run(self, benchmark_returns: pd.Series | None = None) -> BacktestResult:
        """Run the backtest and return results.

        Args:
            benchmark_returns: Optional daily returns series for benchmark comparison.

        Returns:
            ``BacktestResult`` with trades, equity curve, and statistics.
        """
        log.info("Starting backtest …")
        t0 = time.time()

        num_workers = self.config.performance.num_workers if self.config.performance.enable_parallel else 1

        # ── 1. Load universe ──────────────────────────────────────────────
        universe = load_universe(
            validated_dir=self.config.data.validated_dir,
            excluded_symbols=self.config.data.excluded_symbols,
            excluded_symbols_file=self.config.data.excluded_symbols_file,
            start_date=self.config.data.start_date,
            end_date=self.config.data.end_date,
            min_history_days=self.config.data.min_history_days,
            num_workers=num_workers,
        )

        if not universe:
            log.warning("No stocks loaded; returning empty result.")
            from vbacktest.results import BacktestResult
            return BacktestResult.empty()

        log.info("Loaded %d stocks in %.1fs", len(universe), time.time() - t0)

        # ── 2. Precompute indicators ───────────────────────────────────────
        indicator_specs = self.strategy.indicators()
        t1 = time.time()
        universe = precompute_indicators(universe, indicator_specs, num_workers=num_workers)
        log.info("Indicators precomputed in %.1fs", time.time() - t1)

        # ── 3. Convert to fast-access formats ─────────────────────────────
        universe_dict = convert_universe_to_dict_index(universe)
        universe_arrays = convert_universe_to_numpy_arrays(universe)

        # ── 4. Calendar + date index ───────────────────────────────────────
        calendar = build_trading_calendar(universe)
        date_index = build_date_index(universe, num_workers=num_workers)

        warmup = self._warmup_days(indicator_specs)
        log.info("Warmup: %d days. Calendar: %d days.", warmup, len(calendar))

        # ── 5. Day-by-day loop ────────────────────────────────────────────
        for day_idx, trading_day in enumerate(calendar):
            if day_idx < warmup:
                continue

            # Build per-day state
            current_prices: dict[str, dict[str, object]] = {}
            universe_idx: dict[str, int] = {}

            for symbol in universe_dict:
                if trading_day not in date_index.get(symbol, {}):
                    continue
                idx = date_index[symbol][trading_day]
                current_prices[symbol] = universe_dict[symbol][trading_day]
                universe_idx[symbol] = idx

            if not current_prices:
                continue

            # Execute exits at open
            self.portfolio.execute_exits(trading_day, current_prices)  # type: ignore[arg-type]

            # Execute entries at open
            self.portfolio.execute_entries(trading_day, current_prices, universe_idx)  # type: ignore[arg-type]

            # Expire unfilled entry orders (1-day signal lifetime)
            self.portfolio.pending_orders = [
                o for o in self.portfolio.pending_orders if not o.is_entry
            ]

            # Update position prices to close
            for symbol, position in self.portfolio.positions.items():
                if symbol in current_prices:
                    close = current_prices[symbol].get("close")
                    if close is not None:
                        position.update_current_price(float(close))

            # Evaluate exit rules
            self.portfolio.mark_exits(trading_day, universe, universe_idx, current_prices)  # type: ignore[arg-type]

            # Generate entry signals via BarContext
            ctx = BarContext(
                date=trading_day,
                universe=universe,
                universe_idx=universe_idx,
                portfolio=self.portfolio,
                current_prices=current_prices,  # type: ignore[arg-type]
                universe_arrays=universe_arrays,
            )
            signals = self._call_on_bar(ctx)

            # Size + filter → pending orders
            self._process_signals(signals, trading_day, current_prices, universe_arrays, universe_idx)

            # Record equity
            self.portfolio.record_equity(trading_day)

            if (day_idx + 1) % 500 == 0:
                log.info(
                    "Day %d/%d | positions=%d | trades=%d | equity=$%.0f",
                    day_idx + 1, len(calendar),
                    self.portfolio.positions_count,
                    len(self.portfolio.trades),
                    self.portfolio.equity,
                )

        # ── 6. Close remaining positions ──────────────────────────────────
        if calendar:
            self._close_remaining(calendar[-1], universe, date_index)

        # ── 7. Compute results ────────────────────────────────────────────
        from vbacktest.results import compute_stats
        result = compute_stats(self.portfolio, self.config, benchmark_returns=benchmark_returns)

        total = time.time() - t0
        log.info(
            "Backtest done in %.1fs | %d trades | final equity=$%.0f",
            total, len(self.portfolio.trades),
            result.equity_curve["equity"].iloc[-1] if not result.equity_curve.empty else 0,
        )
        return result

    # ------------------------------------------------------------------
    # Strategy compatibility
    # ------------------------------------------------------------------

    def _call_on_bar(self, ctx: BarContext) -> list[Signal]:
        """Call strategy.on_bar(), handling both BarContext and legacy signatures."""
        import inspect

        sig = inspect.signature(self.strategy.on_bar)
        params = list(sig.parameters.keys())

        # Legacy: on_bar(self, date, universe, universe_idx, portfolio, ...)
        if len(params) > 1 and params[1] != "ctx":
            import warnings
            warnings.warn(
                f"{type(self.strategy).__name__}.on_bar() uses the deprecated positional "
                "signature. Update to on_bar(self, ctx: BarContext) -> list[Signal].",
                DeprecationWarning,
                stacklevel=3,
            )
            return self.strategy.on_bar(  # type: ignore[call-arg]
                date=ctx.date,
                universe=ctx.universe,
                universe_idx=ctx.universe_idx,
                portfolio=ctx.portfolio,
                current_prices=ctx.current_prices,
                universe_arrays=ctx.universe_arrays,
            )

        return self.strategy.on_bar(ctx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warmup_days(self, indicator_specs: list) -> int:
        """Determine warmup period from indicator specs."""
        max_period = 0
        for spec in indicator_specs:
            max_period = max(max_period, spec.params.get("period", 0))
            if spec.name == "macd":
                slow = spec.params.get("slow", 26)
                signal = spec.params.get("signal", 9)
                max_period = max(max_period, slow + signal)
            elif spec.name == "stochastic":
                max_period = max(
                    max_period,
                    spec.params.get("k_period", 14) + spec.params.get("d_period", 3),
                )
        return max_period

    def _process_signals(
        self,
        signals: list[Signal],
        trading_day: pd.Timestamp,
        current_prices: dict,
        universe_arrays: dict | None,
        universe_idx: dict[str, int],
    ) -> None:
        """Size and filter signals, create pending entry orders."""
        if not signals:
            return

        pending_symbols = {o.symbol for o in self.portfolio.pending_orders if o.is_entry}

        filtered = [
            s for s in signals
            if not self.portfolio.has_position(s.symbol)
            and s.symbol not in pending_symbols
            and s.symbol in current_prices
        ]
        filtered.sort(key=lambda s: s.score, reverse=True)

        available = self.portfolio.available_slots()
        risk_pct = self.config.position_sizing.risk_per_trade_pct
        max_pos_pct = getattr(self.config.position_sizing, "max_position_pct", 20.0)

        for signal in filtered[:available]:
            close_price = float(current_prices[signal.symbol]["close"])

            shares = self.portfolio.calculate_shares(
                entry_price=close_price,
                stop_price=signal.stop_price,
                risk_pct=risk_pct,
                max_position_pct=max_pos_pct,
            )

            if shares > 0:
                signal.metadata["exit_rules"] = self.strategy.exit_rules()
                self.portfolio.pending_orders.append(Order(
                    symbol=signal.symbol,
                    shares=shares,
                    is_entry=True,
                    signal=signal,
                    created_date=trading_day,
                ))

    def _close_remaining(
        self,
        last_date: pd.Timestamp,
        universe: dict[str, pd.DataFrame],
        date_index: dict[str, dict],
    ) -> None:
        """Liquidate all remaining positions at end of backtest."""
        from vbacktest.strategy import ExitCondition, ExitSignal

        for symbol in list(self.portfolio.positions.keys()):
            if symbol not in universe:
                continue

            if last_date in date_index.get(symbol, {}):
                idx = date_index[symbol][last_date]
            else:
                idx = len(universe[symbol]) - 1

            bar = universe[symbol].iloc[idx]
            exit_signal = ExitSignal(
                condition=ExitCondition.STRATEGY_EXIT,
                fraction=1.0,
                reason="End of backtest",
            )
            position = self.portfolio.positions[symbol]
            self.portfolio.pending_orders.append(Order(
                symbol=symbol,
                shares=position.shares,
                is_entry=False,
                exit_signal=exit_signal,
                created_date=last_date,
            ))
            self.portfolio.execute_exits(last_date, {symbol: {"open": float(bar["close"]), "close": float(bar["close"])}})
