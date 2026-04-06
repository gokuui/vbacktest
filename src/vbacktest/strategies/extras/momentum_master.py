"""Momentum strategy - buy strong performers."""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TimeStopRule


class MomentumMasterStrategy(Strategy):
    """Momentum strategy - buy stocks with strong recent performance.

    Entry conditions:
    - Price has gained > threshold over lookback periods
    - Multiple timeframe confirmation (20d, 60d both positive)
    - Above trend MA (optional)
    - New high breakout (optional)

    Exit conditions:
    - Trailing ATR stop
    - Time stop (momentum fades)

    Scoring:
    - Strongest momentum = highest score
    """

    def __init__(
        self,
        lookback_periods: list[int] = None,
        momentum_threshold_pct: float = 10.0,
        trend_filter_period: int | None = 200,
        new_high_period: int | None = 50,
        atr_period: int = 14,
        atr_stop_multiplier: float = 3.0,
        max_holding_days: int = 60,
    ):
        """Initialize momentum strategy.

        Args:
            lookback_periods: Periods to calculate momentum (default: [20, 60])
            momentum_threshold_pct: Minimum momentum % (default: 10.0)
            trend_filter_period: MA period for trend (None = disabled)
            new_high_period: Period for new high filter (None = disabled)
            atr_period: ATR period for stops (default: 14)
            atr_stop_multiplier: ATR multiplier for trailing stop (default: 3.0)
            max_holding_days: Maximum holding period (default: 60)
        """
        self.lookback_periods = lookback_periods or [20, 60]
        self.momentum_threshold_pct = momentum_threshold_pct
        self.trend_filter_period = trend_filter_period
        self.new_high_period = new_high_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.max_holding_days = max_holding_days

        # Column names
        self.momentum_cols = [f'momentum_{p}' for p in self.lookback_periods]
        self.atr_col = f'atr_{atr_period}'
        self.trend_col = f'sma_{trend_filter_period}' if trend_filter_period else None
        self.high_col = f'high_{new_high_period}' if new_high_period else None

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators."""
        specs = [
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
        ]

        # Add momentum indicators (will compute manually)
        # No built-in momentum indicator, will calculate in on_bar

        if self.trend_filter_period:
            specs.append(IndicatorSpec('sma', {'period': self.trend_filter_period}))

        if self.new_high_period:
            specs.append(IndicatorSpec('rolling_high', {
                'period': self.new_high_period,
                'output_col': self.high_col
            }))

        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate signals for current bar."""
        signals = []

        use_fast_path = ctx.universe_arrays is not None
        max_lookback = max(self.lookback_periods)

        for symbol in ctx.universe_idx:
            # Skip if already in ctx.portfolio
            if ctx.portfolio.has_position(symbol):
                continue

            idx = ctx.universe_idx[symbol]

            # Need enough history for momentum calculation
            if idx < max_lookback:
                continue

            if use_fast_path and ctx.current_prices and symbol in ctx.current_prices and symbol in ctx.universe_arrays:
                # --- Fast path: dict lookups + numpy arrays ---
                bar = ctx.current_prices[symbol]
                arrays = ctx.universe_arrays[symbol]

                # Check required columns
                if self.atr_col not in arrays:
                    continue

                atr_val = arrays[self.atr_col][idx]
                if atr_val != atr_val:  # NaN check
                    continue

                # Calculate momentum for each period
                momentums = []
                close_arr = arrays['close']
                current_close = close_arr[idx]
                for period in self.lookback_periods:
                    past_close = close_arr[idx - period]
                    momentum_pct = ((current_close - past_close) / past_close) * 100
                    momentums.append(momentum_pct)

                # ENTRY CONDITION 1: All momentum periods above threshold
                if not all(m >= self.momentum_threshold_pct for m in momentums):
                    continue

                # ENTRY CONDITION 2: Trend filter (optional)
                if self.trend_col:
                    if self.trend_col not in arrays:
                        continue
                    trend_val = arrays[self.trend_col][idx]
                    if trend_val != trend_val:  # NaN check
                        continue
                    if bar['close'] < trend_val:
                        continue

                # ENTRY CONDITION 3: New high breakout (optional)
                if self.high_col:
                    if self.high_col not in arrays:
                        continue
                    high_val = arrays[self.high_col][idx]
                    if high_val != high_val:  # NaN check
                        continue
                    if bar['close'] < high_val * 0.98:
                        continue

                close = bar['close']
                atr = arrays[self.atr_col][idx]

            else:
                # --- Slow path: pandas iloc fallback ---
                if symbol not in universe:
                    continue

                df = ctx.universe[symbol]

                # Check required columns
                if self.atr_col not in df.columns:
                    continue

                current_bar = df.iloc[idx]

                if pd.isna(current_bar[self.atr_col]):
                    continue

                # Calculate momentum for each period
                momentums = []
                for period in self.lookback_periods:
                    past_bar = df.iloc[idx - period]
                    momentum_pct = ((current_bar['close'] - past_bar['close']) / past_bar['close']) * 100
                    momentums.append(momentum_pct)

                # ENTRY CONDITION 1: All momentum periods above threshold
                if not all(m >= self.momentum_threshold_pct for m in momentums):
                    continue

                # ENTRY CONDITION 2: Trend filter (optional)
                if self.trend_col:
                    if self.trend_col not in df.columns:
                        continue
                    if pd.isna(current_bar[self.trend_col]):
                        continue
                    if current_bar['close'] < current_bar[self.trend_col]:
                        continue

                # ENTRY CONDITION 3: New high breakout (optional)
                if self.high_col:
                    if self.high_col not in df.columns:
                        continue
                    if pd.isna(current_bar[self.high_col]):
                        continue
                    # Current close should be at/near high
                    if current_bar['close'] < current_bar[self.high_col] * 0.98:
                        continue

                close = current_bar['close']
                atr = current_bar[self.atr_col]

            # --- Shared signal generation ---
            # Calculate stop price
            stop_price = close - (self.atr_stop_multiplier * atr)

            # Score based on average momentum
            avg_momentum = sum(momentums) / len(momentums)
            score = avg_momentum

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'atr': atr,  # Store ATR for fallback stop calculation
                    **{f'momentum_{p}d': m for p, m in zip(self.lookback_periods, momentums)}
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """Return exit rules for positions."""
        rules = [
            StopLossRule(),
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier
            ),
            TimeStopRule(max_holding_days=self.max_holding_days),
        ]

        return rules
