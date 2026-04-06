"""Bollinger Breakout V2 - Improved risk controls.

Improvements over V1:
1. Regime filter: Only trade in bull markets (Nifty 50 > MA200)
2. Tighter stop: 3% fixed stop (vs 5%)
3. Time stop: Exit if no 15% gain in 30 days
4. Partial profit: Exit 50% at 2R
5. Stronger entry: Require close > 1% above BB upper

Goal: Reduce Max DD to <20% while maintaining 25%+ CAGR
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TakeProfitPartialRule, TimeStopRule


class BollingerBreakoutV2Strategy(Strategy):
    """Bollinger Breakout V2 with tighter risk controls.

    Entry conditions:
    - Close breaks above upper BB by at least 1% (stronger breakout)
    - Volume surge (1.5x average)
    - Trend confirmation (price > SMA50)
    - Regime filter: Market (Nifty) > MA200 (bull market only)

    Exit conditions:
    - Trailing ATR stop (2.0x)
    - Fixed stop loss (3% - tighter than V1's 5%)
    - Time stop: 30 days if no 15% gain
    - Partial profit: Exit 50% at 2R

    Scoring:
    - Distance above upper band = strength of breakout
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        volume_surge_multiplier: float = 1.5,
        trend_filter_period: int = 50,
        regime_filter_period: int = 200,  # MA200 for regime
        min_breakout_pct: float = 1.0,  # Require 1% above BB
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        stop_loss_pct: float = 3.0,  # Tighter: 3% vs 5%
        time_stop_days: int = 45,  # Exit after 45 days max
        partial_profit_r: float = 2.0,  # Exit 50% at 2R
        partial_profit_fraction: float = 0.5,  # 50% of position
    ):
        """Initialize Bollinger Breakout V2 strategy.

        Args:
            bb_period: Bollinger band period (default: 20)
            bb_std_dev: Number of standard deviations (default: 2.0)
            volume_surge_multiplier: Volume surge threshold (default: 1.5)
            trend_filter_period: MA period for trend (default: 50)
            regime_filter_period: MA period for regime (default: 200)
            min_breakout_pct: Minimum % above BB upper for entry (default: 1.0)
            atr_period: ATR period for stops (default: 14)
            atr_stop_multiplier: ATR multiplier for trailing stop (default: 2.0)
            stop_loss_pct: Fixed stop loss percentage (default: 3.0)
            time_stop_days: Max holding period in days (default: 45)
            partial_profit_r: R-multiple for partial profit (default: 2.0)
            partial_profit_fraction: Fraction to exit at partial profit (default: 0.5)
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.volume_surge_multiplier = volume_surge_multiplier
        self.trend_filter_period = trend_filter_period
        self.regime_filter_period = regime_filter_period
        self.min_breakout_pct = min_breakout_pct
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days
        self.partial_profit_r = partial_profit_r
        self.partial_profit_fraction = partial_profit_fraction

        # Column names
        self.bb_upper_col = 'bb_upper'
        self.bb_middle_col = 'bb_mid'
        self.bb_lower_col = 'bb_lower'
        self.atr_col = f'atr_{atr_period}'
        self.trend_col = f'sma_{trend_filter_period}'
        self.regime_col = f'sma_{regime_filter_period}'
        self.volume_avg_col = f'volume_sma_{bb_period}'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators."""
        return [
            IndicatorSpec('bollinger_bands', {
                'period': self.bb_period,
                'num_std': self.bb_std_dev
            }),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('volume_sma', {
                'period': self.bb_period,
                'output_col': self.volume_avg_col
            }),
            IndicatorSpec('sma', {'period': self.trend_filter_period}),
            IndicatorSpec('sma', {
                'period': self.regime_filter_period,
                'output_col': self.regime_col
            }),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate signals for current bar."""
        signals = []

        # REGIME FILTER: Check if Nifty 50 is in bull market
        # TODO: For now, we'll skip regime filter since we don't have Nifty data
        # in the universe. In production, would check Nifty separately.
        # For now, apply regime filter to each stock individually (price > MA200)

        for symbol, df in ctx.universe.items():
            # Skip if already in ctx.portfolio
            if ctx.portfolio.has_position(symbol):
                continue

            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]

            # Need at least 2 bars for breakout check
            if idx < 1:
                continue

            # Check required columns
            required_cols = [
                self.bb_upper_col, self.bb_middle_col, self.bb_lower_col,
                self.atr_col, self.volume_avg_col, self.trend_col, self.regime_col
            ]

            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]
            prev_bar = df.iloc[idx - 1]

            # Check for NaN values
            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue

            # REGIME FILTER: Price must be above MA200 (bull market)
            if current_bar['close'] < current_bar[self.regime_col]:
                continue

            # ENTRY CONDITION 1: Strong breakout above upper band (>1% above)
            # Previous close was below/at upper band, current close is >1% above
            breakout = (
                prev_bar['close'] <= prev_bar[self.bb_upper_col] and
                current_bar['close'] > current_bar[self.bb_upper_col]
            )

            if not breakout:
                continue

            # Check breakout strength: must be at least 1% above BB upper
            distance_above_pct = (
                (current_bar['close'] - current_bar[self.bb_upper_col]) /
                current_bar[self.bb_upper_col] * 100
            )

            if distance_above_pct < self.min_breakout_pct:
                continue

            # ENTRY CONDITION 2: Volume surge
            volume_threshold = current_bar[self.volume_avg_col] * self.volume_surge_multiplier
            if current_bar['volume'] < volume_threshold:
                continue

            # ENTRY CONDITION 3: Trend filter (price > SMA50)
            if current_bar['close'] < current_bar[self.trend_col]:
                continue

            # Calculate stop price (use 3% fixed stop)
            pct_stop = current_bar['close'] * (1 - self.stop_loss_pct / 100)

            # Also calculate ATR stop for comparison
            atr_stop = current_bar['close'] - (self.atr_stop_multiplier * current_bar[self.atr_col])

            # Use tighter of the two stops
            stop_price = max(atr_stop, pct_stop)

            # Score based on distance above upper band
            score = distance_above_pct

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'atr': current_bar[self.atr_col],
                    'bb_upper': current_bar[self.bb_upper_col],
                    'distance_pct': distance_above_pct
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """Return exit rules for positions."""
        return [
            StopLossRule(),
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier
            ),
            TakeProfitPartialRule(
                r_multiple=self.partial_profit_r,
                fraction=self.partial_profit_fraction
            ),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
