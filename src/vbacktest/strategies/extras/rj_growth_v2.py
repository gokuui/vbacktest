"""Rakesh Jhunjhunwala Growth Momentum v2 - Improved for Mid/Small-Caps.

Improvements over v1:
- Loosened entry criteria (lower ROC threshold)
- Removed volume surge requirement (test separately)
- Optimized for mid/small-cap stocks
- Added partial profit taking

Based on learnings from v1:
- V1 Calmar 0.21 on large-caps
- Good DD control (5.58%) but low returns (1.19% CAGR)
- Win rate too low (38.96%)
- Strategy-market mismatch (tested on mature large-caps)
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TimeStopRule, TakeProfitPartialRule


class RJGrowthV2Strategy(Strategy):
    """RJ Growth Momentum v2 - optimized for mid/small-caps.

    Key changes from v1:
    - Lower ROC threshold (10% vs 15%)
    - Looser "near high" definition (within 5% vs 2%)
    - No volume surge requirement
    - Partial profit taking at 2R
    - Tighter time stop (60 days vs 90)
    """

    def __init__(
        self,
        lookback_period: int = 50,
        roc_period: int = 20,
        roc_threshold: float = 10.0,  # Lowered from 15%
        near_high_pct: float = 5.0,  # Looser from 2%
        ma_trend_period: int = 50,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,
        stop_loss_pct: float = 8.0,
        max_holding_days: int = 60,  # Tighter from 90
        partial_profit_r: float = 2.0,  # Take 50% profit at 2R
    ):
        """Initialize RJ Growth v2.

        Args:
            lookback_period: Period for new highs check
            roc_period: Rate of change period
            roc_threshold: Minimum ROC % for entry (lowered to 10%)
            near_high_pct: % within high to consider "near" (increased to 5%)
            ma_trend_period: MA period for trend filter
            atr_period: ATR period
            atr_stop_multiplier: ATR multiplier for trailing stop
            stop_loss_pct: Fixed stop loss percentage
            max_holding_days: Maximum holding period (reduced to 60)
            partial_profit_r: R-multiple for partial profit (2R)
        """
        self.lookback_period = lookback_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.near_high_pct = near_high_pct
        self.ma_trend_period = ma_trend_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.partial_profit_r = partial_profit_r

        # Column names
        self.roc_col = f'roc_{roc_period}'
        self.ma_col = f'sma_{ma_trend_period}'
        self.atr_col = f'atr_{atr_period}'
        self.high_col = f'rolling_high_{lookback_period}'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators."""
        return [
            IndicatorSpec('roc', {'period': self.roc_period, 'output_col': self.roc_col}),
            IndicatorSpec('sma', {'period': self.ma_trend_period}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': self.lookback_period, 'output_col': self.high_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate entry signals."""
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue

            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]

            # Need enough history
            if idx < self.lookback_period:
                continue

            # Check required columns
            required = [self.roc_col, self.ma_col, self.atr_col, self.high_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            # Check for NaN
            if any(pd.isna(current[col]) for col in required):
                continue

            # ENTRY CONDITION 1: Momentum (LOOSENED)
            # Near high (within 5% of high) OR decent ROC (10%+)
            near_high = current['close'] >= current[self.high_col] * (1 - self.near_high_pct / 100)
            strong_roc = current[self.roc_col] >= self.roc_threshold

            momentum_signal = near_high or strong_roc

            if not momentum_signal:
                continue

            # ENTRY CONDITION 2: Above trend MA (uptrend)
            if current['close'] < current[self.ma_col]:
                continue

            # ENTRY CONDITION 3: Price action (higher close) - KEPT
            if current['close'] <= prev['close']:
                continue

            # Calculate stop price
            atr_stop = current['close'] - (self.atr_stop_multiplier * current[self.atr_col])
            pct_stop = current['close'] * (1 - self.stop_loss_pct / 100)
            stop_price = max(atr_stop, pct_stop)

            # Score: Combine ROC strength + proximity to high
            roc_score = min(current[self.roc_col] / self.roc_threshold, 2.0)  # Cap at 2x
            high_proximity = current['close'] / current[self.high_col]
            score = (roc_score * 0.6) + (high_proximity * 0.4)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'atr': current[self.atr_col],
                    'roc': current[self.roc_col],
                    'near_high': near_high,
                    'strong_roc': strong_roc,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """Return exit rules."""
        return [
            StopLossRule(),
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier
            ),
            TakeProfitPartialRule(
                r_multiple=self.partial_profit_r,
                fraction=0.5  # Take 50% profit at 2R
            ),
            TimeStopRule(max_holding_days=self.max_holding_days),
        ]
