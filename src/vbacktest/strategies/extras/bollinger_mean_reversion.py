"""Bollinger Band Mean Reversion Strategy.

Different approach from trend-following (RJ):
- Enter on oversold conditions (price touches/crosses below lower BB)
- Exit when price reverts to middle BB or resistance
- Typically lower drawdown than momentum strategies
- Good for choppy/sideways markets

Goal: Test if mean-reversion has better Calmar than trend-following

Entry: Close below lower BB + RSI < 30 + Volume confirmation
Exit: Close above middle BB OR stop loss OR time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class BollingerMeanReversionStrategy(Strategy):
    """Mean reversion using Bollinger Bands.

    Counter-trend strategy for better DD control.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        volume_ma_period: int = 20,
        volume_multiplier: float = 1.0,  # No volume requirement
        atr_period: int = 14,
        stop_loss_pct: float = 5.0,  # Tight stop
        max_holding_days: int = 15,  # Quick exits
    ):
        """Initialize Bollinger mean reversion.

        Args:
            bb_period: Bollinger band period
            bb_std_dev: Standard deviations for BB
            rsi_period: RSI period
            rsi_oversold: RSI oversold threshold
            volume_ma_period: Volume MA period
            volume_multiplier: Volume multiplier (1.0 = no filter)
            atr_period: ATR period
            stop_loss_pct: Fixed stop loss %
            max_holding_days: Max holding period
        """
        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days

        # Column names
        self.bb_upper_col = 'bb_upper'
        self.bb_mid_col = 'bb_mid'
        self.bb_lower_col = 'bb_lower'
        self.rsi_col = f'rsi_{rsi_period}'
        self.volume_ma_col = f'volume_sma_{volume_ma_period}'
        self.atr_col = f'atr_{atr_period}'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators."""
        return [
            IndicatorSpec('bollinger_bands', {
                'period': self.bb_period,
                'num_std': self.bb_std_dev
            }),
            IndicatorSpec('rsi', {'period': self.rsi_period, 'output_col': self.rsi_col}),
            IndicatorSpec('volume_sma', {'period': self.volume_ma_period, 'output_col': self.volume_ma_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate mean-reversion signals."""
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue

            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]

            if idx < 1:
                continue

            # Check columns
            required = [self.bb_lower_col, self.bb_mid_col, self.rsi_col, self.volume_ma_col, self.atr_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            # Check NaN
            if any(pd.isna(current[col]) for col in required):
                continue

            # ENTRY 1: Price crossed below lower BB
            below_bb = (prev['close'] >= prev[self.bb_lower_col] and
                       current['close'] < current[self.bb_lower_col])

            if not below_bb:
                continue

            # ENTRY 2: RSI oversold
            if current[self.rsi_col] >= self.rsi_oversold:
                continue

            # ENTRY 3: Optional volume filter
            if self.volume_multiplier > 1.0:
                if current['volume'] < current[self.volume_ma_col] * self.volume_multiplier:
                    continue

            # Calculate stop
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: How oversold (lower RSI = higher score)
            rsi_score = (self.rsi_oversold - current[self.rsi_col]) / self.rsi_oversold
            bb_distance = (current[self.bb_lower_col] - current['close']) / current['close']
            score = (rsi_score * 0.6) + (bb_distance * 100 * 0.4)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=max(score, 0.1),  # Ensure positive
                metadata={
                    'atr': current[self.atr_col],
                    'rsi': current[self.rsi_col],
                    'bb_lower': current[self.bb_lower_col],
                    'bb_mid': current[self.bb_mid_col],
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """Return exit rules - reversion to mean."""
        return [
            StopLossRule(),  # Tight stop
            TrailingMARule(ma_column=self.bb_mid_col),  # Exit when price crosses above middle BB
            TimeStopRule(max_holding_days=self.max_holding_days),  # Quick exit
        ]
