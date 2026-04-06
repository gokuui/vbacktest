"""Short-term Swing strategy (2-5 day holds).

Very short-term momentum bursts. Catches stocks making sharp moves
with volume confirmation and exits quickly. Different timeframe from
Minervini (13 days) and Mean Reversion (7 days).

Entry: 3-day rate of change > 5%, volume surge, above MA50
Exit: Close below previous day's low OR 5 days max
Stop: 3% (very tight for short holds)

Goal: Very short holds, high frequency, uncorrelated timeframe.
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class ShortSwingStrategy(Strategy):
    """Short-term swing trading for NSE markets.

    Entry conditions (ALL must be true):
    1. Trend filter: Price > MA50 (medium-term uptrend)
    2. Momentum burst: 3-day rate of change > 5%
    3. Volume confirmation: Volume > 1.5x 20-day average
    4. Not overextended: Price < 1.10 * MA50 (max 10% above)
    5. Positive close: Close > Open (bullish candle)

    Exit conditions (ANY triggers):
    1. Trailing low: Close < previous day's low
    2. Hard stop: 3% below entry
    3. Time stop: 5 calendar days max
    """

    def __init__(
        self,
        ma_trend: int = 50,
        roc_period: int = 3,
        roc_threshold: float = 5.0,
        volume_surge: float = 1.5,
        max_above_ma_pct: float = 10.0,
        stop_loss_pct: float = 3.0,
        time_stop_days: int = 5,
    ):
        self.ma_trend = ma_trend
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.volume_surge = volume_surge
        self.max_above_ma_pct = max_above_ma_pct
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_col = f'sma_{ma_trend}'
        self.volume_avg_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('volume_sma', {'period': 20, 'output_col': self.volume_avg_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_trend, self.roc_period + 1):
                continue

            required_cols = [self.ma_col, self.volume_avg_col]
            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]
            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue

            # CONDITION 1: Above MA50 (medium-term uptrend)
            if current_bar['close'] <= current_bar[self.ma_col]:
                continue

            # CONDITION 2: Momentum burst (3-day ROC > 5%)
            price_n_ago = df.iloc[idx - self.roc_period]['close']
            if price_n_ago <= 0:
                continue
            roc = (current_bar['close'] / price_n_ago - 1) * 100
            if roc < self.roc_threshold:
                continue

            # CONDITION 3: Volume surge
            if current_bar[self.volume_avg_col] <= 0:
                continue
            vol_ratio = current_bar['volume'] / current_bar[self.volume_avg_col]
            if vol_ratio < self.volume_surge:
                continue

            # CONDITION 4: Not overextended
            distance_from_ma = (current_bar['close'] / current_bar[self.ma_col] - 1) * 100
            if distance_from_ma > self.max_above_ma_pct:
                continue

            # CONDITION 5: Bullish candle
            if current_bar['close'] <= current_bar['open']:
                continue

            # Stop price: 3% below close
            stop_price = current_bar['close'] * (1 - self.stop_loss_pct / 100)

            # Score: higher ROC + volume = stronger signal
            score = roc * vol_ratio

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'roc_3d': roc,
                    'volume_ratio': vol_ratio,
                    'distance_from_ma': distance_from_ma,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        return [
            StopLossRule(),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
