"""MACD Momentum strategy.

Uses MACD histogram crossover as momentum signal, combined with
trend and volume filters. Different indicator set from Minervini
(which uses price levels, MAs, and volatility contraction).

Entry: MACD histogram turns positive + uptrend + volume surge + ADX confirms trend
Exit: 6% stop, trailing SMA10 or 12-day time stop

Goal: Capture momentum acceleration in trending stocks. Uses completely
      different signals (MACD histogram) from Minervini = uncorrelated.
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class MACDMomentumStrategy(Strategy):
    """MACD momentum trading for NSE markets.

    Entry conditions (ALL must be true):
    1. Uptrend: Price > SMA50 (basic trend filter)
    2. MACD crossover: MACD histogram just turned positive
       (current hist > 0 and previous hist <= 0)
    3. MACD direction: MACD line is above zero (bullish regime)
    4. Volume surge: Volume > 1.3x 20-day average
    5. ADX > 20: Some trend strength (not range-bound)

    Exit conditions (ANY triggers):
    1. Hard stop: 6% below entry
    2. Time stop: 12 calendar days max
    """

    def __init__(
        self,
        ma_trend: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        volume_surge: float = 1.3,
        stop_loss_pct: float = 6.0,
        time_stop_days: int = 12,
    ):
        self.ma_trend = ma_trend
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_surge = volume_surge
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_col = f'sma_{ma_trend}'
        self.adx_col = f'adx_{adx_period}'
        self.volume_avg_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('sma', {'period': 10}),  # For trailing MA exit
            IndicatorSpec('macd', {
                'fast': self.macd_fast,
                'slow': self.macd_slow,
                'signal': self.macd_signal,
            }),
            IndicatorSpec('adx', {'period': self.adx_period}),
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
            if idx < max(self.ma_trend, self.macd_slow + self.macd_signal) + 5:
                continue

            required_cols = [self.ma_col, 'macd', 'macd_signal', 'macd_hist',
                            self.adx_col, self.volume_avg_col]
            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]
            prev_bar = df.iloc[idx - 1]

            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue
            if pd.isna(prev_bar['macd_hist']):
                continue

            # CONDITION 1: Above MA50 (uptrend)
            if current_bar['close'] <= current_bar[self.ma_col]:
                continue

            # CONDITION 2: MACD histogram crossover (just turned positive)
            if current_bar['macd_hist'] <= 0 or prev_bar['macd_hist'] > 0:
                continue

            # CONDITION 3: MACD line above zero (bullish regime)
            if current_bar['macd'] <= 0:
                continue

            # CONDITION 4: Volume surge
            if current_bar[self.volume_avg_col] <= 0:
                continue
            vol_ratio = current_bar['volume'] / current_bar[self.volume_avg_col]
            if vol_ratio < self.volume_surge:
                continue

            # CONDITION 5: ADX > threshold (trend strength)
            if current_bar[self.adx_col] < self.adx_threshold:
                continue

            # Stop price: 6% below close
            stop_price = current_bar['close'] * (1 - self.stop_loss_pct / 100)

            # Score: ADX strength * volume ratio * MACD histogram magnitude
            score = current_bar[self.adx_col] * vol_ratio * abs(current_bar['macd_hist'])

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'macd_hist': current_bar['macd_hist'],
                    'adx': current_bar[self.adx_col],
                    'volume_ratio': vol_ratio,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        from vbacktest.exit_rules import TrailingMARule
        return [
            StopLossRule(),
            TrailingMARule(ma_column='sma_10'),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
