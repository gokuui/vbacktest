"""Momentum Pullback strategy (Larry Williams inspired).

Buy pullbacks in strong momentum stocks. UNCORRELATED with Minervini because
it buys during PULLBACKS (RSI 35-50), not breakouts (new highs).

Entry: Stock in uptrend with strong momentum, experiencing a pullback
       (RSI between 35-50), with the first bullish candle signaling reversal.
Exit: 5% stop, RSI>70 target or 10-day time stop
Stop: 5% below entry

Goal: Capture bounce from pullbacks in trending stocks. Different signal
      from Minervini (pullbacks vs breakouts) = uncorrelated.
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class MomentumPullbackStrategy(Strategy):
    """Momentum pullback trading for NSE markets.

    Entry conditions (ALL must be true):
    1. Uptrend: Price > SMA50 (medium-term uptrend)
    2. Strong momentum: 20-day ROC > 5% (stock trending up)
    3. Pullback zone: RSI(14) between 35-50 (pulled back, not oversold)
    4. First bullish candle: Close > Open (reversal signal)
    5. Volume present: Volume > 0.8x 20-day average (not dried up)

    Exit conditions (ANY triggers):
    1. Hard stop: 5% below entry
    2. RSI target: RSI > 70 (overbought = take profit)
    3. Time stop: 10 calendar days max
    """

    def __init__(
        self,
        ma_trend: int = 50,
        roc_period: int = 20,
        roc_threshold: float = 5.0,
        rsi_period: int = 14,
        rsi_low: float = 35.0,
        rsi_high: float = 50.0,
        volume_min: float = 0.8,
        stop_loss_pct: float = 5.0,
        time_stop_days: int = 10,
    ):
        self.ma_trend = ma_trend
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.volume_min = volume_min
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_col = f'sma_{ma_trend}'
        self.roc_col = f'roc_{roc_period}'
        self.rsi_col = f'rsi_{rsi_period}'
        self.volume_avg_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('roc', {'period': self.roc_period}),
            IndicatorSpec('rsi', {'period': self.rsi_period, 'output_col': self.rsi_col}),
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
            if idx < max(self.ma_trend, self.roc_period) + 5:
                continue

            required_cols = [self.ma_col, self.roc_col, self.rsi_col, self.volume_avg_col]
            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]
            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue

            # CONDITION 1: Above MA50 (uptrend)
            if current_bar['close'] <= current_bar[self.ma_col]:
                continue

            # CONDITION 2: Strong momentum (20-day ROC > 5%)
            if current_bar[self.roc_col] < self.roc_threshold:
                continue

            # CONDITION 3: Pullback zone (RSI between 35-50)
            rsi_val = current_bar[self.rsi_col]
            if rsi_val < self.rsi_low or rsi_val > self.rsi_high:
                continue

            # CONDITION 4: Bullish candle (first reversal candle)
            if current_bar['close'] <= current_bar['open']:
                continue

            # CONDITION 5: Volume present (not dried up)
            if current_bar[self.volume_avg_col] <= 0:
                continue
            vol_ratio = current_bar['volume'] / current_bar[self.volume_avg_col]
            if vol_ratio < self.volume_min:
                continue

            # Stop price: 5% below close
            stop_price = current_bar['close'] * (1 - self.stop_loss_pct / 100)

            # Score: higher ROC = stronger momentum = better pullback to buy
            score = current_bar[self.roc_col] * (1 + vol_ratio)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'roc_20d': current_bar[self.roc_col],
                    'rsi': rsi_val,
                    'volume_ratio': vol_ratio,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        return [
            StopLossRule(),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
