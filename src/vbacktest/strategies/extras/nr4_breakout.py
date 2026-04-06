"""NR4 / Inside Bar Breakout strategy.

Edge: Volatility contraction → expansion. When a stock's range gets tight
for several days (narrowest range in 4 days = NR4, or inside bars), a
breakout is imminent. Catches moves before they show up in ROC/momentum.
Completely different entry timing than momentum.

Entry conditions (ALL must be true):
1. NR condition: today's range = narrowest in last 4 days (NR4)
   OR inside bar: today's high < yesterday's high AND today's low > yesterday's low
2. Close > SMA50 (only trade in uptrends)
3. Breakout: close > high of the NR bar (confirmed by close, not intraday)
   — since we act on next bar's open, we enter when price breaks the NR high
4. Volume surge: volume > 1.5x 20-day avg on breakout day

Entry: buy next open when today closes above NR bar's high with volume surge

Exit: Stop below NR4 bar's low + trailing ATR
Score: ATR contraction ratio (tighter base = stronger breakout expected)
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TimeStopRule


class NR4BreakoutStrategy(Strategy):
    """NR4 / Inside Bar volatility contraction breakout."""

    def __init__(
        self,
        nr_period: int = 4,            # narrowest range in last N bars
        ma_trend: int = 50,
        volume_avg_period: int = 20,
        volume_multiplier: float = 1.5,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        stop_loss_pct: float = 6.0,
        max_holding_days: int = 30,    # shorter — these are tactical entries
    ):
        self.nr_period = nr_period
        self.ma_trend = ma_trend
        self.volume_avg_period = volume_avg_period
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days

        self.ma_col = f'sma_{ma_trend}'
        self.vol_avg_col = f'volume_sma_{volume_avg_period}'
        self.atr_col = f'atr_{atr_period}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.vol_avg_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if symbol.startswith('^'):
                continue
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_trend, self.nr_period, self.volume_avg_period) + 5:
                continue

            required = [self.ma_col, self.vol_avg_col, self.atr_col]
            if not all(c in df.columns for c in required):
                continue

            if ctx.universe_arrays and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    high = arrays['high'][idx]
                    low = arrays['low'][idx]
                    volume = arrays['volume'][idx]
                    ma = arrays[self.ma_col][idx]
                    vol_avg = arrays[self.vol_avg_col][idx]
                    atr = arrays[self.atr_col][idx]
                    # Last N bars for NR check (not including today)
                    recent_highs = arrays['high'][idx - self.nr_period:idx]
                    recent_lows = arrays['low'][idx - self.nr_period:idx]
                    prev_high = arrays['high'][idx - 1]
                    prev_low = arrays['low'][idx - 1]
                except (KeyError, IndexError):
                    continue
            else:
                bar = df.iloc[idx]
                close = bar['close']
                high = bar['high']
                low = bar['low']
                volume = bar['volume']
                ma = bar[self.ma_col]
                vol_avg = bar[self.vol_avg_col]
                atr = bar[self.atr_col]
                recent_highs = df['high'].iloc[idx - self.nr_period:idx].values
                recent_lows = df['low'].iloc[idx - self.nr_period:idx].values
                prev_high = df['high'].iloc[idx - 1]
                prev_low = df['low'].iloc[idx - 1]

            if any(np.isnan(v) for v in [close, ma, vol_avg, atr]):
                continue
            if high <= 0 or low <= 0:
                continue

            today_range = high - low

            # CONDITION 1a: NR4 — today's range is narrowest of last nr_period bars
            if len(recent_highs) < self.nr_period or len(recent_lows) < self.nr_period:
                continue
            prior_ranges = recent_highs - recent_lows
            is_nr4 = np.all(prior_ranges > 0) and today_range <= np.min(prior_ranges)

            # CONDITION 1b: Inside bar — today's range inside yesterday's range
            is_inside = (high < prev_high) and (low > prev_low)

            if not (is_nr4 or is_inside):
                continue

            # CONDITION 2: Above SMA50
            if close <= ma:
                continue

            # CONDITION 3: Breakout — close above the NR bar's high
            # The NR bar high is the most recent prior bar's high (yesterday's high)
            nr_bar_high = prev_high
            if close <= nr_bar_high:
                continue

            # CONDITION 4: Volume surge
            if vol_avg <= 0 or volume < vol_avg * self.volume_multiplier:
                continue

            # Stop: below NR bar's low
            nr_bar_low = prev_low
            atr_stop = close - self.atr_stop_multiplier * atr
            pct_stop = close * (1 - self.stop_loss_pct / 100)
            stop_price = max(nr_bar_low, atr_stop, pct_stop)

            # Score: ATR contraction — smaller today_range/atr = tighter compression
            contraction = today_range / atr if atr > 0 else 1.0
            score = 1.0 / contraction  # higher = more compressed = stronger signal

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'is_nr4': bool(is_nr4),
                    'is_inside': bool(is_inside),
                    'range_vs_atr': round(contraction, 2),
                }
            ))

        return signals

    def exit_rules(self) -> list:
        return [
            StopLossRule(),
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier,
            ),
            TimeStopRule(max_holding_days=self.max_holding_days),
        ]
