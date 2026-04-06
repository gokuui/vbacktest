"""Narrow Range Breakout strategy — volatility contraction breakout.

Entry conditions:
1. NR4 or NR7: Today's range is the narrowest of last 4 or 7 bars
2. Stage 2 uptrend: Price > SMA50 > SMA200
3. Close in upper half of range (bullish bias)
4. Price within 20% of 52-week high (strong stock)
5. ATR contraction: 10-day ATR < 20-day ATR (compression)
6. Volume not dead: Volume > 0.5x 50-day average

Entry: Buy at next bar's open when price breaks above NR bar's high

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import pandas as pd
import numpy as np

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class NarrowRangeStrategy(Strategy):
    """Narrow Range Breakout — volatility contraction into breakout."""

    def __init__(
        self,
        nr_period: int = 7,
        ma_fast: int = 50,
        ma_slow: int = 200,
        near_high_pct: float = 20.0,
        atr_period: int = 14,
        volume_avg_period: int = 50,
        volume_min_ratio: float = 0.5,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.nr_period = nr_period
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.near_high_pct = near_high_pct
        self.atr_period = atr_period
        self.volume_avg_period = volume_avg_period
        self.volume_min_ratio = volume_min_ratio
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.volume_avg_col = f'volume_sma_{volume_avg_period}'
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.volume_avg_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_slow, self.nr_period) + 10:
                continue

            required = [self.ma_fast_col, self.ma_slow_col, self.atr_col,
                        self.volume_avg_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: Stage 2 uptrend
            if not (current[self.ma_fast_col] > current[self.ma_slow_col]):
                continue
            if not (current['close'] > current[self.ma_fast_col]):
                continue

            # FILTER 2: Narrow Range — today's range is narrowest of last N bars
            today_range = current['high'] - current['low']
            if today_range <= 0:
                continue

            is_narrow = True
            for j in range(1, self.nr_period):
                prev_bar = df.iloc[idx - j]
                prev_range = prev_bar['high'] - prev_bar['low']
                if today_range >= prev_range:
                    is_narrow = False
                    break

            if not is_narrow:
                continue

            # FILTER 3: Close in upper half of range (bullish bias)
            if today_range > 0:
                position_in_range = (current['close'] - current['low']) / today_range
                if position_in_range < 0.5:
                    continue

            # FILTER 4: Near 52-week high
            price_vs_high = current['close'] / current[self.high_52w_col] * 100
            if price_vs_high < (100 - self.near_high_pct):
                continue

            # FILTER 5: ATR contraction
            if idx >= 20:
                recent_atr = df[self.atr_col].iloc[idx-5:idx+1].mean()
                earlier_atr = df[self.atr_col].iloc[idx-20:idx-5].mean()
                if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
                    continue
                if recent_atr >= earlier_atr:
                    continue

            # FILTER 6: Volume not dead
            if current['volume'] < current[self.volume_avg_col] * self.volume_min_ratio:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: tighter range + closer to high = higher priority
            range_tightness = 1.0 / (today_range / current['close'] * 100 + 0.01)
            score = range_tightness + price_vs_high * 0.1

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'range_pct': today_range / current['close'] * 100,
                    'price_vs_high': price_vs_high,
                    'nr_period': self.nr_period,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        elif self.trailing_exit_type == 'ema10':
            rules.append(TrailingMARule(ma_column='ema_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
