"""Base Breakout strategy — tight consolidation range breakout.

Looks for stocks that formed a 'base' (tight trading range) and break out.
Different from Donchian (channel), Minervini (VCP), or Momentum (ROC).
This focuses on the WIDTH of the consolidation range relative to price.

Entry conditions:
1. Stage 2 uptrend: Price > SMA50 > SMA150
2. Base formation: 20-day range < base_range_pct% of price (tight consolidation)
3. Breakout: Close > highest close of last 20 days
4. Volume confirmation: Volume > 1.2x 50-day average
5. Price within 15% of 52-week high (strong stock building base)
6. ATR contraction: 10-day ATR < 20-day ATR (volatility compression)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class BaseBreakoutStrategy(Strategy):
    """Base Breakout — tight consolidation range breakout."""

    def __init__(
        self,
        base_period: int = 20,
        base_range_pct: float = 10.0,
        ma_fast: int = 50,
        ma_slow: int = 150,
        volume_avg_period: int = 50,
        volume_surge: float = 1.2,
        near_high_pct: float = 15.0,
        atr_period: int = 14,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.base_period = base_period
        self.base_range_pct = base_range_pct
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.volume_avg_period = volume_avg_period
        self.volume_surge = volume_surge
        self.near_high_pct = near_high_pct
        self.atr_period = atr_period
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
            if idx < max(self.ma_slow, self.base_period) + 10:
                continue

            required = [self.ma_fast_col, self.ma_slow_col, self.atr_col,
                        self.volume_avg_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: Stage 2 uptrend
            if not (current['close'] > current[self.ma_fast_col] > current[self.ma_slow_col]):
                continue

            # FILTER 2: Base formation — tight range over base_period days
            lookback = df.iloc[idx - self.base_period:idx]
            base_high = lookback['high'].max()
            base_low = lookback['low'].min()
            if base_low <= 0:
                continue
            base_range_pct = (base_high - base_low) / base_low * 100
            if base_range_pct > self.base_range_pct:
                continue

            # FILTER 3: Breakout — close above the base high
            highest_close = lookback['close'].max()
            if current['close'] <= highest_close:
                continue

            # FILTER 4: Volume confirmation
            if current['volume'] < current[self.volume_avg_col] * self.volume_surge:
                continue

            # FILTER 5: Near 52-week high
            price_vs_high = current['close'] / current[self.high_52w_col] * 100
            if price_vs_high < (100 - self.near_high_pct):
                continue

            # FILTER 6: ATR contraction
            if idx >= 20:
                recent_atr = df[self.atr_col].iloc[idx-5:idx+1].mean()
                earlier_atr = df[self.atr_col].iloc[idx-20:idx-5].mean()
                if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
                    continue
                if recent_atr >= earlier_atr:
                    continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: tighter base + higher price vs high = higher priority
            tightness_score = (self.base_range_pct - base_range_pct)
            score = tightness_score + price_vs_high * 0.1

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'base_range_pct': base_range_pct,
                    'price_vs_high': price_vs_high,
                    'volume_ratio': current['volume'] / current[self.volume_avg_col],
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
