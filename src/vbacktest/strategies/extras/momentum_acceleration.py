"""Momentum Acceleration strategy.

Enters when momentum is ACCELERATING, not just strong. Uses the rate of
change of ROC — second derivative of price. This captures stocks where
trend strength is increasing, which is different from absolute momentum.

Different from existing strategies:
- Momentum: uses absolute ROC thresholds (ROC > X%)
- This: uses ROC acceleration (ROC increasing over short period)
- Minervini/VCP: consolidation breakout (price pattern)
- Donchian: channel breakout (price level)

Entry conditions:
1. Stage 2: Price > SMA50 > SMA200
2. ROC(20) > 10% (minimum momentum — not zero, need some base)
3. ROC acceleration: ROC(20) today > ROC(20) 5 days ago (momentum increasing)
4. ROC(20) today > ROC(20) 10 days ago (sustained acceleration)
5. Near 52-week high: within 15% of 52w high
6. ATR contraction: recent < earlier
7. Close > prior close (positive action)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class MomentumAccelerationStrategy(Strategy):
    """Momentum Acceleration — buy when momentum is increasing."""

    def __init__(
        self,
        roc_period: int = 20,
        roc_min: float = 10.0,
        accel_lookback_short: int = 5,
        accel_lookback_long: int = 10,
        ma_fast: int = 50,
        ma_slow: int = 200,
        near_high_pct: float = 15.0,
        atr_period: int = 14,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.roc_period = roc_period
        self.roc_min = roc_min
        self.accel_lookback_short = accel_lookback_short
        self.accel_lookback_long = accel_lookback_long
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.near_high_pct = near_high_pct
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.roc_col = f'roc_{roc_period}'
        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('roc', {'period': self.roc_period, 'output_col': self.roc_col}),
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
            IndicatorSpec('sma', {'period': 10}),
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
            min_bars = max(self.ma_slow, self.roc_period + self.accel_lookback_long + 5)
            if idx < min_bars:
                continue

            required = [self.roc_col, self.ma_fast_col, self.ma_slow_col,
                        self.atr_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: Stage 2 uptrend
            if not (current['close'] > current[self.ma_fast_col] > current[self.ma_slow_col]):
                continue

            # FILTER 2: Minimum ROC threshold
            current_roc = current[self.roc_col]
            if current_roc < self.roc_min:
                continue

            # FILTER 3: Short-term momentum acceleration
            roc_short_ago = df.iloc[idx - self.accel_lookback_short][self.roc_col]
            if pd.isna(roc_short_ago):
                continue
            if current_roc <= roc_short_ago:
                continue

            # FILTER 4: Sustained acceleration (longer lookback)
            roc_long_ago = df.iloc[idx - self.accel_lookback_long][self.roc_col]
            if pd.isna(roc_long_ago):
                continue
            if current_roc <= roc_long_ago:
                continue

            # FILTER 5: Near 52-week high
            price_vs_high = current['close'] / current[self.high_52w_col] * 100
            if price_vs_high < (100 - self.near_high_pct):
                continue

            # FILTER 6: ATR contraction
            if idx >= 30:
                recent_atr = df[self.atr_col].iloc[idx-5:idx+1].mean()
                earlier_atr = df[self.atr_col].iloc[idx-20:idx-5].mean()
                if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
                    continue
                if recent_atr >= earlier_atr:
                    continue

            # FILTER 7: Positive close
            if current['close'] <= prev['close']:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: acceleration magnitude
            accel = current_roc - roc_short_ago
            score = accel + current_roc * 0.3

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'roc': current_roc,
                    'accel_short': current_roc - roc_short_ago,
                    'accel_long': current_roc - roc_long_ago,
                    'price_vs_high': price_vs_high,
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
