"""Momentum Acceleration Breakout strategy.

Buy when momentum is ACCELERATING — recent returns are stronger than prior
returns. This captures stocks entering the "sweet spot" of their trend
where institutional buying is increasing.

Key insight: ER measures trend quality (static). Elder measures momentum
confirmation (binary). This measures momentum ACCELERATION (derivative) —
the rate at which momentum is increasing.

Entry conditions:
1. Return acceleration: ROC(21d) > ROC(21d, shifted 21d) — recent momentum > prior
2. Stage 2: Close > SMA50 > SMA150
3. Close > EMA10 > EMA21 (short-term aligned)
4. Near 50-day high
5. SMA50 rising
6. ADX > 20 (confirmed trend, not range-bound)

Exit: SMA10 trailing + stop + time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class AccelBreakoutStrategy(Strategy):
    """Buy when momentum is accelerating in a strong uptrend."""

    def __init__(
        self,
        accel_period: int = 21,
        min_roc: float = 5.0,
        adx_threshold: float = 20.0,
        ma_fast: int = 50,
        ma_slow: int = 150,
        breakout_period: int = 50,
        breakout_pct: float = 5.0,
        adx_period: int = 14,
        atr_period: int = 14,
        stop_loss_pct: float = 4.0,
        max_holding_days: int = 21,
        trailing_exit_type: str = 'sma10',
    ):
        self.accel_period = accel_period
        self.min_roc = min_roc
        self.adx_threshold = adx_threshold
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.breakout_period = breakout_period
        self.breakout_pct = breakout_pct
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.adx_col = f'adx_{adx_period}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('ema', {'period': 10}),
            IndicatorSpec('ema', {'period': 21}),
            IndicatorSpec('adx', {'period': self.adx_period}),
            IndicatorSpec('rolling_high', {'period': self.breakout_period, 'output_col': f'high_{self.breakout_period}'}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, self.breakout_period, self.accel_period * 2 + 5) + 30

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue
            idx = ctx.universe_idx[symbol]
            if idx < min_bars:
                continue

            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    ema10 = arrays['ema_10'][idx]
                    ema21 = arrays['ema_21'][idx]
                    adx = arrays[self.adx_col][idx]
                    high_n = arrays[f'high_{self.breakout_period}'][idx]
                    close_arr = arrays['close']
                except (KeyError, IndexError):
                    continue

                vals = [close, ma_fast, ma_slow, ema10, ema21, adx, high_n]
                if any(v != v for v in vals):
                    continue
                if close <= 0:
                    continue

                # FILTER 1: Stage 2 + short-term aligned
                if not (close > ema10 > ema21 > ma_fast > ma_slow):
                    continue

                # FILTER 2: ADX > threshold (confirmed trend)
                if adx < self.adx_threshold:
                    continue

                # FILTER 3: SMA50 rising
                ma_fast_prev = arrays[self.ma_fast_col][idx - 5]
                if ma_fast_prev != ma_fast_prev or ma_fast <= ma_fast_prev:
                    continue

                # FILTER 4: Return acceleration
                # Recent ROC (last accel_period days)
                p = self.accel_period
                c_now = close_arr[idx]
                c_recent_start = close_arr[idx - p]
                c_prior_start = close_arr[idx - 2 * p]
                c_prior_end = close_arr[idx - p]

                if any(v != v or v <= 0 for v in [c_now, c_recent_start, c_prior_start, c_prior_end]):
                    continue

                roc_recent = (c_now - c_recent_start) / c_recent_start * 100
                roc_prior = (c_prior_end - c_prior_start) / c_prior_start * 100

                # Recent ROC must be positive and greater than prior
                if roc_recent < self.min_roc:
                    continue
                if roc_recent <= roc_prior:
                    continue  # Not accelerating

                # FILTER 5: Near N-day high
                if high_n <= 0:
                    continue
                pct_from_high = (high_n - close) / high_n * 100
                if pct_from_high > self.breakout_pct:
                    continue

                stop_price = close * (1 - self.stop_loss_pct / 100)
                # Score: acceleration magnitude
                accel = roc_recent - roc_prior

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=accel,
                    metadata={'roc_recent': roc_recent, 'roc_prior': roc_prior, 'accel': accel}
                ))
            else:
                continue

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        elif self.trailing_exit_type == 'ema10':
            rules.append(TrailingMARule(ma_column='ema_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
