"""Connors RSI Pullback strategy.

Buy oversold pullbacks in strong uptrends using Connors RSI (RSI of RSI + streak + percentile).
This is a mean-reversion strategy that ONLY trades in established uptrends.

Key insight: In strong uptrends, short-term oversold readings (RSI2 < 10) are buying
opportunities, not warnings. Connors RSI combines three components for a more reliable
oversold signal than standard RSI.

Entry conditions:
1. Strong uptrend: Close > SMA50 > SMA200 (Stage 2)
2. RSI(2) < oversold threshold (deeply oversold short-term)
3. Close still above SMA50 (pullback within trend, not breakdown)
4. Today's close > yesterday's low (not in free fall)

Exit: Close > SMA5 (short-term recovery) + hard stop + time stop
This is a FAST mean-reversion play: in and out in 2-7 days typically.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class ConnorsPullbackStrategy(Strategy):
    """Buy RSI(2) oversold pullbacks in Stage 2 uptrends."""

    def __init__(
        self,
        rsi_period: int = 2,
        rsi_oversold: float = 10.0,
        ma_fast: int = 50,
        ma_slow: int = 200,
        exit_ma: int = 5,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 10,
    ):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.exit_ma = exit_ma
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.rsi_col = f'rsi_{rsi_period}'
        self.exit_ma_col = f'sma_{exit_ma}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': self.exit_ma}),
            IndicatorSpec('rsi', {'period': self.rsi_period, 'output_col': self.rsi_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = self.ma_slow + 10

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
                    close_prev = arrays['close'][idx - 1]
                    low_prev = arrays['low'][idx - 1]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    rsi = arrays[self.rsi_col][idx]
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in [close, close_prev, low_prev, ma_fast, ma_slow, rsi]):
                    continue

                # FILTER 1: Stage 2 uptrend
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 2: RSI(2) oversold
                if rsi > self.rsi_oversold:
                    continue

                # FILTER 3: Not in free fall — close > yesterday's low
                if close <= low_prev:
                    continue

                stop_price = close * (1 - self.stop_loss_pct / 100)
                score = self.rsi_oversold - rsi  # Lower RSI = higher priority

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=score,
                    metadata={'rsi2': rsi}
                ))
            else:
                continue

        return signals

    def exit_rules(self) -> list:
        return [
            StopLossRule(),
            TrailingMARule(ma_column=self.exit_ma_col),
            TimeStopRule(max_holding_days=self.max_holding_days),
        ]
