"""ER Pullback strategy.

Buy short-term pullbacks in stocks with high Kaufman ER (clean trends).
Uses Stochastic %K oversold as timing signal within the ER framework.

Key difference from connors_pullback (which FAILED): connors used RSI(2)
WITHOUT Kaufman ER filter. The ER filter is critical — it ensures we're
buying dips in CLEAN trends, not random pullbacks.

This catches different entry timing than TQ/Elder (which buy breakouts).
Pullback entries are earlier and potentially uncorrelated.

Entry:
1. Kaufman ER > threshold (clean trend)
2. Stage 2: Close > SMA50 > SMA150
3. Stochastic %K < oversold (short-term pullback)
4. Close > SMA50 (still above trend, not broken)
5. SMA50 rising

Exit: Close > EMA10 (quick recovery exit) + stop + time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class ERPullbackStrategy(Strategy):
    """Buy Stochastic oversold pullbacks in high-ER uptrends."""

    def __init__(
        self,
        er_period: int = 50,
        er_threshold: float = 0.40,
        stoch_k: int = 14,
        stoch_oversold: float = 20.0,
        ma_fast: int = 50,
        ma_slow: int = 150,
        stop_loss_pct: float = 4.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.er_period = er_period
        self.er_threshold = er_threshold
        self.stoch_k = stoch_k
        self.stoch_oversold = stoch_oversold
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('stochastic', {'k_period': self.stoch_k}),
        ]

    def _kaufman_er(self, close_arr, idx, period):
        if idx < period:
            return float('nan')
        direction = abs(close_arr[idx] - close_arr[idx - period])
        volatility = 0.0
        for j in range(idx - period + 1, idx + 1):
            v = abs(close_arr[j] - close_arr[j - 1])
            if v != v:
                return float('nan')
            volatility += v
        if volatility == 0:
            return float('nan')
        return direction / volatility

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, self.er_period, self.stoch_k) + 30

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
                    stoch_k = arrays['stoch_k'][idx]
                    close_arr = arrays['close']
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in [close, ma_fast, ma_slow, stoch_k]):
                    continue

                # FILTER 1: Stage 2
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 2: SMA50 rising
                ma_fast_prev = arrays[self.ma_fast_col][idx - 5]
                if ma_fast_prev != ma_fast_prev or ma_fast <= ma_fast_prev:
                    continue

                # FILTER 3: Stochastic oversold
                if stoch_k > self.stoch_oversold:
                    continue

                # FILTER 4: Kaufman ER
                er = self._kaufman_er(close_arr, idx, self.er_period)
                if er != er or er < self.er_threshold:
                    continue

                stop_price = close * (1 - self.stop_loss_pct / 100)
                score = (self.stoch_oversold - stoch_k) + er * 50

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=score,
                    metadata={'stoch_k': stoch_k, 'kaufman_er': er}
                ))
            else:
                continue

        return signals

    def exit_rules(self):
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
