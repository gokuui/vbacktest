"""VCP (Volatility Contraction Pattern) Breakout strategy.

Implements Minervini's VCP concept using raw indicators:
- Detect progressively tighter pullbacks (contracting volatility)
- Enter when price breaks above the consolidation with volume

This is DIFFERENT from existing minervini_sepa which uses a Minervini
template score. This directly measures volatility contraction by comparing
recent ATR to prior ATR — the actual VCP mechanism.

Entry conditions:
1. Stage 2: Close > SMA50 > SMA150, SMA50 rising
2. Volatility contraction: ATR(5) < ATR(20) < ATR(50) (progressively tighter)
3. Close > EMA10 (short-term momentum)
4. Near 50-day high (within 5%)
5. Volume above average on breakout day

Exit: SMA10 trailing + stop + time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class VCPBreakoutStrategy(Strategy):
    """Buy when volatility contracts progressively — VCP pattern."""

    def __init__(
        self,
        ma_fast: int = 50,
        ma_slow: int = 150,
        breakout_period: int = 50,
        breakout_pct: float = 5.0,
        contraction_ratio: float = 0.8,  # ATR(5)/ATR(20) must be < this
        volume_mult: float = 1.2,
        atr_period_short: int = 5,
        atr_period_mid: int = 20,
        atr_period_long: int = 50,
        stop_loss_pct: float = 4.0,
        max_holding_days: int = 21,
        trailing_exit_type: str = 'sma10',
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.breakout_period = breakout_period
        self.breakout_pct = breakout_pct
        self.contraction_ratio = contraction_ratio
        self.volume_mult = volume_mult
        self.atr_period_short = atr_period_short
        self.atr_period_mid = atr_period_mid
        self.atr_period_long = atr_period_long
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_short_col = f'atr_{atr_period_short}'
        self.atr_mid_col = f'atr_{atr_period_mid}'
        self.atr_long_col = f'atr_{atr_period_long}'
        self.vol_sma_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('ema', {'period': 10}),
            IndicatorSpec('atr', {'period': self.atr_period_short, 'output_col': self.atr_short_col}),
            IndicatorSpec('atr', {'period': self.atr_period_mid, 'output_col': self.atr_mid_col}),
            IndicatorSpec('atr', {'period': self.atr_period_long, 'output_col': self.atr_long_col}),
            IndicatorSpec('rolling_high', {'period': self.breakout_period, 'output_col': f'high_{self.breakout_period}'}),
            IndicatorSpec('volume_sma', {'period': 20}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, self.atr_period_long, self.breakout_period) + 30

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
                    volume = arrays['volume'][idx]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    ema10 = arrays['ema_10'][idx]
                    atr_s = arrays[self.atr_short_col][idx]
                    atr_m = arrays[self.atr_mid_col][idx]
                    atr_l = arrays[self.atr_long_col][idx]
                    high_n = arrays[f'high_{self.breakout_period}'][idx]
                    vol_sma = arrays[self.vol_sma_col][idx]
                except (KeyError, IndexError):
                    continue

                vals = [close, ma_fast, ma_slow, ema10, atr_s, atr_m, atr_l, high_n, vol_sma]
                if any(v != v for v in vals):
                    continue
                if close <= 0 or atr_m <= 0 or atr_l <= 0:
                    continue

                # FILTER 1: Stage 2 uptrend
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 2: SMA50 rising
                ma_fast_prev = arrays[self.ma_fast_col][idx - 5]
                if ma_fast_prev != ma_fast_prev or ma_fast <= ma_fast_prev:
                    continue

                # FILTER 3: Short-term momentum
                if close < ema10:
                    continue

                # FILTER 4: Volatility contraction — ATR(5) < ATR(20) < ATR(50)
                if not (atr_s < atr_m * self.contraction_ratio):
                    continue
                if not (atr_m < atr_l):
                    continue

                # FILTER 5: Near N-day high
                if high_n <= 0:
                    continue
                pct_from_high = (high_n - close) / high_n * 100
                if pct_from_high > self.breakout_pct:
                    continue

                # FILTER 6: Volume above average
                if vol_sma <= 0 or volume < vol_sma * self.volume_mult:
                    continue

                stop_price = close * (1 - self.stop_loss_pct / 100)
                # Score: tighter contraction = better
                contraction_score = (1 - atr_s / atr_m) * 100

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=contraction_score,
                    metadata={'atr_contraction': atr_s / atr_m, 'pct_from_high': pct_from_high}
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
