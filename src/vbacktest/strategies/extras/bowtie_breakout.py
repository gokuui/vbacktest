"""Bow Tie Breakout strategy (Dave Landry).

The "Bow Tie" occurs when 3 EMAs (10,20,30) converge to within a tight
range then separate in bullish order. This signals the START of a new
trend impulse after a consolidation.

Different from TQ/Elder: those measure ongoing trend quality. Bow Tie
catches the INITIATION of a new trend leg — earlier entry point.

Entry conditions:
1. EMA convergence: max spread of EMA10/20/30 < convergence_pct% of price
   within last convergence_lookback bars
2. Current bullish divergence: EMA10 > EMA20 > EMA30
3. Stage 2: Close > SMA50 > SMA150
4. Close > EMA10 (price above all EMAs)
5. SMA50 rising

Exit: SMA10 trailing + stop + time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class BowtieBreakoutStrategy(Strategy):
    """Buy when 3 EMAs form a Bow Tie pattern (converge then diverge bullishly)."""

    def __init__(
        self,
        ema_fast: int = 10,
        ema_mid: int = 20,
        ema_slow_bt: int = 30,
        convergence_pct: float = 1.5,
        convergence_lookback: int = 10,
        ma_trend: int = 50,
        ma_slow: int = 150,
        stop_loss_pct: float = 4.0,
        max_holding_days: int = 21,
        trailing_exit_type: str = 'sma10',
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow_bt = ema_slow_bt
        self.convergence_pct = convergence_pct
        self.convergence_lookback = convergence_lookback
        self.ma_trend = ma_trend
        self.ma_slow = ma_slow
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ema_fast_col = f'ema_{ema_fast}'
        self.ema_mid_col = f'ema_{ema_mid}'
        self.ema_slow_bt_col = f'ema_{ema_slow_bt}'
        self.ma_trend_col = f'sma_{ma_trend}'
        self.ma_slow_col = f'sma_{ma_slow}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('ema', {'period': self.ema_fast}),
            IndicatorSpec('ema', {'period': self.ema_mid}),
            IndicatorSpec('ema', {'period': self.ema_slow_bt}),
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, self.ema_slow_bt) + self.convergence_lookback + 10

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
                    ef = arrays[self.ema_fast_col][idx]
                    em = arrays[self.ema_mid_col][idx]
                    es = arrays[self.ema_slow_bt_col][idx]
                    mt = arrays[self.ma_trend_col][idx]
                    ms = arrays[self.ma_slow_col][idx]
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in [close, ef, em, es, mt, ms]):
                    continue
                if close <= 0:
                    continue

                # FILTER 1: Stage 2
                if not (close > mt > ms):
                    continue

                # FILTER 2: SMA50 rising
                mt_prev = arrays[self.ma_trend_col][idx - 5]
                if mt_prev != mt_prev or mt <= mt_prev:
                    continue

                # FILTER 3: Current bullish divergence: EMA10 > EMA20 > EMA30
                if not (ef > em > es):
                    continue

                # FILTER 4: Close above all EMAs
                if close < ef:
                    continue

                # FILTER 5: Recent convergence — within last N bars, EMAs were tight
                had_convergence = False
                for lb in range(1, self.convergence_lookback + 1):
                    lb_idx = idx - lb
                    if lb_idx < 0:
                        break
                    try:
                        lb_ef = arrays[self.ema_fast_col][lb_idx]
                        lb_em = arrays[self.ema_mid_col][lb_idx]
                        lb_es = arrays[self.ema_slow_bt_col][lb_idx]
                        lb_close = arrays['close'][lb_idx]
                    except (KeyError, IndexError):
                        break
                    if any(v != v for v in [lb_ef, lb_em, lb_es, lb_close]):
                        continue
                    if lb_close <= 0:
                        continue
                    # Max spread of 3 EMAs as % of close
                    ema_max = max(lb_ef, lb_em, lb_es)
                    ema_min = min(lb_ef, lb_em, lb_es)
                    spread_pct = (ema_max - ema_min) / lb_close * 100
                    if spread_pct < self.convergence_pct:
                        had_convergence = True
                        break

                if not had_convergence:
                    continue

                # Current spread (for scoring — wider = stronger divergence)
                curr_spread = (ef - es) / close * 100

                stop_price = close * (1 - self.stop_loss_pct / 100)

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=curr_spread,
                    metadata={'ema_spread': curr_spread}
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
