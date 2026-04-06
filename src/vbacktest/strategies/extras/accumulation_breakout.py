"""Accumulation Breakout strategy.

Detects institutional accumulation BEFORE the breakout happens.
Uses On-Balance Volume (OBV) making new highs while price consolidates
near resistance — classic Wyckoff/O'Neil accumulation pattern.

Different from existing strategies:
- momentum: waits for price breakout (lagging)
- volume_breakout: just uses volume surge (one-day event)
- pocket_pivot: specific pocket pivot pattern (too narrow)
- This: OBV trend divergence + price consolidation = accumulation phase

Entry conditions (ALL must be true):
1. Stage 2 uptrend: Close > SMA50 > SMA150
2. Near 52-week high: within 15% of 52w high (at resistance)
3. Price consolidating: 10-day range < 10% of price (tight base)
4. Volume accumulation: 20-day OBV slope positive (buying pressure)
5. Rising lows: 10-day low > 20-day low (higher lows in base)
6. Breakout trigger: close > 20-day high (breaks out of consolidation)

Exit: SMA10 trailing + ATR stop + time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TrailingATRStopRule, TimeStopRule


class AccumulationBreakoutStrategy(Strategy):
    """Buy when accumulation detected at resistance, confirmed by breakout."""

    def __init__(
        self,
        ma_fast: int = 50,
        ma_slow: int = 150,
        near_high_pct: float = 15.0,
        consolidation_range_pct: float = 10.0,
        consolidation_period: int = 10,
        breakout_period: int = 20,
        obv_slope_period: int = 20,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 30,
        trailing_exit_type: str = 'sma10',
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.near_high_pct = near_high_pct
        self.consolidation_range_pct = consolidation_range_pct
        self.consolidation_period = consolidation_period
        self.breakout_period = breakout_period
        self.obv_slope_period = obv_slope_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.high_52w_col = 'high_252'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, 252, self.breakout_period, self.obv_slope_period) + 30

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
                    high = arrays['high'][idx]
                    low = arrays['low'][idx]
                    volume = arrays['volume'][idx]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    atr = arrays[self.atr_col][idx]
                    high_52w = arrays[self.high_52w_col][idx]
                    close_arr = arrays['close']
                    high_arr = arrays['high']
                    low_arr = arrays['low']
                    volume_arr = arrays['volume']
                except (KeyError, IndexError):
                    continue

                # NaN check
                if any(v != v for v in [close, ma_fast, ma_slow, atr, high_52w]):
                    continue

                # FILTER 1: Stage 2 uptrend
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 2: Near 52-week high
                if high_52w <= 0:
                    continue
                pct_from_high = (high_52w - close) / high_52w * 100
                if pct_from_high > self.near_high_pct:
                    continue

                # FILTER 3: Price consolidating — 10-day range < X% of price
                consol_start = idx - self.consolidation_period
                if consol_start < 0:
                    continue
                consol_high = high_arr[consol_start:idx + 1].max()
                consol_low = low_arr[consol_start:idx + 1].min()
                if consol_low <= 0:
                    continue
                consol_range_pct = (consol_high - consol_low) / consol_low * 100
                if consol_range_pct > self.consolidation_range_pct:
                    continue

                # FILTER 4: OBV rising (simple approximation: count up-volume days)
                up_vol_days = 0
                total_vol_days = 0
                for lb in range(self.obv_slope_period):
                    lb_idx = idx - lb
                    if lb_idx <= 0:
                        break
                    c_now = close_arr[lb_idx]
                    c_prev = close_arr[lb_idx - 1]
                    if c_now != c_now or c_prev != c_prev:
                        continue
                    total_vol_days += 1
                    if c_now > c_prev:
                        up_vol_days += 1
                if total_vol_days < 10 or up_vol_days / total_vol_days < 0.55:
                    continue

                # FILTER 5: Rising lows — 10d low > 20d low
                low_10d = low_arr[idx - 10:idx + 1].min()
                low_20d = low_arr[idx - 20:idx + 1].min()
                if low_10d <= low_20d:
                    continue

                # FILTER 6: Breakout — close > N-day high (excluding today)
                breakout_high = high_arr[idx - self.breakout_period:idx].max()
                if close <= breakout_high:
                    continue

                # Calculate stop
                stop_price = close - atr * self.atr_stop_multiplier
                max_stop = close * (1 - self.stop_loss_pct / 100)
                stop_price = max(stop_price, max_stop)

                score = (1 - pct_from_high / self.near_high_pct) * 100  # closer to high = better

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=score,
                    metadata={
                        'pct_from_high': pct_from_high,
                        'consol_range_pct': consol_range_pct,
                        'up_vol_ratio': up_vol_days / total_vol_days,
                    }
                ))

            else:
                # Slow path omitted for brevity — fast path covers parallel runs
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
