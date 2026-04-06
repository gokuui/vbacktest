"""Pocket Pivot strategy — institutional accumulation detection.

Based on Chris Kacher/Gil Morales concept: volume on an up day exceeds
the highest volume on any down day in the prior 10 sessions.
This signals institutional buying without waiting for a breakout.

Entry conditions:
1. Up day: Close > Open (bullish candle)
2. Pocket pivot: Today's volume > max(down-day volumes in last 10 bars)
3. Stage 2 uptrend: Price > SMA50, SMA50 > SMA200
4. Price > SMA10 (not too extended from short-term trend)
5. Price within 15% of 52-week high (strong stock)
6. Close in upper 60% of daily range (strong close)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class PocketPivotStrategy(Strategy):
    """Pocket Pivot — detect institutional buying via volume signature."""

    def __init__(
        self,
        lookback: int = 10,
        ma_fast: int = 50,
        ma_slow: int = 200,
        near_high_pct: float = 15.0,
        close_position_pct: float = 60.0,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.lookback = lookback
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.near_high_pct = near_high_pct
        self.close_position_pct = close_position_pct / 100.0
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_slow, self.lookback) + 5:
                continue

            required = [self.ma_fast_col, self.ma_slow_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    open_ = arrays['open'][idx]
                    high = arrays['high'][idx]
                    low = arrays['low'][idx]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    sma10 = arrays['sma_10'][idx]
                    high_52w = arrays[self.high_52w_col][idx]
                    volume = arrays['volume'][idx]
                    # Vectorized pocket pivot: max down-day volume in lookback
                    close_slice = arrays['close'][idx - self.lookback:idx]
                    open_slice = arrays['open'][idx - self.lookback:idx]
                    vol_slice = arrays['volume'][idx - self.lookback:idx]
                except (KeyError, IndexError):
                    continue
                if close != close or ma_fast != ma_fast or ma_slow != ma_slow or high_52w != high_52w:
                    continue
                down_mask = close_slice < open_slice
                if not down_mask.any():
                    continue
                max_down_volume = vol_slice[down_mask].max()
            else:
                current = df.iloc[idx]
                if any(pd.isna(current[col]) for col in required):
                    continue
                close = current['close']
                open_ = current['open']
                high = current['high']
                low = current['low']
                ma_fast = current[self.ma_fast_col]
                ma_slow = current[self.ma_slow_col]
                sma10 = current['sma_10']
                high_52w = current[self.high_52w_col]
                volume = current['volume']
                max_down_volume = 0
                for j in range(1, self.lookback + 1):
                    prev_bar = df.iloc[idx - j]
                    if prev_bar['close'] < prev_bar['open']:
                        if prev_bar['volume'] > max_down_volume:
                            max_down_volume = prev_bar['volume']
                if max_down_volume == 0:
                    continue

            # FILTER 1: Up day (bullish candle)
            if close <= open_:
                continue

            # FILTER 2: Pocket Pivot — volume exceeds max down-day volume
            if volume <= max_down_volume:
                continue

            # FILTER 3: Stage 2 uptrend
            if not (close > ma_fast > ma_slow):
                continue

            # FILTER 4: Price above SMA10
            if close < sma10:
                continue

            # FILTER 5: Near 52-week high
            price_vs_high = close / high_52w * 100
            if price_vs_high < (100 - self.near_high_pct):
                continue

            # FILTER 6: Close in upper portion of range
            daily_range = high - low
            if daily_range <= 0:
                continue
            close_position = (close - low) / daily_range
            if close_position < self.close_position_pct:
                continue

            stop_price = close * (1 - self.stop_loss_pct / 100)
            volume_ratio = volume / max_down_volume
            score = volume_ratio + price_vs_high * 0.05

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'volume_ratio': volume_ratio,
                    'price_vs_high': price_vs_high,
                    'close_position': close_position,
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
