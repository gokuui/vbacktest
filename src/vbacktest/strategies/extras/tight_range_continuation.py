"""Tight Range Continuation strategy.

After a strong move, stocks often consolidate briefly before continuing.
This strategy captures that continuation by looking for tight consolidation
AFTER an initial momentum burst.

Different from existing strategies:
- Minervini: VCP is long consolidation (weeks), this is short (3-5 days)
- Momentum: uses absolute ROC thresholds, this uses relative range contraction
- Donchian/ADX: channel/trend breakouts, this is continuation WITHIN trend

Entry conditions:
1. Recent strong move: ROC(5) > 8% (big burst in last week)
2. Tight consolidation: last 3 bars' range < 3% of price (digesting move)
3. Stage 2: Price > SMA50 > SMA200
4. Near 52-week high: within 15% of 52w high
5. ATR contracting: recent < earlier (confirming consolidation)
6. Breakout from consolidation: close > max(high of last 3 bars)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class TightRangeContinuationStrategy(Strategy):
    """Tight Range Continuation — buy consolidation breakout after strong move."""

    def __init__(
        self,
        burst_period: int = 5,
        burst_min_pct: float = 8.0,
        consolidation_bars: int = 3,
        max_range_pct: float = 3.0,
        ma_fast: int = 50,
        ma_slow: int = 200,
        near_high_pct: float = 15.0,
        atr_period: int = 14,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.burst_period = burst_period
        self.burst_min_pct = burst_min_pct
        self.consolidation_bars = consolidation_bars
        self.max_range_pct = max_range_pct
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.near_high_pct = near_high_pct
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
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
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, self.burst_period + self.consolidation_bars + 30)

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < min_bars:
                continue

            required = [self.ma_fast_col, self.ma_slow_col, self.atr_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close_arr  = arrays['close']
                    high_arr   = arrays['high']
                    low_arr    = arrays['low']
                    close      = close_arr[idx]
                    ma_fast    = arrays[self.ma_fast_col][idx]
                    ma_slow    = arrays[self.ma_slow_col][idx]
                    high_52w   = arrays[self.high_52w_col][idx]
                except (KeyError, IndexError):
                    continue
                if close != close or ma_fast != ma_fast or ma_slow != ma_slow or high_52w != high_52w:
                    continue

                # FILTER 1: Stage 2 uptrend
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 2: Recent strong burst
                burst_end_i   = idx - self.consolidation_bars
                burst_start_i = burst_end_i - self.burst_period
                if burst_start_i < 0:
                    continue
                b_start = close_arr[burst_start_i]
                b_end   = close_arr[burst_end_i]
                if b_start != b_start or b_start == 0:
                    continue
                burst_pct = (b_end - b_start) / b_start * 100
                if burst_pct < self.burst_min_pct:
                    continue

                # FILTER 3: Tight consolidation (vectorized numpy)
                consol_start = idx - self.consolidation_bars
                consol_highs = high_arr[consol_start:idx]
                consol_lows  = low_arr[consol_start:idx]
                consol_max   = consol_highs.max()
                consol_min   = consol_lows.min()
                if consol_min == 0:
                    continue
                consol_range_pct = (consol_max - consol_min) / consol_min * 100
                if consol_range_pct > self.max_range_pct:
                    continue

                # FILTER 4: Breakout above consolidation high
                if close <= consol_max:
                    continue

                # FILTER 5: Near 52-week high
                if high_52w <= 0:
                    continue
                price_vs_high = close / high_52w * 100
                if price_vs_high < (100 - self.near_high_pct):
                    continue

            else:
                current = df.iloc[idx]
                if any(pd.isna(current[col]) for col in required):
                    continue

                # FILTER 1: Stage 2 uptrend
                if not (current['close'] > current[self.ma_fast_col] > current[self.ma_slow_col]):
                    continue

                # FILTER 2: Recent strong burst
                burst_end_idx   = idx - self.consolidation_bars
                burst_start_idx = burst_end_idx - self.burst_period
                if burst_start_idx < 0:
                    continue
                burst_start_close = df.iloc[burst_start_idx]['close']
                burst_end_close   = df.iloc[burst_end_idx]['close']
                if pd.isna(burst_start_close) or burst_start_close == 0:
                    continue
                burst_pct = (burst_end_close - burst_start_close) / burst_start_close * 100
                if burst_pct < self.burst_min_pct:
                    continue

                # FILTER 3: Tight consolidation (Python loop fallback)
                consol_highs = []
                consol_lows  = []
                for i in range(self.consolidation_bars):
                    bar = df.iloc[idx - self.consolidation_bars + i]
                    consol_highs.append(bar['high'])
                    consol_lows.append(bar['low'])
                if not consol_highs:
                    continue
                consol_max = max(consol_highs)
                consol_min = min(consol_lows)
                if consol_min == 0:
                    continue
                consol_range_pct = (consol_max - consol_min) / consol_min * 100
                if consol_range_pct > self.max_range_pct:
                    continue

                # FILTER 4: Breakout above consolidation high
                if current['close'] <= consol_max:
                    continue

                # FILTER 5: Near 52-week high
                price_vs_high = current['close'] / current[self.high_52w_col] * 100
                if price_vs_high < (100 - self.near_high_pct):
                    continue

                close = current['close']

            stop_price = close * (1 - self.stop_loss_pct / 100)
            score      = burst_pct

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'burst_pct': burst_pct,
                    'consol_range_pct': consol_range_pct,
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
