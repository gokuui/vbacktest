"""Filtered Momentum strategy — cross-sectional ranking WITH proven NSE filters.

Combines the best of cross-sectional ranking with the proven NSE winning formula:
- Rank by momentum (like Top Momentum) BUT
- Require ATR contraction (like Donchian/ADX)
- Require near 52-week high (like Minervini/ADX)
- Require Stage 2 uptrend (like all winners)

This should capture the high CAGR of momentum ranking while controlling DD
through the volatility contraction and quality filters.

Entry conditions:
1. Stage 2: Price > SMA50 > SMA200
2. ATR contraction: 10d ATR < 20d ATR (compression before breakout)
3. Near 52-week high: within 10% of 52w high
4. ROC(63) > 20% (strong 3-month momentum — minimum threshold)
5. Rank by ROC(63), buy top N ranked

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class FilteredMomentumStrategy(Strategy):
    """Filtered Momentum — momentum ranking with NSE quality filters."""

    def __init__(
        self,
        roc_period: int = 63,
        roc_min: float = 20.0,
        ma_fast: int = 50,
        ma_slow: int = 200,
        atr_period: int = 14,
        near_high_pct: float = 10.0,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.roc_period = roc_period
        self.roc_min = roc_min
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.atr_period = atr_period
        self.near_high_pct = near_high_pct
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
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        candidates = []

        use_fast = ctx.universe_arrays is not None

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_slow, 30) + 5:
                continue

            required = [self.roc_col, self.ma_fast_col, self.ma_slow_col,
                        self.atr_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            # Fast path: numpy array lookups
            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close      = arrays['close'][idx]
                    ma_fast    = arrays[self.ma_fast_col][idx]
                    ma_slow    = arrays[self.ma_slow_col][idx]
                    roc        = arrays[self.roc_col][idx]
                    high_52w   = arrays[self.high_52w_col][idx]
                    atr_arr    = arrays[self.atr_col]
                except (KeyError, IndexError):
                    continue

                if any(np.isnan(v) for v in [close, ma_fast, ma_slow, roc, high_52w]):
                    continue

                # FILTER 1: Stage 2 uptrend
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 2: Minimum momentum threshold
                if roc < self.roc_min:
                    continue

                # FILTER 3: Near 52-week high
                price_vs_high = close / high_52w * 100
                if price_vs_high < (100 - self.near_high_pct):
                    continue

                # FILTER 4: ATR contraction
                if idx >= 20:
                    recent_atr = atr_arr[idx-5:idx+1].mean()
                    earlier_atr = atr_arr[idx-20:idx-5].mean()
                    if np.isnan(recent_atr) or np.isnan(earlier_atr) or earlier_atr == 0:
                        continue
                    if recent_atr >= earlier_atr:
                        continue

                candidates.append((symbol, roc, close, high_52w))

            else:
                # Slow path: pandas iloc fallback
                current = df.iloc[idx]
                if any(pd.isna(current[col]) for col in required):
                    continue

                if not (current['close'] > current[self.ma_fast_col] > current[self.ma_slow_col]):
                    continue
                if current[self.roc_col] < self.roc_min:
                    continue

                price_vs_high = current['close'] / current[self.high_52w_col] * 100
                if price_vs_high < (100 - self.near_high_pct):
                    continue

                if idx >= 20:
                    recent_atr = df[self.atr_col].iloc[idx-5:idx+1].mean()
                    earlier_atr = df[self.atr_col].iloc[idx-20:idx-5].mean()
                    if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
                        continue
                    if recent_atr >= earlier_atr:
                        continue

                candidates.append((symbol, current[self.roc_col],
                                   current['close'], current[self.high_52w_col]))

        if not candidates:
            return signals

        # Rank by momentum — highest ROC first
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top candidates (2x max positions to give engine choices)
        for symbol, roc, close, high_52w in candidates[:20]:
            stop_price = close * (1 - self.stop_loss_pct / 100)
            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=roc,
                metadata={
                    'roc': roc,
                    'price_vs_high': close / high_52w * 100,
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
