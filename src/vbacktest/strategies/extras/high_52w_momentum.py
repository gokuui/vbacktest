"""52-Week High Momentum strategy.

Edge: Research (George & Hwang 2004) shows hitting a new 52-week high is one
of the strongest predictors of future outperformance. Anchoring bias makes
traders hesitant to buy near 52W highs — causing systematic underreaction.

Entry conditions (ALL must be true):
1. New 52-week high today: close > highest close in prior 252 days
2. Volume > 1.5x 20-day avg (confirming breakout, not a thin-volume spike)
3. Close > SMA50 (minimum trend filter)
4. ROC(20) > 5% (already moving, not a false breakout)

Exit: 8% hard stop + trailing ATR stop + 60-day time stop

Score: ROC(20) — rank by recent momentum strength
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TimeStopRule


class High52WMomentumStrategy(Strategy):
    """New 52-week high breakout with volume confirmation."""

    def __init__(
        self,
        lookback_days: int = 252,      # 52 weeks
        ma_trend: int = 50,            # SMA50 trend filter
        volume_avg_period: int = 20,
        volume_multiplier: float = 1.5,
        roc_period: int = 20,
        roc_min: float = 5.0,          # already moving up
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        stop_loss_pct: float = 8.0,    # wider stop — 52W highs can be volatile
        max_holding_days: int = 60,
        min_price: float = 0.0,
        min_avg_volume: float = 0.0,
    ):
        self.lookback_days = lookback_days
        self.ma_trend = ma_trend
        self.volume_avg_period = volume_avg_period
        self.volume_multiplier = volume_multiplier
        self.roc_period = roc_period
        self.roc_min = roc_min
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume

        self.ma_col = f'sma_{ma_trend}'
        self.vol_avg_col = f'volume_sma_{volume_avg_period}'
        self.roc_col = f'roc_{roc_period}'
        self.high_col = f'rolling_high_{lookback_days}'
        self.atr_col = f'atr_{atr_period}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.vol_avg_col}),
            IndicatorSpec('roc', {'period': self.roc_period, 'output_col': self.roc_col}),
            IndicatorSpec('rolling_high', {'period': self.lookback_days, 'output_col': self.high_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if symbol.startswith('^'):
                continue
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < self.lookback_days + 5:
                continue

            required = [self.ma_col, self.vol_avg_col, self.roc_col, self.high_col, self.atr_col]
            if not all(c in df.columns for c in required):
                continue

            if ctx.universe_arrays and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    ma = arrays[self.ma_col][idx]
                    vol_avg = arrays[self.vol_avg_col][idx]
                    volume = arrays['volume'][idx]
                    roc = arrays[self.roc_col][idx]
                    # 52W high is the highest close in prior 252 days (not including today)
                    high_52w = arrays[self.high_col][idx - 1] if idx > 0 else arrays[self.high_col][idx]
                    atr = arrays[self.atr_col][idx]
                except (KeyError, IndexError):
                    continue
            else:
                bar = df.iloc[idx]
                close = bar['close']
                ma = bar[self.ma_col]
                vol_avg = bar[self.vol_avg_col]
                volume = bar['volume']
                roc = bar[self.roc_col]
                high_52w = df[self.high_col].iloc[idx - 1] if idx > 0 else bar[self.high_col]
                atr = bar[self.atr_col]

            if any(np.isnan(v) for v in [close, ma, vol_avg, volume, roc, high_52w, atr]):
                continue

            # Price filter
            if self.min_price > 0 and close < self.min_price:
                continue

            # Volume filter
            if self.min_avg_volume > 0 and vol_avg < self.min_avg_volume:
                continue

            # CONDITION 1: New 52-week high (close > prior 252-day high)
            if close <= high_52w:
                continue

            # CONDITION 2: Volume confirmation
            if vol_avg <= 0 or volume < vol_avg * self.volume_multiplier:
                continue

            # CONDITION 3: Above SMA50
            if close <= ma:
                continue

            # CONDITION 4: Already moving — ROC(20) > 5%
            if roc < self.roc_min:
                continue

            stop_price = max(
                close - self.atr_stop_multiplier * atr,
                close * (1 - self.stop_loss_pct / 100)
            )

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=roc,
                metadata={'roc_20d': round(roc, 1), 'new_52w_high': True}
            ))

        return signals

    def exit_rules(self) -> list:
        return [
            StopLossRule(),
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier,
            ),
            TimeStopRule(max_holding_days=self.max_holding_days),
        ]
