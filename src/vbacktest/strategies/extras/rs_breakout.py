"""Relative Strength Breakout strategy.

Edge: Buys stocks outperforming Nifty 50 index, not just running fast in
absolute terms. A stock up 15% when market is up 20% is weak. A stock up
15% when market is flat is a monster.

Entry conditions (ALL must be true):
1. RS rating: stock_roc(63d) / nifty_roc(63d) > 1.5  (50% stronger than market)
2. Close > SMA200  (uptrend filter)
3. New 50-day high  (breakout)
4. Volume > 1.5x 20-day avg  (confirming breakout)

Exit: ATR trailing stop + 60-day time stop (same as locked momentum)

Score: RS rating (higher = more outperformance vs market)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TimeStopRule


class RSBreakoutStrategy(Strategy):
    """Relative Strength Breakout — stocks outperforming Nifty 50."""

    def __init__(
        self,
        rs_period: int = 63,          # ~3 months for RS rating
        rs_min_ratio: float = 1.5,    # stock must be 50% stronger than market
        ma_trend: int = 200,          # SMA200 uptrend filter
        high_period: int = 50,        # new 50-day high breakout
        volume_avg_period: int = 20,
        volume_multiplier: float = 1.5,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 60,
        require_positive_nifty: bool = False,  # block entries when Nifty ROC <= 0
    ):
        self.rs_period = rs_period
        self.rs_min_ratio = rs_min_ratio
        self.ma_trend = ma_trend
        self.high_period = high_period
        self.volume_avg_period = volume_avg_period
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.require_positive_nifty = require_positive_nifty

        self.ma_col = f'sma_{ma_trend}'
        self.high_col = f'rolling_high_{high_period}'
        self.vol_avg_col = f'volume_sma_{volume_avg_period}'
        self.atr_col = f'atr_{atr_period}'
        self.roc_col = f'roc_{rs_period}'

        # Pre-load Nifty 50 close prices for RS calculation
        self._nifty_roc: dict[pd.Timestamp, float] = {}
        self._load_nifty_data()

    def _load_nifty_data(self):
        """Pre-load Nifty 50 ROC values for each trading day."""
        # Try multiple possible locations
        candidates = [
            Path(__file__).parent.parent.parent.parent / "data/validated/^nsei.parquet",
            Path("data/validated/^nsei.parquet"),
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_parquet(path)
                df = df.sort_values('date').reset_index(drop=True)
                close = df['close'].values
                dates = pd.to_datetime(df['date']).values

                # Pre-compute rolling ROC for rs_period
                for i in range(self.rs_period, len(close)):
                    if close[i - self.rs_period] > 0:
                        roc = (close[i] / close[i - self.rs_period] - 1) * 100
                        self._nifty_roc[pd.Timestamp(dates[i]).normalize()] = roc
                return

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('rolling_high', {'period': self.high_period, 'output_col': self.high_col}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.vol_avg_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('roc', {'period': self.rs_period, 'output_col': self.roc_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        # Get Nifty ROC for today
        nifty_roc = self._nifty_roc.get(date.normalize())
        if nifty_roc is None:
            return signals  # No benchmark data for today

        # Regime gate: if Nifty ROC is negative, skip all entries for the day
        if self.require_positive_nifty and nifty_roc <= 0:
            return signals

        for symbol, df in ctx.universe.items():
            if symbol.startswith('^'):
                continue
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_trend, self.rs_period) + 1:
                continue

            required = [self.ma_col, self.high_col, self.vol_avg_col, self.atr_col, self.roc_col]
            if not all(c in df.columns for c in required):
                continue

            # Fast path via ctx.universe_arrays
            if ctx.universe_arrays and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    ma = arrays[self.ma_col][idx]
                    high_50d = arrays[self.high_col][idx]
                    vol_avg = arrays[self.vol_avg_col][idx]
                    volume = arrays['volume'][idx]
                    atr = arrays[self.atr_col][idx]
                    stock_roc = arrays[self.roc_col][idx]
                except (KeyError, IndexError):
                    continue
            else:
                bar = df.iloc[idx]
                close = bar['close']
                ma = bar[self.ma_col]
                high_50d = bar[self.high_col]
                vol_avg = bar[self.vol_avg_col]
                volume = bar['volume']
                atr = bar[self.atr_col]
                stock_roc = bar[self.roc_col]

            if any(np.isnan(v) for v in [close, ma, high_50d, vol_avg, volume, atr, stock_roc]):
                continue

            # CONDITION 1: RS rating > threshold
            # When nifty is near 0, avoid division issues
            if abs(nifty_roc) < 0.5:
                # Market flat — only enter if stock strongly positive
                if stock_roc < 5.0:
                    continue
                rs_ratio = stock_roc / 0.5 if stock_roc > 0 else 0
            else:
                rs_ratio = stock_roc / nifty_roc

            if rs_ratio < self.rs_min_ratio:
                continue

            # CONDITION 2: Above SMA200
            if close <= ma:
                continue

            # CONDITION 3: New 50-day high
            if close < high_50d * 0.99:  # small tolerance
                continue

            # CONDITION 4: Volume surge
            if vol_avg <= 0 or volume < vol_avg * self.volume_multiplier:
                continue

            # Stop price
            stop_price = max(
                close - self.atr_stop_multiplier * atr,
                close * (1 - self.stop_loss_pct / 100)
            )

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=rs_ratio,
                metadata={
                    'rs_ratio': round(rs_ratio, 2),
                    'stock_roc_63d': round(stock_roc, 1),
                    'nifty_roc_63d': round(nifty_roc, 1),
                }
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
