"""Keltner Squeeze Breakout strategy.

Signal: Bollinger Bands inside Keltner Channels (squeeze) → expansion breakout.
When BB contracts inside KC, volatility is compressed. The breakout from this
compression tends to be explosive and directional.

This is a VOLATILITY signal (not price/volume/momentum) — different from all
existing strategies, providing uncorrelation.

Entry conditions (ALL must be true):
1. Uptrend: Price > SMA50 (trend filter)
2. Squeeze detected: BB was inside KC within last 3 bars (compressed volatility)
3. Expansion: BB now outside KC (squeeze released)
4. Direction: Close > Keltner mid AND ADX > 20 (breakout is upward + trending)
5. Volume confirmation: Volume ≥ 1.3x average
6. Momentum: Close > Close 5 bars ago (price actually moving up)

Exit conditions:
1. Hard stop: 6% below entry
2. Trailing EMA10 exit
3. Time stop: 14 days max

NSE Winning Formula compliance:
- 6 entry filters ✓
- 6% stop ✓
- 14-day time stop ✓
- EMA10 trailing exit ✓
- Volume confirmation ✓
- Uptrend filter ✓
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class KeltnerSqueezeStrategy(Strategy):
    """Keltner Squeeze breakout — volatility compression → expansion."""

    def __init__(
        self,
        ma_trend: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_ema_period: int = 20,
        kc_atr_period: int = 10,
        kc_atr_mult: float = 1.5,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        squeeze_lookback: int = 3,
        volume_surge: float = 1.3,
        momentum_bars: int = 5,
        stop_loss_pct: float = 6.0,
        time_stop_days: int = 14,
    ):
        self.ma_trend = ma_trend
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_ema_period = kc_ema_period
        self.kc_atr_period = kc_atr_period
        self.kc_atr_mult = kc_atr_mult
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.squeeze_lookback = squeeze_lookback
        self.volume_surge = volume_surge
        self.momentum_bars = momentum_bars
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_trend_col = f'sma_{ma_trend}'
        self.adx_col = f'adx_{adx_period}'
        self.volume_avg_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('ema', {'period': 10}),  # For trailing exit
            IndicatorSpec('bollinger_bands', {
                'period': self.bb_period,
                'num_std': self.bb_std,
            }),
            IndicatorSpec('keltner_channel', {
                'ema_period': self.kc_ema_period,
                'atr_period': self.kc_atr_period,
                'atr_multiplier': self.kc_atr_mult,
            }),
            IndicatorSpec('adx', {'period': self.adx_period}),
            IndicatorSpec('volume_sma', {'period': 20, 'output_col': self.volume_avg_col}),
        ]

    def _is_squeeze(self, bar) -> bool:
        """Check if BB is inside KC (squeeze condition)."""
        return (bar['bb_upper'] < bar['keltner_upper'] and
                bar['bb_lower'] > bar['keltner_lower'])

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_trend, self.bb_period, self.kc_ema_period) + self.squeeze_lookback + 5:
                continue

            required = [self.ma_trend_col, self.adx_col, self.volume_avg_col,
                       'bb_upper', 'bb_lower', 'keltner_upper', 'keltner_lower',
                       'keltner_mid']
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]

            if any(pd.isna(current[col]) for col in required):
                continue

            # CONDITION 1: Uptrend
            if current['close'] <= current[self.ma_trend_col]:
                continue

            # CONDITION 2: Recent squeeze (BB inside KC in lookback)
            was_squeezed = False
            for j in range(1, self.squeeze_lookback + 1):
                bar_j = df.iloc[idx - j]
                if any(pd.isna(bar_j[c]) for c in ['bb_upper', 'bb_lower', 'keltner_upper', 'keltner_lower']):
                    continue
                if self._is_squeeze(bar_j):
                    was_squeezed = True
                    break

            if not was_squeezed:
                continue

            # CONDITION 3: Expansion (BB now outside KC = squeeze released)
            if self._is_squeeze(current):
                continue  # Still in squeeze

            # CONDITION 4: Upward direction + ADX trend
            if current['close'] <= current['keltner_mid']:
                continue
            if current[self.adx_col] < self.adx_threshold:
                continue

            # CONDITION 5: Volume confirmation
            if current[self.volume_avg_col] <= 0:
                continue
            vol_ratio = current['volume'] / current[self.volume_avg_col]
            if vol_ratio < self.volume_surge:
                continue

            # CONDITION 6: Momentum (price rising)
            if idx < self.momentum_bars:
                continue
            past_close = df.iloc[idx - self.momentum_bars]['close']
            if current['close'] <= past_close:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: ADX strength * volume * squeeze duration
            score = current[self.adx_col] * vol_ratio

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'adx': current[self.adx_col],
                    'volume_ratio': vol_ratio,
                    'bb_upper': current['bb_upper'],
                    'kc_upper': current['keltner_upper'],
                }
            ))

        return signals

    def exit_rules(self) -> list:
        from vbacktest.exit_rules import TrailingMARule
        return [
            StopLossRule(),
            TrailingMARule(ma_column='ema_10'),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
