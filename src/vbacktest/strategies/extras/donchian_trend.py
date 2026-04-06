"""Donchian Trend Breakout strategy.

Entry based on Donchian channel breakout — pure price-channel extremes,
fundamentally different from ROC or MA crossover approaches.

Entry conditions:
1. Close > Donchian upper (50-day) — new 50-day high
2. Price > SMA(200) (long-term uptrend)
3. ADX(14) > 20 (trending market, not rangebound)
4. Volume > 1.2x 50-day average (participation)
5. ROC(20) > 0 (positive momentum)
6. ATR contraction: recent ATR < earlier ATR (tightening before breakout)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class DonchianTrendStrategy(Strategy):
    """Donchian Trend Breakout — buy on channel breakout with trend confirmation."""

    def __init__(
        self,
        donchian_period: int = 50,
        trend_ma_period: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 20.0,
        volume_avg_period: int = 50,
        volume_surge: float = 1.2,
        roc_period: int = 20,
        atr_period: int = 14,
        volatility_contraction_ratio: float = 0.8,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.donchian_period = donchian_period
        self.trend_ma_period = trend_ma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_avg_period = volume_avg_period
        self.volume_surge = volume_surge
        self.roc_period = roc_period
        self.atr_period = atr_period
        self.volatility_contraction_ratio = volatility_contraction_ratio
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.trend_ma_col = f'sma_{trend_ma_period}'
        self.adx_col = f'adx_{adx_period}'
        self.volume_avg_col = f'volume_sma_{volume_avg_period}'
        self.roc_col = f'roc_{roc_period}'
        self.atr_col = f'atr_{atr_period}'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('donchian_channel', {'period': self.donchian_period}),
            IndicatorSpec('sma', {'period': self.trend_ma_period}),
            IndicatorSpec('adx', {'period': self.adx_period, 'output_col': self.adx_col}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.volume_avg_col}),
            IndicatorSpec('roc', {'period': self.roc_period, 'output_col': self.roc_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('sma', {'period': 10}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.trend_ma_period, 30):
                continue

            required = ['don_upper', self.trend_ma_col, self.adx_col,
                        self.volume_avg_col, self.roc_col, self.atr_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: Donchian breakout — close above PREVIOUS bar's upper channel
            # (don_upper includes current bar's high, so we compare to prior bar)
            prev = df.iloc[idx - 1]
            if pd.isna(prev['don_upper']):
                continue
            if current['close'] <= prev['don_upper']:
                continue

            # FILTER 2: Long-term uptrend
            if current['close'] < current[self.trend_ma_col]:
                continue

            # FILTER 3: ADX trending (not rangebound)
            if current[self.adx_col] < self.adx_threshold:
                continue

            # FILTER 4: Volume participation
            if current['volume'] < current[self.volume_avg_col] * self.volume_surge:
                continue

            # FILTER 5: Positive momentum
            if current[self.roc_col] <= 0:
                continue

            # FILTER 6: ATR contraction (tightening before breakout)
            if idx >= 30:
                recent_atr = df[self.atr_col].iloc[idx-10:idx].mean()
                earlier_atr = df[self.atr_col].iloc[idx-30:idx-10].mean()
                if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
                    continue
                if recent_atr >= earlier_atr * self.volatility_contraction_ratio:
                    continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: breakout strength (how far above channel) + momentum
            breakout_strength = (current['close'] - current['don_upper']) / current['don_upper'] * 100
            score = breakout_strength + current[self.roc_col] * 0.3

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'don_upper': current['don_upper'],
                    'adx': current[self.adx_col],
                    'roc': current[self.roc_col],
                    'atr': current[self.atr_col],
                    'breakout_strength': breakout_strength,
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
