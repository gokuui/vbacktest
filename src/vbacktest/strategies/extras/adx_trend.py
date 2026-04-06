"""ADX Trend Strength Breakout strategy.

Entry based on ADX trend strength measurement — fundamentally different
from ROC-based (Momentum) or MA hierarchy (Minervini) approaches.

Entry conditions:
1. ADX(14) > 25 (strong trend developing)
2. ADX rising (today > 5 days ago)
3. +DI > -DI (bullish direction)
4. Price > SMA(50) (uptrend confirmation)
5. ROC(10) > 5% (momentum confirmation)
6. Close > prior close (positive action)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class ADXTrendStrategy(Strategy):
    """ADX Trend Strength Breakout — buy when trend strength confirms."""

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        adx_rising_lookback: int = 5,
        trend_ma_period: int = 50,
        roc_period: int = 10,
        roc_threshold: float = 5.0,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.adx_rising_lookback = adx_rising_lookback
        self.trend_ma_period = trend_ma_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.adx_col = f'adx_{adx_period}'
        self.plus_di_col = f'plus_di_{adx_period}'
        self.minus_di_col = f'minus_di_{adx_period}'
        self.trend_ma_col = f'sma_{trend_ma_period}'
        self.roc_col = f'roc_{roc_period}'
        self.atr_col = f'atr_{adx_period}'
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('adx', {'period': self.adx_period, 'output_col': self.adx_col}),
            IndicatorSpec('sma', {'period': self.trend_ma_period}),
            IndicatorSpec('roc', {'period': self.roc_period, 'output_col': self.roc_col}),
            IndicatorSpec('atr', {'period': self.adx_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
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
            if idx < max(self.trend_ma_period, self.adx_rising_lookback + self.adx_period + 10):
                continue

            required = [self.adx_col, self.plus_di_col, self.minus_di_col,
                        self.trend_ma_col, self.roc_col, self.atr_col,
                        self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: ADX above threshold (strong trend)
            if current[self.adx_col] < self.adx_threshold:
                continue

            # FILTER 2: ADX rising (trend strengthening)
            adx_past = df.iloc[idx - self.adx_rising_lookback][self.adx_col]
            if pd.isna(adx_past) or current[self.adx_col] <= adx_past:
                continue

            # FILTER 3: +DI > -DI (bullish direction)
            if current[self.plus_di_col] <= current[self.minus_di_col]:
                continue

            # FILTER 4: Price above trend MA (uptrend)
            if current['close'] < current[self.trend_ma_col]:
                continue

            # FILTER 5: ROC above threshold (momentum)
            if current[self.roc_col] < self.roc_threshold:
                continue

            # FILTER 6: Positive close (action confirmation)
            if current['close'] <= prev['close']:
                continue

            # FILTER 7: Near 52-week high (within 20% — strength filter)
            if current['close'] < current[self.high_52w_col] * 0.80:
                continue

            # FILTER 8: ATR contraction (volatility tightening before breakout)
            if idx >= 30:
                recent_atr = df[self.atr_col].iloc[idx-10:idx].mean()
                earlier_atr = df[self.atr_col].iloc[idx-30:idx-10].mean()
                if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
                    continue
                if recent_atr >= earlier_atr * 0.8:
                    continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: ADX strength + ROC
            score = current[self.adx_col] * 0.5 + current[self.roc_col] * 0.5

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'adx': current[self.adx_col],
                    'plus_di': current[self.plus_di_col],
                    'minus_di': current[self.minus_di_col],
                    'roc': current[self.roc_col],
                    'atr': current[self.atr_col],
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
