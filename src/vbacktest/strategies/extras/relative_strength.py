"""Relative Strength Breakout strategy.

Entry based on stocks showing exceptional relative strength — making
new highs with strong multi-timeframe momentum, while also being in
a clear Stage 2 uptrend. Different from Momentum (which uses absolute
ROC thresholds) and Minervini (which uses VCP contraction).

Entry conditions:
1. Close at new 50-day high (breakout)
2. ROC(20) > 15% (short-term momentum very strong)
3. ROC(60) > ROC(20) (accelerating — longer-term even stronger)
4. Price > SMA(50) > SMA(200) (Stage 2 uptrend)
5. RSI(14) > 60 (momentum confirmed, not overbought < 85)
6. Close > prior close (positive action)

Exit: SMA10 trailing + 5% stop + 21-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class RelativeStrengthStrategy(Strategy):
    """Relative Strength Breakout — buy stocks with accelerating momentum."""

    def __init__(
        self,
        high_period: int = 50,
        roc_short: int = 20,
        roc_long: int = 60,
        roc_min: float = 15.0,
        ma_fast: int = 50,
        ma_slow: int = 200,
        rsi_period: int = 14,
        rsi_min: float = 60.0,
        rsi_max: float = 85.0,
        atr_period: int = 14,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 21,
        trailing_exit_type: str = 'sma10',
    ):
        self.high_period = high_period
        self.roc_short = roc_short
        self.roc_long = roc_long
        self.roc_min = roc_min
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.high_col = f'rolling_high_{high_period}'
        self.roc_short_col = f'roc_{roc_short}'
        self.roc_long_col = f'roc_{roc_long}'
        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.rsi_col = 'rsi'
        self.atr_col = f'atr_{atr_period}'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('rolling_high', {'period': self.high_period, 'output_col': self.high_col}),
            IndicatorSpec('roc', {'period': self.roc_short, 'output_col': self.roc_short_col}),
            IndicatorSpec('roc', {'period': self.roc_long, 'output_col': self.roc_long_col}),
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('rsi', {'period': self.rsi_period}),
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
            if idx < max(self.ma_slow, self.roc_long) + 10:
                continue

            required = [self.high_col, self.roc_short_col, self.roc_long_col,
                        self.ma_fast_col, self.ma_slow_col, self.rsi_col, self.atr_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: New 50-day high breakout
            # Compare to previous bar's high (rolling_high includes current bar)
            prev_high = df.iloc[idx - 1][self.high_col]
            if pd.isna(prev_high):
                continue
            if current['close'] <= prev_high:
                continue

            # FILTER 2: Short-term ROC strong
            if current[self.roc_short_col] < self.roc_min:
                continue

            # FILTER 3: Accelerating momentum (long ROC > short ROC)
            if current[self.roc_long_col] <= current[self.roc_short_col]:
                continue

            # FILTER 4: Stage 2 uptrend (price > MA50 > MA200)
            if not (current['close'] > current[self.ma_fast_col] > current[self.ma_slow_col]):
                continue

            # FILTER 5: RSI in strong momentum zone
            if current[self.rsi_col] < self.rsi_min or current[self.rsi_col] > self.rsi_max:
                continue

            # FILTER 6: Positive close
            if current['close'] <= prev['close']:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: combined momentum strength
            score = current[self.roc_long_col] * 0.5 + current[self.roc_short_col] * 0.3 + current[self.rsi_col] * 0.2

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'roc_short': current[self.roc_short_col],
                    'roc_long': current[self.roc_long_col],
                    'rsi': current[self.rsi_col],
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
