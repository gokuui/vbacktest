"""EMA Stack Expansion Breakout strategy.

Entry based on EMA ribbon alignment + expansion — measures trend
acceleration, fundamentally different from SMA hierarchy (Minervini)
or ROC-based (Momentum) approaches.

Entry conditions:
1. EMA(8) > EMA(21) > EMA(50) (stacked/aligned)
2. EMA ribbon expanding (EMA8-EMA21 gap today > 5 days ago)
3. Price near 20-day high (within 2%)
4. RSI(14) between 50-80 (momentum zone, not overbought)
5. Close > prior close (positive action)
6. Volume > 0.8x average (not dry-up)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class EMAStackStrategy(Strategy):
    """EMA Stack Expansion Breakout — buy when EMA ribbon expands."""

    def __init__(
        self,
        ema_fast: int = 8,
        ema_mid: int = 21,
        ema_slow: int = 50,
        expansion_lookback: int = 5,
        high_period: int = 20,
        near_high_pct: float = 2.0,
        rsi_period: int = 14,
        rsi_min: float = 50.0,
        rsi_max: float = 80.0,
        volume_avg_period: int = 50,
        volume_min_ratio: float = 0.8,
        atr_period: int = 14,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.expansion_lookback = expansion_lookback
        self.high_period = high_period
        self.near_high_pct = near_high_pct
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.volume_avg_period = volume_avg_period
        self.volume_min_ratio = volume_min_ratio
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ema_fast_col = f'ema_{ema_fast}'
        self.ema_mid_col = f'ema_{ema_mid}'
        self.ema_slow_col = f'ema_{ema_slow}'
        self.high_col = f'rolling_high_{high_period}'
        self.high_50_col = 'rolling_high_50'
        self.rsi_col = f'rsi'
        self.volume_avg_col = f'volume_sma_{volume_avg_period}'
        self.atr_col = f'atr_{atr_period}'
        self.roc_col = f'roc_20'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('ema', {'period': self.ema_fast}),
            IndicatorSpec('ema', {'period': self.ema_mid}),
            IndicatorSpec('ema', {'period': self.ema_slow}),
            IndicatorSpec('rolling_high', {'period': self.high_period, 'output_col': self.high_col}),
            IndicatorSpec('rsi', {'period': self.rsi_period}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.volume_avg_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('roc', {'period': 20, 'output_col': self.roc_col}),
            IndicatorSpec('rolling_high', {'period': 50, 'output_col': self.high_50_col}),
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
            if idx < max(self.ema_slow, self.expansion_lookback + 10):
                continue

            required = [self.ema_fast_col, self.ema_mid_col, self.ema_slow_col,
                        self.high_col, self.high_50_col, self.rsi_col,
                        self.volume_avg_col, self.atr_col, self.roc_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: EMA stack (aligned uptrend)
            if not (current[self.ema_fast_col] > current[self.ema_mid_col] > current[self.ema_slow_col]):
                continue

            # FILTER 2: EMA ribbon expanding (trend accelerating)
            past = df.iloc[idx - self.expansion_lookback]
            if pd.isna(past[self.ema_fast_col]) or pd.isna(past[self.ema_mid_col]):
                continue
            current_gap = current[self.ema_fast_col] - current[self.ema_mid_col]
            past_gap = past[self.ema_fast_col] - past[self.ema_mid_col]
            if current_gap <= past_gap:
                continue

            # FILTER 3: Near recent high
            if current['close'] < current[self.high_col] * (1 - self.near_high_pct / 100):
                continue

            # FILTER 4: RSI in momentum zone (not overbought)
            if current[self.rsi_col] < self.rsi_min or current[self.rsi_col] > self.rsi_max:
                continue

            # FILTER 5: Positive close
            if current['close'] <= prev['close']:
                continue

            # FILTER 6: Adequate volume (not dry-up)
            if current['volume'] < current[self.volume_avg_col] * self.volume_min_ratio:
                continue

            # FILTER 7: ROC > 10% (strong momentum confirmation)
            if current[self.roc_col] < 10.0:
                continue

            # FILTER 8: Near 50-day high (within 5%) — strength filter
            if current['close'] < current[self.high_50_col] * 0.95:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: ribbon expansion rate + RSI strength
            expansion_rate = (current_gap - past_gap) / current[self.ema_mid_col] * 100
            score = expansion_rate * 50 + (current[self.rsi_col] - 50) * 0.5

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'ema_gap': current_gap,
                    'expansion_rate': expansion_rate,
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
