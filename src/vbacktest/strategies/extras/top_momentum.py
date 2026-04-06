"""Top Momentum strategy — pure cross-sectional momentum ranking.

Unlike other strategies that use absolute thresholds (ROC > X%, RSI < Y),
this strategy ranks ALL stocks by momentum and buys the top N.
This is fundamentally different — it's always invested, always rotating
into the strongest stocks.

Entry conditions:
1. Rank stocks by 6-month ROC (126 trading days)
2. Filters: Price > SMA50 (uptrend), Volume > 100K avg (liquid)
3. Buy top N ranked stocks when a position slot opens
4. Rebalance: exit when stock drops out of top 2*N ranking

Exit: SMA10 trailing + 7% stop + 21-day review
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class TopMomentumStrategy(Strategy):
    """Top Momentum — buy highest-ranked momentum stocks."""

    def __init__(
        self,
        roc_period: int = 126,
        trend_ma: int = 50,
        min_volume: float = 100000,
        volume_avg_period: int = 50,
        top_n_multiplier: float = 2.0,
        stop_loss_pct: float = 7.0,
        max_holding_days: int = 21,
        trailing_exit_type: str = 'sma10',
    ):
        self.roc_period = roc_period
        self.trend_ma = trend_ma
        self.min_volume = min_volume
        self.volume_avg_period = volume_avg_period
        self.top_n_multiplier = top_n_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.roc_col = f'roc_{roc_period}'
        self.trend_ma_col = f'sma_{trend_ma}'
        self.volume_avg_col = f'volume_sma_{volume_avg_period}'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('roc', {'period': self.roc_period, 'output_col': self.roc_col}),
            IndicatorSpec('sma', {'period': self.trend_ma}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.volume_avg_col}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        # Collect all eligible stocks with their momentum scores
        candidates = []
        for symbol, df in ctx.universe.items():
            if symbol not in ctx.universe_idx:
                continue
            idx = ctx.universe_idx[symbol]
            if idx < max(self.roc_period, self.trend_ma) + 5:
                continue

            required = [self.roc_col, self.trend_ma_col, self.volume_avg_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            if any(pd.isna(current[col]) for col in required):
                continue

            # Basic filters
            if current['close'] <= current[self.trend_ma_col]:
                continue
            if current[self.volume_avg_col] < self.min_volume:
                continue
            if current[self.roc_col] <= 0:
                continue

            candidates.append((symbol, current[self.roc_col], current))

        if not candidates:
            return signals

        # Sort by momentum (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Get max positions from ctx.portfolio config
        max_positions = ctx.portfolio.config.max_positions if hasattr(ctx.portfolio, 'config') else 5
        top_n = int(max_positions * self.top_n_multiplier)

        for symbol, roc, current in candidates[:top_n]:
            if ctx.portfolio.has_position(symbol):
                continue

            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=roc,
                metadata={
                    'roc_126': roc,
                    'rank': candidates.index((symbol, roc, current)) + 1,
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
