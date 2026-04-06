"""Dual Momentum strategy (Absolute + Relative).

Signal: Cross-sectional ranking — buy stocks with BOTH absolute momentum
(positive ROC) and relative momentum (top performers in the universe).
This is architecturally different from all other strategies because it
RANKS stocks against each other, not just filters.

Entry conditions (ALL must be true):
1. Uptrend: Price > SMA50 (basic trend filter)
2. Absolute momentum: 20-day ROC > 15% (strong individual momentum)
3. Relative momentum: Stock ranks in top N by ROC among all candidates
4. Near highs: Price within 15% of 52-week high
5. Volume confirmation: Volume ≥ 1.3x average
6. Price rising: Close > Close 5 bars ago

Exit conditions:
1. Hard stop: 6% below entry
2. Trailing SMA10 exit
3. Time stop: 14 days max

NSE Winning Formula compliance:
- 6 entry filters ✓
- 6% stop ✓
- 14-day time stop ✓
- SMA10 trailing exit ✓
- Volume confirmation ✓
- Uptrend filter ✓
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class DualMomentumStrategy(Strategy):
    """Dual momentum — absolute + relative ranking."""

    def __init__(
        self,
        ma_trend: int = 50,
        roc_period: int = 20,
        roc_threshold: float = 15.0,  # Absolute momentum minimum
        near_high_pct: float = 85.0,  # Within 15% of 52-week high
        volume_surge: float = 1.3,
        momentum_bars: int = 5,
        top_n: int = 20,  # Only consider top N by relative momentum
        stop_loss_pct: float = 6.0,
        time_stop_days: int = 14,
    ):
        self.ma_trend = ma_trend
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.near_high_pct = near_high_pct
        self.volume_surge = volume_surge
        self.momentum_bars = momentum_bars
        self.top_n = top_n
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_col = f'sma_{ma_trend}'
        self.roc_col = f'roc_{roc_period}'
        self.volume_avg_col = 'volume_sma_20'
        self.high_52w_col = 'rolling_high_252'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('sma', {'period': 10}),  # For trailing exit
            IndicatorSpec('roc', {'period': self.roc_period}),
            IndicatorSpec('volume_sma', {'period': 20, 'output_col': self.volume_avg_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        # Phase 1: Collect all candidates with absolute momentum
        candidates = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(252, self.ma_trend + self.momentum_bars) + 5:
                continue

            required = [self.ma_col, self.roc_col, self.volume_avg_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: Uptrend
            if current['close'] <= current[self.ma_col]:
                continue

            # FILTER 2: Absolute momentum
            roc_val = current[self.roc_col]
            if roc_val < self.roc_threshold:
                continue

            # FILTER 3: Near 52-week high
            price_vs_high = (current['close'] / current[self.high_52w_col]) * 100
            if price_vs_high < self.near_high_pct:
                continue

            # FILTER 4: Volume confirmation
            if current[self.volume_avg_col] <= 0:
                continue
            vol_ratio = current['volume'] / current[self.volume_avg_col]
            if vol_ratio < self.volume_surge:
                continue

            # FILTER 5: Price rising (momentum confirmation)
            past_close = df.iloc[idx - self.momentum_bars]['close']
            if current['close'] <= past_close:
                continue

            # FILTER 6: Bullish candle
            if current['close'] <= current['open']:
                continue

            candidates.append({
                'symbol': symbol,
                'roc': roc_val,
                'price_vs_high': price_vs_high,
                'vol_ratio': vol_ratio,
                'close': current['close'],
            })

        # Phase 2: Relative momentum ranking — only take top N
        candidates.sort(key=lambda x: x['roc'], reverse=True)
        top_candidates = candidates[:self.top_n]

        # Phase 3: Generate signals for top ranked
        signals = []
        for c in top_candidates:
            stop_price = c['close'] * (1 - self.stop_loss_pct / 100)

            # Score: ROC rank * proximity to high
            score = c['roc'] * (c['price_vs_high'] / 100) * c['vol_ratio']

            signals.append(Signal(
                symbol=c['symbol'],
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'roc': c['roc'],
                    'price_vs_52w': c['price_vs_high'],
                    'volume_ratio': c['vol_ratio'],
                    'rank': top_candidates.index(c) + 1,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        from vbacktest.exit_rules import TrailingMARule
        return [
            StopLossRule(),
            TrailingMARule(ma_column='sma_10'),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
