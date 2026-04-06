"""Gap-Up Follow-Through strategy.

Completely different signal source from pattern/indicator strategies — based on
gap events which are rare, high-conviction signals.

Signal: True gap up (open > prev high) that holds through the day with volume.
Gaps represent institutional demand that often leads to continuation.

Entry conditions (ALL must be true):
1. Uptrend: Price > SMA50 (basic trend filter)
2. True gap: Open > previous day's high (not just prev close)
3. Gap holds: Close > Open (didn't fill the gap)
4. Gap size: Between 2% and 15% (avoid penny stock noise and extreme gaps)
5. Volume surge: Volume ≥ 1.5x average
6. Close near high: Close within top 30% of day's range (strong close)

Exit conditions:
1. Hard stop: 5% below entry (gaps should work immediately)
2. Trailing SMA7 exit (fast trailing for gap plays)
3. Time stop: 10 days max (gap momentum fades quickly)

NSE Winning Formula compliance:
- 6 entry filters ✓
- 5% stop ✓
- 10-day time stop ✓
- SMA7 trailing exit ✓
- Volume confirmation ✓
- Uptrend filter ✓
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class GapUpFollowStrategy(Strategy):
    """Gap-up follow-through — buys true gap-ups that hold in uptrends."""

    def __init__(
        self,
        ma_trend: int = 50,
        ma_trail: int = 7,
        min_gap_pct: float = 2.0,
        max_gap_pct: float = 15.0,
        volume_surge: float = 1.5,
        close_range_pct: float = 70.0,  # Close in top 30% of range
        stop_loss_pct: float = 5.0,
        time_stop_days: int = 10,
    ):
        self.ma_trend = ma_trend
        self.ma_trail = ma_trail
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.volume_surge = volume_surge
        self.close_range_pct = close_range_pct
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_trend_col = f'sma_{ma_trend}'
        self.ma_trail_col = f'sma_{ma_trail}'
        self.volume_avg_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('sma', {'period': self.ma_trail}),
            IndicatorSpec('volume_sma', {'period': 20, 'output_col': self.volume_avg_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < self.ma_trend + 5:
                continue

            required = [self.ma_trend_col, self.ma_trail_col, self.volume_avg_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # CONDITION 1: Uptrend
            if current['close'] <= current[self.ma_trend_col]:
                continue

            # CONDITION 2: True gap up (open > prev high)
            gap_pct = ((current['open'] - prev['high']) / prev['high']) * 100
            if gap_pct < self.min_gap_pct:
                continue
            if gap_pct > self.max_gap_pct:
                continue

            # CONDITION 3: Gap holds (close > open = green candle)
            if current['close'] <= current['open']:
                continue

            # CONDITION 4: Close near high of day (strong close)
            day_range = current['high'] - current['low']
            if day_range <= 0:
                continue
            close_position = ((current['close'] - current['low']) / day_range) * 100
            if close_position < self.close_range_pct:
                continue

            # CONDITION 5: Volume surge
            if current[self.volume_avg_col] <= 0:
                continue
            vol_ratio = current['volume'] / current[self.volume_avg_col]
            if vol_ratio < self.volume_surge:
                continue

            # CONDITION 6: Close above trailing MA (confirming bounce)
            if current['close'] <= current[self.ma_trail_col]:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: gap size * volume ratio * close position
            score = gap_pct * vol_ratio * (close_position / 100)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'gap_pct': gap_pct,
                    'volume_ratio': vol_ratio,
                    'close_position': close_position,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        from vbacktest.exit_rules import TrailingMARule
        return [
            StopLossRule(),
            TrailingMARule(ma_column=self.ma_trail_col),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
