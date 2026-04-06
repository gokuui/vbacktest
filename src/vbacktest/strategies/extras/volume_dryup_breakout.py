"""Volume Dry-Up Breakout strategy.

Unique signal: volume contraction followed by expansion breakout.
This is different from Minervini's price-based VCP (Volatility Contraction Pattern)
— here we look for volume contraction specifically.

The idea: when volume dries up in an uptrend, it signals that sellers are
exhausted. A subsequent volume surge breakout has high conviction.

Entry conditions (ALL must be true):
1. Uptrend: Price > SMA50 (basic trend filter)
2. Volume dry-up: At least 3 of last 5 bars have volume < 0.5x average
3. Breakout: Close > 10-day high
4. Volume explosion: Current volume ≥ 2.0x average (contrast with dry-up)
5. Bullish candle: Close > Open
6. Near highs: Price within 20% of 52-week high

Exit conditions:
1. Hard stop: 6% below entry
2. Trailing SMA10 exit
3. Time stop: 14 days max

NSE Winning Formula compliance:
- 6 entry filters ✓
- 6% stop ✓
- 14-day time stop ✓
- SMA10 trailing exit ✓
- Volume confirmation ✓ (volume explosion as core signal)
- Uptrend filter ✓
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class VolumeDryUpBreakoutStrategy(Strategy):
    """Volume dry-up followed by breakout — unique volume-based signal."""

    def __init__(
        self,
        ma_trend: int = 50,
        ma_trail: int = 10,
        dryup_threshold: float = 0.5,  # Volume < 0.5x avg = "dry"
        dryup_min_bars: int = 3,  # Min dry bars in lookback
        dryup_lookback: int = 5,  # Days to check for dry-up
        explosion_threshold: float = 2.0,  # Volume ≥ 2x avg
        breakout_period: int = 10,
        near_high_pct: float = 80.0,  # Within 20% of 52-week high
        stop_loss_pct: float = 6.0,
        time_stop_days: int = 14,
    ):
        self.ma_trend = ma_trend
        self.ma_trail = ma_trail
        self.dryup_threshold = dryup_threshold
        self.dryup_min_bars = dryup_min_bars
        self.dryup_lookback = dryup_lookback
        self.explosion_threshold = explosion_threshold
        self.breakout_period = breakout_period
        self.near_high_pct = near_high_pct
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_trend_col = f'sma_{ma_trend}'
        self.ma_trail_col = f'sma_{ma_trail}'
        self.volume_avg_col = 'volume_sma_50'
        self.high_52w_col = 'rolling_high_252'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('sma', {'period': self.ma_trail}),
            IndicatorSpec('volume_sma', {'period': 50, 'output_col': self.volume_avg_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(252, self.ma_trend + self.breakout_period) + 5:
                continue

            required = [self.ma_trend_col, self.ma_trail_col,
                       self.volume_avg_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]

            if any(pd.isna(current[col]) for col in required):
                continue

            # CONDITION 1: Uptrend
            if current['close'] <= current[self.ma_trend_col]:
                continue

            # CONDITION 2: Volume dry-up in recent bars
            avg_vol = current[self.volume_avg_col]
            if avg_vol <= 0:
                continue

            dry_count = 0
            for j in range(1, self.dryup_lookback + 1):
                bar_j = df.iloc[idx - j]
                if bar_j['volume'] < avg_vol * self.dryup_threshold:
                    dry_count += 1

            if dry_count < self.dryup_min_bars:
                continue

            # CONDITION 3: Breakout (close > N-day high)
            high_n = df['high'].iloc[idx - self.breakout_period:idx].max()
            if current['close'] <= high_n:
                continue

            # CONDITION 4: Volume explosion
            vol_ratio = current['volume'] / avg_vol
            if vol_ratio < self.explosion_threshold:
                continue

            # CONDITION 5: Bullish candle
            if current['close'] <= current['open']:
                continue

            # CONDITION 6: Near 52-week high
            price_vs_high = (current['close'] / current[self.high_52w_col]) * 100
            if price_vs_high < self.near_high_pct:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: volume contrast (explosion / dry-up) + proximity to high
            score = vol_ratio * (dry_count / self.dryup_lookback) * (price_vs_high / 100)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'dry_count': dry_count,
                    'volume_ratio': vol_ratio,
                    'price_vs_52w': price_vs_high,
                    'breakout_above': ((current['close'] - high_n) / high_n) * 100,
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
