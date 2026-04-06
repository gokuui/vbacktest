"""Stochastic Oversold Bounce in Uptrend strategy.

Buys pullbacks (not breakouts) — designed to be uncorrelated with Minervini SEPA.

Signal: Stochastic %K < 20 oversold pullback + bullish reversal candle in uptrend.
The key insight: in a strong uptrend, oversold stochastic readings are buying
opportunities, not signs of weakness.

Entry conditions (ALL must be true):
1. Uptrend: Price > SMA50 AND SMA50 > SMA150 (stage 2)
2. Oversold pullback: Stochastic %K was < 20 within last 3 bars
3. Bullish reversal: Close > Open (green candle) AND close > prev close
4. Not at lows: Close > SMA10 (bouncing, not falling)
5. Volume confirmation: Volume ≥ 1.3x average
6. RSI not extreme: RSI > 30 (not in freefall)

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


class StochasticBounceStrategy(Strategy):
    """Stochastic oversold bounce in uptrend.

    Buys pullbacks in strong uptrends when stochastic signals oversold.
    Uncorrelated with breakout strategies (buys weakness, not strength).
    """

    def __init__(
        self,
        ma_fast: int = 50,
        ma_mid: int = 150,
        ma_slow: int = 200,
        ma_trail: int = 10,
        stoch_k_period: int = 14,
        stoch_d_period: int = 3,
        oversold_threshold: float = 20.0,
        oversold_lookback: int = 3,
        rsi_floor: float = 30.0,
        near_high_pct: float = 75.0,  # Must be within 25% of 52w high
        volume_surge: float = 1.5,  # Higher threshold
        stop_loss_pct: float = 6.0,
        time_stop_days: int = 14,
    ):
        self.ma_fast = ma_fast
        self.ma_mid = ma_mid
        self.ma_slow = ma_slow
        self.ma_trail = ma_trail
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.oversold_threshold = oversold_threshold
        self.oversold_lookback = oversold_lookback
        self.rsi_floor = rsi_floor
        self.near_high_pct = near_high_pct
        self.volume_surge = volume_surge
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_mid_col = f'sma_{ma_mid}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.ma_trail_col = f'sma_{ma_trail}'
        self.volume_avg_col = 'volume_sma_20'
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_mid}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': self.ma_trail}),
            IndicatorSpec('stochastic', {
                'k_period': self.stoch_k_period,
                'd_period': self.stoch_d_period,
            }),
            IndicatorSpec('rsi', {'period': 14}),
            IndicatorSpec('volume_sma', {'period': 20, 'output_col': self.volume_avg_col}),
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
            if idx < max(self.ma_slow, 252, self.stoch_k_period + self.oversold_lookback) + 5:
                continue

            required = [self.ma_fast_col, self.ma_mid_col, self.ma_slow_col,
                       self.ma_trail_col, 'stoch_k', 'rsi',
                       self.volume_avg_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # CONDITION 1: Full Stage 2 uptrend (Price > MA50 > MA150 > MA200)
            if current['close'] <= current[self.ma_fast_col]:
                continue
            if current[self.ma_fast_col] <= current[self.ma_mid_col]:
                continue
            if current[self.ma_mid_col] <= current[self.ma_slow_col]:
                continue

            # CONDITION 2: Recent oversold stochastic (within lookback bars)
            was_oversold = False
            for j in range(1, self.oversold_lookback + 1):
                if idx - j < 0:
                    break
                bar_j = df.iloc[idx - j]
                if not pd.isna(bar_j['stoch_k']) and bar_j['stoch_k'] < self.oversold_threshold:
                    was_oversold = True
                    break

            if not was_oversold:
                continue

            # CONDITION 3: Bullish reversal candle
            if current['close'] <= current['open']:
                continue
            if current['close'] <= prev['close']:
                continue

            # CONDITION 4: Bouncing (close above trailing MA)
            if current['close'] <= current[self.ma_trail_col]:
                continue

            # CONDITION 5: Volume confirmation
            if current[self.volume_avg_col] <= 0:
                continue
            vol_ratio = current['volume'] / current[self.volume_avg_col]
            if vol_ratio < self.volume_surge:
                continue

            # CONDITION 6: RSI not in freefall
            if current['rsi'] < self.rsi_floor:
                continue

            # CONDITION 7: Near 52-week high (strength)
            price_vs_high = (current['close'] / current[self.high_52w_col]) * 100
            if price_vs_high < self.near_high_pct:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: prefer deeper pullbacks with stronger bounces
            stoch_depth = max(0, self.oversold_threshold - min(
                df['stoch_k'].iloc[max(0, idx - self.oversold_lookback):idx].min(),
                self.oversold_threshold
            ))
            bounce_strength = (current['close'] - current['open']) / current['open'] * 100
            score = stoch_depth * 0.5 + bounce_strength * 0.3 + vol_ratio * 0.2

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'stoch_k': current['stoch_k'],
                    'rsi': current['rsi'],
                    'volume_ratio': vol_ratio,
                    'bounce_pct': bounce_strength,
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
