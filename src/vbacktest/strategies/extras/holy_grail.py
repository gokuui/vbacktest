"""
Linda Raschke Holy Grail Strategy

Core Principles:
- Trade pullbacks in strong trending markets
- Use ADX to confirm trend strength
- Enter on breakout after pullback to EMA
- Quick exits (1-2 day holding typical)

Entry Logic:
1. ADX > 30 (strong trend confirmation)
2. Price pulls back to touch/cross 20 EMA
3. Buy stop above high of pullback bar
4. Optional: Volume confirmation

Exit Logic:
1. Stop loss: Below recent swing low
2. Time exit: After 2 days maximum
3. Trend exit: Close below 20 EMA

Philosophy:
- "Buy strength, sell weakness"
- Quick in-and-out (1-2 days typical)
- Tight risk control with swing lows
- High capital efficiency

Original Trader: Linda Bradford Raschke (1995)

References:
- https://tradersmastermind.com/linda-raschke-trading-strategy/
- https://www.artoftrading.net/post/swing-trading-the-holy-grail-setup
- "Street Smarts" by Linda Raschke & Laurence Connors (1995)
"""
from __future__ import annotations


import pandas as pd
import numpy as np

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, MaxHoldingBarsRule


class HolyGrailStrategy(Strategy):
    """
    Linda Raschke Holy Grail - Pullback entry in strong trends

    Entry Conditions:
    1. ADX > threshold (default 30) - strong trend confirmation
    2. Price pullback to 20 EMA (low <= EMA <= close)
    3. Breakout above pullback bar high
    4. Optional: Volume surge (1.5x average)

    Exit Conditions:
    1. Stop loss: Below swing low (typically 3-7%)
    2. Time exit: After max_holding_days (default 2)
    3. Trend exit: Close below 20 EMA

    Position Sizing:
    - Risk 1% of equity per trade
    - Position size based on swing low distance
    """

    def __init__(
        self,
        adx_threshold: float = 30.0,
        adx_period: int = 14,
        ema_period: int = 20,
        max_holding_days: int = 2,
        volume_surge_multiplier: float = 1.5,
        require_volume: bool = True,
        swing_lookback: int = 5,  # For swing low detection
        stop_buffer_pct: float = 1.0,  # Buffer below swing low
    ):
        """Initialize Holy Grail strategy.

        Args:
            adx_threshold: Minimum ADX for trend confirmation (default: 30)
            adx_period: Period for ADX calculation (default: 14)
            ema_period: Period for EMA pullback (default: 20)
            max_holding_days: Maximum days to hold position (default: 2)
            volume_surge_multiplier: Volume threshold for confirmation (default: 1.5x)
            require_volume: Whether to require volume confirmation (default: True)
            swing_lookback: Bars to look back for swing low (default: 5)
            stop_buffer_pct: Percentage buffer below swing low (default: 1%)
        """
        self.adx_threshold = adx_threshold
        self.adx_period = adx_period
        self.ema_period = ema_period
        self.max_holding_days = max_holding_days
        self.volume_surge_multiplier = volume_surge_multiplier
        self.require_volume = require_volume
        self.swing_lookback = swing_lookback
        self.stop_buffer_pct = stop_buffer_pct

        # Column names
        self.adx_col = f'adx_{adx_period}'
        self.ema_col = f'ema_{ema_period}'
        self.volume_avg_col = f'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators for Holy Grail."""
        return [
            # ADX for trend strength
            IndicatorSpec('adx', {
                'period': self.adx_period,
                'output_col': self.adx_col
            }),
            # EMA for pullback detection
            IndicatorSpec('ema', {
                'period': self.ema_period,
                'output_col': self.ema_col
            }),
            # Volume SMA for surge detection
            IndicatorSpec('volume_sma', {
                'period': 20,
                'output_col': self.volume_avg_col
            }),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate entry signals for Holy Grail pullback setups."""
        signals = []

        for symbol, df in ctx.universe.items():
            # Skip if already in ctx.portfolio
            if ctx.portfolio.has_position(symbol):
                continue

            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]

            # Need at least swing_lookback + 2 bars of history
            # (+2 for: previous bar to detect pullback, current bar for breakout)
            if idx < self.swing_lookback + 2:
                continue

            # Check required columns
            required_cols = [
                self.adx_col, self.ema_col, self.volume_avg_col,
                'high', 'low', 'close', 'volume'
            ]

            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]
            prev_bar = df.iloc[idx - 1]

            # Check for NaN values
            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue
            if any(pd.isna(prev_bar[col]) for col in required_cols):
                continue

            # ENTRY CONDITION 1: Strong trend (ADX > threshold)
            adx = current_bar[self.adx_col]
            if adx < self.adx_threshold:
                continue

            # ENTRY CONDITION 2: Previous bar pulled back to EMA
            # Pullback = bar touched or crossed EMA (low <= EMA <= close)
            prev_low = prev_bar['low']
            prev_close = prev_bar['close']
            prev_ema = prev_bar[self.ema_col]

            # Check if previous bar was a pullback bar
            # (touched EMA from above or crossed through it)
            pullback_detected = (
                prev_low <= prev_ema <= prev_close or  # Touched from above
                prev_close <= prev_ema <= prev_low      # Crossed through
            )

            if not pullback_detected:
                continue

            # ENTRY CONDITION 3: Current bar breaks above pullback bar high
            # This is the entry trigger
            breakout = current_bar['close'] > prev_bar['high']

            if not breakout:
                continue

            # ENTRY CONDITION 4 (Optional): Volume surge
            if self.require_volume:
                volume_threshold = current_bar[self.volume_avg_col] * self.volume_surge_multiplier
                if current_bar['volume'] < volume_threshold:
                    continue

            # Calculate swing low for stop placement
            # Look back swing_lookback bars for lowest low
            lookback_start = max(0, idx - self.swing_lookback)
            lookback_slice = df.iloc[lookback_start:idx + 1]
            swing_low = lookback_slice['low'].min()

            # Apply buffer below swing low for stop
            stop_price = swing_low * (1 - self.stop_buffer_pct / 100)

            # Ensure stop is reasonable and positive
            if stop_price <= 0 or stop_price >= current_bar['close']:
                continue

            # Calculate risk percentage
            risk_pct = ((current_bar['close'] - stop_price) / current_bar['close']) * 100

            # Skip if risk is too large (>15% is unreasonable)
            if risk_pct > 15.0:
                continue

            # Calculate score based on:
            # 1. ADX strength (higher = stronger trend)
            # 2. Distance from EMA (smaller = better pullback)
            # 3. Volume surge strength
            # 4. Risk/reward (tighter stop = better)
            adx_score = min(adx / 50.0, 1.0) * 40  # Cap at 40 points

            ema_distance = abs(prev_close - prev_ema) / prev_ema
            ema_score = max(0, (1 - ema_distance / 0.02)) * 30  # Prefer <2% from EMA

            volume_strength = current_bar['volume'] / current_bar[self.volume_avg_col]
            volume_score = min(volume_strength / 3.0, 1.0) * 20  # Cap at 20 points

            risk_score = max(0, (1 - risk_pct / 10.0)) * 10  # Prefer <10% risk

            score = adx_score + ema_score + volume_score + risk_score

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'entry_reason': 'holy_grail_pullback',
                    'adx': adx,
                    'ema_value': prev_ema,
                    'pullback_bar_high': prev_bar['high'],
                    'swing_low': swing_low,
                    'risk_pct': risk_pct,
                    'volume_surge': volume_strength,
                    'breakout_pct': ((current_bar['close'] - prev_bar['high']) / prev_bar['high']) * 100,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """
        Holy Grail exit rules:
        1. Stop loss: Below swing low (set at entry)
        2. Time exit: After max_holding_days (default 2)
        3. Trend exit: Handled by EMA breakdown in custom rule
        """
        return [
            StopLossRule(),  # Swing low stop set at entry
            MaxHoldingBarsRule(max_days=self.max_holding_days),  # Quick exit after 2 days
            # TODO: Add EMA breakdown exit rule
        ]
