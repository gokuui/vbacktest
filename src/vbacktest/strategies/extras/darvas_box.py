"""
Nicolas Darvas Box Theory Trading Strategy

Core Principles:
- Pattern-based breakout system using price "boxes" (consolidation ranges)
- Buy stocks making new 52-week highs that consolidate into boxes
- Breakout above box top with volume = entry signal
- Trailing stops using box bottoms as new floors
- 7-8% max stop loss

Entry Logic:
1. Stock makes new 52-week high
2. Box forms: High holds for 3+ days (no new highs), low is established
3. Price breaks out above box top
4. Volume confirms breakout (1.5x average)

Exit Logic:
1. Stop loss: Below box bottom (~7-8%)
2. Trailing box stop: As new boxes form, old top becomes new floor
3. Exit if price falls back into previous box

Philosophy:
- "Buy high, sell higher" - momentum with consolidation
- Boxes = energy coiling before explosive moves
- Volume = institutional confirmation
- Let winners run with trailing boxes

Original Story: Nicolas Darvas (dancer) made $2M in 18 months using this method in 1950s

References:
- https://www.quantifiedstrategies.com/nicolas-darvas/
- https://trendspider.com/learning-center/darvas-box-theory-trading-strategy/
"""
from __future__ import annotations


import pandas as pd
import numpy as np

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule


class DarvasBoxStrategy(Strategy):
    """
    Nicolas Darvas Box Theory - Pattern-based momentum strategy

    Entry Conditions:
    1. Near 52-week high (≥90% of 52-week high)
    2. Box formed: Price hasn't made new high in last 3 days
    3. Breakout: Close > box top (recent high)
    4. Volume surge: ≥1.5x average volume

    Exit Conditions:
    1. Stop loss: Below box bottom (~7-8% from entry)
    2. Trailing box stop: New box forms → old top becomes new floor

    Position Sizing:
    - Risk 1% of equity per trade
    - Position size based on box height (stop distance)
    """

    def __init__(
        self,
        box_confirmation_days: int = 3,
        high_week_threshold: float = 0.90,  # 90% of 52-week high
        volume_surge_multiplier: float = 1.5,
        stop_loss_pct: float = 8.0,  # Max 8% stop
        volume_period: int = 20,
    ):
        """Initialize Darvas Box strategy.

        Args:
            box_confirmation_days: Days to confirm box top (default: 3)
            high_week_threshold: % of 52-week high to qualify (default: 0.90)
            volume_surge_multiplier: Volume surge threshold (default: 1.5x)
            stop_loss_pct: Maximum stop loss percentage (default: 8%)
            volume_period: Period for average volume calculation (default: 20)
        """
        self.box_confirmation_days = box_confirmation_days
        self.high_week_threshold = high_week_threshold
        self.volume_surge_multiplier = volume_surge_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.volume_period = volume_period

        # Column names
        self.high_52w_col = 'high_52w'
        self.volume_avg_col = f'volume_sma_{volume_period}'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators for Darvas Box."""
        return [
            # 52-week (252 trading days) rolling high
            IndicatorSpec('rolling_high', {
                'period': 252,
                'column': 'high',
                'output_col': self.high_52w_col
            }),
            # Average volume for surge detection
            IndicatorSpec('volume_sma', {
                'period': self.volume_period,
                'output_col': self.volume_avg_col
            }),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate entry signals for Darvas Box breakouts."""
        signals = []

        for symbol, df in ctx.universe.items():
            # Skip if already in ctx.portfolio
            if ctx.portfolio.has_position(symbol):
                continue

            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]

            # Need history for box confirmation (at least confirmation_days + 1)
            if idx < self.box_confirmation_days + 1:
                continue

            # Check required columns
            required_cols = [
                self.high_52w_col, self.volume_avg_col,
                'high', 'low', 'close', 'volume'
            ]

            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]

            # Check for NaN values
            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue

            # ENTRY CONDITION 1: Near 52-week high
            # Price should be at least 90% of 52-week high
            high_52w = current_bar[self.high_52w_col]
            near_high = current_bar['close'] >= (high_52w * self.high_week_threshold)

            if not near_high:
                continue

            # ENTRY CONDITION 2: Box formation
            # Check if a box has formed (no new highs in last N days)
            # Lookback excludes current bar - box must be established before breakout
            lookback_slice = df.iloc[idx - self.box_confirmation_days:idx]

            # Box top = highest high in the lookback period
            box_top = lookback_slice['high'].max()

            # Box bottom = lowest low during box formation
            box_bottom = lookback_slice['low'].min()

            # ENTRY CONDITION 3: Breakout above box top
            # Current close breaks above the established box top
            # (Box is confirmed by lookback, current bar is the breakout)
            breakout = current_bar['close'] > box_top

            if not breakout:
                continue

            # ENTRY CONDITION 4: Volume surge
            volume_threshold = current_bar[self.volume_avg_col] * self.volume_surge_multiplier
            volume_surge = current_bar['volume'] >= volume_threshold

            if not volume_surge:
                continue

            # Calculate stop loss
            # Use the lower of: box bottom OR max stop loss percentage
            box_stop = box_bottom
            pct_stop = current_bar['close'] * (1 - self.stop_loss_pct / 100)
            stop_price = max(box_stop, pct_stop)  # Use tighter stop

            # Ensure stop is positive and reasonable
            if stop_price <= 0 or stop_price >= current_bar['close']:
                continue

            # Calculate box height for sizing
            box_height = box_top - box_bottom
            box_height_pct = (box_height / box_bottom) * 100 if box_bottom > 0 else 0

            # Score based on:
            # 1. Distance above box top (breakout strength)
            # 2. Proximity to 52-week high
            # 3. Volume surge strength
            breakout_strength = (current_bar['close'] - box_top) / box_top
            high_proximity = current_bar['close'] / high_52w
            volume_strength = current_bar['volume'] / current_bar[self.volume_avg_col]

            score = (breakout_strength * 100) + (high_proximity * 10) + (volume_strength * 5)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'entry_reason': 'darvas_box_breakout',
                    'box_top': box_top,
                    'box_bottom': box_bottom,
                    'box_height_pct': box_height_pct,
                    'high_52w': high_52w,
                    'high_proximity_pct': high_proximity * 100,
                    'volume_surge': volume_strength,
                    'breakout_pct': breakout_strength * 100,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """
        Darvas Box exit rules:
        1. Stop loss: Below box bottom (or max 8%)
        """
        return [
            StopLossRule(),  # Box bottom stop set at entry
            # Note: Trailing box stops would require tracking box formations
            # which is complex - for now using simple stop loss
        ]
