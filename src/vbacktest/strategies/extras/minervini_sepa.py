"""Mark Minervini SEPA (Specific Entry Point Analysis) strategy.

Simplified implementation for NSE markets:
1. Stage 2 uptrend confirmation (MA hierarchy)
2. Near 52-week highs (strength filter)
3. Volatility contraction (base building)
4. Breakout on volume surge
5. Tight risk control (7% stop, MA10 trailing)

Goal: High win rate (50-60%), Low DD (<15%), Calmar ≥1.5
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TimeStopRule


class MinerviniSEPAStrategy(Strategy):
    """Minervini SEPA strategy for NSE markets.

    Entry conditions:
    1. Stage 2: Price > MA50 > MA150 > MA200 (uptrend hierarchy)
    2. MA200 rising (uptrend confirmed)
    3. Near highs: Price ≥ 70% of 52-week high
    4. Volatility contracted: Recent ATR < earlier ATR
    5. Breakout: Close > 10-day high
    6. Volume surge: ≥1.3x average

    Exit conditions:
    1. Hard stop: 7% below entry (tight risk control)
    2. Trailing MA10 stop (let winners run)
    3. Time stop: 4 weeks if not up 10%

    Scoring:
    - Proximity to 52-week high = strength
    """

    def __init__(
        self,
        ma_fast: int = 50,
        ma_mid: int = 150,
        ma_slow: int = 200,
        high_pct_threshold: float = 70.0,  # % of 52-week high
        volatility_contraction_ratio: float = 0.7,  # ATR recent/earlier
        volume_surge_multiplier: float = 1.3,
        breakout_period: int = 10,  # Days for breakout high
        atr_period: int = 14,
        stop_loss_pct: float = 7.0,  # Minervini's 7-8% max stop
        trailing_ma_period: int = 10,  # MA10 trailing stop
        time_stop_days: int = 28,  # 4 weeks
    ):
        """Initialize Minervini SEPA strategy.

        Args:
            ma_fast: Fast MA period for stage analysis (default: 50)
            ma_mid: Mid MA period for stage analysis (default: 150)
            ma_slow: Slow MA period for stage analysis (default: 200)
            high_pct_threshold: Min % of 52-week high (default: 70%)
            volatility_contraction_ratio: ATR contraction threshold (default: 0.7)
            volume_surge_multiplier: Volume surge threshold (default: 1.3x)
            breakout_period: Days for breakout high check (default: 10)
            atr_period: ATR period (default: 14)
            stop_loss_pct: Hard stop loss % (default: 7%)
            trailing_ma_period: MA period for trailing stop (default: 10)
            time_stop_days: Time stop in days (default: 28 = 4 weeks)
        """
        self.ma_fast = ma_fast
        self.ma_mid = ma_mid
        self.ma_slow = ma_slow
        self.high_pct_threshold = high_pct_threshold
        self.volatility_contraction_ratio = volatility_contraction_ratio
        self.volume_surge_multiplier = volume_surge_multiplier
        self.breakout_period = breakout_period
        self.atr_period = atr_period
        self.stop_loss_pct = stop_loss_pct
        self.trailing_ma_period = trailing_ma_period
        self.time_stop_days = time_stop_days

        # Column names
        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_mid_col = f'sma_{ma_mid}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.trailing_ma_col = f'sma_{trailing_ma_period}'
        self.volume_avg_col = f'volume_sma_50'  # 50-day volume average
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators."""
        return [
            # Stage 2 MAs
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_mid}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            # Trailing stop MA
            IndicatorSpec('sma', {'period': self.trailing_ma_period}),
            # ATR for volatility
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            # Volume average
            IndicatorSpec('volume_sma', {'period': 50, 'output_col': self.volume_avg_col}),
            # 52-week high
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate signals for current bar."""
        signals = []

        for symbol, df in ctx.universe.items():
            # Skip if already in ctx.portfolio
            if ctx.portfolio.has_position(symbol):
                continue

            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]

            # Need history for stage analysis
            if idx < max(self.ma_slow, 30):  # Need at least MA200 + 1 month
                continue

            # Check required columns
            required_cols = [
                self.ma_fast_col, self.ma_mid_col, self.ma_slow_col,
                self.atr_col, self.volume_avg_col, self.high_52w_col,
                self.trailing_ma_col
            ]

            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]
            prev_bar = df.iloc[idx - 1]

            # Check for NaN values
            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue

            # ENTRY CONDITION 1: Stage 2 (Uptrend hierarchy)
            # Price > MA50 > MA150 > MA200
            stage_2 = (
                current_bar['close'] > current_bar[self.ma_fast_col] and
                current_bar[self.ma_fast_col] > current_bar[self.ma_mid_col] and
                current_bar[self.ma_mid_col] > current_bar[self.ma_slow_col]
            )

            if not stage_2:
                continue

            # ENTRY CONDITION 2: MA200 rising (confirmed uptrend)
            # Check MA200 is higher than 1 month ago (~22 trading days)
            if idx < self.ma_slow + 22:
                continue

            ma200_month_ago = df.iloc[idx - 22][self.ma_slow_col]
            ma200_rising = current_bar[self.ma_slow_col] > ma200_month_ago

            if not ma200_rising:
                continue

            # ENTRY CONDITION 3: Near 52-week high (strength)
            price_vs_high = (current_bar['close'] / current_bar[self.high_52w_col]) * 100

            if price_vs_high < self.high_pct_threshold:
                continue

            # ENTRY CONDITION 4: Volatility contraction
            # Recent ATR (last 10 days) < Earlier ATR (10-30 days ago)
            if idx < 30:
                continue

            recent_atr = df[self.atr_col].iloc[idx-10:idx].mean()
            earlier_atr = df[self.atr_col].iloc[idx-30:idx-10].mean()

            volatility_contracted = (
                recent_atr < earlier_atr * self.volatility_contraction_ratio
            )

            if not volatility_contracted:
                continue

            # ENTRY CONDITION 5: Breakout (new 10-day high)
            high_10d = df['high'].iloc[idx-self.breakout_period:idx].max()
            breakout = current_bar['close'] > high_10d

            if not breakout:
                continue

            # ENTRY CONDITION 6: Volume surge
            volume_surge = (
                current_bar['volume'] >=
                current_bar[self.volume_avg_col] * self.volume_surge_multiplier
            )

            if not volume_surge:
                continue

            # Calculate stop price (7% below entry)
            stop_price = current_bar['close'] * (1 - self.stop_loss_pct / 100)

            # Score based on proximity to 52-week high
            score = price_vs_high  # Higher = closer to highs = stronger

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'stage_2': True,
                    'price_vs_52w_high': price_vs_high,
                    'atr': current_bar[self.atr_col],
                    'volume_surge': current_bar['volume'] / current_bar[self.volume_avg_col]
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """Return exit rules for positions."""
        from vbacktest.exit_rules import TrailingMARule

        return [
            # Hard stop loss (7%)
            StopLossRule(),
            # Trailing MA10 stop (Minervini's method)
            TrailingMARule(ma_column=self.trailing_ma_col),
            # Time stop: 4 weeks
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
