"""Rakesh Jhunjhunwala Growth Momentum Strategy.

Based on the investment philosophy of India's Warren Buffett:
- Long-term growth focus (hold winners)
- Fundamental strength (quality businesses)
- High conviction positions
- Cut losses quickly
- Let winners run

Entry Criteria:
- Strong momentum (new 50-day high or strong ROC)
- Volume confirmation (accumulation)
- Above key moving averages (trend)
- Optional: Fundamental filters if data available

Exit Criteria:
- Stop loss (cut losses fast)
- Trailing ATR stop (let winners run)
- Time stop (reasonable holding period)

Position Sizing:
- Higher allocation on stronger signals (high conviction)
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TimeStopRule


class RJGrowthMomentumStrategy(Strategy):
    """Rakesh Jhunjhunwala inspired growth momentum strategy.

    Combines:
    - Price momentum (new highs)
    - Trend confirmation (above MA)
    - Volume strength
    - Quality exits (stop + trailing)
    """

    def __init__(
        self,
        lookback_period: int = 50,
        roc_period: int = 20,
        roc_threshold: float = 15.0,  # 15% gain in 20 days
        ma_trend_period: int = 50,
        volume_ma_period: int = 20,
        volume_surge: float = 1.3,  # 30% above average
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,  # Wider stops (let winners run)
        stop_loss_pct: float = 8.0,  # Quick cut on 8% loss
        max_holding_days: int = 90,  # Long holding period
    ):
        """Initialize RJ Growth Momentum strategy.

        Args:
            lookback_period: Period for new highs check
            roc_period: Rate of change period
            roc_threshold: Minimum ROC % for entry
            ma_trend_period: MA period for trend filter
            volume_ma_period: Volume MA period
            volume_surge: Volume surge multiplier
            atr_period: ATR period
            atr_stop_multiplier: ATR multiplier for trailing stop
            stop_loss_pct: Fixed stop loss percentage
            max_holding_days: Maximum holding period
        """
        self.lookback_period = lookback_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.ma_trend_period = ma_trend_period
        self.volume_ma_period = volume_ma_period
        self.volume_surge = volume_surge
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days

        # Column names
        self.roc_col = f'roc_{roc_period}'
        self.ma_col = f'sma_{ma_trend_period}'
        self.volume_ma_col = f'volume_sma_{volume_ma_period}'
        self.atr_col = f'atr_{atr_period}'
        self.high_col = f'rolling_high_{lookback_period}'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators."""
        return [
            IndicatorSpec('roc', {'period': self.roc_period, 'output_col': self.roc_col}),
            IndicatorSpec('sma', {'period': self.ma_trend_period}),
            IndicatorSpec('volume_sma', {'period': self.volume_ma_period, 'output_col': self.volume_ma_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': self.lookback_period, 'output_col': self.high_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate entry signals."""
        signals = []
        use_fast_path = ctx.universe_arrays is not None

        required = [self.roc_col, self.ma_col, self.volume_ma_col, self.atr_col, self.high_col]

        for symbol in ctx.universe_idx:
            if ctx.portfolio.has_position(symbol):
                continue

            idx = ctx.universe_idx[symbol]
            if idx < self.lookback_period:
                continue

            if use_fast_path and ctx.current_prices and symbol in ctx.current_prices and symbol in ctx.universe_arrays:
                bar = ctx.current_prices[symbol]
                arrays = ctx.universe_arrays[symbol]

                if not all(col in arrays for col in required):
                    continue

                vals = [arrays[col][idx] for col in required]
                if any(v != v for v in vals):
                    continue

                close = bar['close']
                roc_val = arrays[self.roc_col][idx]
                high_val = arrays[self.high_col][idx]
                ma_val = arrays[self.ma_col][idx]
                atr_val = arrays[self.atr_col][idx]
                vol_ma_val = arrays[self.volume_ma_col][idx]

                near_high = close >= high_val * 0.98
                strong_roc = roc_val >= self.roc_threshold
                if not (near_high or strong_roc):
                    continue

                if close < ma_val:
                    continue

                volume_threshold = vol_ma_val * self.volume_surge
                if bar['volume'] < volume_threshold:
                    continue

                prev_close = arrays['close'][idx - 1]
                if close <= prev_close:
                    continue

                atr_stop = close - (self.atr_stop_multiplier * atr_val)
                pct_stop = close * (1 - self.stop_loss_pct / 100)
                stop_price = max(atr_stop, pct_stop)

                roc_score = roc_val / self.roc_threshold
                high_proximity = close / high_val
                score = (roc_score * 0.6) + (high_proximity * 0.4)

                signals.append(Signal(
                    symbol=symbol, action=SignalAction.BUY, date=ctx.date,
                    stop_price=stop_price, score=score,
                    metadata={
                        'atr': atr_val, 'roc': roc_val,
                        'near_high': near_high, 'strong_roc': strong_roc,
                    }
                ))
            else:
                df = ctx.universe[symbol]
                if not all(col in df.columns for col in required):
                    continue

                current = df.iloc[idx]
                prev = df.iloc[idx - 1]

                if any(pd.isna(current[col]) for col in required):
                    continue

                near_high = current['close'] >= current[self.high_col] * 0.98
                strong_roc = current[self.roc_col] >= self.roc_threshold
                if not (near_high or strong_roc):
                    continue

                if current['close'] < current[self.ma_col]:
                    continue

                volume_threshold = current[self.volume_ma_col] * self.volume_surge
                if current['volume'] < volume_threshold:
                    continue

                if current['close'] <= prev['close']:
                    continue

                atr_stop = current['close'] - (self.atr_stop_multiplier * current[self.atr_col])
                pct_stop = current['close'] * (1 - self.stop_loss_pct / 100)
                stop_price = max(atr_stop, pct_stop)

                roc_score = current[self.roc_col] / self.roc_threshold
                high_proximity = current['close'] / current[self.high_col]
                score = (roc_score * 0.6) + (high_proximity * 0.4)

                signals.append(Signal(
                    symbol=symbol, action=SignalAction.BUY, date=ctx.date,
                    stop_price=stop_price, score=score,
                    metadata={
                        'atr': current[self.atr_col], 'roc': current[self.roc_col],
                        'near_high': near_high, 'strong_roc': strong_roc,
                    }
                ))

        return signals

    def exit_rules(self) -> list:
        """Return exit rules - RJ style: cut losses, let winners run."""
        return [
            StopLossRule(),  # Quick loss cutting
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier  # Wide trailing stop
            ),
            TimeStopRule(max_holding_days=self.max_holding_days),  # Long holding period
        ]
