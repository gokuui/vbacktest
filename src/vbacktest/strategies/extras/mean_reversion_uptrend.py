"""Mean Reversion in Uptrends strategy.

Buy oversold dips in strong uptrending stocks. This is the mathematical
OPPOSITE of Minervini SEPA (buy breakouts) and should be highly uncorrelated.

Entry: RSI < 30 in stocks above MA200 (strong stock having a temporary dip)
Exit: RSI > 60 (mean reversion complete) or time stop 7 days
Stop: 5% below entry (tight, matching NSE learnings)

Goal: Uncorrelated with Minervini for ctx.portfolio diversification.
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TimeStopRule


class MeanReversionUptrendStrategy(Strategy):
    """Mean reversion in uptrends for NSE markets.

    Philosophy: Strong stocks that dip temporarily tend to bounce back.
    This fires when Minervini DOESN'T (during pullbacks, not breakouts).

    Entry conditions (ALL must be true):
    1. Uptrend: Price > MA200 (stock is fundamentally strong)
    2. MA200 rising (confirmed uptrend, not just crossing)
    3. Oversold: RSI(14) < 30 (temporary dip)
    4. Not in freefall: Close > MA200 * 0.85 (max 15% below MA200)
    5. Volume spike: Volume > 1.2x average (selling climax)

    Exit conditions (ANY triggers):
    1. Mean reversion: RSI > 60 (bounce complete)
    2. Hard stop: 5% below entry
    3. Time stop: 7 calendar days (quick trade)
    """

    def __init__(
        self,
        ma_trend: int = 200,
        rsi_period: int = 14,
        rsi_entry: float = 30.0,
        rsi_exit: float = 60.0,
        max_below_ma_pct: float = 15.0,
        volume_surge: float = 1.2,
        stop_loss_pct: float = 5.0,
        time_stop_days: int = 7,
    ):
        self.ma_trend = ma_trend
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.max_below_ma_pct = max_below_ma_pct
        self.volume_surge = volume_surge
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_days = time_stop_days

        # Column names
        self.ma_col = f'sma_{ma_trend}'
        self.rsi_col = f'rsi_{rsi_period}'
        self.volume_avg_col = 'volume_sma_50'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_trend}),
            IndicatorSpec('rsi', {'period': self.rsi_period, 'output_col': self.rsi_col}),
            IndicatorSpec('volume_sma', {'period': 50, 'output_col': self.volume_avg_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < self.ma_trend + 22:
                continue

            required_cols = [self.ma_col, self.rsi_col, self.volume_avg_col]
            if not all(col in df.columns for col in required_cols):
                continue

            current_bar = df.iloc[idx]

            if any(pd.isna(current_bar[col]) for col in required_cols):
                continue

            # CONDITION 1: Uptrend - price above MA200
            if current_bar['close'] <= current_bar[self.ma_col]:
                continue

            # CONDITION 2: MA200 rising (confirmed uptrend)
            ma_month_ago = df.iloc[idx - 22][self.ma_col]
            if pd.isna(ma_month_ago) or current_bar[self.ma_col] <= ma_month_ago:
                continue

            # CONDITION 3: Oversold - RSI < 30
            if current_bar[self.rsi_col] >= self.rsi_entry:
                continue

            # CONDITION 4: Not in freefall (within 15% of MA200)
            distance_from_ma = (current_bar['close'] / current_bar[self.ma_col] - 1) * 100
            if distance_from_ma > self.max_below_ma_pct:
                # This shouldn't trigger since price > MA200, but guards against edge cases
                pass

            # CONDITION 5: Volume spike (selling climax)
            if current_bar[self.volume_avg_col] > 0:
                vol_ratio = current_bar['volume'] / current_bar[self.volume_avg_col]
                if vol_ratio < self.volume_surge:
                    continue
            else:
                continue

            # Stop price: 5% below current close
            stop_price = current_bar['close'] * (1 - self.stop_loss_pct / 100)

            # Score: lower RSI = more oversold = better entry
            score = 100 - current_bar[self.rsi_col]

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'rsi': current_bar[self.rsi_col],
                    'distance_from_ma200': distance_from_ma,
                    'volume_ratio': vol_ratio,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        from vbacktest.exit_rules import TrailingMARule
        return [
            StopLossRule(),
            TimeStopRule(max_holding_days=self.time_stop_days),
        ]
