"""Trend Pullback strategy — mean reversion WITHIN a trend.

Buy the dip in confirmed uptrends. Different from pure mean reversion
(which failed on NSE) because it requires Stage 2 uptrend confirmation.

Entry conditions:
1. Stage 2: Price > SMA50 > SMA200 (confirmed uptrend)
2. Price pulled back to near SMA50 (within 3% above)
3. RSI(14) < 40 (oversold relative to trend — NOT absolute oversold)
4. Close > prior close (bounce starting)
5. Volume > 0.8x average (not abandoned)
6. ATR contraction: recent volatility lower than earlier (consolidation)
7. Price within 15% of 52-week high (strong stock pulling back)

Exit: SMA10 trailing + 5% stop + 14-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule, TargetProfitRule, MACrossbackRule


class TrendPullbackStrategy(Strategy):
    """Trend Pullback — buy dips in Stage 2 uptrends."""

    def __init__(
        self,
        ma_fast: int = 50,
        ma_slow: int = 200,
        pullback_pct: float = 3.0,
        rsi_period: int = 14,
        rsi_threshold: float = 40.0,
        volume_avg_period: int = 50,
        volume_min_ratio: float = 0.8,
        atr_period: int = 14,
        volatility_contraction_ratio: float = 0.8,
        near_high_pct: float = 15.0,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.pullback_pct = pullback_pct
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.volume_avg_period = volume_avg_period
        self.volume_min_ratio = volume_min_ratio
        self.atr_period = atr_period
        self.volatility_contraction_ratio = volatility_contraction_ratio
        self.near_high_pct = near_high_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.rsi_col = 'rsi'
        self.volume_avg_col = f'volume_sma_{volume_avg_period}'
        self.atr_col = f'atr_{atr_period}'
        self.high_52w_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('rsi', {'period': self.rsi_period}),
            IndicatorSpec('volume_sma', {'period': self.volume_avg_period, 'output_col': self.volume_avg_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'output_col': self.high_52w_col}),
            IndicatorSpec('sma', {'period': 10}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < max(self.ma_slow, 30) + 10:
                continue

            required = [self.ma_fast_col, self.ma_slow_col, self.rsi_col,
                        self.volume_avg_col, self.atr_col, self.high_52w_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: Stage 2 uptrend (SMA50 > SMA200)
            if not (current[self.ma_fast_col] > current[self.ma_slow_col]):
                continue

            # FILTER 2: Price pulled back NEAR SMA50 (within pullback_pct above)
            distance_to_ma = (current['close'] - current[self.ma_fast_col]) / current[self.ma_fast_col] * 100
            if distance_to_ma < 0 or distance_to_ma > self.pullback_pct:
                continue

            # FILTER 3: RSI oversold relative to trend
            if current[self.rsi_col] >= self.rsi_threshold:
                continue

            # FILTER 4: Bounce starting (close > prior close)
            if current['close'] <= prev['close']:
                continue

            # FILTER 5: Adequate volume
            if current['volume'] < current[self.volume_avg_col] * self.volume_min_ratio:
                continue

            # FILTER 6: ATR contraction (consolidation, not panic)
            if idx >= 30:
                recent_atr = df[self.atr_col].iloc[idx-10:idx].mean()
                earlier_atr = df[self.atr_col].iloc[idx-30:idx-10].mean()
                if pd.isna(recent_atr) or pd.isna(earlier_atr) or earlier_atr == 0:
                    continue
                if recent_atr >= earlier_atr * self.volatility_contraction_ratio:
                    continue

            # FILTER 7: Near 52-week high (strong stock, not broken)
            price_vs_high = current['close'] / current[self.high_52w_col] * 100
            if price_vs_high < (100 - self.near_high_pct):
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score: lower RSI = more oversold = higher priority
            score = (self.rsi_threshold - current[self.rsi_col]) + price_vs_high * 0.1

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'rsi': current[self.rsi_col],
                    'distance_to_ma': distance_to_ma,
                    'price_vs_high': price_vs_high,
                    'atr': current[self.atr_col],
                }
            ))

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'target':
            # Mean reversion exits: target profit + MA crossback
            rules.append(TargetProfitRule(target_pct=8.0))
            rules.append(MACrossbackRule(ma_column=self.ma_fast_col))
        elif self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        elif self.trailing_exit_type == 'ema10':
            rules.append(TrailingMARule(ma_column='ema_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
