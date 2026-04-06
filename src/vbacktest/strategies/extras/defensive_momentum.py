"""Defensive Momentum strategy.

Finds low-volatility stocks with positive momentum — a proxy for
defensive/stable companies (pharma, FMCG, utilities) without sector data.

KEY INSIGHT: All 6 current winner strategies require Stage 2 uptrend
(price > SMA50 > SMA200) AND near 52-week high. The near-52w-high filter
makes them pure bull-market strategies — bear market stocks can be in
Stage 2 but far from their highs (resilient but still fell).

This strategy:
- Keeps Stage 2 (SMA200 required — quality filter is important!)
- REMOVES near-52w-high requirement (finds recovering stage 2 stocks)
- Adds LOW VOLATILITY filter (ATR/price < threshold)
  → Low volatility naturally selects defensive sectors (pharma, FMCG)
  → These sectors maintain Stage 2 even in bear markets
- Uses short time stop (10 days) — shorter holds than pure trend strategies

Entry conditions:
1. Stage 2: price > SMA(50) > SMA(200) — quality structural uptrend
2. ROC(20) > 5% — positive 20-day return (stock going up)
3. ROC(5) > 0% — recent 5 days still positive (momentum not reversed)
4. Low volatility: ATR(14)/close < 3% — defensive, low-beta characteristic
5. RSI(14) between 40-70 — momentum confirmed, not overbought
6. Close > prior close — confirming positive last-day action

Exit: SMA10 trailing + 5% stop + 10-day time stop
"""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class DefensiveMomentumStrategy(Strategy):
    """Defensive Momentum — Stage 2 + low volatility + positive momentum."""

    def __init__(
        self,
        roc_long: int = 20,
        roc_long_min: float = 5.0,
        roc_short: int = 5,
        ma_fast: int = 50,
        ma_slow: int = 200,
        rsi_period: int = 14,
        rsi_min: float = 40.0,
        rsi_max: float = 70.0,
        atr_period: int = 14,
        atr_pct_max: float = 3.0,       # ATR/price ratio max (defensive filter)
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 10,
        trailing_exit_type: str = 'sma10',
    ):
        self.roc_long = roc_long
        self.roc_long_min = roc_long_min
        self.roc_short = roc_short
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rsi_period = rsi_period
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.atr_period = atr_period
        self.atr_pct_max = atr_pct_max
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.roc_long_col = f'roc_{roc_long}'
        self.roc_short_col = f'roc_{roc_short}'
        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.rsi_col = 'rsi'
        self.atr_col = f'atr_{atr_period}'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('roc', {'period': self.roc_long, 'output_col': self.roc_long_col}),
            IndicatorSpec('roc', {'period': self.roc_short, 'output_col': self.roc_short_col}),
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('rsi', {'period': self.rsi_period}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
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
            if idx < max(self.ma_slow, self.roc_long) + 10:
                continue

            required = [self.roc_long_col, self.roc_short_col, self.ma_fast_col,
                        self.ma_slow_col, self.rsi_col, self.atr_col]
            if not all(col in df.columns for col in required):
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            if any(pd.isna(current[col]) for col in required):
                continue

            # FILTER 1: Stage 2 uptrend (quality filter — do NOT remove)
            if not (current['close'] > current[self.ma_fast_col] > current[self.ma_slow_col]):
                continue

            # FILTER 2: Positive 20-day return
            if current[self.roc_long_col] < self.roc_long_min:
                continue

            # FILTER 3: Recent 5-day momentum still positive
            if current[self.roc_short_col] <= 0:
                continue

            # FILTER 4: Low volatility — defensive characteristic
            # ATR/price < threshold selects pharma/FMCG/stable stocks naturally
            atr = current[self.atr_col]
            price = current['close']
            if price <= 0:
                continue
            atr_pct = atr / price * 100
            if atr_pct > self.atr_pct_max:
                continue

            # FILTER 5: RSI in momentum zone (not overbought spike)
            rsi = current[self.rsi_col]
            if rsi < self.rsi_min or rsi > self.rsi_max:
                continue

            # FILTER 6: Positive last-day action
            if current['close'] <= prev['close']:
                continue

            # Stop price
            stop_price = current['close'] * (1 - self.stop_loss_pct / 100)

            # Score by 20-day return — rank highest absolute performers first
            score = current[self.roc_long_col]

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'roc20': current[self.roc_long_col],
                    'roc5': current[self.roc_short_col],
                    'rsi': rsi,
                    'atr_pct': atr_pct,
                }
            ))

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        elif self.trailing_exit_type == 'ema10':
            rules.append(TrailingMARule(ma_column='ema_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
