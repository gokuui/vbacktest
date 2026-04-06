"""NASDAQ Multi-Factor Momentum Strategy.

Faithfully implements Playbook Strategy 1 from arxiv:2603.15848:
  "Vectorised Multi-Factor Momentum" — Sharpe 2.04 out-of-sample on US equities

Entry (ALL required):
  1. Close > SMA(200)           — uptrend filter
  2. EMA(50) > EMA(200)         — golden cross
  3. Close > EMA(50)            — above medium-term trend
  4. Momentum(12-1) > 0         — 12-month momentum skipping last month (avoids reversal)
  5. Rank top-N by Momentum(12-1) — take the strongest stocks

   12-1 month momentum = (close[t-21] / close[t-252]) - 1
                       = (1 + roc_252/100) / (1 + roc_21/100) - 1

Exit (first triggered):
  - Close < SMA(200)            — trend breakdown
  - Trailing ATR 3.5× stop     — volatility-based loss cap

Sizing: Equal weight 1/N via max_position_pct cap (set externally in config).
Liquidity filter: matches validation pipeline ($5 min price, 50k avg vol per bar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import TrailingATRStopRule, TrailingMARule


class NasdaqMomentumStrategy(Strategy):
    """Multi-Factor Momentum strategy tuned for NASDAQ/US equities.

    Uses 12-1 month academic momentum (252-day lookback, skip last 21 days)
    to avoid short-term reversal effects that plagued simpler 63-day momentum.

    Momentum(12-1) = close[t-21] / close[t-252] - 1
                   = (1 + roc_252/100) / (1 + roc_21/100) - 1
    """

    def __init__(
        self,
        sma_trend_period: int = 200,    # long-term trend filter
        ema_fast_period: int = 50,      # medium-term EMA
        ema_slow_period: int = 200,     # long-term EMA (golden cross)
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.5,   # tighter: forces rotation, reduces zombie positions
        top_n: int = 10,                # max stocks to hold simultaneously
        min_price: float = 5.0,         # matches validation pipeline threshold
        min_avg_vol: float = 50_000.0,  # matches validation pipeline threshold
        min_dollar_volume: float = 0.0, # 0 = disabled; set to filter warrants/SPACs
        volume_ma_period: int = 20,
        roc_long_period: int = 252,     # 12-month lookback
        roc_short_period: int = 21,     # 1-month skip to avoid reversal
        roc_medium_period: int = 63,    # 3-month filter: require recent momentum too
    ):
        self.sma_trend_period = sma_trend_period
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.top_n = top_n
        self.min_price = min_price
        self.min_avg_vol = min_avg_vol
        self.min_dollar_volume = min_dollar_volume
        self.volume_ma_period = volume_ma_period
        self.roc_long_period = roc_long_period
        self.roc_short_period = roc_short_period
        self.roc_medium_period = roc_medium_period

        # Indicator column names
        self.roc_long_col = f'roc_{roc_long_period}'
        self.roc_short_col = f'roc_{roc_short_period}'
        self.roc_medium_col = f'roc_{roc_medium_period}'
        self.sma_col = f'sma_{sma_trend_period}'
        self.ema_fast_col = f'ema_{ema_fast_period}'
        self.ema_slow_col = f'ema_{ema_slow_period}'
        self.atr_col = f'atr_{atr_period}'
        self.vol_ma_col = f'volume_sma_{volume_ma_period}'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('roc', {'period': self.roc_long_period, 'output_col': self.roc_long_col}),
            IndicatorSpec('roc', {'period': self.roc_short_period, 'output_col': self.roc_short_col}),
            IndicatorSpec('roc', {'period': self.roc_medium_period, 'output_col': self.roc_medium_col}),
            IndicatorSpec('sma', {'period': self.sma_trend_period}),
            IndicatorSpec('ema', {'period': self.ema_fast_period}),
            IndicatorSpec('ema', {'period': self.ema_slow_period}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('volume_sma', {'period': self.volume_ma_period, 'output_col': self.vol_ma_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate entry signals for top-N momentum stocks passing all filters."""
        candidates = []
        use_fast = ctx.universe_arrays is not None

        required_cols = [
            self.roc_long_col, self.roc_short_col, self.roc_medium_col,
            self.sma_col, self.ema_fast_col, self.ema_slow_col,
            self.atr_col, self.vol_ma_col,
        ]
        min_lookback = max(self.roc_long_period, self.sma_trend_period) + 5

        for symbol in ctx.universe_idx:
            if ctx.portfolio.has_position(symbol):
                continue

            idx = ctx.universe_idx[symbol]
            if idx < min_lookback:
                continue

            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                bar = ctx.current_prices.get(symbol) if ctx.current_prices else None

                if bar is None:
                    continue
                if not all(c in arrays for c in required_cols):
                    continue

                try:
                    close = arrays['close'][idx]
                    roc_long = arrays[self.roc_long_col][idx]
                    roc_short = arrays[self.roc_short_col][idx]
                    roc_medium = arrays[self.roc_medium_col][idx]
                    sma200 = arrays[self.sma_col][idx]
                    ema50 = arrays[self.ema_fast_col][idx]
                    ema200 = arrays[self.ema_slow_col][idx]
                    atr = arrays[self.atr_col][idx]
                    vol_ma = arrays[self.vol_ma_col][idx]
                except (KeyError, IndexError):
                    continue

                vals = (close, roc_long, roc_short, roc_medium, sma200, ema50, ema200, atr, vol_ma)
                if any(v != v for v in vals):
                    continue

            else:
                df = ctx.universe[symbol]
                if not all(c in df.columns for c in required_cols):
                    continue

                bar_row = df.iloc[idx]
                if any(pd.isna(bar_row[c]) for c in required_cols):
                    continue

                close = bar_row['close']
                roc_long = bar_row[self.roc_long_col]
                roc_short = bar_row[self.roc_short_col]
                roc_medium = bar_row[self.roc_medium_col]
                sma200 = bar_row[self.sma_col]
                ema50 = bar_row[self.ema_fast_col]
                ema200 = bar_row[self.ema_slow_col]
                atr = bar_row[self.atr_col]
                vol_ma = bar_row[self.vol_ma_col]

            # ── Liquidity filters ──
            if close < self.min_price:
                continue
            if vol_ma < self.min_avg_vol:
                continue
            # Dollar volume: filters warrants/SPACs without excluding mid-caps
            if close * vol_ma < self.min_dollar_volume:
                continue

            # ── Trend filters ──
            if close <= sma200:
                continue
            if ema50 <= ema200:
                continue
            if close <= ema50:
                continue

            # ── 12-1 month momentum (academic standard) ──
            # mom_12_1 = close[t-21] / close[t-252] - 1
            #           = (1 + roc_252/100) / (1 + roc_21/100) - 1
            denom = 1 + roc_short / 100.0
            if abs(denom) < 1e-9:
                continue
            mom_12_1 = (1 + roc_long / 100.0) / denom - 1

            # Require 3-month momentum also positive: filters spike-then-pullback stocks
            if roc_medium <= 0:
                continue

            # Only enter on positive 12-1 momentum
            if mom_12_1 <= 0:
                continue

            # Optional dollar volume filter (disabled by default)
            if self.min_dollar_volume > 0 and close * vol_ma < self.min_dollar_volume:
                continue

            candidates.append({
                'symbol': symbol,
                'mom': mom_12_1,
                'close': close,
                'atr': atr,
                'stop_price': close - self.atr_stop_multiplier * atr,
            })

        if not candidates:
            return []

        # ── Rank by 6-1 month momentum, take top N available slots ──
        candidates.sort(key=lambda x: x['mom'], reverse=True)

        current_positions = len(ctx.portfolio.positions)
        slots_available = self.top_n - current_positions
        if slots_available <= 0:
            return []

        top = candidates[:slots_available]

        signals = []
        for c in top:
            signals.append(Signal(
                symbol=c['symbol'],
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=c['stop_price'],
                score=c['mom'],
                metadata={'atr': c['atr'], 'mom_6_1': c['mom']},
            ))

        return signals

    def exit_rules(self) -> list:
        """Trend breakdown OR ATR trailing stop."""
        return [
            TrailingMARule(ma_column=self.sma_col),
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier,
            ),
        ]
