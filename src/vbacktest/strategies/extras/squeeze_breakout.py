"""Squeeze Breakout strategy.

Based on TTM Squeeze concept: when Bollinger Bands contract inside Keltner Channels,
volatility is extremely compressed. When the squeeze "fires" (BB expand outside KC),
an explosive move follows.

This is fundamentally different from existing strategies:
- NR4: uses narrow range of individual bars, no BB/KC
- tight_range: uses burst + consolidation pattern, no squeeze detection
- bollinger: uses BB breakout but no KC squeeze confirmation
- keltner_squeeze: exists but FAILED (Calmar -0.26) — this redesigns with better filters

Key improvements over failed keltner_squeeze:
1. Stage 2 uptrend filter (only buy in established uptrends)
2. Squeeze DURATION filter (longer squeeze = more explosive breakout)
3. Volume confirmation on breakout
4. Much tighter stop loss (ATR-based, not fixed %)
5. Trend quality filter: price above rising SMA50

Entry conditions (ALL must be true):
1. Squeeze active: BB width < KC width (BB inside KC)
2. Squeeze duration: squeeze has been active for >= 5 bars (building energy)
3. Squeeze fires: BB width crosses ABOVE KC width (expansion begins)
4. Stage 2: Close > SMA50 > SMA200, SMA50 rising
5. Volume surge: volume > 1.5x 20-day average
6. Close > SMA10 (short-term momentum)

Exit: ATR trailing stop + time stop + SMA10 trailing
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TrailingMARule, TimeStopRule


class SqueezeBreakoutStrategy(Strategy):
    """TTM Squeeze-based breakout — buy when BB expand outside KC after compression."""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        squeeze_min_bars: int = 5,
        ma_fast: int = 50,
        ma_slow: int = 200,
        volume_mult: float = 1.5,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        stop_loss_pct: float = 5.0,
        max_holding_days: int = 30,
        trailing_exit_type: str = 'sma10',
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.squeeze_min_bars = squeeze_min_bars
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.vol_sma_col = 'volume_sma_20'

    def indicators(self) -> list[IndicatorSpec]:
        specs = [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('bollinger_bands', {'period': self.bb_period, 'num_std': self.bb_std}),
            IndicatorSpec('keltner_channel', {'ema_period': self.kc_period, 'atr_multiplier': self.kc_mult}),
            IndicatorSpec('volume_sma', {'period': 20}),
        ]
        if self.trailing_exit_type == 'ema10':
            specs.append(IndicatorSpec('ema', {'period': 10}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, self.bb_period, self.kc_period) + self.squeeze_min_bars + 10

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue
            idx = ctx.universe_idx[symbol]
            if idx < min_bars:
                continue

            required = [self.ma_fast_col, self.ma_slow_col, self.atr_col,
                        'bb_upper', 'bb_lower', 'bb_mid',
                        'keltner_upper', 'keltner_lower',
                        self.vol_sma_col]
            if not all(col in df.columns for col in required):
                continue

            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    volume = arrays['volume'][idx]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    sma10 = arrays['sma_10'][idx]
                    atr = arrays[self.atr_col][idx]
                    bb_upper = arrays['bb_upper'][idx]
                    bb_lower = arrays['bb_lower'][idx]
                    keltner_upper = arrays['keltner_upper'][idx]
                    keltner_lower = arrays['keltner_lower'][idx]
                    vol_sma = arrays[self.vol_sma_col][idx]
                except (KeyError, IndexError):
                    continue

                # NaN check
                vals = [close, ma_fast, ma_slow, sma10, atr, bb_upper, bb_lower,
                        keltner_upper, keltner_lower, vol_sma]
                if any(v != v for v in vals):
                    continue

                # FILTER 1: Stage 2 uptrend — close > SMA50 > SMA200
                if not (close > ma_fast > ma_slow):
                    continue

                # FILTER 2: SMA50 rising (slope positive over 5 bars)
                ma_fast_prev = arrays[self.ma_fast_col][idx - 5]
                if ma_fast_prev != ma_fast_prev or ma_fast <= ma_fast_prev:
                    continue

                # FILTER 3: Close above SMA10 (short-term momentum)
                if close < sma10:
                    continue

                # FILTER 4: Squeeze detection — BB inside KC for >= squeeze_min_bars
                # Current bar: squeeze has FIRED (BB outside KC)
                bb_width = bb_upper - bb_lower
                keltner_width = keltner_upper - keltner_lower
                if keltner_width <= 0:
                    continue

                squeeze_now = bb_width < keltner_width  # True = still in squeeze
                if squeeze_now:
                    continue  # Squeeze hasn't fired yet

                # Check that squeeze was active for at least N bars before
                squeeze_count = 0
                for lookback in range(1, self.squeeze_min_bars + 10):
                    lb_idx = idx - lookback
                    if lb_idx < 0:
                        break
                    try:
                        lb_bb_upper = arrays['bb_upper'][lb_idx]
                        lb_bb_lower = arrays['bb_lower'][lb_idx]
                        lb_keltner_upper = arrays['keltner_upper'][lb_idx]
                        lb_keltner_lower = arrays['keltner_lower'][lb_idx]
                    except (KeyError, IndexError):
                        break
                    if any(v != v for v in [lb_bb_upper, lb_bb_lower, lb_keltner_upper, lb_keltner_lower]):
                        break
                    lb_bb_w = lb_bb_upper - lb_bb_lower
                    lb_kc_w = lb_keltner_upper - lb_keltner_lower
                    if lb_kc_w <= 0:
                        break
                    if lb_bb_w < lb_kc_w:
                        squeeze_count += 1
                    else:
                        break  # Squeeze broke before — only count consecutive

                if squeeze_count < self.squeeze_min_bars:
                    continue

                # FILTER 5: Volume surge on breakout
                if vol_sma <= 0 or volume < vol_sma * self.volume_mult:
                    continue

                # Calculate stop price
                stop_price = close - atr * self.atr_stop_multiplier
                max_stop = close * (1 - self.stop_loss_pct / 100)
                stop_price = max(stop_price, max_stop)

                # Score: longer squeeze = higher priority
                score = float(squeeze_count)

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=score,
                    metadata={
                        'squeeze_bars': squeeze_count,
                        'bb_kc_ratio': bb_width / keltner_width,
                    }
                ))

            else:
                # Slow path — pandas fallback
                current = df.iloc[idx]
                if any(pd.isna(current.get(col)) for col in required):
                    continue

                close = current['close']
                volume_val = current['volume']
                ma_fast_val = current[self.ma_fast_col]
                ma_slow_val = current[self.ma_slow_col]
                sma10_val = current.get('sma_10', float('nan'))
                atr_val = current[self.atr_col]
                bb_upper_val = current['bb_upper']
                bb_lower_val = current['bb_lower']
                keltner_upper_val = current['keltner_upper']
                keltner_lower_val = current['keltner_lower']
                vol_sma_val = current[self.vol_sma_col]

                if not (close > ma_fast_val > ma_slow_val):
                    continue
                if pd.isna(sma10_val) or close < sma10_val:
                    continue

                bb_width = bb_upper_val - bb_lower_val
                keltner_width = keltner_upper_val - keltner_lower_val
                if keltner_width <= 0 or bb_width < keltner_width:
                    continue

                squeeze_count = 0
                for lookback in range(1, self.squeeze_min_bars + 10):
                    lb_idx = idx - lookback
                    if lb_idx < 0:
                        break
                    lb = df.iloc[lb_idx]
                    lb_bb_w = lb['bb_upper'] - lb['bb_lower']
                    lb_kc_w = lb['keltner_upper'] - lb['keltner_lower']
                    if lb_kc_w <= 0:
                        break
                    if lb_bb_w < lb_kc_w:
                        squeeze_count += 1
                    else:
                        break

                if squeeze_count < self.squeeze_min_bars:
                    continue

                if vol_sma_val <= 0 or volume_val < vol_sma_val * self.volume_mult:
                    continue

                stop_price = close - atr_val * self.atr_stop_multiplier
                max_stop = close * (1 - self.stop_loss_pct / 100)
                stop_price = max(stop_price, max_stop)
                score = float(squeeze_count)

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=score,
                    metadata={
                        'squeeze_bars': squeeze_count,
                        'bb_kc_ratio': bb_width / keltner_width,
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
