"""Stan Weinstein Stage 2 Breakout Strategy.

Based on Stan Weinstein's "Secrets for Profiting in Bull and Bear Markets" (1988).

KEY DIFFERENCE from existing strategies:
- Existing strategies (momentum, rs_breakout, etc.) require price ALREADY above SMA200.
  They buy stocks well into Stage 2 (established uptrend).
- Weinstein catches the TRANSITION from Stage 1 (base/flat) to Stage 2 (uptrend START).
  This is the earliest possible entry — when the 30-week SMA slope just turned positive.

Stage Analysis:
  Stage 1: Accumulation (price flat, SMA150 flat) — institutions quietly buying
  Stage 2: Uptrend (price above SMA150, SMA150 slopes up) — public follows ← ENTRY
  Stage 3: Distribution (price fails to make new highs) ← skip
  Stage 4: Downtrend (price below SMA150) ← exit signal

Entry Conditions:
1. SMA150 slope is positive: SMA150[today] > SMA150[20 days ago] (slope just turned up)
2. Price above SMA150: stock has crossed into Stage 2
3. Volume confirmation: volume ≥ 1.2× 30-day average (institutional buying)
4. Not too late: price within 25% of 52-week high (not extended)
5. Price JUST crossed SMA150: SMA150 was below close within last 5 bars (recent transition)

Exit:
1. Price closes below SMA150 (Stage 3/4 entry — full exit)
2. Hard stop: 8% below entry (Weinstein uses ~15%, but NSE volatility warrants tighter)
3. No time stop — Weinstein holds for months

Philosophy:
- Patience: wait for Stage 2 confirmation, not Stage 1 speculation
- Follow institutional money: volume is the key confirmation
- Give trades room: no short time stops, exit only on trend reversal
"""
from __future__ import annotations



import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TrailingMARule


class WeinsteinStage2Strategy(Strategy):
    """Stan Weinstein Stage 2 breakout — buy Stage 1→2 transition."""

    def __init__(
        self,
        sma_period: int = 150,         # 30-week SMA (Weinstein uses 30-week)
        slope_lookback: int = 20,       # days to measure SMA slope
        transition_lookback: int = 5,   # bars to check if price JUST crossed SMA
        volume_period: int = 30,        # volume average period
        volume_surge: float = 1.2,      # volume must be ≥ this × average
        near_high_pct: float = 25.0,    # must be within 25% of 52-week high
        stop_loss_pct: float = 8.0,     # hard stop below entry
        atr_period: int = 20,           # ATR for trailing stop
        atr_stop_multiplier: float = 2.0,  # ATR trailing multiplier (wide for long holds)
        trailing_exit_type: str = 'sma',   # 'sma' = SMA150 trail, 'atr' = ATR trail
    ):
        self.sma_period = sma_period
        self.slope_lookback = slope_lookback
        self.transition_lookback = transition_lookback
        self.volume_period = volume_period
        self.volume_surge = volume_surge
        self.near_high_pct = near_high_pct
        self.stop_loss_pct = stop_loss_pct
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.trailing_exit_type = trailing_exit_type

        self.sma_col = f'sma_{sma_period}'
        self.vol_col = f'volume_sma_{volume_period}'
        self.atr_col = f'atr_{atr_period}'
        self.high52_col = 'high_52w'

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.sma_period}),
            IndicatorSpec('volume_sma', {'period': self.volume_period, 'output_col': self.vol_col}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': 252, 'column': 'high', 'output_col': self.high52_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.sma_period + self.slope_lookback, 260)

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < min_bars:
                continue

            required = [self.sma_col, self.vol_col, self.atr_col, self.high52_col]
            if not all(col in df.columns for col in required):
                continue

            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close     = arrays['close'][idx]
                    volume    = arrays['volume'][idx]
                    sma_now   = arrays[self.sma_col][idx]
                    sma_past  = arrays[self.sma_col][idx - self.slope_lookback]
                    vol_avg   = arrays[self.vol_col][idx]
                    atr       = arrays[self.atr_col][idx]
                    high_52w  = arrays[self.high52_col][idx]
                    # Recent SMA vs close — check if crossing is fresh
                    sma_recent = arrays[self.sma_col][idx - self.transition_lookback:idx]
                    close_recent = arrays['close'][idx - self.transition_lookback:idx]
                except (KeyError, IndexError):
                    continue

                if (close != close or sma_now != sma_now or sma_past != sma_past
                        or vol_avg != vol_avg or atr != atr or high_52w != high_52w):
                    continue
            else:
                current = df.iloc[idx]
                if any(pd.isna(current[col]) for col in required):
                    continue
                close    = current['close']
                volume   = current['volume']
                sma_now  = current[self.sma_col]
                sma_past = df.iloc[idx - self.slope_lookback][self.sma_col]
                vol_avg  = current[self.vol_col]
                atr      = current[self.atr_col]
                high_52w = current[self.high52_col]
                sma_recent   = df[self.sma_col].iloc[idx - self.transition_lookback:idx].values
                close_recent = df['close'].iloc[idx - self.transition_lookback:idx].values

            if sma_now <= 0 or vol_avg <= 0 or high_52w <= 0:
                continue

            # CONDITION 1: SMA slope is positive (trend turned up)
            if sma_now <= sma_past:
                continue

            # CONDITION 2: Price above SMA (in Stage 2)
            if close <= sma_now:
                continue

            # CONDITION 3: RECENT transition (price just crossed above SMA)
            # At least one of the last N bars had close <= SMA (was below, now above)
            was_below = any(
                c <= s for c, s in zip(close_recent, sma_recent)
                if c == c and s == s  # NaN check
            )
            if not was_below:
                continue  # Price has been above SMA150 for too long — not a fresh breakout

            # CONDITION 4: Volume surge (institutional confirmation)
            if volume < vol_avg * self.volume_surge:
                continue

            # CONDITION 5: Not too extended from 52-week high
            if close < high_52w * (1 - self.near_high_pct / 100):
                continue

            # Stop price
            stop_price = close * (1 - self.stop_loss_pct / 100)
            if stop_price <= 0:
                continue

            # Score: SMA slope strength + volume surge
            slope_pct = (sma_now - sma_past) / sma_past * 100
            vol_ratio = volume / vol_avg
            score = slope_pct * 10 + vol_ratio

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'sma_slope_pct': round(slope_pct, 3),
                    'vol_ratio': round(vol_ratio, 2),
                    f'sma_{self.sma_period}': round(sma_now, 2),
                }
            ))

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'sma':
            # Exit when price falls below SMA150 — the Weinstein classic exit
            rules.append(TrailingMARule(ma_column=self.sma_col))
        else:
            # ATR trailing stop for faster exits
            rules.append(TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier,
            ))
        return rules
