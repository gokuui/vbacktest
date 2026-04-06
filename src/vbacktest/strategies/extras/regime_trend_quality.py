"""Regime-Filtered Trend Quality strategy.

Same as trend_quality_breakout but with a market regime filter:
only enters new positions when Nifty 50 is above its SMA(regime_ma_period).
When Nifty is below SMA → no new entries (existing positions still managed by exits).

This should dramatically cut drawdowns by avoiding entries during bear markets.
The DD in trend_quality comes from entering during market corrections when
individual stocks still show "clean trends" but the macro is turning.

For Track B prop trading: combine regime filter with tight stops for ultra-low DD.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingMARule, TimeStopRule


class RegimeTrendQualityStrategy(Strategy):
    """Trend quality breakout with Nifty regime filter."""

    def __init__(
        self,
        er_period: int = 50,
        er_threshold: float = 0.50,
        ma_fast: int = 50,
        ma_slow: int = 150,
        breakout_period: int = 50,
        breakout_pct: float = 3.0,
        volume_mult: float = 1.0,
        atr_period: int = 14,
        atr_stop_multiplier: float = 0.7,
        stop_loss_pct: float = 3.0,
        max_holding_days: int = 14,
        trailing_exit_type: str = 'sma10',
        regime_ma_period: int = 200,
        nifty_data_path: str = 'data/validated/nifty 50.parquet',
    ):
        self.er_period = er_period
        self.er_threshold = er_threshold
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.breakout_period = breakout_period
        self.breakout_pct = breakout_pct
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.trailing_exit_type = trailing_exit_type
        self.regime_ma_period = regime_ma_period
        self.nifty_data_path = nifty_data_path

        self.ma_fast_col = f'sma_{ma_fast}'
        self.ma_slow_col = f'sma_{ma_slow}'
        self.atr_col = f'atr_{atr_period}'
        self.vol_sma_col = 'volume_sma_20'

        # Load Nifty data and compute regime
        self._nifty_regime = self._load_nifty_regime()

    def _load_nifty_regime(self):
        """Load Nifty 50 data and compute regime (above/below SMA)."""
        path = Path(self.nifty_data_path)
        if not path.exists():
            # Try alternative paths
            for p in ['data/validated/nifty 50.parquet',
                       '/home/vinay/code/loser/data/validated/nifty 50.parquet',
                       '/home/vinay/code/loser-revival/data/validated/nifty 50.parquet']:
                if Path(p).exists():
                    path = Path(p)
                    break

        if not path.exists():
            return {}

        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df = df.set_index('date')
        df.index = pd.to_datetime(df.index)

        sma = df['close'].rolling(window=self.regime_ma_period, min_periods=self.regime_ma_period).mean()
        regime = df['close'] > sma  # True = bull, False = bear

        # Convert to dict for O(1) lookup
        return {ts: bool(val) for ts, val in regime.items() if pd.notna(val)}

    def _is_bull_regime(self, date):
        """Check if market is in bull regime on given date."""
        if not self._nifty_regime:
            return True  # If no data, assume bull (fail-open)
        # Find the closest date <= given date
        ts = pd.Timestamp(date)
        if ts in self._nifty_regime:
            return self._nifty_regime[ts]
        # Fallback: check nearby dates
        for offset in range(5):
            check = ts - pd.Timedelta(days=offset)
            if check in self._nifty_regime:
                return self._nifty_regime[check]
        return True  # Default to bull

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec('sma', {'period': self.ma_fast}),
            IndicatorSpec('sma', {'period': self.ma_slow}),
            IndicatorSpec('sma', {'period': 10}),
            IndicatorSpec('ema', {'period': 10}),
            IndicatorSpec('ema', {'period': 21}),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('rolling_high', {'period': self.breakout_period, 'output_col': f'high_{self.breakout_period}'}),
            IndicatorSpec('volume_sma', {'period': 20}),
        ]

    def _kaufman_er(self, close_arr, idx, period):
        if idx < period:
            return float('nan')
        direction = abs(close_arr[idx] - close_arr[idx - period])
        volatility = 0.0
        for j in range(idx - period + 1, idx + 1):
            v = abs(close_arr[j] - close_arr[j - 1])
            if v != v:
                return float('nan')
            volatility += v
        if volatility == 0:
            return float('nan')
        return direction / volatility

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        # REGIME FILTER: No new entries in bear market
        if not self._is_bull_regime(date):
            return []

        signals = []
        use_fast = ctx.universe_arrays is not None
        min_bars = max(self.ma_slow, self.er_period, self.breakout_period) + 30

        for symbol, df in ctx.universe.items():
            if ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue
            idx = ctx.universe_idx[symbol]
            if idx < min_bars:
                continue

            if use_fast and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    close = arrays['close'][idx]
                    volume = arrays['volume'][idx]
                    ma_fast = arrays[self.ma_fast_col][idx]
                    ma_slow = arrays[self.ma_slow_col][idx]
                    ema10 = arrays['ema_10'][idx]
                    ema21 = arrays['ema_21'][idx]
                    atr = arrays[self.atr_col][idx]
                    high_n = arrays[f'high_{self.breakout_period}'][idx]
                    vol_sma = arrays[self.vol_sma_col][idx]
                    close_arr = arrays['close']
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in [close, ma_fast, ma_slow, ema10, ema21, atr, high_n, vol_sma]):
                    continue

                # MA Stack
                if not (close > ema10 > ema21 > ma_fast > ma_slow):
                    continue

                # Kaufman ER
                er = self._kaufman_er(close_arr, idx, self.er_period)
                if er != er or er < self.er_threshold:
                    continue

                # Near N-day high
                if high_n <= 0:
                    continue
                pct_from_high = (high_n - close) / high_n * 100
                if pct_from_high > self.breakout_pct:
                    continue

                # SMA50 rising
                ma_fast_prev = arrays[self.ma_fast_col][idx - 5]
                if ma_fast_prev != ma_fast_prev or ma_fast <= ma_fast_prev:
                    continue

                # Volume
                if vol_sma <= 0 or volume < vol_sma * self.volume_mult:
                    continue

                stop_price = close - atr * self.atr_stop_multiplier
                max_stop = close * (1 - self.stop_loss_pct / 100)
                stop_price = max(stop_price, max_stop)

                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=stop_price,
                    score=er * 100,
                    metadata={'kaufman_er': er, 'pct_from_high': pct_from_high}
                ))
            else:
                continue

        return signals

    def exit_rules(self) -> list:
        rules = [StopLossRule()]
        if self.trailing_exit_type == 'sma10':
            rules.append(TrailingMARule(ma_column='sma_10'))
        elif self.trailing_exit_type == 'ema10':
            rules.append(TrailingMARule(ma_column='ema_10'))
        rules.append(TimeStopRule(max_holding_days=self.max_holding_days))
        return rules
