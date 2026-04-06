"""Moving Average Crossover strategy."""
from __future__ import annotations

import numpy as np

from vbacktest.exit_rules import StopLossRule, TakeProfitPartialRule, TrailingATRStopRule, TrailingMARule
from vbacktest.indicators import IndicatorSpec
from vbacktest.strategy import BarContext, ExitRule, Signal, SignalAction, Strategy


class MACrossoverStrategy(Strategy):
    """MA Crossover with trend filter and volume confirmation.

    Entry:
    - Fast MA crosses above slow MA.
    - Price above trend MA.
    - Volume above average.

    Exit:
    - Stop loss + trailing ATR stop + trailing MA.
    - Optional partial take-profit.

    Score: distance from trend MA (momentum proxy).
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        trend_period: int = 200,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        trailing_ma_period: int = 10,
        volume_period: int = 20,
        partial_exit_r: float | None = None,
        partial_exit_fraction: float = 0.5,
    ) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trailing_ma_period = trailing_ma_period
        self.volume_period = volume_period
        self.partial_exit_r = partial_exit_r
        self.partial_exit_fraction = partial_exit_fraction

        self._fast_col = f"sma_{fast_period}"
        self._slow_col = f"sma_{slow_period}"
        self._trend_col = f"sma_{trend_period}"
        self._atr_col = f"atr_{atr_period}"
        self._trail_col = f"sma_{trailing_ma_period}"
        self._vol_col = f"volume_sma_{volume_period}"

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec("sma", {"period": self.fast_period}),
            IndicatorSpec("sma", {"period": self.slow_period}),
            IndicatorSpec("sma", {"period": self.trend_period}),
            IndicatorSpec("atr", {"period": self.atr_period, "output_col": self._atr_col}),
            IndicatorSpec("sma", {"period": self.trailing_ma_period}),
            IndicatorSpec("volume_sma", {"period": self.volume_period, "output_col": self._vol_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals: list[Signal] = []
        universe_arrays = ctx.universe_arrays

        required = [self._fast_col, self._slow_col, self._trend_col,
                    self._atr_col, self._trail_col, self._vol_col]

        for symbol, df in ctx.universe.items():
            if ctx.portfolio and ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < 1:
                continue

            # Fast path
            if universe_arrays and symbol in universe_arrays:
                arrays = universe_arrays[symbol]
                if not all(c in arrays for c in required):
                    continue
                try:
                    fc, fp = arrays[self._fast_col][idx], arrays[self._fast_col][idx - 1]
                    sc, sp = arrays[self._slow_col][idx], arrays[self._slow_col][idx - 1]
                    trend = arrays[self._trend_col][idx]
                    atr = arrays[self._atr_col][idx]
                    vol_avg = arrays[self._vol_col][idx]
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in (fc, fp, sc, sp, trend, atr, vol_avg)):
                    continue

                if ctx.current_prices and symbol in ctx.current_prices:
                    close = float(ctx.current_prices[symbol]["close"])
                    vol = float(ctx.current_prices[symbol].get("volume", 0))
                else:
                    continue

            else:
                if not all(c in df.columns for c in required):
                    continue
                bar = df.iloc[idx]
                prev = df.iloc[idx - 1]
                import pandas as pd
                if any(pd.isna(bar[c]) for c in required):
                    continue
                fc, fp = float(bar[self._fast_col]), float(prev[self._fast_col])
                sc, sp = float(bar[self._slow_col]), float(prev[self._slow_col])
                trend = float(bar[self._trend_col])
                atr = float(bar[self._atr_col])
                vol_avg = float(bar[self._vol_col])
                close = float(bar["close"])
                vol = float(bar["volume"])

            # Entry conditions
            if not (fp <= sp and fc > sc):
                continue
            if close <= trend:
                continue
            if vol <= vol_avg:
                continue

            stop = close - self.atr_multiplier * atr
            score = (close - trend) / trend * 100

            signals.append(Signal(
                symbol=symbol, action=SignalAction.BUY, date=ctx.date,
                stop_price=stop, score=score, metadata={"atr": atr},
            ))

        return signals

    def exit_rules(self) -> list[ExitRule]:
        rules: list[ExitRule] = [
            StopLossRule(),
            TrailingATRStopRule(atr_column=self._atr_col, multiplier=self.atr_multiplier),
            TrailingMARule(ma_column=self._trail_col),
        ]
        if self.partial_exit_r is not None:
            rules.append(TakeProfitPartialRule(
                r_multiple=self.partial_exit_r, fraction=self.partial_exit_fraction
            ))
        return rules
