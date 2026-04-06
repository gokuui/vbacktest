"""Multi-period momentum strategy."""
from __future__ import annotations

import pandas as pd

from vbacktest.exit_rules import StopLossRule, TimeStopRule, TrailingATRStopRule
from vbacktest.indicators import IndicatorSpec
from vbacktest.strategy import BarContext, ExitRule, Signal, SignalAction, Strategy


class MomentumStrategy(Strategy):
    """Multi-period momentum breakout.

    Entry:
    - ROC above threshold over multiple periods.
    - Price above trend MA.
    - Price at or near 52-week high.

    Exit:
    - Stop loss + trailing ATR stop + time stop.

    Score: sum of ROC values across periods.
    """

    def __init__(
        self,
        roc_periods: tuple[int, ...] = (20, 60, 120),
        roc_threshold: float = 5.0,
        trend_period: int = 200,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        high_period: int = 252,
        max_holding_days: int = 60,
    ) -> None:
        self.roc_periods = roc_periods
        self.roc_threshold = roc_threshold
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.high_period = high_period
        self.max_holding_days = max_holding_days

        self._atr_col = f"atr_{atr_period}"
        self._trend_col = f"sma_{trend_period}"
        self._high_col = f"rolling_high_{high_period}"
        self._roc_cols = [f"roc_{p}" for p in roc_periods]

    def indicators(self) -> list[IndicatorSpec]:
        specs: list[IndicatorSpec] = [
            IndicatorSpec("sma", {"period": self.trend_period}),
            IndicatorSpec("atr", {"period": self.atr_period, "output_col": self._atr_col}),
            IndicatorSpec("rolling_high", {"period": self.high_period}),
        ]
        for p in self.roc_periods:
            specs.append(IndicatorSpec("roc", {"period": p}))
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals: list[Signal] = []
        universe_arrays = ctx.universe_arrays

        needed = (self._trend_col, self._atr_col, self._high_col) + tuple(self._roc_cols)

        for symbol, df in ctx.universe.items():
            if ctx.portfolio and ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]

            if universe_arrays and symbol in universe_arrays:
                arrays = universe_arrays[symbol]
                if not all(c in arrays for c in needed):
                    continue
                try:
                    trend = float(arrays[self._trend_col][idx])
                    atr = float(arrays[self._atr_col][idx])
                    high_52 = float(arrays[self._high_col][idx])
                    rocs = [float(arrays[c][idx]) for c in self._roc_cols]
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in (trend, atr, high_52) + tuple(rocs)):
                    continue

                if ctx.current_prices and symbol in ctx.current_prices:
                    close = float(ctx.current_prices[symbol]["close"])
                else:
                    continue

            else:
                if not all(c in df.columns for c in needed):
                    continue
                bar = df.iloc[idx]
                if any(pd.isna(bar[c]) for c in needed):
                    continue
                trend = float(bar[self._trend_col])
                atr = float(bar[self._atr_col])
                high_52 = float(bar[self._high_col])
                rocs = [float(bar[c]) for c in self._roc_cols]
                close = float(bar["close"])

            if close <= trend:
                continue
            if any(r < self.roc_threshold for r in rocs):
                continue
            if close < high_52 * 0.95:
                continue

            stop = close - self.atr_multiplier * atr
            score = sum(rocs)

            signals.append(Signal(
                symbol=symbol, action=SignalAction.BUY, date=ctx.date,
                stop_price=stop, score=score, metadata={"atr": atr},
            ))

        return signals

    def exit_rules(self) -> list[ExitRule]:
        return [
            StopLossRule(),
            TrailingATRStopRule(atr_column=self._atr_col, multiplier=self.atr_multiplier),
            TimeStopRule(max_holding_days=self.max_holding_days),
        ]
