"""Bollinger Band breakout strategy."""
from __future__ import annotations

from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule
from vbacktest.indicators import IndicatorSpec
from vbacktest.strategy import BarContext, ExitRule, Signal, SignalAction, Strategy


class BollingerBreakoutStrategy(Strategy):
    """Bollinger Band breakout with trend and volume confirmation.

    Entry:
    - Close breaks above upper Bollinger Band.
    - Price above trend MA.
    - Volume above average.

    Exit:
    - Stop loss + trailing ATR stop.

    Score: distance above upper band (breakout strength).
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        trend_period: int = 200,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        volume_period: int = 20,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.volume_period = volume_period

        self._atr_col = f"atr_{atr_period}"
        self._vol_col = f"volume_sma_{volume_period}"
        self._trend_col = f"sma_{trend_period}"

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec("bollinger_bands", {"period": self.bb_period, "num_std": self.bb_std}),
            IndicatorSpec("sma", {"period": self.trend_period}),
            IndicatorSpec("atr", {"period": self.atr_period, "output_col": self._atr_col}),
            IndicatorSpec("volume_sma", {"period": self.volume_period, "output_col": self._vol_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals: list[Signal] = []
        universe_arrays = ctx.universe_arrays

        for symbol, df in ctx.universe.items():
            if ctx.portfolio and ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue

            idx = ctx.universe_idx[symbol]
            if idx < 1:
                continue

            if universe_arrays and symbol in universe_arrays:
                arrays = universe_arrays[symbol]
                needed = ("bb_upper", self._trend_col, self._atr_col, self._vol_col)
                if not all(c in arrays for c in needed):
                    continue
                try:
                    bb_upper = float(arrays["bb_upper"][idx])
                    trend = float(arrays[self._trend_col][idx])
                    atr = float(arrays[self._atr_col][idx])
                    vol_avg = float(arrays[self._vol_col][idx])
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in (bb_upper, trend, atr, vol_avg)):
                    continue

                if ctx.current_prices and symbol in ctx.current_prices:
                    close = float(ctx.current_prices[symbol]["close"])
                    vol = float(ctx.current_prices[symbol].get("volume", 0))
                else:
                    continue

            else:
                import pandas as pd
                needed = ("bb_upper", self._trend_col, self._atr_col, self._vol_col)
                if not all(c in df.columns for c in needed):
                    continue
                bar = df.iloc[idx]
                if any(pd.isna(bar[c]) for c in needed):
                    continue
                bb_upper = float(bar["bb_upper"])
                trend = float(bar[self._trend_col])
                atr = float(bar[self._atr_col])
                vol_avg = float(bar[self._vol_col])
                close = float(bar["close"])
                vol = float(bar["volume"])

            if close <= bb_upper:
                continue
            if close <= trend:
                continue
            if vol <= vol_avg:
                continue

            stop = close - self.atr_multiplier * atr
            score = (close - bb_upper) / bb_upper * 100

            signals.append(Signal(
                symbol=symbol, action=SignalAction.BUY, date=ctx.date,
                stop_price=stop, score=score, metadata={"atr": atr},
            ))

        return signals

    def exit_rules(self) -> list[ExitRule]:
        return [
            StopLossRule(),
            TrailingATRStopRule(atr_column=self._atr_col, multiplier=self.atr_multiplier),
        ]
