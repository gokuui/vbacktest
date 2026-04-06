"""Volume breakout strategy."""
from __future__ import annotations

from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule
from vbacktest.indicators import IndicatorSpec
from vbacktest.strategy import BarContext, ExitRule, Signal, SignalAction, Strategy


class VolumeBreakoutStrategy(Strategy):
    """Volume surge + price breakout above N-day high.

    Entry:
    - Volume > volume_surge_multiplier × average volume.
    - Close > N-day rolling high (breakout).

    Exit:
    - Stop loss + trailing ATR stop.

    Score: volume surge ratio × price momentum.
    """

    def __init__(
        self,
        volume_surge_multiplier: float = 3.0,
        volume_period: int = 20,
        breakout_period: int = 50,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ) -> None:
        self.volume_surge_multiplier = volume_surge_multiplier
        self.volume_period = volume_period
        self.breakout_period = breakout_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

        self._vol_col = f"volume_sma_{volume_period}"
        self._high_col = f"rolling_high_{breakout_period}"
        self._atr_col = f"atr_{atr_period}"

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec("volume_sma", {"period": self.volume_period, "output_col": self._vol_col}),
            IndicatorSpec("rolling_high", {"period": self.breakout_period}),
            IndicatorSpec("atr", {"period": self.atr_period, "output_col": self._atr_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals: list[Signal] = []
        universe_arrays = ctx.universe_arrays
        needed = (self._vol_col, self._high_col, self._atr_col)

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
                    vol_avg = float(arrays[self._vol_col][idx])
                    high_n = float(arrays[self._high_col][idx])
                    atr = float(arrays[self._atr_col][idx])
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in (vol_avg, high_n, atr)):
                    continue

                if ctx.current_prices and symbol in ctx.current_prices:
                    close = float(ctx.current_prices[symbol]["close"])
                    vol = float(ctx.current_prices[symbol].get("volume", 0))
                else:
                    continue

            else:
                import pandas as pd
                if not all(c in df.columns for c in needed):
                    continue
                bar = df.iloc[idx]
                if any(pd.isna(bar[c]) for c in needed):
                    continue
                vol_avg = float(bar[self._vol_col])
                high_n = float(bar[self._high_col])
                atr = float(bar[self._atr_col])
                close = float(bar["close"])
                vol = float(bar["volume"])

            if vol <= self.volume_surge_multiplier * vol_avg:
                continue
            if close <= high_n:
                continue

            stop = close - self.atr_multiplier * atr
            vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0
            score = vol_ratio * (close / high_n - 1.0) * 100

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
