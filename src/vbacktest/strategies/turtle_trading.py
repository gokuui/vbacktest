"""Richard Dennis Turtle Trading System 2 (55-day breakout)."""
from __future__ import annotations

from vbacktest.exit_rules import StopLossRule, TrailingLowRule
from vbacktest.indicators import IndicatorSpec
from vbacktest.strategy import BarContext, ExitRule, Signal, SignalAction, Strategy


class TurtleTradingStrategy(Strategy):
    """Turtle Trading System 2.

    Entry:
    - Close breaks above 55-day rolling high.
    - Volume > 0.

    Exit:
    - Hard stop: 2 ATR below entry.
    - Trend reversal: Close below 20-day rolling low.

    Score: breakout magnitude relative to ATR.
    """

    def __init__(
        self,
        entry_breakout_days: int = 55,
        exit_breakdown_days: int = 20,
        atr_period: int = 20,
        atr_stop_multiplier: float = 2.0,
    ) -> None:
        self.entry_breakout_days = entry_breakout_days
        self.exit_breakdown_days = exit_breakdown_days
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier

        self._high_col = f"high_{entry_breakout_days}d"
        self._low_col = f"low_{exit_breakdown_days}d"
        self._atr_col = f"atr_{atr_period}"

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec("rolling_high", {
                "period": self.entry_breakout_days, "column": "high",
                "output_col": self._high_col,
            }),
            IndicatorSpec("rolling_low", {
                "period": self.exit_breakdown_days, "column": "low",
                "output_col": self._low_col,
            }),
            IndicatorSpec("atr", {"period": self.atr_period, "output_col": self._atr_col}),
        ]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals: list[Signal] = []
        universe_arrays = ctx.universe_arrays
        needed = (self._high_col, self._low_col, self._atr_col)

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
                if not all(c in arrays for c in needed):
                    continue
                try:
                    high_n = float(arrays[self._high_col][idx])
                    high_n_prev = float(arrays[self._high_col][idx - 1])
                    atr = float(arrays[self._atr_col][idx])
                except (KeyError, IndexError):
                    continue

                if any(v != v for v in (high_n, high_n_prev, atr)):
                    continue

                if ctx.current_prices and symbol in ctx.current_prices:
                    close = float(ctx.current_prices[symbol]["close"])
                    vol = float(ctx.current_prices[symbol].get("volume", 0))
                else:
                    continue

                # Breakout: prev close <= prev high AND current close > prev high
                try:
                    prev_close = float(arrays["close"][idx - 1])
                except (KeyError, IndexError):
                    continue

            else:
                import pandas as pd
                if not all(c in df.columns for c in needed):
                    continue
                bar = df.iloc[idx]
                prev = df.iloc[idx - 1]
                if any(pd.isna(bar[c]) for c in needed):
                    continue
                high_n_prev = float(prev[self._high_col])
                atr = float(bar[self._atr_col])
                close = float(bar["close"])
                vol = float(bar["volume"])
                prev_close = float(prev["close"])

            if not (prev_close <= high_n_prev < close):
                continue
            if vol <= 0:
                continue

            stop = close - self.atr_stop_multiplier * atr
            score = (close - high_n_prev) / atr if atr > 0 else 0.0

            signals.append(Signal(
                symbol=symbol, action=SignalAction.BUY, date=ctx.date,
                stop_price=stop, score=score, metadata={"atr": atr},
            ))

        return signals

    def exit_rules(self) -> list[ExitRule]:
        return [
            StopLossRule(),
            TrailingLowRule(lookback_days=self.exit_breakdown_days, low_column=self._low_col),
        ]
