"""RSI mean reversion strategy."""
from __future__ import annotations

from vbacktest.exit_rules import StopLossRule, TakeProfitPartialRule, TimeStopRule
from vbacktest.indicators import IndicatorSpec
from vbacktest.strategy import BarContext, ExitRule, Signal, SignalAction, Strategy


class RSIMeanReversionStrategy(Strategy):
    """RSI-based mean reversion.

    Entry:
    - RSI crosses below oversold threshold.
    - Optional trend filter (price above slow MA).
    - Optional volume filter.

    Exit:
    - Stop loss + time stop.
    - Optional partial take-profit.

    Score: inverse RSI (more oversold = higher score).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        trend_period: int | None = 200,
        volume_period: int | None = 20,
        stop_loss_pct: float = 8.0,
        max_holding_days: int = 20,
        partial_exit_r: float | None = 2.0,
        partial_exit_fraction: float = 0.5,
    ) -> None:
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.trend_period = trend_period
        self.volume_period = volume_period
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_days = max_holding_days
        self.partial_exit_r = partial_exit_r
        self.partial_exit_fraction = partial_exit_fraction

        self._trend_col = f"sma_{trend_period}" if trend_period else None
        self._vol_col = f"volume_sma_{volume_period}" if volume_period else None

    def indicators(self) -> list[IndicatorSpec]:
        specs: list[IndicatorSpec] = [IndicatorSpec("rsi", {"period": self.rsi_period})]
        if self.trend_period:
            specs.append(IndicatorSpec("sma", {"period": self.trend_period}))
        if self.volume_period:
            specs.append(IndicatorSpec("volume_sma", {
                "period": self.volume_period, "output_col": self._vol_col
            }))
        return specs

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
                if "rsi" not in arrays:
                    continue
                try:
                    rsi_cur = arrays["rsi"][idx]
                    rsi_prev = arrays["rsi"][idx - 1]
                except (KeyError, IndexError):
                    continue

                if rsi_cur != rsi_cur or rsi_prev != rsi_prev:
                    continue

                if ctx.current_prices and symbol in ctx.current_prices:
                    close = float(ctx.current_prices[symbol]["close"])
                    vol = float(ctx.current_prices[symbol].get("volume", 0))
                else:
                    continue

                if self._trend_col and self._trend_col in arrays:
                    trend = arrays[self._trend_col][idx]
                    if trend != trend or close <= float(trend):
                        continue

                if self._vol_col and self._vol_col in arrays:
                    vol_avg = arrays[self._vol_col][idx]
                    if vol_avg != vol_avg or vol <= float(vol_avg):
                        continue

            else:
                if "rsi" not in df.columns:
                    continue
                import pandas as pd
                bar = df.iloc[idx]
                prev = df.iloc[idx - 1]
                if pd.isna(bar["rsi"]) or pd.isna(prev["rsi"]):
                    continue
                rsi_cur = float(bar["rsi"])
                rsi_prev = float(prev["rsi"])
                close = float(bar["close"])
                vol = float(bar["volume"])

                if self._trend_col and self._trend_col in df.columns:
                    if pd.isna(bar[self._trend_col]) or close <= float(bar[self._trend_col]):
                        continue
                if self._vol_col and self._vol_col in df.columns:
                    if pd.isna(bar[self._vol_col]) or vol <= float(bar[self._vol_col]):
                        continue

            # Entry: RSI crosses below oversold
            if not (rsi_prev >= self.oversold_threshold > rsi_cur):
                continue

            stop = close * (1 - self.stop_loss_pct / 100)
            score = self.oversold_threshold - rsi_cur  # more oversold = higher score

            signals.append(Signal(
                symbol=symbol, action=SignalAction.BUY, date=ctx.date,
                stop_price=stop, score=score,
            ))

        return signals

    def exit_rules(self) -> list[ExitRule]:
        rules: list[ExitRule] = [
            StopLossRule(),
            TimeStopRule(max_holding_days=self.max_holding_days),
        ]
        if self.partial_exit_r is not None:
            rules.append(TakeProfitPartialRule(
                r_multiple=self.partial_exit_r, fraction=self.partial_exit_fraction
            ))
        return rules
