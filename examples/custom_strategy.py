#!/usr/bin/env python3
"""Example: creating a custom strategy — Golden Cross (SMA50 crosses SMA200).

Usage:
    python examples/custom_strategy.py --data /path/to/validated/
"""
from __future__ import annotations

import argparse

from vbacktest import BarContext, BacktestConfig, BacktestEngine, Signal, SignalAction, Strategy
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule
from vbacktest.indicators import IndicatorSpec
from vbacktest.results import print_report


class GoldenCrossStrategy(Strategy):
    """Buy when the 50-day SMA crosses above the 200-day SMA.

    This is the classic Golden Cross pattern.  Stops are set 2 × ATR below
    entry with a trailing ATR stop to lock in profits.
    """

    def indicators(self) -> list[IndicatorSpec]:
        return [
            IndicatorSpec("sma", {"period": 50}),
            IndicatorSpec("sma", {"period": 200}),
            IndicatorSpec("atr", {"period": 14}),
        ]

    def exit_rules(self) -> list:
        return [StopLossRule(), TrailingATRStopRule(multiplier=2.5)]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals: list[Signal] = []
        use_fast = ctx.universe_arrays is not None

        for symbol, df in ctx.universe.items():
            if ctx.portfolio and ctx.portfolio.has_position(symbol):
                continue
            if symbol not in ctx.universe_idx:
                continue
            idx = ctx.universe_idx[symbol]
            if idx < 201:
                continue

            if use_fast and ctx.universe_arrays and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    sma50 = float(arrays["sma_50"][idx])
                    sma50_prev = float(arrays["sma_50"][idx - 1])
                    sma200 = float(arrays["sma_200"][idx])
                    sma200_prev = float(arrays["sma_200"][idx - 1])
                    close = float(arrays["close"][idx])
                    atr_val = float(arrays["atr"][idx])
                except (KeyError, IndexError):
                    continue
                if any(v != v for v in (sma50, sma200)):
                    continue
            else:
                import pandas as pd
                bar = df.iloc[idx]
                prev = df.iloc[idx - 1]
                if any(pd.isna(bar[c]) for c in ("sma_50", "sma_200", "atr")):
                    continue
                sma50 = float(bar["sma_50"])
                sma50_prev = float(prev["sma_50"])
                sma200 = float(bar["sma_200"])
                sma200_prev = float(prev["sma_200"])
                close = float(bar["close"])
                atr_val = float(bar["atr"])

            # Golden cross: SMA50 crosses above SMA200
            if sma50_prev <= sma200_prev and sma50 > sma200:
                stop = close - 2.0 * atr_val
                signals.append(
                    Signal(
                        symbol=symbol,
                        action=SignalAction.BUY,
                        date=ctx.date,
                        stop_price=stop,
                        score=sma50 / sma200,  # how far above = conviction
                    )
                )

        return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden Cross strategy example")
    parser.add_argument("--data", required=True, help="Path to validated data directory")
    parser.add_argument("--capital", type=float, default=100_000)
    args = parser.parse_args()

    config = BacktestConfig.simple(args.data, capital=args.capital)
    engine = BacktestEngine(config, GoldenCrossStrategy())
    result = engine.run()
    print_report(result)


if __name__ == "__main__":
    main()
