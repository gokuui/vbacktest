#!/usr/bin/env python3
"""Basic backtest example using a built-in strategy.

Usage:
    python examples/basic_backtest.py --data /path/to/validated/
"""
from __future__ import annotations

import argparse
from vbacktest import BacktestConfig, BacktestEngine
from vbacktest.config import DataConfig
from vbacktest.results import print_report
from vbacktest.strategies import MACrossoverStrategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic backtest example")
    parser.add_argument("--data", required=True, help="Path to validated data directory")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--max-positions", type=int, default=10)
    args = parser.parse_args()

    config = BacktestConfig.simple(
        args.data, capital=args.capital, max_positions=args.max_positions
    )
    strategy = MACrossoverStrategy(fast_period=10, slow_period=30, trend_period=200)
    engine = BacktestEngine(config, strategy)
    result = engine.run()
    print_report(result)


if __name__ == "__main__":
    main()
