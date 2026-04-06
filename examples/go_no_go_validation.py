#!/usr/bin/env python3
"""Example: running go/no-go validation on backtest results.

Usage:
    python examples/go_no_go_validation.py --data /path/to/validated/ --out reports/
"""
from __future__ import annotations

import argparse
import sys

from vbacktest import BacktestConfig, BacktestEngine
from vbacktest.analysis import GoNoGo
from vbacktest.config import DataConfig
from vbacktest.strategies import MomentumStrategy


def main() -> int:
    parser = argparse.ArgumentParser(description="Go/no-go validation example")
    parser.add_argument("--data", required=True, help="Path to validated data directory")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--mc-sims", type=int, default=5000)
    parser.add_argument("--out", type=str, default=None, help="Write markdown report to path")
    args = parser.parse_args()

    # Run backtest
    config = BacktestConfig.simple(args.data, capital=args.capital)
    strategy = MomentumStrategy(roc_periods=(20, 60, 120), trend_period=200)
    engine = BacktestEngine(config, strategy)
    result = engine.run()

    if result.stats.total_trades == 0:
        print("No trades generated — nothing to validate.")
        return 1

    # Run go/no-go analysis
    report = GoNoGo(
        result.trades,
        result.equity_series(),
        initial_capital=args.capital,
        mc_sims=args.mc_sims,
    ).run(name="Momentum Strategy")

    report.print_terminal()

    if args.out:
        report.write_markdown(args.out)
        print(f"\nMarkdown report saved to: {args.out}")

    print(f"\nOverall verdict: {report.overall}")
    return 0 if report.overall != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
