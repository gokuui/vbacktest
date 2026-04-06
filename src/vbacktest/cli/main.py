"""vbacktest CLI entry point."""
from __future__ import annotations

import argparse
import sys

from vbacktest import __version__


def _cmd_strategies(args: argparse.Namespace) -> int:
    import vbacktest.strategies  # noqa: F401 — registers core 6
    from vbacktest.registry import strategy_registry

    for name in strategy_registry.keys():
        print(name)
    return 0


def _cmd_gonogo(args: argparse.Namespace) -> int:
    from vbacktest.analysis import GoNoGo
    from vbacktest.results import BacktestResult

    result = BacktestResult.from_json(args.results_path)
    equity = result.equity_series()
    initial_capital = float(
        result.config_snapshot.get("portfolio", {}).get("initial_capital", 100_000)
    )
    report = GoNoGo(
        result.trades,
        equity,
        initial_capital=initial_capital,
        mc_sims=args.mc_sims,
        seed=args.seed,
    ).run(name=args.results_path)
    report.print_terminal()
    if args.out:
        report.write_markdown(args.out)
    return 0 if report.overall != "FAIL" else 1


def _cmd_run(args: argparse.Namespace) -> int:
    import vbacktest.strategies  # noqa: F401 — registers core 6
    from vbacktest.config import BacktestConfig
    from vbacktest.engine import BacktestEngine
    from vbacktest.registry import strategy_registry
    from vbacktest.results import print_report

    strategy_cls = strategy_registry.get(args.strategy)
    config = BacktestConfig.simple(args.data, capital=args.capital)
    engine = BacktestEngine(config, strategy_cls())
    result = engine.run()
    print_report(result)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(result.to_json())
    return 0


def main(argv: list[str] | None = None) -> int:
    """vbacktest CLI."""
    parser = argparse.ArgumentParser(
        prog="vbacktest",
        description="vbacktest — market-agnostic backtesting framework",
    )
    parser.add_argument(
        "--version", action="version", version=f"vbacktest {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    # vbacktest strategies
    subparsers.add_parser("strategies", help="List registered strategies")

    # vbacktest gonogo
    gonogo = subparsers.add_parser("gonogo", help="Run go/no-go validation on saved results")
    gonogo.add_argument("results_path", help="Path to backtest results JSON")
    gonogo.add_argument("--mc-sims", type=int, default=5000, help="MC iterations (default 5000)")
    gonogo.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    gonogo.add_argument("--out", type=str, default=None, help="Write markdown report to path")

    # vbacktest run
    run = subparsers.add_parser("run", help="Run a backtest")
    run.add_argument("--strategy", required=True, help="Strategy name (e.g. ma_crossover)")
    run.add_argument("--data", required=True, help="Path to validated data directory")
    run.add_argument("--capital", type=float, default=100_000, help="Initial capital")
    run.add_argument("--output", type=str, default=None, help="Save results to JSON path")

    args = parser.parse_args(argv)

    if args.command == "strategies":
        return _cmd_strategies(args)
    elif args.command == "gonogo":
        return _cmd_gonogo(args)
    elif args.command == "run":
        return _cmd_run(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
