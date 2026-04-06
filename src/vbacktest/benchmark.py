"""Benchmark comparison utilities.

Compares strategy performance against a market benchmark (e.g. index returns).
The main function is :func:`compute_benchmark_comparison`.

Example::

    import pandas as pd
    from vbacktest.benchmark import compute_benchmark_comparison

    benchmark_returns = pd.Series(...)  # daily returns, date-indexed
    comparison = compute_benchmark_comparison(
        strategy_returns, benchmark_returns, stats
    )
    print(comparison["alpha"], comparison["beta"])
"""
from __future__ import annotations

import pandas as pd

from vbacktest.results import BacktestStats, _compute_benchmark_comparison


def compute_benchmark_comparison(
    strategy_returns: "pd.Series[float]",
    benchmark_returns: "pd.Series[float]",
    stats: BacktestStats,
) -> dict[str, float]:
    """Compare strategy returns against a benchmark.

    Args:
        strategy_returns: Date-indexed daily strategy returns (pct, not decimal).
        benchmark_returns: Date-indexed daily benchmark returns (pct, not decimal).
        stats: Computed :class:`~vbacktest.results.BacktestStats` for the strategy.

    Returns:
        Dict with keys: ``strategy_return``, ``benchmark_return``,
        ``total_return_diff``, ``alpha``, ``beta``, ``strategy_sharpe``,
        ``benchmark_sharpe``, ``sharpe_diff``, ``benchmark_max_dd``.
    """
    return _compute_benchmark_comparison(strategy_returns, benchmark_returns, stats)


__all__ = ["compute_benchmark_comparison"]
