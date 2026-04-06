"""Monte Carlo simulation tests for go/no-go analysis."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vbacktest.analysis.defaults import MonteCarloThresholds
from vbacktest.analysis.report import TestResult


def _compound_cagr(
    pnl_pcts: np.ndarray, initial_capital: float, calendar_days: int
) -> float:
    """Compute CAGR from sequential compounding of portfolio-pct array."""
    if len(pnl_pcts) == 0:
        return 0.0
    factors = 1 + pnl_pcts / 100
    final = initial_capital * np.prod(factors)
    return float(((final / initial_capital) ** (365 / calendar_days) - 1) * 100)


def _portfolio_pnl_pcts(trades: list[Any], equity: pd.Series) -> np.ndarray:
    """Convert position PnL to portfolio-level % return per trade.

    Uses equity at each trade's exit date so compounding reflects the actual
    portfolio size — the correct input for MC simulations.

    Args:
        trades: Objects with ``.pnl`` (absolute) and ``.exit_date`` attributes.
        equity: Date-indexed equity Series.

    Returns:
        Array of portfolio-level % returns, one per trade.
    """
    out = []
    eq_index = equity.index
    for t in trades:
        pos = eq_index.searchsorted(t.exit_date, side="left")
        eq_val = float(equity.iloc[max(pos - 1, 0)])
        if eq_val > 0:
            out.append((t.pnl / eq_val) * 100)
        else:
            out.append(0.0)
    return np.array(out)


def run_monte_carlo(
    trades: list[Any],
    equity: pd.Series,
    initial_capital: float,
    calendar_days: int,
    n_sims: int = 5000,
    seed: int = 42,
    thresholds: MonteCarloThresholds | None = None,
) -> tuple[list[TestResult], list[Any]]:
    """Run three Monte Carlo methods.

    Returns:
        ``(test_results, [])`` — the empty list is retained for API compatibility.

    MC methods:
    - **Reshuffle**: 95th-percentile drawdown across random trade orderings.
    - **Bootstrap**: 5th-percentile CAGR when trades are resampled with replacement.
    - **Slip-inject**: 5th-percentile CAGR with 0–0.5% extra round-trip slippage.
    """
    if thresholds is None:
        thresholds = MonteCarloThresholds()

    rng = np.random.default_rng(seed)

    if not trades:
        empty = [
            TestResult(f"MC-{m}", "N/A", "FAIL")
            for m in ("Reshuffle", "Bootstrap", "Slip-inject")
        ]
        return empty, []

    port_pnls = _portfolio_pnl_pcts(trades, equity)
    n = len(port_pnls)
    results: list[TestResult] = []

    # 1. MC-Reshuffle: 95th-percentile max drawdown across random orderings
    dd_sims = []
    for _ in range(n_sims):
        sim = rng.permutation(port_pnls)
        eq = initial_capital * np.cumprod(1 + sim / 100)
        running_max = np.maximum.accumulate(eq)
        dd = float(abs(((eq - running_max) / running_max * 100).min()))
        dd_sims.append(dd)
    dd_arr = np.array(dd_sims)
    p5_dd = float(np.percentile(dd_arr, 5))
    p50_dd = float(np.percentile(dd_arr, 50))
    p95_dd = float(np.percentile(dd_arr, 95))
    reshuffle_status = (
        "PASS"
        if p95_dd < thresholds.max_reshuffle_dd_pass
        else ("WARN" if p95_dd < thresholds.max_reshuffle_dd_warn else "FAIL")
    )
    results.append(
        TestResult(
            "MC-Reshuffle (max DD)",
            f"P5/P50/P95 DD: {p5_dd:.1f}% / {p50_dd:.1f}% / {p95_dd:.1f}%",
            reshuffle_status,
        )
    )

    # 2. MC-Bootstrap: CAGR distribution from resampling with replacement
    cagrs = np.array(
        [
            _compound_cagr(
                rng.choice(port_pnls, size=n, replace=True), initial_capital, calendar_days
            )
            for _ in range(n_sims)
        ]
    )
    p5 = float(np.percentile(cagrs, 5))
    p50 = float(np.percentile(cagrs, 50))
    p95 = float(np.percentile(cagrs, 95))
    boot_status = "PASS" if p5 > thresholds.min_bootstrap_cagr_p5 else "FAIL"
    results.append(
        TestResult(
            "MC-Bootstrap",
            f"P5/P50/P95 CAGR: {p5:.1f}% / {p50:.1f}% / {p95:.1f}%",
            boot_status,
        )
    )

    # 3. MC-Slip-inject: 0–0.5% random extra round-trip slippage per trade
    notionals = np.array([t.shares * t.entry_price for t in trades])
    final_eq = float(equity.iloc[-1])
    cagrs_slip = []
    for _ in range(n_sims):
        extra_rts = rng.uniform(0, 0.5, size=n) / 100
        total_extra = float((notionals * extra_rts).sum())
        adj_final = final_eq - total_extra
        if adj_final <= 0:
            cagrs_slip.append(-100.0)
        else:
            cagrs_slip.append(
                ((adj_final / initial_capital) ** (365 / calendar_days) - 1) * 100
            )
    slip_arr = np.array(cagrs_slip)
    p5s = float(np.percentile(slip_arr, 5))
    p50s = float(np.percentile(slip_arr, 50))
    p95s = float(np.percentile(slip_arr, 95))
    slip_status = "PASS" if p5s > thresholds.min_slip_inject_cagr_p5 else "FAIL"
    results.append(
        TestResult(
            "MC-Slip-inject",
            f"P5/P50/P95 CAGR: {p5s:.1f}% / {p50s:.1f}% / {p95s:.1f}%",
            slip_status,
        )
    )

    return results, []
