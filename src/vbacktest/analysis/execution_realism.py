"""Post-hoc execution realism tests: slippage and commission ladders."""
from __future__ import annotations

from typing import Any

import pandas as pd

from vbacktest.analysis.report import TestResult


def run_execution_realism(
    trades: list[Any],
    equity: pd.Series,
    initial_capital: float,
    calendar_days: int,
) -> list[TestResult]:
    """Slippage and commission ladders based on position notional.

    Extra cost per trade = shares × entry_price × extra_slip_rt / 100.
    Subtracted from final equity to recompute CAGR. This slightly overstates
    robustness for high-trade-count strategies (~5–10% CAGR) because friction
    paid early reduces compounding. Adequate for go/no-go screening.
    """
    if not trades:
        return [
            TestResult("Slippage ladder", "N/A", "FAIL"),
            TestResult("Commission ladder", "N/A", "FAIL"),
        ]

    final_equity = float(equity.iloc[-1])
    total_notional = sum(t.shares * t.entry_price for t in trades)

    def _adjusted_cagr(extra_cost_total: float) -> float:
        adj_final = final_equity - extra_cost_total
        if adj_final <= 0:
            return -100.0
        return float(((adj_final / initial_capital) ** (365 / calendar_days) - 1) * 100)

    results: list[TestResult] = []

    # Slippage ladder
    slip_levels = [
        (0.0, "baseline"),
        (0.1, "0.1% RT"),
        (0.5, "0.5% RT"),
        (0.9, "0.9% RT"),
        (1.9, "~1.9% RT (hard)"),
    ]
    slip_rows = []
    hard_nogo = False
    for extra_rt, label in slip_levels:
        extra_cost = total_notional * extra_rt / 100
        cagr = _adjusted_cagr(extra_cost)
        pf = "PASS" if cagr > 0 else "FAIL"
        if label == "~1.9% RT (hard)" and cagr <= 0:
            hard_nogo = True
        slip_rows.append(f"{label}: {cagr:.1f}% ({pf})")
    results.append(
        TestResult(
            "Slippage ladder",
            " | ".join(slip_rows),
            "FAIL" if hard_nogo else "PASS",
            hard_nogo=hard_nogo,
            detail="Hard NO-GO if ~1.9% RT slippage turns CAGR negative",
        )
    )

    # Commission ladder
    comm_levels = [
        (0.0, "baseline"),
        (0.1, "+0.1% RT"),
        (0.2, "+0.2% RT (threshold)"),
        (0.3, "+0.3% RT (stress)"),
    ]
    comm_rows = []
    comm_fail = False
    for extra_rt, label in comm_levels:
        extra_cost = total_notional * extra_rt / 100
        cagr = _adjusted_cagr(extra_cost)
        pf = "PASS" if cagr > 0 else "FAIL"
        if label == "+0.2% RT (threshold)" and cagr <= 0:
            comm_fail = True
        comm_rows.append(f"{label}: {cagr:.1f}% ({pf})")
    results.append(
        TestResult(
            "Commission ladder",
            " | ".join(comm_rows),
            "FAIL" if comm_fail else "PASS",
        )
    )

    return results
