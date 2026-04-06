"""Year-by-year return breakdown."""
from __future__ import annotations

from typing import cast

import pandas as pd

from vbacktest.analysis.report import TestResult


def run_annual_breakdown(
    equity: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> list[TestResult]:
    """Year-by-year strategy returns with optional benchmark cross-reference.

    Args:
        benchmark_returns: Optional date-indexed daily benchmark returns.
                           Bear years are flagged when benchmark return < -10%.
    """
    eq_df = equity.to_frame("equity")
    eq_df["year"] = pd.DatetimeIndex(eq_df.index).year

    bench_annual: dict[int, float] = {}
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bench_cum = (1 + benchmark_returns).cumprod()
        bench_df = bench_cum.to_frame("bench")
        bench_df["year"] = pd.DatetimeIndex(bench_df.index).year
        for yr, grp in bench_df.groupby("year"):
            bench_annual[int(cast(int, yr))] = (
                grp["bench"].iloc[-1] / grp["bench"].iloc[0] - 1
            ) * 100

    yearly = eq_df.groupby("year")["equity"].agg(["first", "last"])
    yearly["ret"] = (yearly["last"] / yearly["first"] - 1) * 100

    current_year = pd.Timestamp.now().year
    rows: list[str] = []
    n_negative = 0
    for yr, row in yearly.iterrows():  # type: ignore[assignment]
        yr_int = int(cast(int, yr))
        ret = float(row["ret"])
        bench_ret = bench_annual.get(yr_int)
        bench_str = f" (bench {bench_ret:+.0f}%)" if bench_ret is not None else ""
        bear_flag = (
            " [BEAR YEAR]" if (bench_ret is not None and bench_ret < -10) else ""
        )
        is_partial = yr_int == current_year
        partial_flag = " [YTD]" if is_partial else ""
        rows.append(f"{yr_int}: {ret:+.1f}%{bench_str}{bear_flag}{partial_flag}")
        if ret < 0 and not is_partial:
            n_negative += 1

    n_years = len(yearly) - 1  # exclude current partial year
    status = "PASS" if n_negative == 0 else ("WARN" if n_negative == 1 else "FAIL")

    return [
        TestResult(
            "Annual returns",
            f"{n_negative} negative year(s) of {n_years} | " + " | ".join(rows),
            status,
        )
    ]
