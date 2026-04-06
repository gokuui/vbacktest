"""Rolling and regime-based validation tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from vbacktest.analysis.report import TestResult


def run_regime_rolling(
    equity: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> list[TestResult]:
    """Rolling Sharpe, annual consistency, alpha/beta, and overall Sharpe.

    Args:
        benchmark_returns: Optional date-indexed daily benchmark returns.
                           Replaces the NSE-specific ``nifty_returns`` parameter.
    """
    if len(equity) < 252:
        return [TestResult("Regime tests", "insufficient data", "WARN")]

    port_returns = equity.pct_change().dropna()
    results: list[TestResult] = []

    # Rolling 12-month Sharpe (3-month stride)
    window = 252
    stride = 63
    sharpes = []
    for start in range(0, len(port_returns) - window, stride):
        w = port_returns.iloc[start : start + window]
        ann_std = float(w.std() * np.sqrt(252))
        if ann_std > 0:
            sharpes.append(float(w.mean() * 252) / ann_std)
    pct_pos = sum(1 for s in sharpes if s > 0) / len(sharpes) * 100 if sharpes else 0.0
    roll_status = "PASS" if pct_pos >= 60 else ("WARN" if pct_pos >= 50 else "FAIL")
    results.append(
        TestResult("Rolling Sharpe (12m)", f"{pct_pos:.0f}% positive windows", roll_status)
    )

    # Annual consistency
    eq_df = equity.to_frame("equity")
    eq_df["year"] = eq_df.index.year
    yearly = eq_df.groupby("year")["equity"].agg(["first", "last"])
    yearly["ret"] = (yearly["last"] / yearly["first"] - 1) * 100
    pct_pos_years = float((yearly["ret"] > 0).mean() * 100)
    yr_status = "PASS" if pct_pos_years >= 60 else ("WARN" if pct_pos_years >= 50 else "FAIL")
    results.append(
        TestResult("Annual consistency", f"{pct_pos_years:.0f}% profitable years", yr_status)
    )

    # Alpha / Beta vs benchmark
    if benchmark_returns is not None and len(benchmark_returns) > 50:
        aligned = pd.DataFrame(
            {"port": port_returns, "bench": benchmark_returns}
        ).dropna()
        if len(aligned) > 50:
            slope, intercept, _, _, _ = stats.linregress(
                aligned["bench"].values, aligned["port"].values
            )
            beta = float(slope)
            alpha_ann = float(intercept) * 252 * 100
            results.append(
                TestResult(
                    "Alpha vs benchmark",
                    f"{alpha_ann:.2f}% ann.",
                    "PASS" if alpha_ann > 0 else "FAIL",
                )
            )
            results.append(
                TestResult(
                    "Beta vs benchmark",
                    f"{beta:.2f}",
                    "PASS" if beta < 0.8 else ("WARN" if beta < 1.0 else "FAIL"),
                )
            )
        else:
            results.append(TestResult("Alpha vs benchmark", "N/A (insufficient overlap)", "WARN"))
            results.append(TestResult("Beta vs benchmark", "N/A (insufficient overlap)", "WARN"))
    else:
        results.append(TestResult("Alpha vs benchmark", "N/A (no benchmark data)", "WARN"))
        results.append(TestResult("Beta vs benchmark", "N/A (no benchmark data)", "WARN"))

    # Overall Sharpe
    ann_std = float(port_returns.std() * np.sqrt(252))
    sharpe = float(port_returns.mean() * 252) / ann_std if ann_std > 0 else 0.0
    sharpe_status = "PASS" if sharpe >= 1.0 else ("WARN" if sharpe >= 0.7 else "FAIL")
    results.append(TestResult("Sharpe ratio", f"{sharpe:.2f}", sharpe_status))

    return results
