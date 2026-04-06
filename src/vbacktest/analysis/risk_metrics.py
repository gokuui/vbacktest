"""Risk metrics tests: drawdown, Kelly, risk-of-ruin, Calmar, and advanced risk."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from vbacktest.analysis.defaults import RiskThresholds
from vbacktest.analysis.report import TestResult


def run_risk_metrics(
    trades: list[Any],
    equity: pd.Series,
    mc_shuffled_sims: list[Any],  # retained for API compat — not used
    initial_capital: float,
    thresholds: RiskThresholds | None = None,
) -> list[TestResult]:
    """Core risk metrics: drawdown, Kelly, risk-of-ruin, streaks, Calmar.

    Args:
        mc_shuffled_sims: Unused — retained for API compatibility.
    """
    if thresholds is None:
        thresholds = RiskThresholds()

    if not trades or len(equity) < 2:
        return [TestResult("Risk metrics", "insufficient data", "FAIL")]

    pnl_pcts = np.array([t.pnl_pct for t in trades])
    winners = pnl_pcts[pnl_pcts > 0]
    losers = pnl_pcts[pnl_pcts < 0]
    results: list[TestResult] = []

    # Max drawdown
    running_max = equity.cummax()
    dd_series = (equity - running_max) / running_max * 100
    max_dd = float(abs(dd_series.min()))
    if max_dd <= thresholds.max_drawdown_pass:
        dd_status = "PASS"
    elif max_dd <= thresholds.max_drawdown_warn:
        dd_status = "WARN"
    else:
        dd_status = "FAIL"
    results.append(
        TestResult(
            "Max drawdown",
            f"{max_dd:.1f}%",
            dd_status,
            hard_nogo=(max_dd > thresholds.max_drawdown_hard_nogo),
        )
    )

    # Max drawdown duration
    in_dd = equity < running_max
    max_dd_days = 0
    current = 0
    for v in in_dd:
        current = current + 1 if v else 0
        max_dd_days = max(max_dd_days, current)
    dd_months = max_dd_days / 30
    if dd_months <= thresholds.max_dd_duration_pass_months:
        dur_status = "PASS"
    elif dd_months <= thresholds.max_dd_duration_warn_months:
        dur_status = "WARN"
    else:
        dur_status = "FAIL"
    results.append(
        TestResult(
            "Max DD duration",
            f"{dd_months:.1f} months ({max_dd_days}d)",
            dur_status,
        )
    )

    # Kelly fraction
    if len(winners) > 0 and len(losers) > 0:
        win_rate = len(winners) / len(pnl_pcts)
        avg_win = float(winners.mean())
        avg_loss = float(abs(losers.mean()))
        kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss) if avg_loss > 0 else 0.0
    else:
        kelly = 0.0
    if 0 < kelly <= thresholds.max_kelly_pass:
        k_status = "PASS"
    elif kelly > thresholds.max_kelly_pass:
        k_status = "WARN"
    else:
        k_status = "FAIL"
    results.append(
        TestResult("Kelly fraction", f"{kelly:.3f} (1/4K={kelly / 4:.3f})", k_status)
    )

    # Risk of ruin — bootstrap daily equity returns
    daily_returns = equity.pct_change().dropna().values
    n_days = len(daily_returns)
    ruin_threshold = initial_capital * 0.75
    rng = np.random.default_rng(42)
    ruin_count = 0
    n_ruin_sims = 1000
    for _ in range(n_ruin_sims):
        sim_returns = rng.choice(np.asarray(daily_returns), size=n_days, replace=True)
        eq_path = initial_capital * np.cumprod(1 + sim_returns)
        if np.any(eq_path < ruin_threshold):
            ruin_count += 1
    ror_pct = ruin_count / n_ruin_sims * 100
    if ror_pct <= thresholds.max_risk_of_ruin_pass:
        ror_status = "PASS"
    elif ror_pct <= thresholds.max_risk_of_ruin_warn:
        ror_status = "WARN"
    else:
        ror_status = "FAIL"
    results.append(
        TestResult(
            "Risk of ruin (25%)",
            f"{ror_pct:.1f}%",
            ror_status,
            hard_nogo=(ror_pct > thresholds.max_risk_of_ruin_warn),
        )
    )

    # Max consecutive losses
    risk_per_trade_pct = 1.0
    max_cons = curr_cons = 0
    for p in pnl_pcts:
        if p < 0:
            curr_cons += 1
            max_cons = max(max_cons, curr_cons)
        else:
            curr_cons = 0
    portfolio_impact = max_cons * risk_per_trade_pct
    if portfolio_impact <= 15:
        cons_status = "PASS"
    elif portfolio_impact <= 25:
        cons_status = "WARN"
    else:
        cons_status = "FAIL"
    results.append(
        TestResult(
            "Max consec. losses",
            f"{max_cons} ({portfolio_impact:.0f}% impact @ 1% risk)",
            cons_status,
        )
    )

    # Calmar ratio
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    cagr_pct = ((float(equity.iloc[-1]) / float(equity.iloc[0])) ** (365 / days) - 1) * 100
    calmar = cagr_pct / max_dd if max_dd > 0 else 0.0
    if calmar >= thresholds.min_calmar_pass:
        cal_status = "PASS"
    elif calmar >= thresholds.min_calmar_warn:
        cal_status = "WARN"
    else:
        cal_status = "FAIL"
    results.append(TestResult("Calmar ratio", f"{calmar:.2f}", cal_status))

    return results


def run_advanced_risk(
    trades: list[Any],
    equity: pd.Series,
    initial_capital: float,
    benchmark_returns: pd.Series | None = None,
) -> list[TestResult]:
    """Professional quant risk: tail risk, concentration, turnover, autocorrelation.

    Args:
        benchmark_returns: Optional date-indexed daily returns for regime analysis.
                           Replaces the NSE-specific ``nifty_returns`` parameter.
    """
    if not trades or len(equity) < 100:
        return [TestResult("Advanced risk", "insufficient data", "WARN")]

    daily_returns = equity.pct_change().dropna()
    pnl_pcts = np.array([t.pnl_pct for t in trades])
    pnl_amounts = np.array([t.pnl for t in trades])
    results: list[TestResult] = []

    # CVaR / Expected Shortfall (5%)
    var_5 = daily_returns.quantile(0.05)
    cvar_5 = daily_returns[daily_returns <= var_5].mean() * 100
    cvar_status = "PASS" if cvar_5 > -3.0 else ("WARN" if cvar_5 > -5.0 else "FAIL")
    results.append(TestResult("CVaR (5%)", f"{cvar_5:.2f}% daily", cvar_status))

    # Tail ratio (95th / |5th|)
    p95 = daily_returns.quantile(0.95)
    p5_abs = abs(daily_returns.quantile(0.05))
    tail_ratio = p95 / p5_abs if p5_abs > 0 else 0.0
    tail_status = "PASS" if tail_ratio >= 1.0 else ("WARN" if tail_ratio >= 0.7 else "FAIL")
    results.append(TestResult("Tail ratio (95/5)", f"{tail_ratio:.2f}", tail_status))

    # Stock concentration
    by_stock: dict[str, float] = defaultdict(float)
    for t in trades:
        by_stock[t.symbol] += t.pnl
    sorted_stocks = sorted(by_stock.values(), reverse=True)
    total_pnl = sum(sorted_stocks)
    if total_pnl > 0:
        top1_pct = sorted_stocks[0] / total_pnl * 100
        top5_pct = sum(sorted_stocks[:5]) / total_pnl * 100 if len(sorted_stocks) >= 5 else 100.0
        top10_pct = sum(sorted_stocks[:10]) / total_pnl * 100 if len(sorted_stocks) >= 10 else 100.0
    else:
        top1_pct = top5_pct = top10_pct = 100.0
    conc_status = "PASS" if top1_pct <= 20 else ("WARN" if top1_pct <= 30 else "FAIL")
    results.append(
        TestResult(
            "Stock concentration",
            f"Top1={top1_pct:.0f}% Top5={top5_pct:.0f}% Top10={top10_pct:.0f}%",
            conc_status,
            hard_nogo=(top1_pct > 50),
        )
    )

    # Annual turnover
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1.0)
    trades_per_year = len(trades) / years
    median_trade_value = float(
        np.median([abs(t.entry_price * t.shares) for t in trades])
    ) if trades else 0.0
    annual_turnover = trades_per_year * median_trade_value * 2 / initial_capital
    turn_status = "PASS" if annual_turnover <= 30 else "WARN"
    results.append(
        TestResult(
            "Annual turnover",
            f"{annual_turnover:.1f}x ({trades_per_year:.0f} trades/yr)",
            turn_status,
        )
    )

    # Return autocorrelation
    if len(daily_returns) > 20:
        autocorr_1 = float(daily_returns.autocorr(lag=1))
        autocorr_5 = float(daily_returns.autocorr(lag=5))
        ac_status = (
            "PASS" if abs(autocorr_1) < 0.1 and abs(autocorr_5) < 0.1 else "WARN"
        )
        results.append(
            TestResult(
                "Return autocorrelation",
                f"lag1={autocorr_1:.3f} lag5={autocorr_5:.3f}",
                ac_status,
            )
        )

    # Sortino ratio
    downside_dev = np.sqrt(np.mean(np.minimum(np.asarray(daily_returns.values), 0) ** 2)) * np.sqrt(252)
    ann_return = daily_returns.mean() * 252
    sortino = ann_return / downside_dev if downside_dev > 0 else 0.0
    sort_status = "PASS" if sortino >= 2.0 else ("WARN" if sortino >= 1.0 else "FAIL")
    results.append(TestResult("Sortino ratio", f"{sortino:.2f}", sort_status))

    # Worst drawdown recovery time
    running_max = equity.cummax()
    in_dd = (equity - running_max) < 0
    dd_starts: list[int] = []
    dd_ends: list[int] = []
    in_drawdown = False
    for i, val in enumerate(in_dd):
        if val and not in_drawdown:
            dd_starts.append(i)
            in_drawdown = True
        elif not val and in_drawdown:
            dd_ends.append(i)
            in_drawdown = False
    if in_drawdown:
        dd_ends.append(len(in_dd) - 1)
    if dd_starts:
        recovery_days = [
            dd_ends[i] - dd_starts[i]
            for i in range(min(len(dd_starts), len(dd_ends)))
        ]
        max_recovery = max(recovery_days)
        avg_recovery = float(np.mean(recovery_days))
        rec_status = (
            "PASS" if max_recovery <= 180 else ("WARN" if max_recovery <= 500 else "FAIL")
        )
        results.append(
            TestResult(
                "DD recovery time",
                f"max={max_recovery}d avg={avg_recovery:.0f}d ({len(dd_starts)} drawdowns)",
                rec_status,
            )
        )

    # Ulcer Index
    dd_pct = (equity - running_max) / running_max * 100
    ulcer_index = float(np.sqrt((dd_pct**2).mean()))
    ui_status = "PASS" if ulcer_index <= 5 else ("WARN" if ulcer_index <= 10 else "FAIL")
    results.append(TestResult("Ulcer Index", f"{ulcer_index:.2f}", ui_status))

    # Gain-to-pain ratio
    gains = float(pnl_amounts[pnl_amounts > 0].sum())
    losses = float(abs(pnl_amounts[pnl_amounts < 0].sum()))
    gtp = gains / losses if losses > 0 else 0.0
    gtp_status = "PASS" if gtp >= 2.0 else ("WARN" if gtp >= 1.5 else "FAIL")
    results.append(TestResult("Gain-to-pain ratio", f"{gtp:.2f}", gtp_status))

    # Monthly win rate
    monthly_returns = equity.resample("ME").last().pct_change().dropna()
    monthly_win_rate = (monthly_returns > 0).mean() * 100
    mw_status = (
        "PASS" if monthly_win_rate >= 60 else ("WARN" if monthly_win_rate >= 50 else "FAIL")
    )
    results.append(
        TestResult(
            "Monthly win rate",
            f"{monthly_win_rate:.0f}% ({(monthly_returns > 0).sum()}/{len(monthly_returns)} months)",
            mw_status,
        )
    )

    # Trade distribution (skew/kurt)
    trade_skew = float(pd.Series(pnl_pcts).skew())  # type: ignore[arg-type]
    trade_kurt = float(pd.Series(pnl_pcts).kurtosis())  # type: ignore[arg-type]
    skew_status = (
        "PASS" if trade_skew > 0 else ("WARN" if trade_skew > -0.5 else "FAIL")
    )
    results.append(
        TestResult(
            "Trade distribution",
            f"skew={trade_skew:.2f} kurt={trade_kurt:.2f}",
            skew_status,
            detail="Positive skew preferred. Kurt>3 = fat tails.",
        )
    )

    # Worst single trade
    worst_trade = float(pnl_pcts.min()) if len(pnl_pcts) > 0 else 0.0
    wt_status = (
        "PASS" if worst_trade > -20 else ("WARN" if worst_trade > -50 else "FAIL")
    )
    results.append(
        TestResult(
            "Worst single trade",
            f"{worst_trade:.1f}%",
            wt_status,
            hard_nogo=(worst_trade < -50),
        )
    )

    # Worst/best month
    if len(monthly_returns) > 0:
        worst_month = float(monthly_returns.min() * 100)
        best_month = float(monthly_returns.max() * 100)
        wm_status = (
            "PASS" if worst_month > -10 else ("WARN" if worst_month > -20 else "FAIL")
        )
        results.append(
            TestResult(
                "Worst/best month",
                f"worst={worst_month:.1f}% best={best_month:.1f}%",
                wm_status,
            )
        )

    # Win/loss streaks
    max_win_streak = max_loss_streak = curr_win = curr_loss = 0
    for p in pnl_pcts:
        if p > 0:
            curr_win += 1
            curr_loss = 0
            max_win_streak = max(max_win_streak, curr_win)
        else:
            curr_loss += 1
            curr_win = 0
            max_loss_streak = max(max_loss_streak, curr_loss)
    streak_status = (
        "PASS" if max_loss_streak <= 15 else ("WARN" if max_loss_streak <= 30 else "FAIL")
    )
    results.append(
        TestResult(
            "Win/loss streaks",
            f"max_win={max_win_streak} max_loss={max_loss_streak}",
            streak_status,
        )
    )

    # Time in drawdown
    time_in_dd = (equity < equity.cummax()).mean() * 100
    tidd_status = (
        "PASS" if time_in_dd <= 60 else ("WARN" if time_in_dd <= 90 else "FAIL")
    )
    results.append(
        TestResult("Time in drawdown", f"{time_in_dd:.0f}% of days", tidd_status)
    )

    # IS/OOS consistency (first-half vs second-half Calmar)
    def _calmar(eq: pd.Series) -> float:
        days_ = max((eq.index[-1] - eq.index[0]).days, 1)
        cagr_ = ((float(eq.iloc[-1]) / float(eq.iloc[0])) ** (365 / days_) - 1) * 100
        dd_ = float(abs(((eq - eq.cummax()) / eq.cummax() * 100).min()))
        return cagr_ / dd_ if dd_ > 0 else 0.0

    mid = len(equity) // 2
    cal_first = _calmar(equity.iloc[:mid])
    cal_second = _calmar(equity.iloc[mid:])
    ratio = cal_second / cal_first if cal_first > 0 else 0.0
    oos_status = "PASS" if ratio >= 0.5 else ("WARN" if ratio >= 0.25 else "FAIL")
    results.append(
        TestResult(
            "IS/OOS consistency",
            f"1st half Cal={cal_first:.2f} 2nd half Cal={cal_second:.2f} ratio={ratio:.2f}",
            oos_status,
        )
    )

    # Yearly Calmar
    yearly_equity = equity.resample("YE").last()
    yearly_calmars = []
    for i in range(1, len(yearly_equity)):
        yr_eq = equity[
            (equity.index >= yearly_equity.index[i - 1])
            & (equity.index <= yearly_equity.index[i])
        ]
        if len(yr_eq) > 50:
            yearly_calmars.append(_calmar(yr_eq))
    if yearly_calmars:
        pct_good_years = sum(1 for c in yearly_calmars if c > 1) / len(yearly_calmars) * 100
        median_yr_cal = float(np.median(yearly_calmars))
        yr_status = (
            "PASS" if pct_good_years >= 70 else ("WARN" if pct_good_years >= 50 else "FAIL")
        )
        results.append(
            TestResult(
                "Yearly Calmar",
                f"{pct_good_years:.0f}% years Cal>1, median={median_yr_cal:.2f}",
                yr_status,
            )
        )

    # Deflated Sharpe Ratio (De Prado 2018, N=80 strategies tested)
    N_STRATEGIES_TESTED = 80
    ann_return_pct = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    observed_sharpe = ann_return_pct / ann_vol if ann_vol > 0 else 0.0
    expected_max_sharpe = np.sqrt(2 * np.log(N_STRATEGIES_TESTED))
    if observed_sharpe > expected_max_sharpe * 1.5:
        dsr_status = "PASS"
    elif observed_sharpe > expected_max_sharpe:
        dsr_status = "WARN"
    else:
        dsr_status = "FAIL"
    results.append(
        TestResult(
            f"Deflated Sharpe (N={N_STRATEGIES_TESTED})",
            f"SR={observed_sharpe:.2f} vs threshold={expected_max_sharpe:.2f}",
            dsr_status,
            detail=(
                f"Multiple testing correction: with {N_STRATEGIES_TESTED} strategies tested, "
                f"expected best SR by chance = {expected_max_sharpe:.2f}"
            ),
        )
    )

    # Bull/bear regime (requires benchmark_returns)
    if benchmark_returns is not None and len(benchmark_returns) > 100:
        bench_annual = benchmark_returns.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        strat_annual = equity.resample("YE").last().pct_change().dropna()
        common_idx = bench_annual.index.intersection(strat_annual.index)
        if len(common_idx) >= 4:
            b_ann = bench_annual.loc[common_idx]
            s_ann = strat_annual.loc[common_idx]
            bull_mask = b_ann > 0
            bear_returns = s_ann.loc[~bull_mask]  # type: ignore[index]
            bull_returns = s_ann.loc[bull_mask]  # type: ignore[index]
            bull_avg = float(bull_returns.mean() * 100) if len(bull_returns) > 0 else 0.0
            bear_avg = float(bear_returns.mean() * 100) if len(bear_returns) > 0 else 0.0
            bear_positive = (
                float((bear_returns > 0).mean() * 100) if len(bear_returns) > 0 else 0.0
            )
            regime_status = (
                "PASS"
                if bear_avg > 0 or bear_positive >= 50
                else ("WARN" if bear_avg > -10 else "FAIL")
            )
            results.append(
                TestResult(
                    "Bull/bear regime",
                    f"bull={bull_avg:.1f}%/yr bear={bear_avg:.1f}%/yr "
                    f"({bear_positive:.0f}% bear yrs positive)",
                    regime_status,
                )
            )

    return results
