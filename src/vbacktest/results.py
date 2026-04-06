"""Backtest result dataclasses and statistics computation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vbacktest.config import BacktestConfig
    from vbacktest.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestStats:
    """Scalar performance metrics for a completed backtest."""

    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_days: int = 0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_holding_days: float = 0.0
    max_consecutive_losses: int = 0
    expectancy: float = 0.0
    avg_exposure_pct: float = 0.0
    total_trades: int = 0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Complete backtest output including equity curve, trades, and metrics.

    Class attributes:
        schema_version: Bumped when the serialised format changes.
    """

    schema_version: str = field(default="1.0", init=False, repr=False)

    equity_curve: pd.DataFrame
    trades: list[Any]
    trade_df: pd.DataFrame
    stats: BacktestStats
    monthly_returns: pd.DataFrame
    yearly_returns: pd.DataFrame
    config_snapshot: dict[str, Any]
    benchmark_comparison: dict[str, float] = field(default_factory=dict)

    def equity_series(self) -> pd.Series:
        """Return equity as a ``pd.Series`` indexed by date.

        Returns:
            Series with ``pd.Timestamp`` index and float values, or an empty
            Series if the equity curve is empty.
        """
        if self.equity_curve.empty:
            return pd.Series(dtype=float)
        return self.equity_curve.set_index("date")["equity"]

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "config_snapshot": self.config_snapshot,
            "stats": self.stats.__dict__,
            "benchmark_comparison": self.benchmark_comparison,
            "equity_curve": self.equity_curve.to_dict(orient="records"),
            "trades": self.trade_df.to_dict(orient="records") if not self.trade_df.empty else [],
        }
        return json.dumps(payload, default=str)

    @classmethod
    def from_json(cls, data: str | dict[str, Any]) -> BacktestResult:
        """Deserialise from a JSON string or dict.

        Schema migration:
        - ``config`` key (pre-1.0) is renamed to ``config_snapshot``.
        """
        payload: dict[str, Any] = json.loads(data) if isinstance(data, str) else data

        # Schema migration: old key name
        if "config" in payload and "config_snapshot" not in payload:
            payload["config_snapshot"] = payload.pop("config")

        equity_curve = pd.DataFrame(payload.get("equity_curve", []))
        if not equity_curve.empty and "date" in equity_curve.columns:
            equity_curve["date"] = pd.to_datetime(equity_curve["date"])

        trade_records = payload.get("trades", [])
        trade_df = pd.DataFrame(trade_records)

        stats_dict = payload.get("stats", {})
        stats = BacktestStats(**{k: v for k, v in stats_dict.items() if k in BacktestStats.__dataclass_fields__})

        return cls(
            equity_curve=equity_curve,
            trades=[],  # raw Trade objects not preserved in JSON
            trade_df=trade_df,
            stats=stats,
            monthly_returns=pd.DataFrame(),
            yearly_returns=pd.DataFrame(),
            config_snapshot=payload.get("config_snapshot", {}),
            benchmark_comparison=payload.get("benchmark_comparison", {}),
        )

    @classmethod
    def empty(cls) -> BacktestResult:
        """Create an empty result (useful as a sentinel)."""
        return cls(
            equity_curve=pd.DataFrame(columns=["date", "equity"]),
            trades=[],
            trade_df=pd.DataFrame(),
            stats=BacktestStats(),
            monthly_returns=pd.DataFrame(),
            yearly_returns=pd.DataFrame(),
            config_snapshot={},
        )


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_stats(
    portfolio: Portfolio,
    config: BacktestConfig,
    benchmark_returns: pd.Series | None = None,
) -> BacktestResult:
    """Compute backtest statistics from a completed portfolio run.

    Args:
        portfolio: Finished portfolio.
        config: Backtest configuration (for config snapshot and sizing params).
        benchmark_returns: Optional daily returns series for benchmark comparison.

    Returns:
        ``BacktestResult`` with full statistics.
    """
    equity_curve = (
        pd.DataFrame(portfolio.equity_history, columns=["date", "equity"])
        if portfolio.equity_history
        else pd.DataFrame(columns=["date", "equity"])
    )

    trade_df = (
        pd.DataFrame([vars(t) for t in portfolio.trades])
        if portfolio.trades
        else pd.DataFrame()
    )

    stats = BacktestStats()

    if len(equity_curve) > 0:
        initial = config.portfolio.initial_capital
        final = equity_curve["equity"].iloc[-1]

        stats.total_return_pct = (final - initial) / initial * 100

        days = (equity_curve["date"].iloc[-1] - equity_curve["date"].iloc[0]).days
        years = days / 365.25
        stats.cagr = (((final / initial) ** (1 / years)) - 1) * 100 if years > 0 else 0.0

        equity_curve["returns"] = equity_curve["equity"].pct_change()
        returns = equity_curve["returns"].dropna()

        if len(returns) > 0 and returns.std() > 0:
            stats.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        downside = np.minimum(returns.values, 0.0)
        downside_dev = np.sqrt(np.mean(downside ** 2))
        if downside_dev > 0:
            stats.sortino_ratio = (returns.mean() / downside_dev) * np.sqrt(252)

        running_max = equity_curve["equity"].cummax()
        drawdown = (equity_curve["equity"] - running_max) / running_max * 100
        stats.max_drawdown_pct = abs(drawdown.min())

        if stats.max_drawdown_pct > 0:
            in_dd = drawdown < 0
            periods: list[int] = []
            start: int | None = None
            for i, is_dd in enumerate(in_dd):
                if is_dd and start is None:
                    start = i
                elif not is_dd and start is not None:
                    periods.append(i - start)
                    start = None
            if start is not None:
                periods.append(len(in_dd) - start)
            stats.max_drawdown_days = max(periods) if periods else 0

        if stats.max_drawdown_pct > 0:
            stats.calmar_ratio = stats.cagr / stats.max_drawdown_pct

    if len(trade_df) > 0:
        stats.total_trades = len(trade_df)
        wins = trade_df[trade_df["pnl"] > 0]
        losses = trade_df[trade_df["pnl"] <= 0]
        stats.win_rate = len(wins) / len(trade_df) * 100
        if len(wins) > 0:
            stats.avg_win_pct = wins["pnl_pct"].mean()
        if len(losses) > 0:
            stats.avg_loss_pct = losses["pnl_pct"].mean()
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
        stats.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0
            else (999.99 if gross_profit > 0 else 0.0)
        )
        stats.avg_holding_days = trade_df["holding_days"].mean()
        consecutive = 0
        max_consec = 0
        for pnl in trade_df["pnl"]:
            if pnl <= 0:
                consecutive += 1
                max_consec = max(max_consec, consecutive)
            else:
                consecutive = 0
        stats.max_consecutive_losses = max_consec
        stats.expectancy = trade_df["pnl"].mean()

        if len(equity_curve) > 0:
            total_days = len(equity_curve)
            total_pos_days = trade_df["holding_days"].sum()
            max_pos = config.portfolio.max_positions
            if total_days > 0 and max_pos > 0:
                stats.avg_exposure_pct = (total_pos_days / (total_days * max_pos)) * 100

    # Monthly and yearly returns
    monthly_returns: pd.DataFrame = pd.DataFrame()
    yearly_returns: pd.DataFrame = pd.DataFrame()
    if len(equity_curve) > 0:
        equity_curve["year"] = equity_curve["date"].dt.year
        equity_curve["month"] = equity_curve["date"].dt.to_period("M")
        monthly_eq = equity_curve.groupby("month")["equity"].last()
        monthly_returns = pd.DataFrame({
            "month": monthly_eq.index.astype(str),
            "return_pct": monthly_eq.pct_change() * 100,
        })
        yearly_eq = equity_curve.groupby("year")["equity"].last()
        yearly_returns = pd.DataFrame({
            "year": yearly_eq.index,
            "return_pct": yearly_eq.pct_change() * 100,
        })

    # Config snapshot
    config_snapshot: dict[str, Any] = {
        "initial_capital": config.portfolio.initial_capital,
        "max_positions": config.portfolio.max_positions,
        "risk_per_trade_pct": config.position_sizing.risk_per_trade_pct,
        "commission_pct": config.execution.commission_pct,
        "slippage_pct": config.execution.slippage_pct,
        "start_date": config.data.start_date,
        "end_date": config.data.end_date,
    }

    # Benchmark comparison
    benchmark_comparison: dict[str, float] = {}
    if benchmark_returns is not None and len(equity_curve) > 0:
        # Build a date-indexed returns series for alignment
        strat_returns = equity_curve.set_index("date")["returns"].dropna()
        benchmark_comparison = _compute_benchmark_comparison(
            strat_returns, benchmark_returns, stats
        )

    return BacktestResult(
        equity_curve=equity_curve,
        trades=portfolio.trades,
        trade_df=trade_df,
        stats=stats,
        monthly_returns=monthly_returns,
        yearly_returns=yearly_returns,
        config_snapshot=config_snapshot,
        benchmark_comparison=benchmark_comparison,
    )


def _compute_benchmark_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    stats: BacktestStats,
) -> dict[str, float]:
    """Compute alpha, beta and other comparison metrics."""
    aligned = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")], axis=1
    ).dropna()

    if len(aligned) < 2:
        return {}

    s = aligned["s"]
    b = aligned["b"]

    # Beta via covariance / variance
    cov = np.cov(s.values, b.values)
    bench_var = float(np.var(b.values, ddof=1))
    beta = float(cov[0, 1] / bench_var) if bench_var > 0 else 0.0

    # Annualised benchmark return
    bench_total = (1 + b).prod() - 1
    bench_cagr = float((1 + bench_total) ** (252 / len(b)) - 1) * 100

    # Benchmark Sharpe
    bench_sharpe = (
        (b.mean() / b.std()) * np.sqrt(252)
        if b.std() > 0
        else 0.0
    )

    # Benchmark max drawdown
    bench_cum = (1 + b).cumprod()
    bench_dd = (bench_cum / bench_cum.cummax() - 1).min() * 100

    # Alpha (annualised, CAPM)
    risk_free_daily = 0.0
    alpha_daily = s.mean() - (risk_free_daily + beta * (b.mean() - risk_free_daily))
    alpha = float(alpha_daily * 252 * 100)

    return {
        "strategy_return": stats.total_return_pct,
        "benchmark_return": float(bench_total * 100),
        "total_return_diff": stats.total_return_pct - float(bench_total * 100),
        "alpha": alpha,
        "beta": beta,
        "strategy_sharpe": stats.sharpe_ratio,
        "benchmark_sharpe": float(bench_sharpe),
        "sharpe_diff": stats.sharpe_ratio - float(bench_sharpe),
        "strategy_max_dd": stats.max_drawdown_pct,
        "benchmark_max_dd": float(abs(bench_dd)),
        "max_dd_diff": stats.max_drawdown_pct - float(abs(bench_dd)),
        "strategy_cagr": stats.cagr,
        "benchmark_cagr": bench_cagr,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report to stdout."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print("\nPERFORMANCE METRICS:")
    print(f"  Total Return:        {result.stats.total_return_pct:>10.2f}%")
    print(f"  CAGR:                {result.stats.cagr:>10.2f}%")
    print(f"  Sharpe Ratio:        {result.stats.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:       {result.stats.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:        {result.stats.max_drawdown_pct:>10.2f}%")
    print(f"  Max DD Duration:     {result.stats.max_drawdown_days:>10} days")
    print(f"  Calmar Ratio:        {result.stats.calmar_ratio:>10.2f}")

    print("\nTRADE STATISTICS:")
    print(f"  Total Trades:        {result.stats.total_trades:>10}")
    print(f"  Win Rate:            {result.stats.win_rate:>10.2f}%")
    print(f"  Profit Factor:       {result.stats.profit_factor:>10.2f}")
    print(f"  Avg Win:             {result.stats.avg_win_pct:>10.2f}%")
    print(f"  Avg Loss:            {result.stats.avg_loss_pct:>10.2f}%")
    print(f"  Avg Holding Days:    {result.stats.avg_holding_days:>10.1f}")
    print(f"  Max Consec Losses:   {result.stats.max_consecutive_losses:>10}")
    print(f"  Expectancy:          ${result.stats.expectancy:>10.2f}")
    print(f"  Avg Exposure:        {result.stats.avg_exposure_pct:>10.2f}%")

    if result.benchmark_comparison:
        b = result.benchmark_comparison
        print("\nBENCHMARK COMPARISON:")
        print(f"  Strategy Return:     {b.get('strategy_return', 0):>10.2f}%")
        print(f"  Benchmark Return:    {b.get('benchmark_return', 0):>10.2f}%")
        print(f"  Outperformance:      {b.get('total_return_diff', 0):>10.2f}%")
        print(f"  Alpha:               {b.get('alpha', 0):>10.2f}%")
        print(f"  Beta:                {b.get('beta', 0):>10.2f}")
        print(f"  Strategy Sharpe:     {b.get('strategy_sharpe', 0):>10.2f}")
        print(f"  Benchmark Sharpe:    {b.get('benchmark_sharpe', 0):>10.2f}")

    if len(result.equity_curve) > 0:
        print("\nEQUITY CURVE:")
        print(f"  Start Date:          {result.equity_curve['date'].iloc[0].strftime('%Y-%m-%d')}")
        print(f"  End Date:            {result.equity_curve['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print(f"  Initial Equity:      ${result.equity_curve['equity'].iloc[0]:>12,.2f}")
        print(f"  Final Equity:        ${result.equity_curve['equity'].iloc[-1]:>12,.2f}")

    print("\n" + "=" * 60)


def save_report(result: BacktestResult, output_dir: Path | str) -> None:
    """Save backtest report to ``output_dir``.

    Creates ``equity_curve.csv``, ``trades.csv`` (if any), and ``summary.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)

    if not result.trade_df.empty:
        result.trade_df.to_csv(output_dir / "trades.csv", index=False)

    summary = {"stats": vars(result.stats), "config_snapshot": result.config_snapshot}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
