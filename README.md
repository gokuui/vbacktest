# vbacktest

Production-grade day-by-day backtesting framework for equities.

Market-agnostic, strict no-lookahead, installable via pip. Extracted from a live NSE trading system with 330+ tests.

```python
from vbacktest import BacktestConfig, BacktestEngine
from vbacktest.strategies import MomentumStrategy

config = BacktestConfig.simple("data/validated/", capital=100_000)
result = BacktestEngine(config, MomentumStrategy()).run()
result.print_report()
```

## Install

```bash
pip install vbacktest
```

Development:
```bash
git clone https://github.com/yourusername/vbacktest
cd vbacktest
pip install -e ".[dev]"
```

## Features

- **Day-by-day simulation** — entries and exits at next bar's open, strict no-lookahead
- **Risk-based position sizing** — size positions as a fixed % of equity risk per trade
- **6 built-in strategies** + 47 extras, all extensible
- **Go/no-go validation** — 42 statistical tests before going live (Monte Carlo, slippage ladders, Kelly, t-test, Calmar, regime analysis)
- **Fast execution** — numpy array fast path, 12–20× speedup over pandas-only
- **Strict typing** — mypy strict, 363 tests
- **Market-agnostic** — works with any daily OHLCV parquet data (NSE, NYSE, crypto, etc.)

## Built-in Strategies

| Key | Strategy | Entry Logic |
|-----|----------|-------------|
| `ma_crossover` | MA Crossover | Fast MA > Slow MA + trend filter |
| `momentum` | Momentum | Multi-period ROC + new high + trend |
| `rsi_mean_reversion` | RSI Mean Reversion | RSI crosses oversold |
| `bollinger_breakout` | Bollinger Breakout | Close breaks upper BB + volume |
| `volume_breakout` | Volume Breakout | 3× volume surge + price breakout |
| `turtle_trading` | Turtle Trading | Donchian channel breakout |

## Data Format

vbacktest loads `.parquet` files from a directory. Each file is one symbol:

```
data/validated/
    RELIANCE.parquet
    TCS.parquet
    AAPL.parquet
    ...
```

Required columns: `date` (datetime64), `open`, `high`, `low`, `close`, `volume` (float64).

See [docs/data_contract.md](docs/data_contract.md) for full constraints.

## Configuration

```python
from vbacktest import BacktestConfig
from vbacktest.config import DataConfig, ExecutionConfig, PortfolioConfig

config = BacktestConfig(
    data=DataConfig(
        validated_dir="data/validated/",
        start_date="2015-01-01",
        end_date="2024-12-31",
        min_history_days=200,
        min_price=1.0,
        min_avg_volume=50_000,
        excluded_symbols=["BROKEN1"],
        excluded_symbols_file="data/excluded.txt",
    ),
    portfolio=PortfolioConfig(
        initial_capital=100_000,
        max_positions=10,
        risk_per_trade_pct=1.0,       # 1% of equity per trade
    ),
    execution=ExecutionConfig(
        commission_pct=0.1,           # 0.1% per side
        slippage_pct=0.05,
    ),
)
```

Or via `BacktestConfig.simple()` for quick runs:

```python
config = BacktestConfig.simple("data/validated/", capital=100_000, max_positions=10)
```

## Custom Strategy

```python
from vbacktest import BacktestConfig, BacktestEngine, Signal, SignalAction, Strategy, BarContext
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule
from vbacktest.indicators import IndicatorSpec

class GoldenCrossStrategy(Strategy):
    def indicators(self):
        return [
            IndicatorSpec("sma", {"period": 50}),
            IndicatorSpec("sma", {"period": 200}),
            IndicatorSpec("atr", {"period": 14}),
        ]

    def exit_rules(self):
        return [StopLossRule(), TrailingATRStopRule(multiplier=2.5)]

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        signals = []
        for symbol, df in ctx.universe.items():
            if ctx.portfolio and ctx.portfolio.has_position(symbol):
                continue
            idx = ctx.universe_idx.get(symbol, 0)
            if idx < 201:
                continue

            # Fast path: use pre-computed numpy arrays
            if ctx.universe_arrays and symbol in ctx.universe_arrays:
                arrays = ctx.universe_arrays[symbol]
                try:
                    sma50 = float(arrays["sma_50"][idx])
                    sma50_prev = float(arrays["sma_50"][idx - 1])
                    sma200 = float(arrays["sma_200"][idx])
                    sma200_prev = float(arrays["sma_200"][idx - 1])
                    close = float(arrays["close"][idx])
                    atr = float(arrays["atr"][idx])
                except (KeyError, IndexError):
                    continue
                if sma50 != sma50 or sma200 != sma200:  # NaN check
                    continue
            else:
                import pandas as pd
                bar = df.iloc[idx]
                if pd.isna(bar["sma_50"]) or pd.isna(bar["sma_200"]):
                    continue
                sma50, sma50_prev = float(bar["sma_50"]), float(df.iloc[idx-1]["sma_50"])
                sma200, sma200_prev = float(bar["sma_200"]), float(df.iloc[idx-1]["sma_200"])
                close = float(bar["close"])
                atr = float(bar["atr"])

            if sma50_prev <= sma200_prev and sma50 > sma200:
                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    date=ctx.date,
                    stop_price=close - 2.0 * atr,
                    score=sma50 / sma200,
                ))
        return signals

config = BacktestConfig.simple("data/validated/")
result = BacktestEngine(config, GoldenCrossStrategy()).run()
result.print_report()
```

## Go/No-Go Validation

Before going live, run the 42-test validation suite:

```python
from vbacktest.analysis import GoNoGo

report = GoNoGo(
    trades=result.trades,
    equity=result.equity_series(),
    initial_capital=100_000,
    mc_sims=5000,
).run(name="Golden Cross")

report.print_terminal()        # coloured terminal output
report.write_markdown("report.md")  # save to file
print(report.verdict())        # "PASS", "WARN", or "FAIL"
```

Test categories:

| Category | Tests |
|----------|-------|
| Statistical Foundation | Trade count, CAGR, profit factor, expectancy, t-test, win rate |
| Monte Carlo | MC-Reshuffle (DD), MC-Bootstrap (CAGR), MC-Slip-inject |
| Execution Realism | Slippage ladder (0–1.9% RT), commission ladder |
| Risk Metrics | Max drawdown, DD duration, Kelly, risk-of-ruin, Calmar |
| Advanced Risk | CVaR, tail ratio, Sortino, Ulcer Index, autocorrelation, streaks |
| Regime & Rolling | Rolling Sharpe, annual consistency, alpha/beta vs benchmark |
| Annual Breakdown | Year-by-year returns, bear year flagging |

## CLI

```bash
# List available strategies
vbacktest strategies

# Run a backtest
vbacktest run --strategy momentum --data data/validated/ --capital 100000

# Run go/no-go on a saved result
vbacktest gonogo result.json --out report.md
```

## Saving and Loading Results

```python
# Save
with open("result.json", "w") as f:
    f.write(result.to_json())

# Load
from vbacktest.results import BacktestResult
with open("result.json") as f:
    result = BacktestResult.from_json(f.read())
```

## Performance Notes

Strategies should implement the numpy fast path for ~10–20× speedup:

```python
if ctx.universe_arrays and symbol in ctx.universe_arrays:
    arrays = ctx.universe_arrays[symbol]
    val = float(arrays["sma_50"][idx])     # O(1) scalar lookup
    window = arrays["close"][idx-20:idx].mean()  # numpy slice
```

See [examples/custom_strategy.py](examples/custom_strategy.py) for the full pattern.

## Registry

Strategies and indicators can be discovered and extended via the registry:

```python
from vbacktest.registry import list_strategies, list_indicators, register_strategy

print(list_strategies())  # ['ma_crossover', 'momentum', ...]

@register_strategy("my_strategy")
class MyStrategy(Strategy):
    ...
```

## Running Tests

```bash
make test           # all 363 tests
make test-fast      # skip slow tests
make typecheck      # mypy strict
```

## Contributing

1. Fork and clone
2. `pip install -e ".[dev]"`
3. Write failing tests first (TDD)
4. Run `make test` and `make typecheck` before submitting a PR

All strategies must implement the numpy fast path (see `CLAUDE.md` for the checklist).

## License

MIT
