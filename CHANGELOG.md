# Changelog

All notable changes to vbacktest will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-04-06

Initial open-source release.

### Added

**Core framework**
- `BacktestEngine` — day-by-day execution with warmup, commission, slippage, no lookahead
- `Portfolio` — risk-based position sizing, stop loss enforcement, equity curve tracking
- `Strategy` base class with `BarContext`, `Signal`, `ExitRule` interfaces
- `BacktestConfig` — dataclass-based config with YAML loading and `BacktestConfig.simple()` helper

**Indicators** (15 built-in)
- SMA, EMA, ATR, RSI, Bollinger Bands, Donchian Channel
- MACD, Stochastic, Volume SMA, Relative Volume, ROC, Rolling High, ADX

**Exit rules** (9 built-in)
- StopLoss, TrailingATRStop, TrailingMA, MA10Exit, TakeProfitPartial, TimeStop
- Stateful rules support deep copy for per-position instances

**Strategies** (53 total)
- 6 core: MA Crossover, RSI Mean Reversion, Bollinger Breakout, Momentum, Volume Breakout, Turtle Trading
- 47 extras: Elder Impulse, Minervini SEPA, VCP, ADX Trend, Filtered Momentum, and more

**Go/No-Go analysis**
- 7 test categories: Statistical Foundation, Monte Carlo (5000 sims), Execution Realism, Risk Metrics, Advanced Risk, Regime & Rolling, Annual Breakdown
- Deflated Sharpe Ratio with multiple-testing correction
- PASS / WARN / FAIL verdict with hard no-go flags

**Registry**
- Thread-safe `strategy_registry` and `indicator_registry`
- `@register_strategy` / `@register_indicator` decorators
- `list_strategies()`, `list_indicators()`, `get_strategy()`, `get_indicator()`

**CLI**
- `vbacktest strategies` — list all registered strategies
- `vbacktest gonogo` — run go/no-go analysis on a saved backtest result
- `vbacktest run` — run a backtest from config file

**Quality**
- 363 tests, 0 mypy strict errors across 79 source files
- CI: GitHub Actions on Python 3.10 / 3.11 / 3.12
- MIT License
