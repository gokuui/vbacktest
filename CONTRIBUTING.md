# Contributing to vbacktest

Thank you for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/gokuui/vbacktest.git
cd vbacktest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

```bash
make test        # run all 363 tests
make typecheck   # mypy strict (must be 0 errors)
make test-fast   # skip slow tests
```

All PRs must pass the full test suite and `mypy --strict` before merging.

## Adding a Strategy

1. Create `src/vbacktest/strategies/my_strategy.py` subclassing `Strategy`
2. Implement `indicators()`, `on_bar()`, and `exit_rules()`
3. Register with `@strategy_registry.register("my_strategy")`
4. Export from `src/vbacktest/strategies/__init__.py`
5. Add tests in `tests/test_strategies/`

See `src/vbacktest/strategies/rsi_mean_reversion.py` as a reference implementation.

## Adding an Indicator

1. Add a function `compute_my_indicator(df, **params) -> pd.DataFrame` to `src/vbacktest/indicators.py`
2. Register: `indicator_registry.register("my_indicator")(compute_my_indicator)`
3. Add tests in `tests/test_indicators.py`

## Code Style

- Strict type annotations throughout (mypy strict)
- No external linter configured — follow the existing style
- Keep files focused; prefer small, single-responsibility modules

## Reporting Issues

Open an issue at https://github.com/gokuui/vbacktest/issues with:
- Python version and OS
- Minimal reproducible example
- Expected vs actual behaviour
