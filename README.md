# vbacktest

Production-grade day-by-day backtesting framework for equities.

## Install

```bash
pip install vbacktest
# or from source:
pip install -e ".[dev]"
```

## Quick Start

```python
from vbacktest import BacktestEngine, BacktestConfig
from vbacktest.strategies import MACrossover

config = BacktestConfig.simple("data/validated/", capital=100000)
engine = BacktestEngine(config)
result = engine.run(MACrossover())
result.print_report()
```
