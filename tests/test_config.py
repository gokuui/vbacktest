import tempfile
import yaml
import pytest
from pathlib import Path
from vbacktest.config import (
    DataConfig, PortfolioConfig, ExecutionConfig,
    PositionSizingConfig, PerformanceConfig, BacktestConfig,
)
from vbacktest.exceptions import ConfigError


class TestDataConfig:
    def test_defaults(self):
        dc = DataConfig(validated_dir="/tmp/data")
        assert dc.validated_dir == Path("/tmp/data")
        assert dc.min_history_days == 200
        assert dc.excluded_symbols == []
        assert dc.excluded_symbols_file is None
        assert dc.required_columns == ["date", "open", "high", "low", "close", "volume"]
        assert dc.start_date is None
        assert dc.end_date is None

    def test_validated_dir_converted_to_path(self):
        dc = DataConfig(validated_dir="/tmp/data")
        assert isinstance(dc.validated_dir, Path)

    def test_excluded_symbols_list(self):
        dc = DataConfig(validated_dir="/tmp", excluded_symbols=["AAPL", "GOOG"])
        assert dc.excluded_symbols == ["AAPL", "GOOG"]


class TestPortfolioConfig:
    def test_defaults(self):
        pc = PortfolioConfig()
        assert pc.initial_capital == 100_000
        assert pc.max_positions == 10

    def test_invalid_capital_raises(self):
        with pytest.raises(ConfigError):
            PortfolioConfig(initial_capital=-1)


class TestExecutionConfig:
    def test_defaults(self):
        ec = ExecutionConfig()
        assert ec.commission_pct == 0.1
        assert ec.slippage_pct == 0.05
        assert ec.entry_on == "open"
        assert ec.exit_on == "open"


class TestPositionSizingConfig:
    def test_defaults(self):
        ps = PositionSizingConfig()
        assert ps.risk_per_trade_pct == 1.0

    def test_invalid_risk_raises(self):
        with pytest.raises(ConfigError):
            PositionSizingConfig(risk_per_trade_pct=-1)


class TestPerformanceConfig:
    def test_defaults(self):
        pc = PerformanceConfig()
        assert pc.enable_parallel is True
        assert pc.num_workers is None

    def test_invalid_workers_raises(self):
        with pytest.raises(ConfigError):
            PerformanceConfig(num_workers=0)


class TestBacktestConfig:
    def test_simple_constructor(self):
        bc = BacktestConfig.simple("/tmp/data", capital=50000, max_positions=5)
        assert bc.data.validated_dir == Path("/tmp/data")
        assert bc.portfolio.initial_capital == 50000
        assert bc.portfolio.max_positions == 5

    def test_simple_defaults(self):
        bc = BacktestConfig.simple("/tmp/data")
        assert bc.portfolio.initial_capital == 100_000
        assert bc.execution.commission_pct == 0.1

    def test_from_yaml(self):
        cfg = {
            "data": {"validated_dir": "/tmp/data"},
            "portfolio": {"initial_capital": 200000, "max_positions": 15},
            "execution": {"commission_pct": 0.2},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(cfg, f)
            f.flush()
            bc = BacktestConfig.from_yaml(f.name)
        assert bc.portfolio.initial_capital == 200000
        assert bc.execution.commission_pct == 0.2
        assert bc.data.validated_dir == Path("/tmp/data")

    def test_aggregate_has_all_sub_configs(self):
        bc = BacktestConfig.simple("/tmp/data")
        assert hasattr(bc, "data")
        assert hasattr(bc, "portfolio")
        assert hasattr(bc, "execution")
        assert hasattr(bc, "performance")
