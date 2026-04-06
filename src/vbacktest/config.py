"""Configuration dataclasses for vbacktest."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vbacktest.exceptions import ConfigError


@dataclass
class DataConfig:
    validated_dir: str | Path
    min_history_days: int = 200
    excluded_symbols: list[str] = field(default_factory=list)
    excluded_symbols_file: str | None = None
    required_columns: list[str] = field(
        default_factory=lambda: ["date", "open", "high", "low", "close", "volume"]
    )
    start_date: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        self.validated_dir = Path(self.validated_dir)


@dataclass
class PortfolioConfig:
    initial_capital: float = 100_000
    max_positions: int = 10

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ConfigError("initial_capital must be > 0")


@dataclass
class ExecutionConfig:
    commission_pct: float = 0.1
    slippage_pct: float = 0.05
    entry_on: str = "open"
    exit_on: str = "open"


@dataclass
class PositionSizingConfig:
    risk_per_trade_pct: float = 1.0

    def __post_init__(self) -> None:
        if self.risk_per_trade_pct <= 0:
            raise ConfigError("risk_per_trade_pct must be > 0")


@dataclass
class PerformanceConfig:
    enable_parallel: bool = True
    num_workers: int | None = None

    def __post_init__(self) -> None:
        if self.num_workers is not None and self.num_workers <= 0:
            raise ConfigError("num_workers must be > 0")


@dataclass
class BacktestConfig:
    data: DataConfig
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    @classmethod
    def simple(
        cls,
        validated_dir: str | Path,
        *,
        capital: float = 100_000,
        max_positions: int = 10,
        **kwargs: Any,
    ) -> BacktestConfig:
        return cls(
            data=DataConfig(validated_dir=validated_dir),
            portfolio=PortfolioConfig(initial_capital=capital, max_positions=max_positions),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> BacktestConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            portfolio=PortfolioConfig(**raw.get("portfolio", {})),
            execution=ExecutionConfig(**raw.get("execution", {})),
            position_sizing=PositionSizingConfig(**raw.get("position_sizing", {})),
            performance=PerformanceConfig(**raw.get("performance", {})),
        )
