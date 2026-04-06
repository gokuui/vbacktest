"""Exception taxonomy for vbacktest."""
from __future__ import annotations
from typing import Any


class VBacktestError(Exception):
    """Base exception for all vbacktest errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigError(VBacktestError):
    """Invalid or missing configuration."""


class DataError(VBacktestError):
    """Missing, malformed, or invalid market data."""


class StrategyError(VBacktestError):
    """Error during strategy execution."""


class ValidationError(VBacktestError):
    """Go/no-go validation error."""


class RegistryError(VBacktestError):
    """Strategy or indicator registry error (e.g. duplicate name)."""
