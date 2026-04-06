"""Shared test fixtures for vbacktest."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _registry_snapshot():
    """Snapshot and restore global registries around each test.

    Prevents test-registered strategies/indicators from leaking into subsequent
    tests and causing non-deterministic count assertions.
    """
    from vbacktest.registry import strategy_registry, indicator_registry

    # Snapshot current state
    strat_snapshot = dict(strategy_registry._entries)
    ind_snapshot = dict(indicator_registry._entries)

    yield

    # Restore
    strategy_registry._entries.clear()
    strategy_registry._entries.update(strat_snapshot)
    indicator_registry._entries.clear()
    indicator_registry._entries.update(ind_snapshot)
