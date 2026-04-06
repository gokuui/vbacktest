"""Indicator deduplication utilities."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vbacktest.indicators import IndicatorSpec


def deduplicate_indicators(specs: list[IndicatorSpec]) -> list[IndicatorSpec]:
    """Remove duplicate indicator specifications.

    Two specs are considered duplicates if they have the same name and params.
    Preserves order of first occurrence.

    Args:
        specs: List of IndicatorSpec (may contain duplicates)

    Returns:
        List of unique IndicatorSpec (duplicates removed)

    Examples:
        >>> specs = [
        ...     IndicatorSpec('sma', {'period': 20}),
        ...     IndicatorSpec('sma', {'period': 20}),  # Duplicate
        ...     IndicatorSpec('ema', {'period': 10}),
        ... ]
        >>> deduped = deduplicate_indicators(specs)
        >>> len(deduped)
        2
    """
    if not specs:
        return []

    seen = set()
    unique = []

    for spec in specs:
        # Create hashable key from name and params using JSON serialization
        # This handles unhashable types like lists and nested dicts
        params_json = json.dumps(spec.params, sort_keys=True)
        key = (spec.name, params_json)

        if key not in seen:
            seen.add(key)
            unique.append(spec)

    return unique
