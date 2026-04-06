"""Tests for indicator deduplication utilities."""
from __future__ import annotations

from vbacktest.indicators import IndicatorSpec
from vbacktest.indicator_dedup import deduplicate_indicators


class TestDeduplicateIndicators:
    def test_empty_list(self) -> None:
        assert deduplicate_indicators([]) == []

    def test_no_duplicates(self) -> None:
        specs = [
            IndicatorSpec("sma", {"period": 10}),
            IndicatorSpec("ema", {"period": 20}),
        ]
        result = deduplicate_indicators(specs)
        assert len(result) == 2

    def test_exact_duplicates_removed(self) -> None:
        specs = [
            IndicatorSpec("sma", {"period": 20}),
            IndicatorSpec("sma", {"period": 20}),
        ]
        result = deduplicate_indicators(specs)
        assert len(result) == 1
        assert result[0].name == "sma"
        assert result[0].params == {"period": 20}

    def test_different_params_not_deduplicated(self) -> None:
        specs = [
            IndicatorSpec("sma", {"period": 10}),
            IndicatorSpec("sma", {"period": 20}),
        ]
        result = deduplicate_indicators(specs)
        assert len(result) == 2

    def test_first_occurrence_preserved(self) -> None:
        """Order of first occurrence is maintained."""
        specs = [
            IndicatorSpec("rsi", {"period": 14}),
            IndicatorSpec("sma", {"period": 10}),
            IndicatorSpec("rsi", {"period": 14}),  # duplicate
            IndicatorSpec("ema", {"period": 20}),
        ]
        result = deduplicate_indicators(specs)
        assert len(result) == 3
        assert result[0].name == "rsi"
        assert result[1].name == "sma"
        assert result[2].name == "ema"

    def test_same_name_different_params(self) -> None:
        specs = [
            IndicatorSpec("atr", {"period": 14}),
            IndicatorSpec("atr", {"period": 7}),
            IndicatorSpec("atr", {"period": 14}),  # duplicate of first
        ]
        result = deduplicate_indicators(specs)
        assert len(result) == 2
        periods = {r.params["period"] for r in result}
        assert periods == {14, 7}

    def test_empty_params_deduplication(self) -> None:
        specs = [
            IndicatorSpec("macd"),
            IndicatorSpec("macd"),
        ]
        result = deduplicate_indicators(specs)
        assert len(result) == 1

    def test_many_duplicates_large_list(self) -> None:
        specs = [IndicatorSpec("sma", {"period": 20})] * 100
        result = deduplicate_indicators(specs)
        assert len(result) == 1

    def test_mixed_bulk(self) -> None:
        specs = [
            IndicatorSpec("sma", {"period": 10}),
            IndicatorSpec("sma", {"period": 20}),
            IndicatorSpec("ema", {"period": 10}),
            IndicatorSpec("sma", {"period": 10}),  # dup
            IndicatorSpec("rsi", {"period": 14}),
            IndicatorSpec("ema", {"period": 10}),  # dup
        ]
        result = deduplicate_indicators(specs)
        assert len(result) == 4
