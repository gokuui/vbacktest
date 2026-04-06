"""Tests for data loading and preprocessing."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vbacktest.data_loader import (
    convert_universe_to_dict_index,
    convert_universe_to_numpy_arrays,
    build_trading_calendar,
    build_date_index,
    precompute_indicators,
    load_universe,
    _validate_ohlc,
)
from vbacktest.indicators import IndicatorSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 30, close_start: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = close_start + np.arange(n, dtype=float)
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": [1_000_000.0] * n,
        "source": ["test"] * n,
        "confidence": [2] * n,
    })


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def _universe(n_symbols: int = 3, n_bars: int = 30) -> dict[str, pd.DataFrame]:
    return {
        f"SYM{i}": _make_df(n_bars, close_start=100.0 + i * 10)
        for i in range(n_symbols)
    }


# ---------------------------------------------------------------------------
# _validate_ohlc
# ---------------------------------------------------------------------------

class TestValidateOHLC:
    def test_valid_data_passes(self) -> None:
        df = _make_df()
        assert _validate_ohlc(df, "TEST") is True

    def test_nan_values_fail(self) -> None:
        df = _make_df()
        df.loc[5, "close"] = float("nan")
        assert _validate_ohlc(df, "TEST") is False

    def test_negative_price_fails(self) -> None:
        df = _make_df()
        df.loc[5, "close"] = -1.0
        assert _validate_ohlc(df, "TEST") is False

    def test_low_above_open_fails(self) -> None:
        df = _make_df()
        df.loc[5, "low"] = df.loc[5, "open"] + 5.0
        assert _validate_ohlc(df, "TEST") is False

    def test_high_below_close_fails(self) -> None:
        df = _make_df()
        df.loc[5, "high"] = df.loc[5, "close"] - 5.0
        assert _validate_ohlc(df, "TEST") is False

    def test_negative_volume_fails(self) -> None:
        df = _make_df()
        df.loc[3, "volume"] = -1.0
        assert _validate_ohlc(df, "TEST") is False

    def test_infinite_value_fails(self) -> None:
        df = _make_df()
        df.loc[0, "close"] = float("inf")
        assert _validate_ohlc(df, "TEST") is False

    def test_missing_columns_fails(self) -> None:
        df = pd.DataFrame({"close": [100.0], "volume": [1e6]})
        assert _validate_ohlc(df, "TEST") is False


# ---------------------------------------------------------------------------
# convert_universe_to_dict_index
# ---------------------------------------------------------------------------

class TestConvertUniverseToDictIndex:
    def test_returns_nested_dict(self) -> None:
        u = _universe(2, 5)
        # Set date as index
        for sym in u:
            u[sym] = u[sym].set_index("date", drop=False)

        result = convert_universe_to_dict_index(u)
        assert "SYM0" in result
        assert "SYM1" in result
        for sym, date_dict in result.items():
            assert len(date_dict) == 5

    def test_values_are_dicts(self) -> None:
        u = _universe(1, 3)
        u["SYM0"] = u["SYM0"].set_index("date", drop=False)
        result = convert_universe_to_dict_index(u)
        first_key = list(result["SYM0"].keys())[0]
        assert isinstance(result["SYM0"][first_key], dict)

    def test_auto_sets_date_index(self) -> None:
        """Should work even if date is a column, not the index."""
        u = _universe(1, 3)
        result = convert_universe_to_dict_index(u)
        assert "SYM0" in result


# ---------------------------------------------------------------------------
# convert_universe_to_numpy_arrays
# ---------------------------------------------------------------------------

class TestConvertUniverseToNumpyArrays:
    def test_returns_arrays(self) -> None:
        u = _universe(2, 10)
        result = convert_universe_to_numpy_arrays(u)
        assert "SYM0" in result
        assert "close" in result["SYM0"]
        assert isinstance(result["SYM0"]["close"], np.ndarray)

    def test_date_column_excluded(self) -> None:
        u = _universe(1, 5)
        result = convert_universe_to_numpy_arrays(u)
        assert "date" not in result["SYM0"]

    def test_array_length_matches_df(self) -> None:
        u = _universe(1, 20)
        result = convert_universe_to_numpy_arrays(u)
        assert len(result["SYM0"]["close"]) == 20


# ---------------------------------------------------------------------------
# build_trading_calendar
# ---------------------------------------------------------------------------

class TestBuildTradingCalendar:
    def test_empty_universe(self) -> None:
        assert build_trading_calendar({}) == []

    def test_sorted_unique_dates(self) -> None:
        u = {
            "A": _make_df(5),
            "B": _make_df(5),
        }
        calendar = build_trading_calendar(u)
        assert calendar == sorted(set(calendar))
        assert len(calendar) == 5  # same 5 business days

    def test_calendar_type(self) -> None:
        u = _universe(1, 5)
        cal = build_trading_calendar(u)
        assert all(isinstance(d, pd.Timestamp) for d in cal)

    def test_union_of_all_dates(self) -> None:
        df_a = _make_df(3)
        df_b = pd.DataFrame({
            "date": pd.date_range("2020-02-01", periods=3, freq="B"),
            "open": [100.0] * 3, "high": [101.0] * 3,
            "low": [99.0] * 3, "close": [100.0] * 3, "volume": [1e6] * 3,
        })
        u = {"A": df_a, "B": df_b}
        cal = build_trading_calendar(u)
        assert len(cal) == 6  # 3 Jan + 3 Feb = 6 unique dates


# ---------------------------------------------------------------------------
# build_date_index
# ---------------------------------------------------------------------------

class TestBuildDateIndex:
    def test_empty_universe(self) -> None:
        assert build_date_index({}) == {}

    def test_structure(self) -> None:
        u = _universe(2, 5)
        result = build_date_index(u, num_workers=1)
        assert "SYM0" in result
        for key, val in result["SYM0"].items():
            assert isinstance(key, pd.Timestamp)
            assert isinstance(val, int)

    def test_index_is_zero_based(self) -> None:
        u = _universe(1, 5)
        result = build_date_index(u, num_workers=1)
        indices = sorted(result["SYM0"].values())
        assert indices == [0, 1, 2, 3, 4]

    def test_sequential_equals_parallel(self) -> None:
        u = _universe(3, 10)
        seq = build_date_index(u, num_workers=1)
        par = build_date_index(u, num_workers=2)
        for sym in seq:
            assert seq[sym] == par[sym]


# ---------------------------------------------------------------------------
# precompute_indicators
# ---------------------------------------------------------------------------

class TestPrecomputeIndicators:
    def test_empty_specs_returns_unchanged(self) -> None:
        u = _universe(2, 10)
        original_cols = set(u["SYM0"].columns)
        result = precompute_indicators(u, [], num_workers=1)
        assert set(result["SYM0"].columns) == original_cols

    def test_sma_added(self) -> None:
        u = _universe(1, 30)
        spec = IndicatorSpec(name="sma", params={"period": 10})
        result = precompute_indicators(u, [spec], num_workers=1)
        assert "sma_10" in result["SYM0"].columns

    def test_deduplication_applied(self) -> None:
        u = _universe(1, 30)
        specs = [
            IndicatorSpec(name="sma", params={"period": 10}),
            IndicatorSpec(name="sma", params={"period": 10}),  # duplicate
        ]
        result = precompute_indicators(u, specs, num_workers=1)
        # Should still succeed with single column added
        assert "sma_10" in result["SYM0"].columns

    def test_unknown_indicator_skipped(self) -> None:
        u = _universe(1, 30)
        spec = IndicatorSpec(name="nonexistent_indicator_xyz")
        # Should not raise; unknown indicators are skipped
        result = precompute_indicators(u, [spec], num_workers=1)
        assert "nonexistent_indicator_xyz" not in result["SYM0"].columns

    def test_multiple_indicators(self) -> None:
        u = _universe(2, 50)
        specs = [
            IndicatorSpec(name="sma", params={"period": 20}),
            IndicatorSpec(name="atr", params={"period": 14}),
        ]
        result = precompute_indicators(u, specs, num_workers=1)
        for sym in result:
            assert "sma_20" in result[sym].columns
            assert "atr" in result[sym].columns


# ---------------------------------------------------------------------------
# load_universe (filesystem)
# ---------------------------------------------------------------------------

class TestLoadUniverse:
    def _setup_dir(self, symbols: dict[str, pd.DataFrame]) -> Path:
        tmp = Path(tempfile.mkdtemp())
        for sym, df in symbols.items():
            _write_parquet(df, tmp / f"{sym}.parquet")
        return tmp

    def test_loads_all_files(self) -> None:
        syms = {"AAPL": _make_df(250), "MSFT": _make_df(250)}
        d = self._setup_dir(syms)
        u = load_universe(d, num_workers=1)
        assert set(u.keys()) == {"AAPL", "MSFT"}

    def test_nonexistent_dir_returns_empty(self) -> None:
        u = load_universe(Path("/nonexistent/path/xyz"), num_workers=1)
        assert u == {}

    def test_empty_dir_returns_empty(self) -> None:
        d = Path(tempfile.mkdtemp())
        u = load_universe(d, num_workers=1)
        assert u == {}

    def test_min_history_filter(self) -> None:
        """Stock with fewer bars than min_history_days should be excluded."""
        syms = {
            "LONG": _make_df(300),
            "SHORT": _make_df(10),  # too short
        }
        d = self._setup_dir(syms)
        u = load_universe(d, min_history_days=200, num_workers=1)
        assert "LONG" in u
        assert "SHORT" not in u

    def test_symbols_allowlist(self) -> None:
        syms = {"AAPL": _make_df(250), "MSFT": _make_df(250), "GOOG": _make_df(250)}
        d = self._setup_dir(syms)
        u = load_universe(d, symbols=["AAPL", "MSFT"], num_workers=1)
        assert "AAPL" in u
        assert "MSFT" in u
        assert "GOOG" not in u

    def test_excluded_symbols(self) -> None:
        syms = {"AAPL": _make_df(250), "BAD": _make_df(250)}
        d = self._setup_dir(syms)
        u = load_universe(d, excluded_symbols=["BAD"], num_workers=1)
        assert "BAD" not in u
        assert "AAPL" in u

    def test_excluded_symbols_file(self) -> None:
        import tempfile as tf
        syms = {"AAPL": _make_df(250), "BAD": _make_df(250)}
        d = self._setup_dir(syms)
        exc_file = d / "excluded.txt"
        exc_file.write_text("BAD\n")
        u = load_universe(d, excluded_symbols_file=str(exc_file), num_workers=1)
        assert "BAD" not in u

    def test_date_as_index(self) -> None:
        syms = {"AAPL": _make_df(250)}
        d = self._setup_dir(syms)
        u = load_universe(d, num_workers=1)
        assert u["AAPL"].index.name == "date"

    def test_start_date_filter(self) -> None:
        syms = {"AAPL": _make_df(300)}
        d = self._setup_dir(syms)
        u = load_universe(d, start_date="2020-03-01", num_workers=1)
        assert u["AAPL"]["date"].min() >= pd.Timestamp("2020-03-01")

    def test_ohlc_violation_excluded(self) -> None:
        """Stock with OHLC violations should be dropped."""
        bad_df = _make_df(250)
        bad_df.loc[5, "low"] = bad_df.loc[5, "open"] + 100.0  # low > open
        syms = {"BAD": bad_df, "GOOD": _make_df(250)}
        d = self._setup_dir(syms)
        u = load_universe(d, num_workers=1)
        assert "BAD" not in u
        assert "GOOD" in u
