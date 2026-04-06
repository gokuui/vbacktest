"""Data loading and preprocessing for backtesting."""
from __future__ import annotations

import logging
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vbacktest.indicators import IndicatorSpec

log = logging.getLogger(__name__)

# Required OHLCV columns
_OHLCV = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Universe conversion helpers (performance optimisations)
# ---------------------------------------------------------------------------

def convert_universe_to_dict_index(
    universe: dict[str, pd.DataFrame],
) -> dict[str, dict[pd.Timestamp, dict[str, object]]]:
    """Convert universe DataFrames to nested dicts for O(1) bar lookups.

    Converts ``{symbol: DataFrame}`` → ``{symbol: {Timestamp: {col: value}}}``.

    Args:
        universe: Symbol → DataFrame (date must be the index or a column).

    Returns:
        Symbol → date-keyed dict of column dicts.
    """
    result: dict[str, dict[pd.Timestamp, dict[str, object]]] = {}
    for symbol, df in universe.items():
        if "date" not in df.index.names and "date" in df.columns:
            df = df.set_index("date", drop=False)
        result[symbol] = df.to_dict("index")  # type: ignore[assignment]
    log.debug("Converted %d stocks to dict-indexed format", len(result))
    return result


def convert_universe_to_numpy_arrays(
    universe: dict[str, pd.DataFrame],
) -> dict[str, dict[str, np.ndarray]]:
    """Convert each stock's columns to numpy arrays for fast indexed access.

    Numpy array indexing (``arr[idx]``) is ~50–100x faster than pandas
    ``iloc``. Useful for lookback operations like ``arr[idx-1]``.

    Args:
        universe: Symbol → DataFrame.

    Returns:
        Symbol → column-name → numpy ndarray.
    """
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for symbol, df in universe.items():
        arrays[symbol] = {
            col: np.asarray(df[col].values)
            for col in df.columns
            if col != "date"
        }
    log.debug("Converted %d stocks to numpy arrays", len(arrays))
    return arrays


# ---------------------------------------------------------------------------
# OHLC validation
# ---------------------------------------------------------------------------

def _validate_ohlc(df: pd.DataFrame, symbol: str) -> bool:
    """Return True if OHLCV data satisfies basic sanity constraints."""
    ohlcv = df[_OHLCV] if all(c in df.columns for c in _OHLCV) else None
    if ohlcv is None:
        log.warning("%s: missing OHLCV columns", symbol)
        return False

    if not ohlcv.notna().all().all():
        log.warning("%s: contains NaN values", symbol)
        return False

    if np.isinf(ohlcv.values).any():
        log.warning("%s: contains infinite values", symbol)
        return False

    price_cols = df[["open", "high", "low", "close"]]
    if (price_cols <= 0).any().any():
        log.warning("%s: contains non-positive prices", symbol)
        return False

    if (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any():
        log.warning("%s: low > open or close", symbol)
        return False

    if (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any():
        log.warning("%s: high < open or close", symbol)
        return False

    if (df["volume"] < 0).any():
        log.warning("%s: negative volume", symbol)
        return False

    return True


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def _load_stock_worker(
    parquet_file: Path,
    symbols: list[str] | None,
    excluded: set[str],
    start_date: str | None,
    end_date: str | None,
    min_history_days: int,
    min_confidence: int,
) -> tuple[str, pd.DataFrame | None]:
    """Load and filter a single parquet file.

    Returns ``(symbol, DataFrame)`` or ``(symbol, None)`` if filtered out.
    """
    symbol = parquet_file.stem

    if symbol in excluded:
        return symbol, None

    if symbols and symbol.lower() not in {s.lower() for s in symbols}:
        return symbol, None

    try:
        df = pd.read_parquet(parquet_file)

        # Normalise timestamps — sources differ on time-of-day component
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        if not _validate_ohlc(df, symbol):
            return symbol, None

        if min_confidence > 1 and "confidence" in df.columns:
            df = df[df["confidence"] >= min_confidence]

        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        if len(df) < min_history_days:
            return symbol, None

        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date", drop=False)
        return symbol, df

    except Exception as exc:
        log.warning("Failed to load %s: %s", parquet_file, exc)
        return symbol, None


def _load_stock_worker_wrapper(args: Any) -> tuple[str, pd.DataFrame | None]:
    return _load_stock_worker(*args)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_universe(
    validated_dir: Path | str,
    symbols: list[str] | None = None,
    excluded_symbols: list[str] | None = None,
    excluded_symbols_file: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_history_days: int = 200,
    min_confidence: int = 1,
    num_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Load universe of stocks from validated parquet files.

    Args:
        validated_dir: Directory containing ``*.parquet`` files.
        symbols: Allowlist of symbols to load (``None`` = all).
        excluded_symbols: Symbols to skip entirely.
        excluded_symbols_file: Path to newline-separated exclusion file.
        start_date: Start date filter ``YYYY-MM-DD``.
        end_date: End date filter ``YYYY-MM-DD``.
        min_history_days: Minimum number of trading days required.
        min_confidence: Minimum confidence column value.
        num_workers: Parallel workers (``None`` = auto, ``1`` = sequential).

    Returns:
        Symbol → DataFrame mapping (date as index).
    """
    validated_dir = Path(validated_dir)
    if not validated_dir.exists():
        log.warning("Validated directory does not exist: %s", validated_dir)
        return {}

    parquet_files = list(validated_dir.glob("*.parquet"))
    if not parquet_files:
        log.warning("No parquet files found in %s", validated_dir)
        return {}

    # Build exclusion set
    excluded: set[str] = set(excluded_symbols or [])
    if excluded_symbols_file:
        exc_path = Path(excluded_symbols_file)
        if exc_path.exists():
            excluded.update(
                s.strip() for s in exc_path.read_text().splitlines() if s.strip()
            )
        else:
            log.warning("excluded_symbols_file not found: %s", exc_path)

    log.info("Loading universe from %s (%d files)", validated_dir, len(parquet_files))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    work_items = [
        (pf, symbols, excluded, start_date, end_date, min_history_days, min_confidence)
        for pf in parquet_files
    ]

    if num_workers == 1:
        universe: dict[str, pd.DataFrame] = {}
        for item in work_items:
            sym, df = _load_stock_worker(*item)
            if df is not None:
                universe[sym] = df
        log.info("Loaded %d stocks (sequential)", len(universe))
        return universe

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            universe = {}
            for sym, df in pool.imap_unordered(_load_stock_worker_wrapper, work_items):
                if df is not None:
                    universe[sym] = df
        log.info("Loaded %d stocks (parallel, workers=%d)", len(universe), num_workers)
        return universe
    except Exception as exc:
        log.error("Parallel load failed: %s — falling back to sequential", exc)
        return load_universe(
            validated_dir, symbols, excluded_symbols, excluded_symbols_file,
            start_date, end_date, min_history_days, min_confidence, num_workers=1,
        )


def build_trading_calendar(universe: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Build a sorted list of all unique trading dates across the universe.

    Args:
        universe: Symbol → DataFrame.

    Returns:
        Sorted list of unique ``pd.Timestamp`` dates.
    """
    if not universe:
        return []

    all_dates: set[pd.Timestamp] = set()
    for df in universe.values():
        all_dates.update(pd.Timestamp(d) for d in df["date"].values)

    calendar = sorted(all_dates)
    log.info(
        "Trading calendar: %d dates from %s to %s",
        len(calendar), calendar[0], calendar[-1],
    )
    return calendar


def precompute_indicators(
    universe: dict[str, pd.DataFrame],
    indicator_specs: list[IndicatorSpec],
    num_workers: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Precompute technical indicators for all stocks in the universe.

    Args:
        universe: Symbol → DataFrame.
        indicator_specs: Indicators to compute (duplicates are removed).
        num_workers: Parallel workers (``None`` = auto, ``1`` = sequential).

    Returns:
        The same universe dict with indicator columns added to each DataFrame.
    """
    if not indicator_specs:
        return universe

    from vbacktest.indicator_dedup import deduplicate_indicators
    from vbacktest.indicators import INDICATOR_REGISTRY

    original_count = len(indicator_specs)
    indicator_specs = deduplicate_indicators(indicator_specs)
    if len(indicator_specs) < original_count:
        log.info(
            "Deduplicated %d → %d indicators", original_count, len(indicator_specs)
        )

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    if num_workers == 1:
        for symbol, df in universe.items():
            df_copy = df.copy(deep=True)
            for spec in indicator_specs:
                if spec.name not in INDICATOR_REGISTRY:
                    log.warning("Unknown indicator %s for %s", spec.name, symbol)
                    continue
                try:
                    INDICATOR_REGISTRY[spec.name](df_copy, **spec.params)
                except Exception as exc:
                    log.warning("Failed to compute %s for %s: %s", spec.name, symbol, exc)
            universe[symbol] = df_copy
        return universe

    def _worker(item: Any) -> tuple[str, pd.DataFrame | None]:
        sym, df, specs = item
        df_copy = df.copy(deep=True)
        for spec in specs:
            if spec.name not in INDICATOR_REGISTRY:
                continue
            try:
                INDICATOR_REGISTRY[spec.name](df_copy, **spec.params)
            except Exception:
                pass
        return sym, df_copy

    try:
        work_items = [(sym, df, indicator_specs) for sym, df in universe.items()]
        with multiprocessing.Pool(processes=num_workers) as pool:
            for sym, df_with_ind in pool.imap_unordered(_worker, work_items):
                if df_with_ind is not None:
                    universe[sym] = df_with_ind
        return universe
    except Exception as exc:
        log.error("Parallel indicator precomputation failed: %s — falling back", exc)
        return precompute_indicators(universe, indicator_specs, num_workers=1)


def build_date_index(
    universe: dict[str, pd.DataFrame],
    num_workers: int | None = None,
) -> dict[str, dict[pd.Timestamp, int]]:
    """Build date → integer-index mapping for O(1) lookups.

    Args:
        universe: Symbol → DataFrame.
        num_workers: Parallel workers (``None`` = auto, ``1`` = sequential).

    Returns:
        Symbol → {``pd.Timestamp`` → integer row index}.
    """
    if not universe:
        return {}

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    if num_workers == 1:
        date_index: dict[str, dict[pd.Timestamp, int]] = {}
        for symbol, df in universe.items():
            date_index[symbol] = {
                pd.Timestamp(d): i for i, d in enumerate(df["date"].values)
            }
        return date_index

    def _worker(item: Any) -> tuple[str, dict[pd.Timestamp, int]]:
        sym, df = item
        return sym, {pd.Timestamp(d): i for i, d in enumerate(df["date"].values)}

    try:
        work_items = list(universe.items())
        date_index = {}
        with multiprocessing.Pool(processes=num_workers) as pool:
            for sym, idx in pool.imap_unordered(_worker, work_items):
                date_index[sym] = idx
        return date_index
    except Exception as exc:
        log.error("Parallel date index build failed: %s — falling back", exc)
        return build_date_index(universe, num_workers=1)
