# Test & Quality Gap Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 4 known test/quality gaps: pytest warning on TestResult, silent unknown kwargs in BacktestConfig.simple(), shallow extras strategy smoke tests, and missing data loader edge case tests.

**Architecture:** Four independent, TDD-first tasks. Each produces a commit. No strategy implementation logic changes — only tests added and two targeted production fixes. Target: 363 → ~393 tests, 0 mypy errors maintained.

**Post-Codex-review changes:**
- Task 3: Added multi-symbol ranking test for FilteredMomentum + fast-path-absent contract tests for ElderImpulse and VCPBreakout
- Task 4: Reduced from 4 tests to 2 — empty_dir, min_history, and excluded_symbols are already covered at lines 285/290/309 of test_data_loader.py; only corrupt parquet + empty-after-date-filter are genuinely new
- Final verification: grep targets `PytestCollectionWarning` specifically instead of all warnings

**Tech Stack:** pytest, numpy, pandas, vbacktest internals (BarContext, Signal, load_universe)

---

## File Map

| File | Change |
|---|---|
| `src/vbacktest/analysis/report.py` | Add `__test__: ClassVar[bool] = False` to `TestResult` |
| `src/vbacktest/config.py` | Raise `ConfigError` for unknown kwargs in `BacktestConfig.simple()` |
| `tests/test_extras_strategies.py` | Add 5 strategy logic test classes (~25 new tests) |
| `tests/test_extras_strategies.py` | Add fast-path-absent contract tests + multi-symbol FilteredMomentum ranking test |
| `tests/test_data_loader.py` | Add 2 genuinely new edge case tests (corrupt parquet, empty-after-date-filter) |

---

## Task 1: Fix PytestCollectionWarning on TestResult

**Files:**
- Modify: `src/vbacktest/analysis/report.py:26`

pytest tries to collect `TestResult` as a test class because its name starts with `Test` and it has an `__init__`. Fix: add `__test__: ClassVar[bool] = False` — a standard pytest opt-out that dataclasses support via `ClassVar`.

- [ ] **Step 1: Write the failing test (verify warning exists)**

Run:
```bash
cd ~/code/vbacktest && source .venv/bin/activate
pytest tests/test_analysis/test_go_no_go.py -v 2>&1 | grep -i "warning\|PytestCollection"
```
Expected output contains:
```
PytestCollectionWarning: cannot collect test class 'TestResult' because it has a __init__ constructor
```

- [ ] **Step 2: Apply the fix to report.py**

In `src/vbacktest/analysis/report.py`, add `ClassVar` to the import and `__test__` to the dataclass:

```python
# Change this import line (line 7):
from typing import Any

# To:
from typing import Any, ClassVar
```

Then in the `TestResult` dataclass (around line 26), add the `__test__` class variable as the first field:

```python
@dataclass
class TestResult:
    """Single test outcome within a validation category."""

    __test__: ClassVar[bool] = False  # prevent pytest from collecting this as a test class

    name: str
    value: str       # formatted display value
    status: str      # "PASS", "WARN", or "FAIL"
    hard_nogo: bool = False
    detail: str = ""
```

- [ ] **Step 3: Verify warning is gone**

Run:
```bash
pytest tests/ -q 2>&1 | grep "PytestCollectionWarning"
```
Expected: no output (warning is gone). The test suite still passes 363 tests.

Run:
```bash
pytest tests/ -q 2>&1 | tail -5
```
Expected:
```
363 passed in ...
```

- [ ] **Step 4: Verify mypy still clean**

Run:
```bash
python -m mypy src/vbacktest/analysis/report.py --no-error-summary
```
Expected: `Success: no issues found in 1 source file`

- [ ] **Step 5: Commit**

```bash
cd ~/code/vbacktest
git add src/vbacktest/analysis/report.py
git commit -m "fix: suppress PytestCollectionWarning on TestResult dataclass"
```

---

## Task 2: BacktestConfig.simple() raises ConfigError for unknown kwargs

**Files:**
- Modify: `src/vbacktest/config.py`
- Modify: `tests/test_config.py`

Currently `BacktestConfig.simple(validated_dir, foo=123)` silently ignores `foo`. This should raise `ConfigError` so users don't typo a parameter and wonder why it has no effect.

- [ ] **Step 1: Write the failing test**

Add this test class to `tests/test_config.py` (append after `TestBacktestConfig`):

```python
class TestBacktestConfigSimpleValidation:
    def test_unknown_kwarg_raises_config_error(self):
        with pytest.raises(ConfigError, match="Unknown parameters"):
            BacktestConfig.simple("/tmp/data", typo_param=123)

    def test_multiple_unknown_kwargs_listed_in_error(self):
        with pytest.raises(ConfigError, match="foo") as exc_info:
            BacktestConfig.simple("/tmp/data", foo=1, bar=2)
        assert "bar" in str(exc_info.value)

    def test_known_kwargs_still_work(self):
        bc = BacktestConfig.simple("/tmp/data", capital=50_000, max_positions=3)
        assert bc.portfolio.initial_capital == 50_000
        assert bc.portfolio.max_positions == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
pytest tests/test_config.py::TestBacktestConfigSimpleValidation -v
```
Expected: 2 FAILED (unknown kwarg tests), 1 PASSED (known kwargs test).

- [ ] **Step 3: Implement the fix in config.py**

In `src/vbacktest/config.py`, update `BacktestConfig.simple()`:

```python
@classmethod
def simple(
    cls,
    validated_dir: str | Path,
    *,
    capital: float = 100_000,
    max_positions: int = 10,
    **kwargs: Any,
) -> BacktestConfig:
    if kwargs:
        raise ConfigError(
            f"Unknown parameters: {sorted(kwargs)}. "
            f"Valid parameters are: capital, max_positions"
        )
    return cls(
        data=DataConfig(validated_dir=validated_dir),
        portfolio=PortfolioConfig(initial_capital=capital, max_positions=max_positions),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
pytest tests/test_config.py -v
```
Expected: all tests PASS including the 3 new ones.

- [ ] **Step 5: Verify full suite still green**

Run:
```bash
pytest tests/ -q 2>&1 | tail -3
```
Expected: `366 passed in ...`

- [ ] **Step 6: Commit**

```bash
cd ~/code/vbacktest
git add src/vbacktest/config.py tests/test_config.py
git commit -m "fix: BacktestConfig.simple() raises ConfigError for unknown kwargs"
```

---

## Task 3: Extras strategy logic tests

**Files:**
- Modify: `tests/test_extras_strategies.py`

Add logic tests for 5 extras strategies with distinct entry mechanisms. Each gets a class with two tests: signal fires when conditions are met, signal does NOT fire when one key condition is broken.

**Important context per strategy:**
- `ElderImpulseBreakoutStrategy`: fast path only (`else: continue`) — must pass `universe_arrays`
- `MinerviniSEPAStrategy`: slow path only (uses `df.iloc`) — pass `universe_arrays=None`
- `ADXTrendStrategy`: slow path only (uses `df.iloc`) — pass `universe_arrays=None`
- `VCPBreakoutStrategy`: fast path preferred — pass `universe_arrays`
- `FilteredMomentumStrategy`: has slow path — pass `universe_arrays=None` for test simplicity

- [ ] **Step 1: Write all failing tests**

Append the following to `tests/test_extras_strategies.py`:

```python
import numpy as np
import pandas as pd
from vbacktest.strategy import BarContext, SignalAction


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _DummyPortfolio:
    def has_position(self, s: str) -> bool:
        return False


def _make_base_df(n: int = 300) -> pd.DataFrame:
    """Synthetic uptrend: close = 100 + i*0.1"""
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100.0 + np.arange(n, dtype=float) * 0.1
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 2_000_000.0),
    })


def _make_arrays(df: pd.DataFrame, extra: dict) -> dict:
    """Build numpy arrays dict from df columns + extra precomputed columns."""
    arrays = {col: df[col].to_numpy(dtype=float) for col in df.columns if col != "date"}
    arrays.update(extra)
    return arrays


def _ctx_slow(symbol: str, df: pd.DataFrame, idx: int) -> BarContext:
    """BarContext using slow path (universe_arrays=None)."""
    bar = {col: df[col].iloc[idx] for col in df.columns if col != "date"}
    return BarContext(
        date=df["date"].iloc[idx],
        universe={symbol: df},
        universe_idx={symbol: idx},
        portfolio=_DummyPortfolio(),  # type: ignore[arg-type]
        current_prices={symbol: bar},
        universe_arrays=None,
    )


def _ctx_fast(symbol: str, df: pd.DataFrame, idx: int, arrays: dict) -> BarContext:
    """BarContext using fast path (universe_arrays provided)."""
    bar = {col: df[col].iloc[idx] for col in df.columns if col != "date"}
    return BarContext(
        date=df["date"].iloc[idx],
        universe={symbol: df},
        universe_idx={symbol: idx},
        portfolio=_DummyPortfolio(),  # type: ignore[arg-type]
        current_prices={symbol: bar},
        universe_arrays={symbol: arrays},
    )


# ---------------------------------------------------------------------------
# ElderImpulseBreakoutStrategy — fast path only
# ---------------------------------------------------------------------------

class TestElderImpulseLogic:
    """Entry: Elder Impulse positive + Stage2 + Kaufman ER + near high."""

    def _make_ctx(self, signal: bool) -> BarContext:
        from vbacktest.strategies.extras import ElderImpulseBreakoutStrategy
        n = 300
        df = _make_base_df(n)
        idx = n - 1
        close = df["close"].to_numpy()

        ema13 = np.full(n, 120.0)
        ema13[idx - 1] = 121.0
        ema13[idx] = 122.0 if signal else 120.0  # rising if signal else flat

        macd_hist = np.full(n, 0.5)
        macd_hist[idx - 1] = 0.8
        macd_hist[idx] = 1.0  # always rising (test only Elder EMA branch)

        extra = {
            "ema_13": ema13,
            "macd_hist": macd_hist,
            "ema_10": np.full(n, 128.0),   # close[299]=129.9 > 128 ✓
            "ema_21": np.full(n, 126.0),   # 128 > 126 ✓
            "sma_50": np.full(n, 120.0),   # 126 > 120 ✓
            "sma_150": np.full(n, 110.0),  # 120 > 110 ✓
            "atr_14": np.full(n, 2.0),
            "high_50": np.full(n, 131.0),  # 129.9 within 3% of 131 ✓
        }
        arrays = _make_arrays(df, extra)
        s = ElderImpulseBreakoutStrategy()
        return _ctx_fast("SYM", df, idx, arrays)

    def test_signal_fires_when_all_conditions_met(self) -> None:
        from vbacktest.strategies.extras import ElderImpulseBreakoutStrategy
        s = ElderImpulseBreakoutStrategy()
        ctx = self._make_ctx(signal=True)
        signals = s.on_bar(ctx)
        assert len(signals) == 1
        assert signals[0].symbol == "SYM"
        assert signals[0].action == SignalAction.BUY

    def test_no_signal_when_elder_impulse_fails(self) -> None:
        from vbacktest.strategies.extras import ElderImpulseBreakoutStrategy
        s = ElderImpulseBreakoutStrategy()
        ctx = self._make_ctx(signal=False)  # ema13 flat — Elder Impulse not rising
        signals = s.on_bar(ctx)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# MinerviniSEPAStrategy — slow path only
# ---------------------------------------------------------------------------

class TestMinerviniSEPALogic:
    """Entry: Stage2 + MA200 rising + near 52w high + ATR contraction + breakout + volume."""

    def _make_ctx(self, volume_surge: bool) -> BarContext:
        n = 300
        df = _make_base_df(n)
        idx = n - 1

        # Force close to a known value at idx
        df.at[df.index[idx], "close"] = 105.0

        # Stage 2: close(105) > sma50(100) > sma150(95) > sma200(90)
        df["sma_50"] = 100.0
        df["sma_150"] = 95.0
        df["sma_200"] = 90.0
        # MA200 rising: current(90) > 22 bars ago(85)
        df.at[df.index[idx - 22], "sma_200"] = 85.0

        # Near 52w high: 105/110 = 95.5% > 70%
        df["high_52w"] = 110.0

        # ATR contraction: recent(idx-10:idx) mean=1.0 < earlier(idx-30:idx-10) mean*0.7=5*0.7=3.5
        df["atr_14"] = 5.0  # baseline (earlier)
        df.iloc[idx - 10:idx, df.columns.get_loc("atr_14")] = 1.0  # recent = low

        # Breakout: close(105) > 10-day high. Set high[idx-10:idx] = 104
        df.iloc[idx - 10:idx, df.columns.get_loc("high")] = 104.0

        # Volume surge: need volume > volume_sma_50 * 1.3
        df["volume_sma_50"] = 1_000_000.0
        df.at[df.index[idx], "volume"] = 2_000_000.0 if volume_surge else 500_000.0

        # Trailing MA needed for exit rule but not entry
        df["sma_10"] = 103.0

        return _ctx_slow("SYM", df, idx)

    def test_signal_fires_when_all_conditions_met(self) -> None:
        from vbacktest.strategies.extras import MinerviniSEPAStrategy
        s = MinerviniSEPAStrategy()
        ctx = self._make_ctx(volume_surge=True)
        signals = s.on_bar(ctx)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.BUY

    def test_no_signal_when_volume_insufficient(self) -> None:
        from vbacktest.strategies.extras import MinerviniSEPAStrategy
        s = MinerviniSEPAStrategy()
        ctx = self._make_ctx(volume_surge=False)  # volume 500k < 1.3M threshold
        signals = s.on_bar(ctx)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# ADXTrendStrategy — slow path only
# ---------------------------------------------------------------------------

class TestADXTrendLogic:
    """Entry: ADX > 25 + rising + +DI > -DI + price above MA50 + ROC > 5% + near high."""

    def _make_ctx(self, bullish_di: bool) -> BarContext:
        n = 300
        df = _make_base_df(n)
        idx = n - 1

        df.at[df.index[idx], "close"] = 105.0
        df.at[df.index[idx - 1], "close"] = 104.0  # previous close (filter 6: close > prev)

        # ADX > 25 and rising vs 5 bars ago
        df["adx_14"] = 30.0
        df.at[df.index[idx - 5], "adx_14"] = 25.0  # 30 > 25 → rising ✓

        # +DI > -DI (bullish) or reversed (no signal)
        df["plus_di_14"] = 20.0 if bullish_di else 5.0
        df["minus_di_14"] = 10.0

        # Price above SMA50
        df["sma_50"] = 100.0  # close(105) > 100 ✓

        # ROC(10) > 5%
        df["roc_10"] = 6.0

        # Near 52w high: close(105) >= high_52w(110)*0.80=88 ✓
        df["high_52w"] = 110.0

        # ATR contraction (filter 8): recent < earlier * 0.8
        df["atr_14"] = 3.0  # earlier
        df.iloc[idx - 10:idx, df.columns.get_loc("atr_14")] = 1.0  # recent=1.0 < 3.0*0.8=2.4 ✓

        # Required for exit rule column lookup
        df["sma_10"] = 103.0

        return _ctx_slow("SYM", df, idx)

    def test_signal_fires_when_all_conditions_met(self) -> None:
        from vbacktest.strategies.extras import ADXTrendStrategy
        s = ADXTrendStrategy()
        ctx = self._make_ctx(bullish_di=True)
        signals = s.on_bar(ctx)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.BUY

    def test_no_signal_when_di_bearish(self) -> None:
        from vbacktest.strategies.extras import ADXTrendStrategy
        s = ADXTrendStrategy()
        ctx = self._make_ctx(bullish_di=False)  # +DI(5) <= -DI(10) → no signal
        signals = s.on_bar(ctx)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# VCPBreakoutStrategy — fast path
# ---------------------------------------------------------------------------

class TestVCPBreakoutLogic:
    """Entry: Stage2 + SMA50 rising + close>EMA10 + ATR(5)<ATR(20)*0.8 + ATR(20)<ATR(50) + near high + volume."""

    def _make_ctx(self, atr_contracted: bool) -> BarContext:
        n = 300
        df = _make_base_df(n)
        idx = n - 1
        close = df["close"].to_numpy()

        # close[299]=129.9, sma50=120, sma150=110 → Stage2 ✓
        sma50 = np.full(n, 120.0)
        sma50[idx - 5] = 119.0  # rising: 120 > 119 ✓

        extra = {
            "sma_50": sma50,
            "sma_150": np.full(n, 110.0),
            "ema_10": np.full(n, 128.0),         # close(129.9) > 128 ✓
            "atr_5": np.full(n, 0.5 if atr_contracted else 2.0),
            "atr_20": np.full(n, 1.0),            # atr5(0.5) < atr20(1.0)*0.8=0.8 ✓ when contracted
            "atr_50": np.full(n, 2.0),            # atr20(1.0) < atr50(2.0) ✓
            "high_50": np.full(n, 131.0),         # 129.9 within 5% of 131 ✓
            "volume_sma_20": np.full(n, 1_000_000.0),  # vol(2M) > 1M*1.2=1.2M ✓
            "sma_10": np.full(n, 128.5),
        }
        arrays = _make_arrays(df, extra)
        return _ctx_fast("SYM", df, idx, arrays)

    def test_signal_fires_when_all_conditions_met(self) -> None:
        from vbacktest.strategies.extras import VCPBreakoutStrategy
        s = VCPBreakoutStrategy()
        ctx = self._make_ctx(atr_contracted=True)
        signals = s.on_bar(ctx)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.BUY

    def test_no_signal_when_atr_not_contracted(self) -> None:
        from vbacktest.strategies.extras import VCPBreakoutStrategy
        s = VCPBreakoutStrategy()
        ctx = self._make_ctx(atr_contracted=False)  # atr5(2.0) >= atr20(1.0)*0.8=0.8 → fails
        signals = s.on_bar(ctx)
        assert len(signals) == 0


# ---------------------------------------------------------------------------
# FilteredMomentumStrategy — slow path
# ---------------------------------------------------------------------------

class TestFilteredMomentumLogic:
    """Entry: Stage2 + ROC > 20% + near 52w high + ATR contraction. Ranked by ROC."""

    def _make_ctx(self, stage2_ok: bool) -> BarContext:
        n = 300
        df = _make_base_df(n)
        idx = n - 1

        close_val = 109.0 if stage2_ok else 80.0
        df.at[df.index[idx], "close"] = close_val

        # Stage 2: close > sma50 > sma200
        df["sma_50"] = 100.0    # 109 > 100 (ok) or 80 < 100 (fails)
        df["sma_200"] = 90.0

        # ROC(63) > 20%
        df["roc_63"] = 25.0

        # Near 52w high: close/high >= 90%  → 109/110=99.1% ✓
        df["high_52w"] = 110.0

        # ATR contraction: recent < earlier
        df["atr_14"] = 3.0
        df.iloc[idx - 5:idx + 1, df.columns.get_loc("atr_14")] = 1.0  # recent
        df.iloc[idx - 20:idx - 5, df.columns.get_loc("atr_14")] = 3.0  # earlier

        df["sma_10"] = 108.0

        return _ctx_slow("SYM", df, idx)

    def test_signal_fires_when_all_conditions_met(self) -> None:
        from vbacktest.strategies.extras import FilteredMomentumStrategy
        s = FilteredMomentumStrategy()
        ctx = self._make_ctx(stage2_ok=True)
        signals = s.on_bar(ctx)
        assert len(signals) == 1
        assert signals[0].action == SignalAction.BUY

    def test_no_signal_when_stage2_fails(self) -> None:
        from vbacktest.strategies.extras import FilteredMomentumStrategy
        s = FilteredMomentumStrategy()
        ctx = self._make_ctx(stage2_ok=False)  # close(80) < sma50(100) → no stage 2
        signals = s.on_bar(ctx)
        assert len(signals) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```bash
pytest tests/test_extras_strategies.py::TestElderImpulseLogic \
       tests/test_extras_strategies.py::TestMinerviniSEPALogic \
       tests/test_extras_strategies.py::TestADXTrendLogic \
       tests/test_extras_strategies.py::TestVCPBreakoutLogic \
       tests/test_extras_strategies.py::TestFilteredMomentumLogic -v
```
Expected: 10 FAILED (tests exist but entry conditions are not yet verified to produce signals). The failures confirm we're testing real behavior, not just smoke.

> **Note:** If tests PASS immediately, the synthetic data is working correctly and the strategy logic fires as expected — this is acceptable. The intent is to lock in the behavior.

- [ ] **Step 3: Run full suite to ensure no regressions**

Run:
```bash
pytest tests/ -q 2>&1 | tail -5
```
Expected: `~376 passed in ...` (363 original + 10 new + 3 from Task 2)

- [ ] **Step 4: Add fast-path-absent contract tests and multi-symbol FilteredMomentum test**

Append additionally to `tests/test_extras_strategies.py`:

```python
class TestFastPathAbsentContract:
    """ElderImpulse and VCPBreakout are fast-path-only. When universe_arrays is None,
    no signals should be emitted (else: continue). This locks in the contract so
    a future regression in engine context wiring doesn't silently disable entries."""

    def test_elder_impulse_no_signals_without_arrays(self) -> None:
        from vbacktest.strategies.extras import ElderImpulseBreakoutStrategy
        n = 300
        df = _make_base_df(n)
        # Apply indicators so df has required columns
        from vbacktest.indicators import INDICATOR_REGISTRY
        from vbacktest.strategies.extras import ElderImpulseBreakoutStrategy as E
        s = E()
        for spec in s.indicators():
            if spec.name in INDICATOR_REGISTRY:
                INDICATOR_REGISTRY[spec.name](df, **spec.params)
        ctx = _ctx_slow("SYM", df, n - 1)  # universe_arrays=None
        signals = s.on_bar(ctx)
        assert signals == [], "ElderImpulse must not fire without universe_arrays"

    def test_vcp_no_signals_without_arrays(self) -> None:
        from vbacktest.strategies.extras import VCPBreakoutStrategy
        n = 300
        df = _make_base_df(n)
        from vbacktest.indicators import INDICATOR_REGISTRY
        s = VCPBreakoutStrategy()
        for spec in s.indicators():
            if spec.name in INDICATOR_REGISTRY:
                INDICATOR_REGISTRY[spec.name](df, **spec.params)
        ctx = _ctx_slow("SYM", df, n - 1)  # universe_arrays=None
        signals = s.on_bar(ctx)
        assert signals == [], "VCPBreakout must not fire without universe_arrays"


class TestFilteredMomentumRanking:
    """FilteredMomentum ranks by ROC and emits top-N. With 2 qualifying symbols,
    the one with higher ROC should rank first."""

    def test_multi_symbol_ranking(self) -> None:
        from vbacktest.strategies.extras import FilteredMomentumStrategy
        n = 300
        idx = n - 1
        s = FilteredMomentumStrategy()

        def _make_qualifying_df(roc_val: float) -> pd.DataFrame:
            df = _make_base_df(n)
            df.at[df.index[idx], "close"] = 109.0
            df["sma_50"] = 100.0
            df["sma_200"] = 90.0
            df["roc_63"] = roc_val
            df["high_52w"] = 110.0
            df["atr_14"] = 3.0
            df.iloc[idx - 5:idx + 1, df.columns.get_loc("atr_14")] = 1.0
            df.iloc[idx - 20:idx - 5, df.columns.get_loc("atr_14")] = 3.0
            df["sma_10"] = 108.0
            return df

        df_a = _make_qualifying_df(roc_val=40.0)  # higher ROC
        df_b = _make_qualifying_df(roc_val=25.0)  # lower ROC

        ctx = BarContext(
            date=df_a["date"].iloc[idx],
            universe={"SYMA": df_a, "SYMB": df_b},
            universe_idx={"SYMA": idx, "SYMB": idx},
            portfolio=_DummyPortfolio(),  # type: ignore[arg-type]
            current_prices={
                "SYMA": {col: df_a[col].iloc[idx] for col in df_a.columns if col != "date"},
                "SYMB": {col: df_b[col].iloc[idx] for col in df_b.columns if col != "date"},
            },
            universe_arrays=None,
        )
        signals = s.on_bar(ctx)
        # Both should qualify; SYMA has higher ROC so higher score
        assert len(signals) == 2
        assert signals[0].symbol == "SYMA"
        assert signals[0].score >= signals[1].score
```

- [ ] **Step 5: Run full suite**

Run:
```bash
pytest tests/ -q 2>&1 | tail -5
```
Expected: `~393 passed in ...` (363 + 3 from Task 2 + 10 from strategy logic + 3 from contract/ranking + 2 from Task 4 + 4 from final verification)

> The exact count depends on how many tests pass. The important thing: 0 failures.

- [ ] **Step 6: Commit**

```bash
cd ~/code/vbacktest
git add tests/test_extras_strategies.py
git commit -m "test: logic tests for 5 extras strategies — signal fires, no-signal, fast-path contract, ranking"
```

---

## Task 4: Data loader edge case tests (genuinely new only)

**Files:**
- Modify: `tests/test_data_loader.py`

**Note:** `tests/test_data_loader.py` already covers empty dir (line 285), min history filter (line 290), excluded symbols (line 309), and OHLC violations (line 337). Only two cases are genuinely untested: corrupt parquet files in a mixed load, and symbols that become empty after a date range filter.

- [ ] **Step 1: Write the failing tests**

Append this class to `tests/test_data_loader.py` (after the existing `TestLoadUniverse` class):

```python
class TestLoadUniverseNewEdgeCases:
    """Edge cases not covered by existing TestLoadUniverse tests."""

    def test_corrupt_parquet_skipped_valid_symbols_loaded(self, tmp_path):
        """Two valid parquet files + one corrupt binary → returns only the 2 valid symbols."""
        _write_parquet(_make_df(n=250), tmp_path / "GOOD1.parquet")
        _write_parquet(_make_df(n=250), tmp_path / "GOOD2.parquet")
        # Corrupt parquet: random bytes, not valid parquet magic bytes
        (tmp_path / "CORRUPT.parquet").write_bytes(b"not a parquet file at all")

        result = load_universe(tmp_path, min_history_days=200, num_workers=1)
        assert "GOOD1" in result
        assert "GOOD2" in result
        assert "CORRUPT" not in result

    def test_symbol_excluded_after_date_range_filter(self, tmp_path):
        """Symbol with data only outside the requested date range is excluded."""
        # EARLY has data from 2015 only; RECENT has data from 2020+
        def _make_dated_df(start: str, n: int = 250) -> pd.DataFrame:
            df = _make_df(n=n)
            df["date"] = pd.date_range(start, periods=n, freq="B")
            return df

        _write_parquet(_make_dated_df("2015-01-01"), tmp_path / "EARLY.parquet")
        _write_parquet(_make_dated_df("2020-01-01"), tmp_path / "RECENT.parquet")

        result = load_universe(
            tmp_path,
            start_date="2020-01-01",
            min_history_days=200,
            num_workers=1,
        )
        # EARLY has no data from 2020 onwards → below min_history_days after filter
        assert "EARLY" not in result
        assert "RECENT" in result
```

- [ ] **Step 2: Run tests to verify state**

Run:
```bash
pytest tests/test_data_loader.py::TestLoadUniverseNewEdgeCases -v
```
Expected: 2 tests. The corrupt parquet test may fail if load_universe doesn't currently handle corrupt files gracefully — that would reveal a genuine bug to fix. The date-range test should pass if start_date filtering + min_history_days work together correctly.

- [ ] **Step 3: Run full suite**

Run:
```bash
pytest tests/ -q 2>&1 | tail -5
```
Expected: `~380 passed in ...` (cumulative from all 4 tasks)

- [ ] **Step 4: Confirm mypy still clean**

Run:
```bash
python -m mypy src/vbacktest/ --no-error-summary 2>&1 | tail -3
```
Expected: `Success: no issues found in 79 source files`

- [ ] **Step 5: Commit**

```bash
cd ~/code/vbacktest
git add tests/test_data_loader.py
git commit -m "test: data loader edge cases — corrupt parquet in mixed load, empty-after-date-filter"
```

---

## Final Verification

After all 4 tasks:

```bash
cd ~/code/vbacktest && source .venv/bin/activate

# No PytestCollectionWarning
pytest tests/ -q 2>&1 | grep "PytestCollectionWarning" || echo "No collection warnings — good"

# All tests pass
pytest tests/ -q 2>&1 | tail -5

# Zero mypy errors
python -m mypy src/vbacktest/ --no-error-summary 2>&1 | tail -3

# Push
git push origin master
```

Expected final state: **~390+ tests, 0 warnings, 0 mypy errors**.
