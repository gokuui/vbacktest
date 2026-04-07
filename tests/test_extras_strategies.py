"""Smoke tests for the extras strategies package."""
from __future__ import annotations

import vbacktest.strategies  # noqa: F401 — registers core 6
import vbacktest.strategies.extras  # noqa: F401 — registers extras


def test_extras_importable() -> None:
    """All extras strategies are importable without error."""
    from vbacktest.strategies.extras import (
        AccelBreakoutStrategy,
        DarvasBoxStrategy,
        DonchianTrendStrategy,
        MinerviniSEPAStrategy,
        RJGrowthMomentumStrategy,
        VCPBreakoutStrategy,
    )
    # Spot-check a few
    for cls in (
        AccelBreakoutStrategy,
        DarvasBoxStrategy,
        DonchianTrendStrategy,
        MinerviniSEPAStrategy,
        RJGrowthMomentumStrategy,
        VCPBreakoutStrategy,
    ):
        s = cls()
        assert hasattr(s, "indicators")
        assert hasattr(s, "on_bar")
        assert hasattr(s, "exit_rules")


def test_extras_registered_in_strategy_registry() -> None:
    """Importing extras registers all strategies."""
    from vbacktest.registry import strategy_registry

    keys = strategy_registry.keys()
    # 6 core + 47 extras = 53 total
    assert len(keys) >= 53, f"Expected ≥53 registered strategies, got {len(keys)}: {keys}"

    assert "rj_growth_momentum" in keys
    assert "vcp_breakout" in keys
    assert "darvas_box" in keys
    assert "minervini_sepa" in keys


def test_extras_indicators_return_list() -> None:
    """Every extras strategy returns a list of IndicatorSpec from indicators()."""
    from vbacktest.strategies import extras
    from vbacktest.indicators import IndicatorSpec

    modules = [
        extras.AccelBreakoutStrategy,
        extras.DonchianTrendStrategy,
        extras.MomentumMasterStrategy,
        extras.WeinsteinStage2Strategy,
    ]
    for cls in modules:
        s = cls()
        specs = s.indicators()
        assert isinstance(specs, list), f"{cls.__name__}.indicators() not a list"
        for sp in specs:
            assert isinstance(sp, IndicatorSpec), f"{cls.__name__} returned non-IndicatorSpec"


def test_extras_exit_rules_fresh_per_call() -> None:
    """exit_rules() returns a new list on each call."""
    from vbacktest.strategies.extras import RJGrowthMomentumStrategy

    s = RJGrowthMomentumStrategy()
    r1 = s.exit_rules()
    r2 = s.exit_rules()
    assert r1 is not r2


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


# ---------------------------------------------------------------------------
# Fast-path absent contract tests
# ---------------------------------------------------------------------------

class TestFastPathAbsentContract:
    """ElderImpulse and VCPBreakout are fast-path-only. When universe_arrays is None,
    no signals should be emitted (else: continue). This locks in the contract so
    a future regression in engine context wiring doesn't silently disable entries."""

    def test_elder_impulse_no_signals_without_arrays(self) -> None:
        from vbacktest.strategies.extras import ElderImpulseBreakoutStrategy
        from vbacktest.indicators import INDICATOR_REGISTRY
        n = 300
        df = _make_base_df(n)
        s = ElderImpulseBreakoutStrategy()
        for spec in s.indicators():
            if spec.name in INDICATOR_REGISTRY:
                INDICATOR_REGISTRY[spec.name](df, **spec.params)
        ctx = _ctx_slow("SYM", df, n - 1)  # universe_arrays=None
        signals = s.on_bar(ctx)
        assert signals == [], "ElderImpulse must not fire without universe_arrays"

    def test_vcp_no_signals_without_arrays(self) -> None:
        from vbacktest.strategies.extras import VCPBreakoutStrategy
        from vbacktest.indicators import INDICATOR_REGISTRY
        n = 300
        df = _make_base_df(n)
        s = VCPBreakoutStrategy()
        for spec in s.indicators():
            if spec.name in INDICATOR_REGISTRY:
                INDICATOR_REGISTRY[spec.name](df, **spec.params)
        ctx = _ctx_slow("SYM", df, n - 1)  # universe_arrays=None
        signals = s.on_bar(ctx)
        assert signals == [], "VCPBreakout must not fire without universe_arrays"


# ---------------------------------------------------------------------------
# FilteredMomentum multi-symbol ranking test
# ---------------------------------------------------------------------------

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
