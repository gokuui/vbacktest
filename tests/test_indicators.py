"""Tests for indicators module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vbacktest.indicators import (
    IndicatorSpec,
    INDICATOR_REGISTRY,
    apply_indicator,
    sma,
    ema,
    atr,
    rsi,
    bollinger_bands,
    donchian_channel,
    macd,
    stochastic,
    volume_sma,
    relative_volume,
    roc,
    rolling_high,
    rolling_low,
    adx,
    keltner_channel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with realistic price action."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = np.maximum(close, 1.0)  # prices stay positive
    noise = rng.uniform(0.5, 2.0, n)
    high = close + noise
    low = close - noise
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# IndicatorSpec
# ---------------------------------------------------------------------------

class TestIndicatorSpec:
    def test_name_required(self) -> None:
        spec = IndicatorSpec(name="sma")
        assert spec.name == "sma"

    def test_params_default_empty(self) -> None:
        spec = IndicatorSpec(name="atr")
        assert spec.params == {}

    def test_params_stored(self) -> None:
        spec = IndicatorSpec(name="sma", params={"period": 20})
        assert spec.params == {"period": 20}

    def test_params_mutable_default_isolated(self) -> None:
        """Each IndicatorSpec gets its own params dict (no shared mutable default)."""
        s1 = IndicatorSpec(name="sma")
        s2 = IndicatorSpec(name="ema")
        s1.params["period"] = 10
        assert "period" not in s2.params


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_output_column_default(self) -> None:
        df = _make_df()
        sma(df, period=20)
        assert "sma_20" in df.columns

    def test_output_column_custom(self) -> None:
        df = _make_df()
        sma(df, period=10, output_col="my_sma")
        assert "my_sma" in df.columns

    def test_nan_count(self) -> None:
        df = _make_df(50)
        sma(df, period=20)
        assert df["sma_20"].isna().sum() == 19  # first 19 bars are NaN

    def test_correctness_spot_check(self) -> None:
        """SMA of a constant series equals that constant."""
        df = pd.DataFrame({"close": [10.0] * 30})
        sma(df, period=10)
        assert (df["sma_10"].dropna() == 10.0).all()

    def test_returns_dataframe(self) -> None:
        df = _make_df()
        result = sma(df, period=5)
        assert result is df  # in-place modification


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_output_column_default(self) -> None:
        df = _make_df()
        ema(df, period=10)
        assert "ema_10" in df.columns

    def test_nan_count(self) -> None:
        """min_periods=period means first period-1 values are NaN."""
        df = _make_df(50)
        ema(df, period=20)
        assert df["ema_20"].isna().sum() == 19

    def test_ema_responds_faster_than_sma(self) -> None:
        """EMA should track price changes more quickly than SMA."""
        n = 60
        prices = [100.0] * 30 + [110.0] * 30  # step up at bar 30
        df = pd.DataFrame({
            "open": prices, "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices], "close": prices,
            "volume": [1e6] * n,
        })
        sma(df, period=10)
        ema(df, period=10)
        # At bar 35 (5 bars after step), EMA should be closer to 110 than SMA
        ema_val = df["ema_10"].iloc[35]
        sma_val = df["sma_10"].iloc[35]
        assert ema_val > sma_val


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_column_added(self) -> None:
        df = _make_df()
        atr(df, period=14)
        assert "atr" in df.columns

    def test_all_positive(self) -> None:
        df = _make_df()
        atr(df, period=14)
        values = df["atr"].dropna()
        assert (values > 0).all()

    def test_nan_prefix(self) -> None:
        df = _make_df(50)
        atr(df, period=14)
        assert df["atr"].isna().sum() == 14


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_column_added(self) -> None:
        df = _make_df()
        rsi(df, period=14)
        assert "rsi" in df.columns

    def test_bounded(self) -> None:
        df = _make_df()
        rsi(df, period=14)
        values = df["rsi"].dropna()
        assert (values >= 0).all() and (values <= 100).all()

    def test_constant_price_rsi_edge(self) -> None:
        """Constant prices (no change) should not produce NaN after warmup."""
        prices = [100.0] * 30
        df = pd.DataFrame({"close": prices})
        rsi(df, period=14)
        # After warmup, should be defined (50 for no change)
        assert not df["rsi"].iloc[-1] != df["rsi"].iloc[-1]  # not NaN


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_columns_added(self) -> None:
        df = _make_df()
        bollinger_bands(df, period=20)
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_mid" in df.columns

    def test_upper_above_lower(self) -> None:
        df = _make_df()
        bollinger_bands(df, period=20)
        valid = df.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_mid_between_bands(self) -> None:
        df = _make_df()
        bollinger_bands(df, period=20)
        valid = df.dropna(subset=["bb_upper", "bb_lower", "bb_mid"])
        assert (valid["bb_mid"] <= valid["bb_upper"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

class TestDonchianChannel:
    def test_columns_added(self) -> None:
        df = _make_df()
        donchian_channel(df, period=20)
        assert "don_upper" in df.columns
        assert "don_lower" in df.columns
        assert "don_mid" in df.columns

    def test_upper_ge_lower(self) -> None:
        df = _make_df()
        donchian_channel(df, period=20)
        valid = df.dropna(subset=["don_upper", "don_lower"])
        assert (valid["don_upper"] >= valid["don_lower"]).all()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_columns_added(self) -> None:
        df = _make_df()
        macd(df)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns

    def test_histogram_is_difference(self) -> None:
        df = _make_df()
        macd(df)
        valid = df.dropna(subset=["macd", "macd_signal", "macd_hist"])
        diff = (valid["macd"] - valid["macd_signal"]).round(8)
        hist = valid["macd_hist"].round(8)
        pd.testing.assert_series_equal(diff, hist, check_names=False)


# ---------------------------------------------------------------------------
# Stochastic
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_columns_added(self) -> None:
        df = _make_df()
        stochastic(df)
        assert "stoch_k" in df.columns
        assert "stoch_d" in df.columns

    def test_bounded(self) -> None:
        df = _make_df()
        stochastic(df)
        for col in ["stoch_k", "stoch_d"]:
            values = df[col].dropna()
            assert (values >= 0).all() and (values <= 100).all()


# ---------------------------------------------------------------------------
# Volume SMA / Relative Volume
# ---------------------------------------------------------------------------

class TestVolumeSMA:
    def test_column_added(self) -> None:
        df = _make_df()
        volume_sma(df, period=20)
        assert "volume_sma_20" in df.columns

    def test_positive(self) -> None:
        df = _make_df()
        volume_sma(df, period=10)
        assert (df["volume_sma_10"].dropna() > 0).all()


class TestRelativeVolume:
    def test_column_added(self) -> None:
        df = _make_df()
        relative_volume(df, period=20)
        assert "relative_volume" in df.columns

    def test_ratio_positive(self) -> None:
        df = _make_df()
        relative_volume(df, period=10)
        assert (df["relative_volume"].dropna() > 0).all()


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

class TestROC:
    def test_column_added(self) -> None:
        df = _make_df()
        roc(df, period=10)
        assert "roc_10" in df.columns

    def test_direction(self) -> None:
        """Increasing price should give positive ROC."""
        prices = list(range(1, 31))
        df = pd.DataFrame({"close": [float(p) for p in prices]})
        roc(df, period=5)
        # ROC at bar 10 (close=11, 5 bars ago close=6): (11-6)/6*100 = 83.3%
        val = df["roc_5"].iloc[10]
        assert val > 0


# ---------------------------------------------------------------------------
# Rolling High / Rolling Low
# ---------------------------------------------------------------------------

class TestRollingHighLow:
    def test_rolling_high_column(self) -> None:
        df = _make_df()
        rolling_high(df, period=20)
        assert "rolling_high_20" in df.columns

    def test_rolling_low_column(self) -> None:
        df = _make_df()
        rolling_low(df, period=20)
        assert "rolling_low_20" in df.columns

    def test_high_ge_low(self) -> None:
        df = _make_df()
        rolling_high(df, period=20)
        rolling_low(df, period=20)
        valid = df.dropna(subset=["rolling_high_20", "rolling_low_20"])
        assert (valid["rolling_high_20"] >= valid["rolling_low_20"]).all()

    def test_high_ge_close(self) -> None:
        df = _make_df()
        rolling_high(df, period=5)
        valid = df.dropna(subset=["rolling_high_5"])
        assert (valid["rolling_high_5"] >= valid["close"]).all()


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

class TestADX:
    def test_column_added(self) -> None:
        df = _make_df()
        adx(df, period=14)
        assert "adx_14" in df.columns

    def test_bounded_positive(self) -> None:
        df = _make_df()
        adx(df, period=14)
        values = df["adx_14"].dropna()
        assert (values >= 0).all()


# ---------------------------------------------------------------------------
# Keltner Channel
# ---------------------------------------------------------------------------

class TestKeltnerChannel:
    def test_columns_added(self) -> None:
        df = _make_df()
        keltner_channel(df)
        assert "keltner_upper" in df.columns
        assert "keltner_lower" in df.columns
        assert "keltner_mid" in df.columns

    def test_upper_above_lower(self) -> None:
        df = _make_df()
        keltner_channel(df)
        valid = df.dropna(subset=["keltner_upper", "keltner_lower"])
        assert (valid["keltner_upper"] >= valid["keltner_lower"]).all()


# ---------------------------------------------------------------------------
# INDICATOR_REGISTRY
# ---------------------------------------------------------------------------

class TestIndicatorRegistry:
    def test_all_expected_keys_present(self) -> None:
        expected = {
            "sma", "ema", "atr", "rsi", "bollinger_bands", "donchian_channel",
            "macd", "stochastic", "volume_sma", "relative_volume", "roc",
            "rolling_high", "rolling_low", "adx", "keltner_channel",
        }
        assert expected.issubset(set(INDICATOR_REGISTRY.keys()))

    def test_all_values_callable(self) -> None:
        for name, fn in INDICATOR_REGISTRY.items():
            assert callable(fn), f"INDICATOR_REGISTRY['{name}'] is not callable"


# ---------------------------------------------------------------------------
# apply_indicator
# ---------------------------------------------------------------------------

class TestApplyIndicator:
    def test_applies_sma(self) -> None:
        df = _make_df()
        spec = IndicatorSpec(name="sma", params={"period": 10})
        apply_indicator(df, spec)
        assert "sma_10" in df.columns

    def test_applies_atr(self) -> None:
        df = _make_df()
        spec = IndicatorSpec(name="atr", params={"period": 14})
        apply_indicator(df, spec)
        assert "atr" in df.columns

    def test_unknown_indicator_raises(self) -> None:
        df = _make_df()
        spec = IndicatorSpec(name="nonexistent_indicator")
        with pytest.raises(KeyError):
            apply_indicator(df, spec)

    def test_returns_dataframe(self) -> None:
        df = _make_df()
        spec = IndicatorSpec(name="rsi", params={"period": 14})
        result = apply_indicator(df, spec)
        assert result is df
