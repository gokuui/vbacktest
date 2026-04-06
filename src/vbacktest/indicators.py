"""Technical indicators for backtesting.

All functions take a DataFrame and add columns in-place, returning the DataFrame.
Pure functions - no side effects beyond column addition.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
import numpy as np


@dataclass
class IndicatorSpec:
    """Specification for an indicator needed by a strategy."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)


def sma(df: pd.DataFrame, period: int, column: str = 'close', output_col: str | None = None) -> pd.DataFrame:
    """Calculate Simple Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for moving average
        column: Column to calculate SMA on (default: 'close')
        output_col: Output column name (default: f'sma_{period}')

    Returns:
        DataFrame with SMA column added
    """
    if output_col is None:
        output_col = f'sma_{period}'

    df[output_col] = df[column].rolling(window=period, min_periods=period).mean()
    return df


def ema(df: pd.DataFrame, period: int, column: str = 'close', output_col: str | None = None) -> pd.DataFrame:
    """Calculate Exponential Moving Average.

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for EMA
        column: Column to calculate EMA on (default: 'close')
        output_col: Output column name (default: f'ema_{period}')

    Returns:
        DataFrame with EMA column added
    """
    if output_col is None:
        output_col = f'ema_{period}'

    df[output_col] = df[column].ewm(span=period, adjust=False, min_periods=period).mean()
    return df


def atr(df: pd.DataFrame, period: int = 14, output_col: str = 'atr') -> pd.DataFrame:
    """Calculate Average True Range.

    True Range is the maximum of:
    - High - Low
    - abs(High - Previous Close)
    - abs(Low - Previous Close)

    ATR is the moving average of True Range.

    Args:
        df: DataFrame with OHLCV data (must have 'high', 'low', 'close')
        period: Number of periods for ATR (default: 14)
        output_col: Output column name (default: 'atr')

    Returns:
        DataFrame with ATR column added
    """
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=df.index)

    # Use Wilder's smoothing (alpha=1/period, NOT span=period)
    df[output_col] = true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return df


def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close', output_col: str = 'rsi') -> pd.DataFrame:
    """Calculate Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for RSI (default: 14)
        column: Column to calculate RSI on (default: 'close')
        output_col: Output column name (default: 'rsi')

    Returns:
        DataFrame with RSI column added
    """
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Use Wilder's smoothing (alpha=1/period, NOT span=period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    # Handle edge cases: when avg_loss is zero RSI=100, when avg_gain is zero RSI=0
    rsi_val = rsi_val.where(avg_loss != 0, 100.0)
    rsi_val = rsi_val.where(avg_gain != 0, 0.0)
    df[output_col] = rsi_val

    return df


def bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0,
                    column: str = 'close') -> pd.DataFrame:
    """Calculate Bollinger Bands.

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for moving average (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
        column: Column to calculate bands on (default: 'close')

    Returns:
        DataFrame with bb_mid, bb_upper, bb_lower columns added
    """
    df['bb_mid'] = df[column].rolling(window=period, min_periods=period).mean()
    # Use population std (ddof=0) per Bollinger's canonical definition
    std = df[column].rolling(window=period, min_periods=period).std(ddof=0)

    df['bb_upper'] = df['bb_mid'] + (num_std * std)
    df['bb_lower'] = df['bb_mid'] - (num_std * std)

    return df


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Calculate Donchian Channel.

    Upper = Highest high over period
    Lower = Lowest low over period
    Mid = (Upper + Lower) / 2

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for channel (default: 20)

    Returns:
        DataFrame with don_upper, don_lower, don_mid columns added
    """
    df['don_upper'] = df['high'].rolling(window=period, min_periods=period).max()
    df['don_lower'] = df['low'].rolling(window=period, min_periods=period).min()
    df['don_mid'] = (df['don_upper'] + df['don_lower']) / 2

    return df


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
         column: str = 'close') -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence).

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line

    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line EMA period (default: 9)
        column: Column to calculate MACD on (default: 'close')

    Returns:
        DataFrame with macd, macd_signal, macd_hist columns added
    """
    ema_fast = df[column].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False, min_periods=slow).mean()

    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False, min_periods=signal).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator.

    %K = 100 * (Close - Low_n) / (High_n - Low_n)
    %D = SMA(%K, d_period)

    where Low_n and High_n are the lowest low and highest high over k_period.

    Args:
        df: DataFrame with OHLCV data
        k_period: Period for %K calculation (default: 14)
        d_period: Period for %D smoothing (default: 3)

    Returns:
        DataFrame with stoch_k, stoch_d columns added
    """
    low_min = df['low'].rolling(window=k_period, min_periods=k_period).min()
    high_max = df['high'].rolling(window=k_period, min_periods=k_period).max()

    # Guard against division by zero when high == low (flat price period)
    price_range = high_max - low_min
    df['stoch_k'] = np.where(
        price_range == 0,
        50.0,  # Neutral value when range is zero
        100 * (df['close'] - low_min) / price_range
    )
    # Preserve NaN for warmup period
    df.loc[high_max.isna(), 'stoch_k'] = np.nan
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period, min_periods=d_period).mean()

    return df


def volume_sma(df: pd.DataFrame, period: int = 20, output_col: str | None = None) -> pd.DataFrame:
    """Calculate Simple Moving Average of volume.

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for volume SMA (default: 20)
        output_col: Output column name (default: f'volume_sma_{period}')

    Returns:
        DataFrame with volume SMA column added
    """
    if output_col is None:
        output_col = f'volume_sma_{period}'

    df[output_col] = df['volume'].rolling(window=period, min_periods=period).mean()
    return df


def relative_volume(df: pd.DataFrame, period: int = 20, output_col: str = 'relative_volume') -> pd.DataFrame:
    """Calculate relative volume (current volume / average volume).

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for average volume (default: 20)
        output_col: Output column name (default: 'relative_volume')

    Returns:
        DataFrame with relative volume column added
    """
    avg_volume = df['volume'].rolling(window=period, min_periods=period).mean()
    df[output_col] = df['volume'] / avg_volume.replace(0, np.nan)

    return df


def roc(df: pd.DataFrame, period: int = 10, column: str = 'close', output_col: str | None = None) -> pd.DataFrame:
    """Calculate Rate of Change (ROC).

    ROC = ((Current - Previous) / Previous) * 100

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods to look back (default: 10)
        column: Column to calculate ROC on (default: 'close')
        output_col: Output column name (default: f'roc_{period}')

    Returns:
        DataFrame with ROC column added
    """
    if output_col is None:
        output_col = f'roc_{period}'

    shifted = df[column].shift(period)
    df[output_col] = ((df[column] - shifted) / shifted.replace(0, np.nan)) * 100

    return df


def rolling_high(df: pd.DataFrame, period: int = 50, column: str = 'high', output_col: str | None = None) -> pd.DataFrame:
    """Calculate rolling maximum (highest high over period).

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for rolling max (default: 50)
        column: Column to calculate rolling max on (default: 'high')
        output_col: Output column name (default: f'rolling_high_{period}')

    Returns:
        DataFrame with rolling high column added
    """
    if output_col is None:
        output_col = f'rolling_high_{period}'

    df[output_col] = df[column].rolling(window=period, min_periods=period).max()

    return df


def rolling_low(df: pd.DataFrame, period: int = 50, column: str = 'low', output_col: str | None = None) -> pd.DataFrame:
    """Calculate rolling minimum (lowest low over period).

    Args:
        df: DataFrame with OHLCV data
        period: Number of periods for rolling min (default: 50)
        column: Column to calculate rolling min on (default: 'low')
        output_col: Output column name (default: f'rolling_low_{period}')

    Returns:
        DataFrame with rolling low column added
    """
    if output_col is None:
        output_col = f'rolling_low_{period}'

    df[output_col] = df[column].rolling(window=period, min_periods=period).min()

    return df


def adx(df: pd.DataFrame, period: int = 14, output_col: str | None = None) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX) using Wilder's smoothing.

    ADX measures trend strength (not direction). Values:
    - < 20: Weak/no trend
    - 20-40: Developing trend
    - 40-60: Strong trend
    - > 60: Very strong trend

    Args:
        df: DataFrame with OHLCV data (must have 'high', 'low', 'close')
        period: Number of periods for ADX (default: 14)
        output_col: Output column name prefix (default: uses period suffix)

    Returns:
        DataFrame with adx_{period}, plus_di_{period}, minus_di_{period} columns added
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Step 1: Calculate +DM and -DM (Directional Movement)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm_arr = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm_arr = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm_arr, index=df.index)
    minus_dm = pd.Series(minus_dm_arr, index=df.index)

    # Step 2: Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=df.index)

    # Step 3: Wilder's smoothing (alpha=1/period) for TR, +DM, -DM
    alpha = 1.0 / period
    smoothed_tr = true_range.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # Step 4: +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

    # Step 5: DX = |+DI - -DI| / (+DI + -DI) * 100
    di_sum = plus_di + minus_di
    dx = (plus_di - minus_di).abs() / di_sum.replace(0, np.nan) * 100

    # Step 6: ADX = Wilder's smoothed DX
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    adx_col = output_col if output_col else f'adx_{period}'
    df[adx_col] = adx_val
    df[f'plus_di_{period}'] = plus_di
    df[f'minus_di_{period}'] = minus_di

    return df


def keltner_channel(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10,
                    atr_multiplier: float = 1.5, column: str = 'close') -> pd.DataFrame:
    """Calculate Keltner Channel (EMA-based with ATR bands).

    Keltner Channel:
    - Mid = EMA(close, ema_period)
    - Upper = Mid + atr_multiplier * ATR(atr_period)
    - Lower = Mid - atr_multiplier * ATR(atr_period)

    Used for squeeze detection when Bollinger Bands contract inside Keltner Channels.

    Args:
        df: DataFrame with OHLCV data
        ema_period: Period for center EMA (default: 20)
        atr_period: Period for ATR calculation (default: 10)
        atr_multiplier: ATR multiplier for band width (default: 1.5)
        column: Column for EMA calculation (default: 'close')

    Returns:
        DataFrame with keltner_mid, keltner_upper, keltner_lower columns added
    """
    # Center line: EMA
    df['keltner_mid'] = df[column].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()

    # ATR for band width (using Wilder's smoothing)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3), index=df.index)
    keltner_atr = true_range.ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean()

    df['keltner_upper'] = df['keltner_mid'] + atr_multiplier * keltner_atr
    df['keltner_lower'] = df['keltner_mid'] - atr_multiplier * keltner_atr

    return df


# Registry of all indicator functions
INDICATOR_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {
    'sma': sma,
    'ema': ema,
    'atr': atr,
    'rsi': rsi,
    'bollinger_bands': bollinger_bands,
    'donchian_channel': donchian_channel,
    'macd': macd,
    'stochastic': stochastic,
    'volume_sma': volume_sma,
    'relative_volume': relative_volume,
    'roc': roc,
    'rolling_high': rolling_high,
    'rolling_low': rolling_low,
    'adx': adx,
    'keltner_channel': keltner_channel,
}


def apply_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    """Apply a single indicator to a DataFrame using INDICATOR_REGISTRY.

    Args:
        df: OHLCV DataFrame to modify in-place.
        spec: IndicatorSpec describing the indicator and its params.

    Returns:
        The modified DataFrame.

    Raises:
        KeyError: If spec.name is not found in INDICATOR_REGISTRY.
    """
    fn = INDICATOR_REGISTRY[spec.name]
    result: pd.DataFrame = fn(df, **spec.params)
    return result
