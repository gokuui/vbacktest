"""Multi-indicator confluence strategy."""
from __future__ import annotations


import pandas as pd

from vbacktest.strategy import BarContext, Strategy, Signal, SignalAction
from vbacktest.indicators import IndicatorSpec
from vbacktest.exit_rules import StopLossRule, TrailingATRStopRule, TrailingMARule


class MultiIndicatorStrategy(Strategy):
    """Multi-indicator confluence strategy.

    Entry requires multiple bullish signals:
    - RSI recovery from oversold
    - MACD bullish crossover
    - Price above key moving averages
    - Stochastic bullish crossover

    Exit conditions:
    - Trailing stops (ATR and MA)
    - Fixed stop loss

    Scoring:
    - Number of bullish signals = score
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_recovery_threshold: float = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        fast_ma_period: int = 20,
        slow_ma_period: int = 50,
        stoch_k_period: int = 14,
        stoch_d_period: int = 3,
        stoch_threshold: float = 50,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        trailing_ma_period: int = 20,
        stop_loss_pct: float = 7.0,
        min_signals: int = 3,
    ):
        """Initialize multi-indicator strategy.

        Args:
            rsi_period: RSI period (default: 14)
            rsi_recovery_threshold: RSI threshold for bullish (default: 50)
            macd_fast: MACD fast period (default: 12)
            macd_slow: MACD slow period (default: 26)
            macd_signal: MACD signal period (default: 9)
            fast_ma_period: Fast MA period (default: 20)
            slow_ma_period: Slow MA period (default: 50)
            stoch_k_period: Stochastic %K period (default: 14)
            stoch_d_period: Stochastic %D period (default: 3)
            stoch_threshold: Stochastic threshold (default: 50)
            atr_period: ATR period (default: 14)
            atr_stop_multiplier: ATR multiplier (default: 2.0)
            trailing_ma_period: Trailing MA period (default: 20)
            stop_loss_pct: Stop loss percentage (default: 7.0)
            min_signals: Minimum number of bullish signals required (default: 3)
        """
        self.rsi_period = rsi_period
        self.rsi_recovery_threshold = rsi_recovery_threshold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_threshold = stoch_threshold
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.trailing_ma_period = trailing_ma_period
        self.stop_loss_pct = stop_loss_pct
        self.min_signals = min_signals

        # Column names
        self.rsi_col = f'rsi_{rsi_period}'
        self.macd_col = 'macd'
        self.macd_signal_col = 'macd_signal'
        self.fast_ma_col = f'sma_{fast_ma_period}'
        self.slow_ma_col = f'sma_{slow_ma_period}'
        self.stoch_k_col = 'stoch_k'
        self.stoch_d_col = 'stoch_d'
        self.atr_col = f'atr_{atr_period}'
        self.trailing_col = f'sma_{trailing_ma_period}'

    def indicators(self) -> list[IndicatorSpec]:
        """Return required indicators."""
        specs = [
            IndicatorSpec('rsi', {'period': self.rsi_period, 'output_col': self.rsi_col}),
            IndicatorSpec('macd', {
                'fast': self.macd_fast,
                'slow': self.macd_slow,
                'signal': self.macd_signal
            }),
            IndicatorSpec('sma', {'period': self.fast_ma_period}),
            IndicatorSpec('sma', {'period': self.slow_ma_period}),
            IndicatorSpec('stochastic', {
                'k_period': self.stoch_k_period,
                'd_period': self.stoch_d_period
            }),
            IndicatorSpec('atr', {'period': self.atr_period, 'output_col': self.atr_col}),
            IndicatorSpec('sma', {'period': self.trailing_ma_period}),
        ]
        return specs

    def on_bar(self, ctx: BarContext) -> list[Signal]:
        """Generate signals for current bar."""
        signals = []

        use_fast_path = ctx.universe_arrays is not None

        # Constant across the loop -- hoist outside
        required_cols = [
            self.rsi_col, self.macd_col, self.macd_signal_col,
            self.fast_ma_col, self.slow_ma_col,
            self.stoch_k_col, self.stoch_d_col, self.atr_col
        ]

        for symbol in ctx.universe_idx:
            # Skip if already in ctx.portfolio
            if ctx.portfolio.has_position(symbol):
                continue

            idx = ctx.universe_idx[symbol]

            # Need history for crossover checks
            if idx < 1:
                continue

            if use_fast_path and ctx.current_prices and symbol in ctx.current_prices and symbol in ctx.universe_arrays:
                # --- Fast path: dict lookups + numpy arrays ---
                bar = ctx.current_prices[symbol]
                arrays = ctx.universe_arrays[symbol]

                # Check required columns exist in arrays
                if not all(col in arrays for col in required_cols):
                    continue

                # Load current values from numpy arrays
                cur_rsi = arrays[self.rsi_col][idx]
                cur_macd = arrays[self.macd_col][idx]
                cur_macd_sig = arrays[self.macd_signal_col][idx]
                cur_fast_ma = arrays[self.fast_ma_col][idx]
                cur_slow_ma = arrays[self.slow_ma_col][idx]
                cur_stoch_k = arrays[self.stoch_k_col][idx]
                cur_stoch_d = arrays[self.stoch_d_col][idx]
                cur_atr = arrays[self.atr_col][idx]

                # Load previous values from numpy arrays
                prev_rsi = arrays[self.rsi_col][idx - 1]
                prev_macd = arrays[self.macd_col][idx - 1]
                prev_macd_sig = arrays[self.macd_signal_col][idx - 1]
                prev_fast_ma = arrays[self.fast_ma_col][idx - 1]
                prev_slow_ma = arrays[self.slow_ma_col][idx - 1]
                prev_stoch_k = arrays[self.stoch_k_col][idx - 1]
                prev_stoch_d = arrays[self.stoch_d_col][idx - 1]
                prev_atr = arrays[self.atr_col][idx - 1]

                # NaN check using x != x (works for numpy scalars)
                if (cur_rsi != cur_rsi or cur_macd != cur_macd or
                        cur_macd_sig != cur_macd_sig or cur_fast_ma != cur_fast_ma or
                        cur_slow_ma != cur_slow_ma or cur_stoch_k != cur_stoch_k or
                        cur_stoch_d != cur_stoch_d or cur_atr != cur_atr):
                    continue
                if (prev_rsi != prev_rsi or prev_macd != prev_macd or
                        prev_macd_sig != prev_macd_sig or prev_fast_ma != prev_fast_ma or
                        prev_slow_ma != prev_slow_ma or prev_stoch_k != prev_stoch_k or
                        prev_stoch_d != prev_stoch_d or prev_atr != prev_atr):
                    continue

                # Count bullish signals
                bullish_signals = 0

                # Signal 1: RSI above recovery threshold
                if cur_rsi > self.rsi_recovery_threshold:
                    bullish_signals += 1

                # Signal 2: MACD bullish crossover
                if prev_macd <= prev_macd_sig and cur_macd > cur_macd_sig:
                    bullish_signals += 1

                # Signal 3: Price above fast MA
                if bar['close'] > cur_fast_ma:
                    bullish_signals += 1

                # Signal 4: Fast MA above slow MA
                if cur_fast_ma > cur_slow_ma:
                    bullish_signals += 1

                # Signal 5: Stochastic bullish crossover or above threshold
                stoch_cross = prev_stoch_k <= prev_stoch_d and cur_stoch_k > cur_stoch_d
                if stoch_cross or cur_stoch_k > self.stoch_threshold:
                    bullish_signals += 1

                close = bar['close']
                atr_val = cur_atr

            else:
                # --- Slow path: pandas iloc ---
                if symbol not in universe:
                    continue

                df = ctx.universe[symbol]

                if not all(col in df.columns for col in required_cols):
                    continue

                current_bar = df.iloc[idx]
                prev_bar = df.iloc[idx - 1]

                # Check for NaN values
                if any(pd.isna(current_bar[col]) for col in required_cols):
                    continue
                if any(pd.isna(prev_bar[col]) for col in required_cols):
                    continue

                # Count bullish signals
                bullish_signals = 0

                # Signal 1: RSI above recovery threshold
                if current_bar[self.rsi_col] > self.rsi_recovery_threshold:
                    bullish_signals += 1

                # Signal 2: MACD bullish crossover
                macd_cross = (
                    prev_bar[self.macd_col] <= prev_bar[self.macd_signal_col] and
                    current_bar[self.macd_col] > current_bar[self.macd_signal_col]
                )
                if macd_cross:
                    bullish_signals += 1

                # Signal 3: Price above fast MA
                if current_bar['close'] > current_bar[self.fast_ma_col]:
                    bullish_signals += 1

                # Signal 4: Fast MA above slow MA
                if current_bar[self.fast_ma_col] > current_bar[self.slow_ma_col]:
                    bullish_signals += 1

                # Signal 5: Stochastic bullish crossover or above threshold
                stoch_cross = (
                    prev_bar[self.stoch_k_col] <= prev_bar[self.stoch_d_col] and
                    current_bar[self.stoch_k_col] > current_bar[self.stoch_d_col]
                )
                if stoch_cross or current_bar[self.stoch_k_col] > self.stoch_threshold:
                    bullish_signals += 1

                close = current_bar['close']
                atr_val = current_bar[self.atr_col]

            # --- Shared: entry check, stop price, signal construction ---
            if bullish_signals < self.min_signals:
                continue

            # Calculate stop price
            atr_stop = close - (self.atr_stop_multiplier * atr_val)
            pct_stop = close * (1 - self.stop_loss_pct / 100)
            stop_price = max(atr_stop, pct_stop)

            # Score = number of bullish signals
            score = float(bullish_signals)

            signals.append(Signal(
                symbol=symbol,
                action=SignalAction.BUY,
                date=ctx.date,
                stop_price=stop_price,
                score=score,
                metadata={
                    'atr': atr_val,  # Store ATR for fallback stop calculation
                    'bullish_signals': bullish_signals
                }
            ))

        return signals

    def exit_rules(self) -> list:
        """Return exit rules for positions."""
        rules = [
            StopLossRule(),
            TrailingATRStopRule(
                atr_column=self.atr_col,
                multiplier=self.atr_stop_multiplier
            ),
            TrailingMARule(ma_column=self.trailing_col),
        ]

        return rules
