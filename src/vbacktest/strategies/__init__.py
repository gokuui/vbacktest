"""Built-in trading strategies for vbacktest."""
from __future__ import annotations

from vbacktest.registry import strategy_registry as _sr
from vbacktest.strategies.bollinger_breakout import BollingerBreakoutStrategy
from vbacktest.strategies.ma_crossover import MACrossoverStrategy
from vbacktest.strategies.momentum import MomentumStrategy
from vbacktest.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from vbacktest.strategies.turtle_trading import TurtleTradingStrategy
from vbacktest.strategies.volume_breakout import VolumeBreakoutStrategy

_sr.register_fn("bollinger_breakout", BollingerBreakoutStrategy)
_sr.register_fn("ma_crossover", MACrossoverStrategy)
_sr.register_fn("momentum", MomentumStrategy)
_sr.register_fn("rsi_mean_reversion", RSIMeanReversionStrategy)
_sr.register_fn("turtle_trading", TurtleTradingStrategy)
_sr.register_fn("volume_breakout", VolumeBreakoutStrategy)

__all__ = [
    "BollingerBreakoutStrategy",
    "MACrossoverStrategy",
    "MomentumStrategy",
    "RSIMeanReversionStrategy",
    "TurtleTradingStrategy",
    "VolumeBreakoutStrategy",
]
