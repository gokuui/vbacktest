"""Built-in trading strategies for vbacktest."""
from vbacktest.strategies.bollinger_breakout import BollingerBreakoutStrategy
from vbacktest.strategies.ma_crossover import MACrossoverStrategy
from vbacktest.strategies.momentum import MomentumStrategy
from vbacktest.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from vbacktest.strategies.turtle_trading import TurtleTradingStrategy
from vbacktest.strategies.volume_breakout import VolumeBreakoutStrategy

__all__ = [
    "BollingerBreakoutStrategy",
    "MACrossoverStrategy",
    "MomentumStrategy",
    "RSIMeanReversionStrategy",
    "TurtleTradingStrategy",
    "VolumeBreakoutStrategy",
]
