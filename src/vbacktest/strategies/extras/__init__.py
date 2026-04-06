"""Extra strategies — 47 market-agnostic trading strategies."""
from __future__ import annotations

from vbacktest.registry import strategy_registry as _sr

_reg = _sr.register_fn

from .accel_breakout import AccelBreakoutStrategy
from .accumulation_breakout import AccumulationBreakoutStrategy
from .adx_trend import ADXTrendStrategy
from .base_breakout import BaseBreakoutStrategy
from .bollinger_breakout_v2 import BollingerBreakoutV2Strategy
from .bollinger_mean_reversion import BollingerMeanReversionStrategy
from .bowtie_breakout import BowtieBreakoutStrategy
from .connors_pullback import ConnorsPullbackStrategy
from .darvas_box import DarvasBoxStrategy
from .defensive_momentum import DefensiveMomentumStrategy
from .donchian_trend import DonchianTrendStrategy
from .dual_momentum import DualMomentumStrategy
from .elder_impulse_breakout import ElderImpulseBreakoutStrategy
from .ema_stack import EMAStackStrategy
from .er_pullback import ERPullbackStrategy
from .filtered_momentum import FilteredMomentumStrategy
from .gap_up_follow import GapUpFollowStrategy
from .high_52w_momentum import High52WMomentumStrategy
from .holy_grail import HolyGrailStrategy
from .keltner_squeeze import KeltnerSqueezeStrategy
from .low_vol_trend import LowVolTrendStrategy
from .macd_momentum import MACDMomentumStrategy
from .mean_reversion_uptrend import MeanReversionUptrendStrategy
from .minervini_sepa import MinerviniSEPAStrategy
from .momentum_acceleration import MomentumAccelerationStrategy
from .momentum_master import MomentumMasterStrategy
from .momentum_pullback import MomentumPullbackStrategy
from .multi_indicator import MultiIndicatorStrategy
from .narrow_range import NarrowRangeStrategy
from .nasdaq_momentum import NasdaqMomentumStrategy
from .nr4_breakout import NR4BreakoutStrategy
from .pocket_pivot import PocketPivotStrategy
from .regime_trend_quality import RegimeTrendQualityStrategy
from .relative_strength import RelativeStrengthStrategy
from .rj_growth_momentum import RJGrowthMomentumStrategy
from .rj_growth_v2 import RJGrowthV2Strategy
from .rs_breakout import RSBreakoutStrategy
from .short_swing import ShortSwingStrategy
from .squeeze_breakout import SqueezeBreakoutStrategy
from .stochastic_bounce import StochasticBounceStrategy
from .tight_range_continuation import TightRangeContinuationStrategy
from .top_momentum import TopMomentumStrategy
from .trend_pullback import TrendPullbackStrategy
from .trend_quality_breakout import TrendQualityBreakoutStrategy
from .vcp_breakout import VCPBreakoutStrategy
from .volume_dryup_breakout import VolumeDryUpBreakoutStrategy
from .weinstein_stage2 import WeinsteinStage2Strategy

_reg("accel_breakout", AccelBreakoutStrategy)
_reg("accumulation_breakout", AccumulationBreakoutStrategy)
_reg("adx_trend", ADXTrendStrategy)
_reg("base_breakout", BaseBreakoutStrategy)
_reg("bollinger_breakout_v2", BollingerBreakoutV2Strategy)
_reg("bollinger_mean_reversion", BollingerMeanReversionStrategy)
_reg("bowtie_breakout", BowtieBreakoutStrategy)
_reg("connors_pullback", ConnorsPullbackStrategy)
_reg("darvas_box", DarvasBoxStrategy)
_reg("defensive_momentum", DefensiveMomentumStrategy)
_reg("donchian_trend", DonchianTrendStrategy)
_reg("dual_momentum", DualMomentumStrategy)
_reg("elder_impulse_breakout", ElderImpulseBreakoutStrategy)
_reg("ema_stack", EMAStackStrategy)
_reg("er_pullback", ERPullbackStrategy)
_reg("filtered_momentum", FilteredMomentumStrategy)
_reg("gap_up_follow", GapUpFollowStrategy)
_reg("high_52w_momentum", High52WMomentumStrategy)
_reg("holy_grail", HolyGrailStrategy)
_reg("keltner_squeeze", KeltnerSqueezeStrategy)
_reg("low_vol_trend", LowVolTrendStrategy)
_reg("macd_momentum", MACDMomentumStrategy)
_reg("mean_reversion_uptrend", MeanReversionUptrendStrategy)
_reg("minervini_sepa", MinerviniSEPAStrategy)
_reg("momentum_acceleration", MomentumAccelerationStrategy)
_reg("momentum_master", MomentumMasterStrategy)
_reg("momentum_pullback", MomentumPullbackStrategy)
_reg("multi_indicator", MultiIndicatorStrategy)
_reg("narrow_range", NarrowRangeStrategy)
_reg("nasdaq_momentum", NasdaqMomentumStrategy)
_reg("nr4_breakout", NR4BreakoutStrategy)
_reg("pocket_pivot", PocketPivotStrategy)
_reg("regime_trend_quality", RegimeTrendQualityStrategy)
_reg("relative_strength", RelativeStrengthStrategy)
_reg("rj_growth_momentum", RJGrowthMomentumStrategy)
_reg("rj_growth_v2", RJGrowthV2Strategy)
_reg("rs_breakout", RSBreakoutStrategy)
_reg("short_swing", ShortSwingStrategy)
_reg("squeeze_breakout", SqueezeBreakoutStrategy)
_reg("stochastic_bounce", StochasticBounceStrategy)
_reg("tight_range_continuation", TightRangeContinuationStrategy)
_reg("top_momentum", TopMomentumStrategy)
_reg("trend_pullback", TrendPullbackStrategy)
_reg("trend_quality_breakout", TrendQualityBreakoutStrategy)
_reg("vcp_breakout", VCPBreakoutStrategy)
_reg("volume_dryup_breakout", VolumeDryUpBreakoutStrategy)
_reg("weinstein_stage2", WeinsteinStage2Strategy)

__all__ = [
    "AccelBreakoutStrategy",
    "AccumulationBreakoutStrategy",
    "ADXTrendStrategy",
    "BaseBreakoutStrategy",
    "BollingerBreakoutV2Strategy",
    "BollingerMeanReversionStrategy",
    "BowtieBreakoutStrategy",
    "ConnorsPullbackStrategy",
    "DarvasBoxStrategy",
    "DefensiveMomentumStrategy",
    "DonchianTrendStrategy",
    "DualMomentumStrategy",
    "ElderImpulseBreakoutStrategy",
    "EMAStackStrategy",
    "ERPullbackStrategy",
    "FilteredMomentumStrategy",
    "GapUpFollowStrategy",
    "High52WMomentumStrategy",
    "HolyGrailStrategy",
    "KeltnerSqueezeStrategy",
    "LowVolTrendStrategy",
    "MACDMomentumStrategy",
    "MeanReversionUptrendStrategy",
    "MinerviniSEPAStrategy",
    "MomentumAccelerationStrategy",
    "MomentumMasterStrategy",
    "MomentumPullbackStrategy",
    "MultiIndicatorStrategy",
    "NarrowRangeStrategy",
    "NasdaqMomentumStrategy",
    "NR4BreakoutStrategy",
    "PocketPivotStrategy",
    "RegimeTrendQualityStrategy",
    "RelativeStrengthStrategy",
    "RJGrowthMomentumStrategy",
    "RJGrowthV2Strategy",
    "RSBreakoutStrategy",
    "ShortSwingStrategy",
    "SqueezeBreakoutStrategy",
    "StochasticBounceStrategy",
    "TightRangeContinuationStrategy",
    "TopMomentumStrategy",
    "TrendPullbackStrategy",
    "TrendQualityBreakoutStrategy",
    "VCPBreakoutStrategy",
    "VolumeDryUpBreakoutStrategy",
    "WeinsteinStage2Strategy",
]
