"""
TradePilot Engine Layers - Complete 18-Layer Analysis System
"""

# Core Technical Analysis Layers (1-10)
from .layer_1_momentum import Layer1Momentum
from .layer_2_volume import Layer2Volume
from .layer_3_divergence import Layer3Divergence
from .layer_4_volume_strength import Layer4VolumeStrength
from .layer_5_trend import Layer5Trend
from .layer_6_structure import Layer6Structure
from .layer_7_liquidity import Layer7Liquidity
from .layer_8_volatility_regime import Layer8VolatilityRegime
from .layer_9_confirmation import Layer9Confirmation
from .layer_10_candle_intelligence import Layer10CandleIntelligence

# Advanced Analysis Layers (11-13)
from .layer_11_support_resistance import Layer11SupportResistance
from .layer_12_vwap_analysis import Layer12VWAPAnalysis
from .layer_13_volume_profile import Layer13VolumeProfile

# Options Analysis Layers (14-17)
from .layer_14_iv_analysis import Layer14IVAnalysis
from .layer_15_gamma_max_pain import Layer15GammaMaxPain
from .layer_16_put_call_ratio import Layer16PutCallRatio
# Layer 17: Greeks Analysis (JSON config, loaded at runtime by Layer 18)

# Master Aggregator (Layer 18)
from .layer_18_master_aggregator import (
    Layer18MasterAggregator,
    MasterAggregatorResult,
    TradeMode,
    # Backward compatibility aliases
    Layer18BrainV3,
    Layer18DataAggregator
)

__all__ = [
    # Core Technical Analysis (Layers 1-10)
    "Layer1Momentum",
    "Layer2Volume",
    "Layer3Divergence",
    "Layer4VolumeStrength",
    "Layer5Trend",
    "Layer6Structure",
    "Layer7Liquidity",
    "Layer8VolatilityRegime",
    "Layer9Confirmation",
    "Layer10CandleIntelligence",
    
    # Advanced Analysis (Layers 11-13)
    "Layer11SupportResistance",
    "Layer12VWAPAnalysis",
    "Layer13VolumeProfile",
    
    # Options Analysis (Layers 14-16)
    "Layer14IVAnalysis",
    "Layer15GammaMaxPain",
    "Layer16PutCallRatio",
    # Layer 17: Greeks - JSON config, no class export
    
    # Master Aggregator (Layer 18)
    "Layer18MasterAggregator",
    "MasterAggregatorResult",
    "TradeMode",
    
    # Backward compatibility
    "Layer18BrainV3",
    "Layer18DataAggregator"
]
