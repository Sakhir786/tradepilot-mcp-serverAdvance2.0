"""
================================================================================
LAYER 18: PURE DATA AGGREGATOR FOR AI ANALYSIS
================================================================================

PURPOSE: Extract 100% of data from Layers 1-17, organize intelligently,
         calculate derived metrics, and output clean structured data for AI.

DESIGN PHILOSOPHY:
- ZERO playbooks, ZERO scoring, ZERO decisions
- AI receives pure organized data and makes ALL trading decisions
- Data grouped by analytical category for efficient AI processing
- Derived calculations that AI would need anyway
- Complete data quality transparency

TRADING MODES:
┌──────────────┬─────────────┬────────────────┬────────────────────────────────┐
│ Mode         │ DTE Range   │ Timeframe      │ Description                    │
├──────────────┼─────────────┼────────────────┼────────────────────────────────┤
│ SCALP        │ 0-2 DTE     │ 5-minute bars  │ Quick momentum plays           │
│ INTRADAY     │ 0-1 DTE     │ 15-minute bars │ Same day directional trades    │
│ SWING        │ 7-45 DTE    │ Daily bars     │ Multi-day trend following      │
│ LEAPS        │ 180-720 DTE │ Daily bars     │ Long-term directional/hedges   │
└──────────────┴─────────────┴────────────────┴────────────────────────────────┘

OUTPUT: Structured data for AI consumption via JSON/Dict

Author: TradePilot MCP Server
Version: 6.0 - Pure Data Architecture
================================================================================
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import numpy as np


# =============================================================================
# TRADING MODES
# =============================================================================

class TradeMode(Enum):
    """Trading mode with DTE range and timeframe configuration"""
    SCALP = "SCALP"         # 0-2 DTE, 5-min bars
    INTRADAY = "INTRADAY"   # 0-1 DTE, 15-min bars
    SWING = "SWING"         # 7-45 DTE, Daily bars
    LEAPS = "LEAPS"         # 180-720 DTE, Daily bars


MODE_CONFIG = {
    TradeMode.SCALP: {
        "dte_min": 0,
        "dte_max": 2,
        "timeframe": "5min",
        "description": "Quick momentum plays",
        "ideal_delta_range": (0.40, 0.60),
        "focus": ["momentum", "volume", "vwap", "structure"]
    },
    TradeMode.INTRADAY: {
        "dte_min": 0,
        "dte_max": 1,
        "timeframe": "15min",
        "description": "Same day directional trades",
        "ideal_delta_range": (0.45, 0.55),
        "focus": ["momentum", "structure", "vwap", "liquidity"]
    },
    TradeMode.SWING: {
        "dte_min": 7,
        "dte_max": 45,
        "timeframe": "daily",
        "description": "Multi-day trend following",
        "ideal_delta_range": (0.50, 0.70),
        "focus": ["trend", "structure", "divergences", "mtf_confirmation"]
    },
    TradeMode.LEAPS: {
        "dte_min": 180,
        "dte_max": 720,
        "timeframe": "daily",
        "description": "Long-term directional positions",
        "ideal_delta_range": (0.60, 0.80),
        "focus": ["trend", "iv_analysis", "support_resistance", "structure"]
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_float(value: Any, default: float = None) -> Optional[float]:
    """Safely convert to float, handling numpy types"""
    if value is None:
        return default
    try:
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = None) -> Optional[int]:
    """Safely convert to int, handling numpy types"""
    if value is None:
        return default
    try:
        if isinstance(value, (np.floating, np.integer)):
            return int(value)
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any, default: bool = None) -> Optional[bool]:
    """Safely convert to bool, handling numpy types"""
    if value is None:
        return default
    try:
        if isinstance(value, (np.bool_)):
            return bool(value)
        return bool(value)
    except (ValueError, TypeError):
        return default


def safe_list(value: Any) -> List:
    """Safely convert to list, handling numpy arrays"""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [safe_float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
    return []


def calculate_distance(price: float, level: float) -> Optional[float]:
    """Calculate absolute distance between price and level"""
    if price is None or level is None or level == 0:
        return None
    return round(price - level, 4)


def calculate_distance_pct(price: float, level: float) -> Optional[float]:
    """Calculate percentage distance between price and level"""
    if price is None or level is None or level == 0:
        return None
    return round((price - level) / level * 100, 4)


def classify_direction(current: float, previous: float) -> Optional[str]:
    """Classify direction based on current vs previous value"""
    if current is None or previous is None:
        return None
    if current > previous:
        return "rising"
    elif current < previous:
        return "falling"
    return "flat"


def numpy_safe_serializer(obj: Any) -> Any:
    """JSON serializer that handles numpy types"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# =============================================================================
# MAIN AGGREGATOR CLASS
# =============================================================================

class Layer18PureDataAggregator:
    """
    Pure Data Aggregator - Extracts 100% of layer data for AI analysis.
    
    NO playbooks, NO scoring, NO decisions - AI handles everything.
    
    Layer Categories:
    - Technical (1-5): Momentum, Volume, Divergence, Volume Strength, Trend
    - Structure (6-7): Market Structure (ICT), Liquidity
    - Regime (8-10): Volatility, MTF Confirmation, Candle Patterns
    - Price Action (11-13): S/R, VWAP, Volume Profile
    - Options (14-17): IV, Gamma/MaxPain, PCR, Greeks
    """
    
    VERSION = "6.0"
    
    def __init__(self):
        """Initialize aggregator with version info"""
        self.version = self.VERSION
    
    def analyze(
        self,
        ticker: str,
        layer_results: Dict[str, Any],
        current_price: float,
        mode: TradeMode = TradeMode.SWING
    ) -> Dict[str, Any]:
        """
        Aggregate all layer data into organized structure for AI.
        
        Args:
            ticker: Stock/ETF symbol
            layer_results: Dict with keys "layer_1" through "layer_17"
            current_price: Current price of the underlying
            mode: Trading mode (SCALP, INTRADAY, SWING, LEAPS)
            
        Returns:
            Dict containing 100% of organized data for AI consumption
        """
        current_price = safe_float(current_price, 0.0)
        mode_config = MODE_CONFIG.get(mode, MODE_CONFIG[TradeMode.SWING])
        
        # Build the complete data package
        result = {
            # ========== META ==========
            "meta": {
                "ticker": ticker,
                "current_price": current_price,
                "mode": mode.value if isinstance(mode, TradeMode) else str(mode),
                "mode_config": {
                    "dte_min": mode_config["dte_min"],
                    "dte_max": mode_config["dte_max"],
                    "timeframe": mode_config["timeframe"],
                    "description": mode_config["description"],
                    "ideal_delta_range": mode_config["ideal_delta_range"],
                    "focus_layers": mode_config["focus"]
                },
                "timestamp": datetime.now().isoformat(),
                "aggregator_version": self.VERSION
            },
            
            # ========== TECHNICAL LAYERS (1-5) ==========
            "momentum": self._extract_momentum(layer_results),
            "volume": self._extract_volume(layer_results),
            "divergences": self._extract_divergences(layer_results),
            "volume_strength": self._extract_volume_strength(layer_results),
            "trend": self._extract_trend(layer_results, current_price),
            
            # ========== STRUCTURE LAYERS (6-7) ==========
            "market_structure": self._extract_market_structure(layer_results, current_price),
            "liquidity": self._extract_liquidity(layer_results, current_price),
            
            # ========== REGIME LAYERS (8-10) ==========
            "volatility_regime": self._extract_volatility_regime(layer_results),
            "mtf_confirmation": self._extract_mtf_confirmation(layer_results),
            "candle_patterns": self._extract_candle_patterns(layer_results),
            
            # ========== PRICE ACTION LAYERS (11-13) ==========
            "support_resistance": self._extract_support_resistance(layer_results, current_price),
            "vwap": self._extract_vwap(layer_results, current_price),
            "volume_profile": self._extract_volume_profile(layer_results, current_price),
            
            # ========== OPTIONS LAYERS (14-17) ==========
            "iv_analysis": self._extract_iv_analysis(layer_results),
            "gamma_max_pain": self._extract_gamma_max_pain(layer_results, current_price),
            "put_call_ratio": self._extract_put_call_ratio(layer_results),
            "greeks": self._extract_greeks(layer_results, current_price),
            
            # ========== DERIVED CALCULATIONS ==========
            "derived": self._calculate_derived_metrics(layer_results, current_price, mode),
            
            # ========== DATA QUALITY ==========
            "data_quality": self._assess_data_quality(layer_results)
        }
        
        return result
    
    # =========================================================================
    # LAYER 1: MOMENTUM
    # =========================================================================
    def _extract_momentum(self, layers: Dict) -> Dict:
        """Extract Layer 1 - Momentum indicators"""
        l1 = layers.get("layer_1", {}) or {}
        
        rsi_14 = safe_float(l1.get("rsi_14"))
        rsi_prev = safe_float(l1.get("rsi_prev"))
        stoch_k = safe_float(l1.get("stoch_k"))
        stoch_k_prev = safe_float(l1.get("stoch_k_prev"))
        stoch_d = safe_float(l1.get("stoch_d"))
        macd_hist = safe_float(l1.get("macd_histogram"))
        macd_hist_prev = safe_float(l1.get("macd_histogram_prev"))
        
        return {
            # RSI
            "rsi": {
                "value_14": rsi_14,
                "value_7": safe_float(l1.get("rsi_7")),
                "previous": rsi_prev,
                "direction": classify_direction(rsi_14, rsi_prev),
                "zone": self._classify_rsi_zone(rsi_14)
            },
            # MACD
            "macd": {
                "line": safe_float(l1.get("macd_line")),
                "signal_line": safe_float(l1.get("macd_signal_line")),
                "histogram": macd_hist,
                "histogram_prev": macd_hist_prev,
                "histogram_rising": safe_bool(l1.get("macd_histogram_rising")),
                "above_signal": macd_hist > 0 if macd_hist is not None else None
            },
            # Stochastic
            "stochastic": {
                "k": stoch_k,
                "d": stoch_d,
                "k_prev": stoch_k_prev,
                "zone": self._classify_stoch_zone(stoch_k),
                "k_above_d": stoch_k > stoch_d if stoch_k is not None and stoch_d is not None else None
            },
            # CMF
            "cmf": {
                "value": safe_float(l1.get("cmf")),
                "previous": safe_float(l1.get("cmf_prev")),
                "positive": l1.get("cmf", 0) > 0 if l1.get("cmf") is not None else None
            },
            # ADX/DMI
            "adx": {
                "value": safe_float(l1.get("adx")),
                "plus_di": safe_float(l1.get("plus_di")),
                "minus_di": safe_float(l1.get("minus_di")),
                "di_diff": safe_float(l1.get("di_diff")),
                "trend_strength": self._classify_adx_strength(safe_float(l1.get("adx")))
            },
            # Ichimoku
            "ichimoku": {
                "conv_line": safe_float(l1.get("ichimoku_conv")),
                "base_line": safe_float(l1.get("ichimoku_base")),
                "lead1": safe_float(l1.get("ichimoku_lead1")),
                "lead2": safe_float(l1.get("ichimoku_lead2")),
                "price_vs_cloud_top": safe_float(l1.get("price_vs_cloud_top")),
                "price_vs_cloud_bottom": safe_float(l1.get("price_vs_cloud_bottom"))
            }
        }
    
    # =========================================================================
    # LAYER 2: VOLUME
    # =========================================================================
    def _extract_volume(self, layers: Dict) -> Dict:
        """Extract Layer 2 - Volume indicators"""
        l2 = layers.get("layer_2", {}) or {}
        
        vol_ratio = safe_float(l2.get("volume_ratio"))
        
        return {
            # Basic Volume
            "current": safe_float(l2.get("current_volume")),
            "average_20": safe_float(l2.get("avg_volume_20")),
            "ratio": vol_ratio,
            "change_5bar_pct": safe_float(l2.get("volume_change_5bar_pct")),
            "interpretation": self._classify_volume(vol_ratio),
            # OBV
            "obv": {
                "value": safe_float(l2.get("obv")),
                "ma": safe_float(l2.get("obv_ma")),
                "slope": safe_float(l2.get("obv_slope")),
                "vs_ma": safe_float(l2.get("obv_vs_ma")),
                "trend": classify_direction(
                    safe_float(l2.get("obv")),
                    safe_float(l2.get("obv_ma"))
                )
            },
            # A/D Line
            "ad_line": {
                "value": safe_float(l2.get("ad_line")),
                "ma": safe_float(l2.get("ad_ma")),
                "slope": safe_float(l2.get("ad_slope")),
                "vs_ma": safe_float(l2.get("ad_vs_ma"))
            },
            # Price/Volume Relationship
            "price_slope": safe_float(l2.get("price_slope")),
            "price_rising_volume_falling": safe_bool(l2.get("price_rising_volume_falling")),
            "price_falling_volume_rising": safe_bool(l2.get("price_falling_volume_rising"))
        }
    
    # =========================================================================
    # LAYER 3: DIVERGENCES
    # =========================================================================
    def _extract_divergences(self, layers: Dict) -> Dict:
        """Extract Layer 3 - Divergence detection"""
        l3 = layers.get("layer_3", {}) or {}
        
        total_bull = safe_int(l3.get("total_bullish_divergences"), 0)
        total_bear = safe_int(l3.get("total_bearish_divergences"), 0)
        
        return {
            # Summary
            "total_bullish": total_bull,
            "total_bearish": total_bear,
            "net_divergence": total_bull - total_bear,
            "bias": "bullish" if total_bull > total_bear else "bearish" if total_bear > total_bull else "neutral",
            # MACD Divergences
            "macd": {
                "regular_bullish": safe_int(l3.get("macd_regular_bullish_count"), 0),
                "hidden_bullish": safe_int(l3.get("macd_hidden_bullish_count"), 0),
                "regular_bearish": safe_int(l3.get("macd_regular_bearish_count"), 0),
                "hidden_bearish": safe_int(l3.get("macd_hidden_bearish_count"), 0),
                "total_bullish": safe_int(l3.get("macd_total_bullish"), 0),
                "total_bearish": safe_int(l3.get("macd_total_bearish"), 0)
            },
            # RSI Divergences
            "rsi": {
                "regular_bullish": safe_int(l3.get("rsi_regular_bullish_count"), 0),
                "hidden_bullish": safe_int(l3.get("rsi_hidden_bullish_count"), 0),
                "regular_bearish": safe_int(l3.get("rsi_regular_bearish_count"), 0),
                "hidden_bearish": safe_int(l3.get("rsi_hidden_bearish_count"), 0),
                "total_bullish": safe_int(l3.get("rsi_total_bullish"), 0),
                "total_bearish": safe_int(l3.get("rsi_total_bearish"), 0)
            },
            # Latest Values
            "latest": {
                "rsi": safe_float(l3.get("latest_rsi")),
                "macd_1h": safe_float(l3.get("latest_macd_1h")),
                "macd_4h": safe_float(l3.get("latest_macd_4h")),
                "macd_1d": safe_float(l3.get("latest_macd_1d"))
            }
        }
    
    # =========================================================================
    # LAYER 4: VOLUME STRENGTH
    # =========================================================================
    def _extract_volume_strength(self, layers: Dict) -> Dict:
        """Extract Layer 4 - CVD and volume pressure"""
        l4 = layers.get("layer_4", {}) or {}
        
        buying_pct = safe_float(l4.get("buying_volume_pct"))
        selling_pct = safe_float(l4.get("selling_volume_pct"))
        
        return {
            # CVD
            "cvd": {
                "value": safe_float(l4.get("cvd")),
                "previous": safe_float(l4.get("cvd_prev")),
                "trend": classify_direction(
                    safe_float(l4.get("cvd")),
                    safe_float(l4.get("cvd_prev"))
                )
            },
            # Volume Split
            "buying_volume_pct": buying_pct,
            "selling_volume_pct": selling_pct,
            "pressure": "buying" if buying_pct and buying_pct > 55 else "selling" if selling_pct and selling_pct > 55 else "neutral",
            # Cumulative
            "cumulative_buying": safe_float(l4.get("cumulative_buying_volume")),
            "cumulative_selling": safe_float(l4.get("cumulative_selling_volume")),
            # Strength Wave
            "strength_wave": safe_float(l4.get("volume_strength_wave")),
            "strength_wave_ema": safe_float(l4.get("ema_volume_strength_wave")),
            # Latest Bar
            "latest_buying": safe_float(l4.get("latest_buying_volume")),
            "latest_selling": safe_float(l4.get("latest_selling_volume")),
            # EOM
            "eom": {
                "value": safe_float(l4.get("eom")),
                "previous": safe_float(l4.get("eom_prev")),
                "hl2_change": safe_float(l4.get("eom_hl2_change")),
                "distance": safe_float(l4.get("eom_distance"))
            }
        }
    
    # =========================================================================
    # LAYER 5: TREND
    # =========================================================================
    def _extract_trend(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 5 - SuperTrend and trend analysis"""
        l5 = layers.get("layer_5", {}) or {}
        
        st_value = safe_float(l5.get("supertrend_value"))
        
        return {
            # SuperTrend
            "supertrend": {
                "value": st_value,
                "direction": safe_int(l5.get("supertrend_direction")),
                "bullish": safe_bool(l5.get("supertrend_bullish")),
                "bearish": safe_bool(l5.get("supertrend_bearish")),
                "price_vs_supertrend": calculate_distance(current_price, st_value),
                "price_vs_supertrend_pct": calculate_distance_pct(current_price, st_value),
                "trend_changed": safe_bool(l5.get("trend_changed")),
                "raw_buy_signal": safe_bool(l5.get("raw_buy_signal")),
                "raw_sell_signal": safe_bool(l5.get("raw_sell_signal"))
            },
            # ATR
            "atr": {
                "value": safe_float(l5.get("atr")),
                "percent": safe_float(l5.get("atr_percent")),
                "adaptive_multiplier": safe_float(l5.get("adaptive_multiplier"))
            },
            # Volatility State
            "volatility": {
                "high": safe_bool(l5.get("volatility_high")),
                "low": safe_bool(l5.get("volatility_low")),
                "normal": safe_bool(l5.get("volatility_normal"))
            },
            # Market Regime
            "regime": {
                "trending": safe_bool(l5.get("trending")),
                "weak_trend": safe_bool(l5.get("weak_trend")),
                "choppy": safe_bool(l5.get("choppy"))
            },
            # Whipsaw Detection
            "whipsaw": {
                "mode": safe_bool(l5.get("whipsaw_mode")),
                "flip_count": safe_int(l5.get("flip_count")),
                "bars_since_flip": safe_int(l5.get("bars_since_flip"))
            },
            # Volume Confirmation
            "volume_confirmed": safe_bool(l5.get("volume_confirmed")),
            "volume_ratio": safe_float(l5.get("volume_ratio")),
            # HTF Alignment
            "htf": {
                "timeframe": l5.get("htf_timeframe"),
                "bullish": safe_bool(l5.get("htf_bullish")),
                "bearish": safe_bool(l5.get("htf_bearish")),
                "aligned": safe_bool(l5.get("htf_aligned"))
            },
            # RSI Alignment
            "rsi": safe_float(l5.get("rsi")),
            "rsi_aligned": safe_bool(l5.get("rsi_aligned")),
            # Persistence
            "bars_in_trend": safe_int(l5.get("bars_in_trend")),
            # Time of Day
            "time_context": {
                "is_first_30min": safe_bool(l5.get("is_first_30min")),
                "is_lunch_hours": safe_bool(l5.get("is_lunch_hours")),
                "is_close_risk": safe_bool(l5.get("is_close_risk")),
                "is_optimal_hours": safe_bool(l5.get("is_optimal_hours"))
            }
        }
    
    # =========================================================================
    # LAYER 6: MARKET STRUCTURE (ICT)
    # =========================================================================
    def _extract_market_structure(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 6 - Market structure (CHoCH, BOS, OB, FVG)"""
        l6 = layers.get("layer_6", {}) or {}
        
        last_ph = safe_float(l6.get("last_pivot_high"))
        last_pl = safe_float(l6.get("last_pivot_low"))
        
        return {
            # Pivot Points
            "pivots": {
                "last_high": last_ph,
                "last_low": last_pl,
                "last_high_index": safe_int(l6.get("last_pivot_high_index")),
                "last_low_index": safe_int(l6.get("last_pivot_low_index")),
                "high_count": safe_int(l6.get("pivot_high_count")),
                "low_count": safe_int(l6.get("pivot_low_count")),
                "recent_highs": safe_list(l6.get("recent_pivot_highs")),
                "recent_lows": safe_list(l6.get("recent_pivot_lows"))
            },
            # Price vs Pivots
            "price_position": {
                "vs_last_high": calculate_distance(current_price, last_ph),
                "vs_last_high_pct": calculate_distance_pct(current_price, last_ph),
                "vs_last_low": calculate_distance(current_price, last_pl),
                "vs_last_low_pct": calculate_distance_pct(current_price, last_pl),
                "above_last_high": current_price > last_ph if last_ph else None,
                "below_last_low": current_price < last_pl if last_pl else None
            },
            # Swing Range
            "swing_range": {
                "value": (last_ph - last_pl) if last_ph and last_pl else None,
                "pct": ((last_ph - last_pl) / last_pl * 100) if last_ph and last_pl else None,
                "price_position_in_range": ((current_price - last_pl) / (last_ph - last_pl) * 100) if last_ph and last_pl and last_ph != last_pl else None
            },
            # Consecutive Pattern Detection
            "consecutive": {
                "higher_highs": safe_int(l6.get("consecutive_higher_highs")),
                "higher_lows": safe_int(l6.get("consecutive_higher_lows")),
                "lower_highs": safe_int(l6.get("consecutive_lower_highs")),
                "lower_lows": safe_int(l6.get("consecutive_lower_lows"))
            },
            # Pattern Facts
            "pattern_facts": {
                "highs_ascending": safe_bool(l6.get("highs_ascending")),
                "lows_ascending": safe_bool(l6.get("lows_ascending")),
                "highs_descending": safe_bool(l6.get("highs_descending")),
                "lows_descending": safe_bool(l6.get("lows_descending"))
            },
            # CHoCH (Change of Character)
            "choch": {
                "bull_detected": safe_bool(l6.get("choch_bull_detected")),
                "bear_detected": safe_bool(l6.get("choch_bear_detected")),
                "bull_quality": safe_float(l6.get("choch_bull_quality")),
                "bear_quality": safe_float(l6.get("choch_bear_quality")),
                "bull_delta": safe_float(l6.get("choch_bull_delta")),
                "bear_delta": safe_float(l6.get("choch_bear_delta")),
                "total_bull": safe_int(l6.get("total_choch_bull")),
                "total_bear": safe_int(l6.get("total_choch_bear"))
            },
            # BOS (Break of Structure)
            "bos": {
                "bull_detected": safe_bool(l6.get("bos_bull_detected")),
                "bear_detected": safe_bool(l6.get("bos_bear_detected")),
                "bull_quality": safe_float(l6.get("bos_bull_quality")),
                "bear_quality": safe_float(l6.get("bos_bear_quality")),
                "bull_delta": safe_float(l6.get("bos_bull_delta")),
                "bear_delta": safe_float(l6.get("bos_bear_delta")),
                "total_bull": safe_int(l6.get("total_bos_bull")),
                "total_bear": safe_int(l6.get("total_bos_bear"))
            },
            # Order Blocks
            "order_blocks": {
                "bull_detected": safe_bool(l6.get("ob_bull_detected")),
                "bear_detected": safe_bool(l6.get("ob_bear_detected")),
                "bull_quality": safe_float(l6.get("ob_bull_quality")),
                "bear_quality": safe_float(l6.get("ob_bear_quality")),
                "bull_zone": {
                    "top": safe_float(l6.get("ob_bull_top")),
                    "bottom": safe_float(l6.get("ob_bull_btm"))
                },
                "bear_zone": {
                    "top": safe_float(l6.get("ob_bear_top")),
                    "bottom": safe_float(l6.get("ob_bear_btm"))
                },
                "total_bull": safe_int(l6.get("total_ob_bull")),
                "total_bear": safe_int(l6.get("total_ob_bear"))
            },
            # Fair Value Gaps
            "fvg": {
                "bull_detected": safe_bool(l6.get("fvg_bull_detected")),
                "bear_detected": safe_bool(l6.get("fvg_bear_detected")),
                "bull_quality": safe_float(l6.get("fvg_bull_quality")),
                "bear_quality": safe_float(l6.get("fvg_bear_quality")),
                "bull_zone": {
                    "top": safe_float(l6.get("fvg_bull_top")),
                    "bottom": safe_float(l6.get("fvg_bull_btm"))
                },
                "bear_zone": {
                    "top": safe_float(l6.get("fvg_bear_top")),
                    "bottom": safe_float(l6.get("fvg_bear_btm"))
                },
                "total_bull": safe_int(l6.get("total_fvg_bull")),
                "total_bear": safe_int(l6.get("total_fvg_bear"))
            },
            # Structure Totals
            "totals": {
                "bullish_patterns": safe_int(l6.get("total_bullish_patterns")),
                "bearish_patterns": safe_int(l6.get("total_bearish_patterns")),
                "current_trend": safe_int(l6.get("current_trend"))  # 1=bull, -1=bear, 0=neutral
            },
            # Trend EMA Context
            "trend_ema": {
                "value": safe_float(l6.get("trend_ema")),
                "price_vs_ema": safe_float(l6.get("price_vs_trend_ema")),
                "price_vs_ema_pct": safe_float(l6.get("price_vs_trend_ema_pct")),
                "is_above": safe_bool(l6.get("is_above_trend_ema"))
            },
            # Liquidity Levels (from L6)
            "liquidity_levels": {
                "buy_level": safe_float(l6.get("liq_buy_level")),
                "sell_level": safe_float(l6.get("liq_sell_level"))
            }
        }
    
    # =========================================================================
    # LAYER 7: LIQUIDITY
    # =========================================================================
    def _extract_liquidity(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 7 - Liquidity sweeps and grabs"""
        l7 = layers.get("layer_7", {}) or {}
        
        return {
            # LuxAlgo Sweeps
            "sweeps": {
                "bull_detected": safe_bool(l7.get("bull_sweep_detected")),
                "bear_detected": safe_bool(l7.get("bear_sweep_detected")),
                "bull_type": l7.get("bull_sweep_type"),
                "bear_type": l7.get("bear_sweep_type"),
                "bull_zone": {
                    "top": safe_float(l7.get("bull_sweep_zone_top")),
                    "bottom": safe_float(l7.get("bull_sweep_zone_bottom"))
                },
                "bear_zone": {
                    "top": safe_float(l7.get("bear_sweep_zone_top")),
                    "bottom": safe_float(l7.get("bear_sweep_zone_bottom"))
                },
                "total_bull": safe_int(l7.get("total_bull_sweeps")),
                "total_bear": safe_int(l7.get("total_bear_sweeps"))
            },
            # ICT Liquidity
            "ict": {
                "buy_detected": safe_bool(l7.get("ict_buy_liq_detected")),
                "sell_detected": safe_bool(l7.get("ict_sell_liq_detected")),
                "buy_level": safe_float(l7.get("ict_buy_liq_level")),
                "sell_level": safe_float(l7.get("ict_sell_liq_level"))
            },
            # Liquidity Grabs / Stop Hunts
            "grabs": {
                "bullish_detected": safe_bool(l7.get("bullish_grab_detected")),
                "bearish_detected": safe_bool(l7.get("bearish_grab_detected")),
                "bull": {
                    "level": safe_float(l7.get("bull_grab_level")),
                    "strength": safe_float(l7.get("bull_grab_strength")),
                    "retrace": safe_float(l7.get("bull_grab_retrace")),
                    "type": l7.get("bull_grab_type"),
                    "is_round_number": safe_bool(l7.get("bull_is_round_number")),
                    "is_equal_level": safe_bool(l7.get("bull_is_equal_level"))
                },
                "bear": {
                    "level": safe_float(l7.get("bear_grab_level")),
                    "strength": safe_float(l7.get("bear_grab_strength")),
                    "retrace": safe_float(l7.get("bear_grab_retrace")),
                    "type": l7.get("bear_grab_type"),
                    "is_round_number": safe_bool(l7.get("bear_is_round_number")),
                    "is_equal_level": safe_bool(l7.get("bear_is_equal_level"))
                },
                "total_bull": safe_int(l7.get("total_bull_grabs")),
                "total_bear": safe_int(l7.get("total_bear_grabs")),
                "successful_bull": safe_int(l7.get("successful_bull_grabs")),
                "successful_bear": safe_int(l7.get("successful_bear_grabs"))
            },
            # Statistics
            "stats": {
                "total_sweeps": safe_int(l7.get("total_sweeps")),
                "total_grabs": safe_int(l7.get("total_grabs"))
            },
            # Context
            "context": {
                "volume_spike": safe_bool(l7.get("volume_spike")),
                "volume_ratio": safe_float(l7.get("volume_ratio"))
            }
        }
    
    # =========================================================================
    # LAYER 8: VOLATILITY REGIME
    # =========================================================================
    def _extract_volatility_regime(self, layers: Dict) -> Dict:
        """Extract Layer 8 - Volatility regime classification"""
        l8 = layers.get("layer_8", {}) or {}
        
        return {
            # ATR
            "atr": {
                "value": safe_float(l8.get("atr")),
                "previous": safe_float(l8.get("atr_prev")),
                "change_pct": safe_float(l8.get("atr_change_pct"))
            },
            # ATRP
            "atrp": {
                "value": safe_float(l8.get("atrp")),
                "previous": safe_float(l8.get("atrp_prev")),
                "smoothed": safe_float(l8.get("atrp_smoothed")),
                "trend_5bar": safe_float(l8.get("atrp_trend_5bar"))
            },
            # Percentile Ranking
            "percentile": {
                "rank": safe_float(l8.get("percentile_rank")),
                "bucket": l8.get("percentile_bucket"),
                "thresholds": {
                    "p20": safe_float(l8.get("p20_threshold")),
                    "p40": safe_float(l8.get("p40_threshold")),
                    "p60": safe_float(l8.get("p60_threshold")),
                    "p80": safe_float(l8.get("p80_threshold"))
                }
            },
            # State Flags
            "state": {
                "is_below_p20": safe_bool(l8.get("is_below_p20")),
                "is_below_p40": safe_bool(l8.get("is_below_p40")),
                "is_above_p60": safe_bool(l8.get("is_above_p60")),
                "is_above_p80": safe_bool(l8.get("is_above_p80")),
                "expanding": safe_bool(l8.get("volatility_expanding")),
                "contracting": safe_bool(l8.get("volatility_contracting"))
            },
            # Regime Classification
            "regime": l8.get("regime"),  # very_low/low/normal/high/extreme
            "regime_duration": safe_int(l8.get("regime_duration"))
        }
    
    # =========================================================================
    # LAYER 9: MTF CONFIRMATION
    # =========================================================================
    def _extract_mtf_confirmation(self, layers: Dict) -> Dict:
        """Extract Layer 9 - Multi-timeframe confirmation"""
        l9 = layers.get("layer_9", {}) or {}
        
        return {
            # Current Timeframe
            "current": {
                "timeframe": l9.get("current_timeframe"),
                "direction": safe_int(l9.get("current_st_direction")),
                "bullish": safe_bool(l9.get("current_st_bullish")),
                "bearish": safe_bool(l9.get("current_st_bearish")),
                "adx": safe_float(l9.get("current_adx")),
                "supertrend_line": safe_float(l9.get("current_st_line"))
            },
            # Per-Timeframe Data
            "timeframes": {
                "5min": {
                    "direction": safe_int(l9.get("tf_5min_direction")),
                    "bullish": safe_bool(l9.get("tf_5min_bullish")),
                    "adx": safe_float(l9.get("tf_5min_adx")),
                    "weight": safe_int(l9.get("tf_5min_weight"), 25)
                },
                "15min": {
                    "direction": safe_int(l9.get("tf_15min_direction")),
                    "bullish": safe_bool(l9.get("tf_15min_bullish")),
                    "adx": safe_float(l9.get("tf_15min_adx")),
                    "weight": safe_int(l9.get("tf_15min_weight"), 25)
                },
                "1h": {
                    "direction": safe_int(l9.get("tf_1h_direction")),
                    "bullish": safe_bool(l9.get("tf_1h_bullish")),
                    "adx": safe_float(l9.get("tf_1h_adx")),
                    "weight": safe_int(l9.get("tf_1h_weight"), 25)
                },
                "4h": {
                    "direction": safe_int(l9.get("tf_4h_direction")),
                    "bullish": safe_bool(l9.get("tf_4h_bullish")),
                    "adx": safe_float(l9.get("tf_4h_adx")),
                    "weight": safe_int(l9.get("tf_4h_weight"), 15)
                },
                "1d": {
                    "direction": safe_int(l9.get("tf_1d_direction")),
                    "bullish": safe_bool(l9.get("tf_1d_bullish")),
                    "adx": safe_float(l9.get("tf_1d_adx")),
                    "weight": safe_int(l9.get("tf_1d_weight"), 10)
                }
            },
            # Alignment Metrics
            "alignment": {
                "bull_count": safe_int(l9.get("bull_count")),
                "bear_count": safe_int(l9.get("bear_count")),
                "total_timeframes": safe_int(l9.get("total_timeframes")),
                "aligned_with_current": safe_int(l9.get("aligned_with_current_count")),
                "not_aligned": safe_int(l9.get("not_aligned_count")),
                "weighted_aligned": safe_float(l9.get("weighted_aligned")),
                "weighted_not_aligned": safe_float(l9.get("weighted_not_aligned")),
                "total_weight": safe_float(l9.get("total_weight")),
                "alignment_pct": safe_float(l9.get("alignment_pct"))
            },
            # HTF Context
            "htf": {
                "bullish_count": safe_int(l9.get("htf_bullish_count")),
                "bearish_count": safe_int(l9.get("htf_bearish_count")),
                "aligned": safe_bool(l9.get("htf_aligned"))
            },
            # Dominant Trend
            "dominant_trend": l9.get("mtf_dominant_trend")  # bullish/bearish/mixed
        }
    
    # =========================================================================
    # LAYER 10: CANDLE PATTERNS
    # =========================================================================
    def _extract_candle_patterns(self, layers: Dict) -> Dict:
        """Extract Layer 10 - Candlestick pattern intelligence"""
        l10 = layers.get("layer_10", {}) or {}
        
        return {
            # Pattern Lists
            "detected": safe_list(l10.get("patterns_detected")),
            "bullish_list": safe_list(l10.get("bullish_patterns")),
            "bearish_list": safe_list(l10.get("bearish_patterns")),
            # Pattern Counts
            "counts": {
                "total": safe_int(l10.get("total_patterns_detected")),
                "bullish": safe_int(l10.get("bullish_pattern_count")),
                "bearish": safe_int(l10.get("bearish_pattern_count")),
                "neutral": safe_int(l10.get("neutral_pattern_count")),
                "total_bullish": safe_int(l10.get("total_bullish_patterns")),
                "total_bearish": safe_int(l10.get("total_bearish_patterns"))
            },
            # Specific Patterns Detected
            "patterns": {
                "doji": safe_bool(l10.get("doji_detected")),
                "hammer": safe_bool(l10.get("hammer_detected")),
                "inverted_hammer": safe_bool(l10.get("inverted_hammer_detected")),
                "hanging_man": safe_bool(l10.get("hanging_man_detected")),
                "shooting_star": safe_bool(l10.get("shooting_star_detected")),
                "morning_star": safe_bool(l10.get("morning_star_detected")),
                "evening_star": safe_bool(l10.get("evening_star_detected")),
                "bullish_engulfing": safe_bool(l10.get("bullish_engulfing_detected")),
                "bearish_engulfing": safe_bool(l10.get("bearish_engulfing_detected")),
                "bullish_harami": safe_bool(l10.get("bullish_harami_detected")),
                "bearish_harami": safe_bool(l10.get("bearish_harami_detected")),
                "piercing_line": safe_bool(l10.get("piercing_line_detected")),
                "dark_cloud": safe_bool(l10.get("dark_cloud_detected")),
                "bullish_kicker": safe_bool(l10.get("bullish_kicker_detected")),
                "bearish_kicker": safe_bool(l10.get("bearish_kicker_detected"))
            },
            # Three White Soldiers (High-Quality Bullish)
            "three_white_soldiers": {
                "detected": safe_bool(l10.get("tws_detected")),
                "quality": safe_float(l10.get("tws_quality")),
                "high_quality": safe_bool(l10.get("tws_high_quality")),
                "volume_strong": safe_bool(l10.get("tws_volume_strong")),
                "after_downtrend": safe_bool(l10.get("tws_after_downtrend")),
                "near_support": safe_bool(l10.get("tws_near_support")),
                "levels": {
                    "entry": safe_float(l10.get("tws_entry")),
                    "stop": safe_float(l10.get("tws_stop")),
                    "target1": safe_float(l10.get("tws_target1")),
                    "target2": safe_float(l10.get("tws_target2"))
                }
            },
            # Inside Bar
            "inside_bar": {
                "detected": safe_bool(l10.get("ib_detected")),
                "bullish_breakout": safe_bool(l10.get("ib_bullish_breakout")),
                "bearish_breakout": safe_bool(l10.get("ib_bearish_breakout")),
                "quality": safe_float(l10.get("ib_quality")),
                "mother_high": safe_float(l10.get("ib_mother_high")),
                "mother_low": safe_float(l10.get("ib_mother_low")),
                "inside_ratio_pct": safe_float(l10.get("ib_inside_ratio_pct")),
                "bull_levels": {
                    "entry": safe_float(l10.get("ib_bull_entry")),
                    "stop": safe_float(l10.get("ib_bull_stop")),
                    "target1": safe_float(l10.get("ib_bull_target1")),
                    "target2": safe_float(l10.get("ib_bull_target2"))
                },
                "bear_levels": {
                    "entry": safe_float(l10.get("ib_bear_entry")),
                    "stop": safe_float(l10.get("ib_bear_stop")),
                    "target1": safe_float(l10.get("ib_bear_target1")),
                    "target2": safe_float(l10.get("ib_bear_target2"))
                }
            },
            # Morning/Evening Star Pro
            "morning_star_pro": {
                "detected": safe_bool(l10.get("morning_star_pro_detected")),
                "quality": safe_float(l10.get("morning_star_quality")),
                "high_quality": safe_bool(l10.get("morning_star_high_quality")),
                "levels": {
                    "entry": safe_float(l10.get("morning_star_entry")),
                    "stop": safe_float(l10.get("morning_star_stop")),
                    "target1": safe_float(l10.get("morning_star_target1")),
                    "target2": safe_float(l10.get("morning_star_target2"))
                }
            },
            "evening_star_pro": {
                "detected": safe_bool(l10.get("evening_star_pro_detected")),
                "quality": safe_float(l10.get("evening_star_quality")),
                "high_quality": safe_bool(l10.get("evening_star_high_quality")),
                "levels": {
                    "entry": safe_float(l10.get("evening_star_entry")),
                    "stop": safe_float(l10.get("evening_star_stop")),
                    "target1": safe_float(l10.get("evening_star_target1")),
                    "target2": safe_float(l10.get("evening_star_target2"))
                }
            },
            # Pattern Quality
            "quality": {
                "bullish": safe_float(l10.get("patterns_bull_quality")),
                "bearish": safe_float(l10.get("patterns_bear_quality"))
            },
            # Current Candle Context
            "current_candle": {
                "bullish": safe_bool(l10.get("current_candle_bullish")),
                "bearish": safe_bool(l10.get("current_candle_bearish")),
                "body_size": safe_float(l10.get("current_body_size")),
                "range": safe_float(l10.get("current_range")),
                "upper_wick": safe_float(l10.get("current_upper_wick")),
                "lower_wick": safe_float(l10.get("current_lower_wick")),
                "avg_body": safe_float(l10.get("avg_body")),
                "body_vs_avg_ratio": safe_float(l10.get("body_vs_avg_ratio"))
            }
        }
    
    # =========================================================================
    # LAYER 11: SUPPORT/RESISTANCE
    # =========================================================================
    def _extract_support_resistance(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 11 - Support/Resistance levels"""
        l11 = layers.get("layer_11", {}) or {}
        
        return {
            # Fractals
            "fractals": {
                "high_current": safe_float(l11.get("fractal_high_current")),
                "low_current": safe_float(l11.get("fractal_low_current")),
                "high_count": safe_int(l11.get("fractal_high_count")),
                "low_count": safe_int(l11.get("fractal_low_count")),
                "highs_recent": safe_list(l11.get("fractal_highs_recent")),
                "lows_recent": safe_list(l11.get("fractal_lows_recent"))
            },
            # S/R Channels
            "channels": {
                "count": safe_int(l11.get("sr_channel_count")),
                "strongest": {
                    "top": safe_float(l11.get("sr_strongest_top")),
                    "bottom": safe_float(l11.get("sr_strongest_bottom")),
                    "strength": safe_float(l11.get("sr_strongest_strength")),
                    "type": l11.get("sr_strongest_type")
                },
                "resistance_broken": safe_bool(l11.get("sr_resistance_broken")),
                "support_broken": safe_bool(l11.get("sr_support_broken")),
                "all_channels": l11.get("sr_channels", [])
            },
            # Daily Pivots
            "daily_pivots": {
                "pp": safe_float(l11.get("daily_pp")),
                "r1": safe_float(l11.get("daily_r1")),
                "r2": safe_float(l11.get("daily_r2")),
                "r3": safe_float(l11.get("daily_r3")),
                "s1": safe_float(l11.get("daily_s1")),
                "s2": safe_float(l11.get("daily_s2")),
                "s3": safe_float(l11.get("daily_s3"))
            },
            # Weekly Pivots
            "weekly_pivots": {
                "pp": safe_float(l11.get("weekly_pp")),
                "r1": safe_float(l11.get("weekly_r1")),
                "r2": safe_float(l11.get("weekly_r2")),
                "r3": safe_float(l11.get("weekly_r3")),
                "s1": safe_float(l11.get("weekly_s1")),
                "s2": safe_float(l11.get("weekly_s2")),
                "s3": safe_float(l11.get("weekly_s3"))
            },
            # Monthly Pivots
            "monthly_pivots": {
                "pp": safe_float(l11.get("monthly_pp")),
                "r1": safe_float(l11.get("monthly_r1")),
                "r2": safe_float(l11.get("monthly_r2")),
                "r3": safe_float(l11.get("monthly_r3")),
                "s1": safe_float(l11.get("monthly_s1")),
                "s2": safe_float(l11.get("monthly_s2")),
                "s3": safe_float(l11.get("monthly_s3"))
            },
            # MTF Levels
            "mtf_levels": {
                "pdh": safe_float(l11.get("pdh")),  # Previous Day High
                "pdl": safe_float(l11.get("pdl")),  # Previous Day Low
                "pwh": safe_float(l11.get("pwh")),  # Previous Week High
                "pwl": safe_float(l11.get("pwl")),  # Previous Week Low
                "pmh": safe_float(l11.get("pmh")),  # Previous Month High
                "pml": safe_float(l11.get("pml")),  # Previous Month Low
                "ath": safe_float(l11.get("ath")),  # All Time High
                "atl": safe_float(l11.get("atl"))   # All Time Low
            },
            # MTF Touches
            "mtf_touches": {
                "pdh": safe_bool(l11.get("pdh_touch")),
                "pdl": safe_bool(l11.get("pdl_touch")),
                "pwh": safe_bool(l11.get("pwh_touch")),
                "pwl": safe_bool(l11.get("pwl_touch")),
                "pmh": safe_bool(l11.get("pmh_touch")),
                "pml": safe_bool(l11.get("pml_touch"))
            },
            # MTF Breaks
            "mtf_breaks": {
                "pdh": safe_bool(l11.get("pdh_break")),
                "pdl": safe_bool(l11.get("pdl_break")),
                "pwh": safe_bool(l11.get("pwh_break")),
                "pwl": safe_bool(l11.get("pwl_break"))
            },
            # Confluence Zones
            "confluence": {
                "count": safe_int(l11.get("confluence_zone_count")),
                "zones": l11.get("confluence_zones", [])
            },
            # Nearest Levels
            "nearest": {
                "support": safe_float(l11.get("nearest_support")),
                "resistance": safe_float(l11.get("nearest_resistance")),
                "distance_to_support": safe_float(l11.get("distance_to_support")),
                "distance_to_resistance": safe_float(l11.get("distance_to_resistance")),
                "distance_to_support_pct": safe_float(l11.get("distance_to_support_pct")),
                "distance_to_resistance_pct": safe_float(l11.get("distance_to_resistance_pct"))
            },
            # Price vs Pivots
            "price_position": {
                "above_daily_pp": safe_bool(l11.get("price_above_daily_pp")),
                "above_weekly_pp": safe_bool(l11.get("price_above_weekly_pp")),
                "above_pdh": safe_bool(l11.get("price_above_pdh")),
                "below_pdl": safe_bool(l11.get("price_below_pdl"))
            }
        }
    
    # =========================================================================
    # LAYER 12: VWAP ANALYSIS
    # =========================================================================
    def _extract_vwap(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 12 - VWAP and standard deviation bands"""
        l12 = layers.get("layer_12", {}) or {}
        
        vwap = safe_float(l12.get("vwap"))
        
        return {
            # Core VWAP
            "vwap": vwap,
            "stdev": safe_float(l12.get("stdev")),
            # Bands
            "bands": {
                "upper_1sd": safe_float(l12.get("upper_1sd")),
                "lower_1sd": safe_float(l12.get("lower_1sd")),
                "upper_2sd": safe_float(l12.get("upper_2sd")),
                "lower_2sd": safe_float(l12.get("lower_2sd"))
            },
            # Price vs VWAP
            "price_position": {
                "vs_vwap": safe_float(l12.get("price_vs_vwap")),
                "vs_vwap_pct": safe_float(l12.get("price_vs_vwap_pct")),
                "above_vwap": safe_bool(l12.get("price_above_vwap")),
                "below_vwap": safe_bool(l12.get("price_below_vwap")),
                "stdev_distance": safe_float(l12.get("stdev_distance"))
            },
            # Band Positions
            "band_positions": {
                "at_upper_1sd": safe_bool(l12.get("at_upper_1sd")),
                "at_lower_1sd": safe_bool(l12.get("at_lower_1sd")),
                "at_upper_2sd": safe_bool(l12.get("at_upper_2sd")),
                "at_lower_2sd": safe_bool(l12.get("at_lower_2sd")),
                "between_bands": safe_bool(l12.get("between_bands"))
            },
            # Crossovers
            "crossovers": {
                "crossed_above_vwap": safe_bool(l12.get("crossed_above_vwap")),
                "crossed_below_vwap": safe_bool(l12.get("crossed_below_vwap")),
                "crossed_above_lower_1sd": safe_bool(l12.get("crossed_above_lower_1sd")),
                "crossed_below_upper_1sd": safe_bool(l12.get("crossed_below_upper_1sd")),
                "crossed_above_upper_2sd": safe_bool(l12.get("crossed_above_upper_2sd")),
                "crossed_below_lower_2sd": safe_bool(l12.get("crossed_below_lower_2sd"))
            },
            # Slope
            "slope": {
                "value": safe_float(l12.get("vwap_slope")),
                "pct": safe_float(l12.get("vwap_slope_pct")),
                "bullish": safe_bool(l12.get("slope_above_bull_threshold")),
                "bearish": safe_bool(l12.get("slope_below_bear_threshold")),
                "neutral": safe_bool(l12.get("slope_neutral"))
            },
            # Support/Resistance
            "as_level": {
                "rejection_count": safe_int(l12.get("rejection_count")),
                "is_strong_level": safe_bool(l12.get("is_strong_level"))
            },
            # Zone
            "zone": {
                "current": l12.get("current_zone"),
                "bars_in_zone": safe_int(l12.get("bars_in_zone")),
                "accepted_above": safe_bool(l12.get("accepted_above")),
                "accepted_below": safe_bool(l12.get("accepted_below"))
            },
            # Distance to Bands
            "distances": {
                "to_upper_1sd": safe_float(l12.get("distance_to_upper_1sd")),
                "to_lower_1sd": safe_float(l12.get("distance_to_lower_1sd")),
                "to_upper_2sd": safe_float(l12.get("distance_to_upper_2sd")),
                "to_lower_2sd": safe_float(l12.get("distance_to_lower_2sd"))
            },
            # Volume Context
            "volume": {
                "current": safe_float(l12.get("current_volume")),
                "avg_20": safe_float(l12.get("avg_volume_20")),
                "ratio": safe_float(l12.get("volume_ratio")),
                "is_high": safe_bool(l12.get("is_high_volume"))
            }
        }
    
    # =========================================================================
    # LAYER 13: VOLUME PROFILE
    # =========================================================================
    def _extract_volume_profile(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 13 - Volume profile analysis"""
        l13 = layers.get("layer_13", {}) or {}
        
        poc = safe_float(l13.get("poc_price"))
        vah = safe_float(l13.get("vah_price"))
        val = safe_float(l13.get("val_price"))
        
        return {
            # POC (Point of Control)
            "poc": {
                "price": poc,
                "level": safe_int(l13.get("poc_level")),
                "volume": safe_float(l13.get("poc_volume")),
                "volume_pct": safe_float(l13.get("poc_volume_pct")),
                "price_vs_poc": calculate_distance(current_price, poc),
                "price_vs_poc_pct": calculate_distance_pct(current_price, poc),
                "above_poc": current_price > poc if poc else None,
                "touching": safe_bool(l13.get("touching_poc")),
                "touch_count": safe_int(l13.get("poc_touch_count")),
                "is_strong": safe_bool(l13.get("poc_is_strong")),
                "distance_pct": safe_float(l13.get("distance_to_poc_pct"))
            },
            # Value Area
            "value_area": {
                "high": vah,
                "low": val,
                "volume_pct": safe_float(l13.get("va_volume_pct")),
                "width": safe_float(l13.get("va_width")),
                "in_value_area": safe_bool(l13.get("in_value_area")),
                "above_value_area": safe_bool(l13.get("above_value_area")),
                "below_value_area": safe_bool(l13.get("below_value_area")),
                "position": l13.get("position_location"),  # ABOVE_VA/IN_VA/BELOW_VA
                "touching_vah": safe_bool(l13.get("touching_vah")),
                "touching_val": safe_bool(l13.get("touching_val"))
            },
            # Profile Range
            "profile": {
                "high": safe_float(l13.get("profile_high")),
                "low": safe_float(l13.get("profile_low")),
                "range": safe_float(l13.get("profile_range")),
                "levels": safe_int(l13.get("profile_levels")),
                "total_volume": safe_float(l13.get("total_volume"))
            },
            # Rejection Data
            "rejections": {
                "from_above": safe_bool(l13.get("rejection_from_above")),
                "from_below": safe_bool(l13.get("rejection_from_below")),
                "poc_rejection_bull": safe_bool(l13.get("poc_rejection_bull")),
                "poc_rejection_bear": safe_bool(l13.get("poc_rejection_bear")),
                "upper_wick_pct": safe_float(l13.get("upper_wick_pct")),
                "lower_wick_pct": safe_float(l13.get("lower_wick_pct"))
            },
            # Acceptance
            "acceptance": {
                "bars_at_poc": safe_int(l13.get("bars_at_poc")),
                "bars_above_vah": safe_int(l13.get("bars_above_vah")),
                "bars_below_val": safe_int(l13.get("bars_below_val")),
                "accepted_at_poc": safe_bool(l13.get("accepted_at_poc")),
                "accepted_above_va": safe_bool(l13.get("accepted_above_va")),
                "accepted_below_va": safe_bool(l13.get("accepted_below_va"))
            },
            # Buy Pressure
            "pressure": {
                "buy_pct": safe_float(l13.get("buy_pressure_pct")),
                "strong_buying": safe_bool(l13.get("strong_buying")),
                "strong_selling": safe_bool(l13.get("strong_selling")),
                "current_level_volume": safe_float(l13.get("current_level_volume")),
                "current_level_buy_volume": safe_float(l13.get("current_level_buy_volume"))
            },
            # Crossovers
            "crossovers": {
                "crossed_above_vah": safe_bool(l13.get("crossed_above_vah")),
                "crossed_below_vah": safe_bool(l13.get("crossed_below_vah")),
                "crossed_above_val": safe_bool(l13.get("crossed_above_val")),
                "crossed_below_val": safe_bool(l13.get("crossed_below_val")),
                "crossed_above_poc": safe_bool(l13.get("crossed_above_poc")),
                "crossed_below_poc": safe_bool(l13.get("crossed_below_poc"))
            },
            # Volume Context
            "volume_context": {
                "current": safe_float(l13.get("current_volume")),
                "avg_20": safe_float(l13.get("avg_volume_20")),
                "ratio": safe_float(l13.get("volume_ratio")),
                "is_high": safe_bool(l13.get("is_high_volume")),
                "spike": safe_bool(l13.get("volume_spike"))
            }
        }
    
    # =========================================================================
    # LAYER 14: IV ANALYSIS
    # =========================================================================
    def _extract_iv_analysis(self, layers: Dict) -> Dict:
        """Extract Layer 14 - Implied volatility analysis"""
        l14 = layers.get("layer_14", {}) or {}
        
        return {
            # HV (Historical Volatility)
            "hv": {
                "current": safe_float(l14.get("hv_current")),
                "smoothed": safe_float(l14.get("hv_smoothed")),
                "high_52w": safe_float(l14.get("hv_high_52w")),
                "low_52w": safe_float(l14.get("hv_low_52w")),
                "range_52w": safe_float(l14.get("hv_range_52w")),
                "rising": safe_bool(l14.get("hv_rising")),
                "falling": safe_bool(l14.get("hv_falling")),
                "stable": safe_bool(l14.get("hv_stable")),
                "change_5d": safe_float(l14.get("hv_5d_change")),
                "change_10d": safe_float(l14.get("hv_10d_change")),
                "vs_avg": safe_float(l14.get("hv_vs_avg"))
            },
            # IV Rank & Percentile
            "iv_metrics": {
                "rank": safe_float(l14.get("iv_rank")),
                "rank_valid": safe_bool(l14.get("iv_rank_valid")),
                "percentile": safe_float(l14.get("iv_percentile")),
                "percentile_valid": safe_bool(l14.get("iv_percentile_valid")),
                "vs_hv": safe_float(l14.get("iv_vs_hv"))
            },
            # Threshold Comparisons
            "thresholds": {
                "above_extreme_high": safe_bool(l14.get("iv_above_extreme_high")),
                "above_high": safe_bool(l14.get("iv_above_high")),
                "below_low": safe_bool(l14.get("iv_below_low")),
                "below_extreme_low": safe_bool(l14.get("iv_below_extreme_low")),
                "in_normal_range": safe_bool(l14.get("iv_in_normal_range")),
                "values": {
                    "extreme_high": safe_float(l14.get("threshold_extreme_high"), 80),
                    "high": safe_float(l14.get("threshold_high"), 60),
                    "low": safe_float(l14.get("threshold_low"), 40),
                    "extreme_low": safe_float(l14.get("threshold_extreme_low"), 20)
                }
            },
            # IV State
            "state": l14.get("iv_state"),  # VERY_HIGH/HIGH/NORMAL/LOW/VERY_LOW
            # Expected Move (Current DTE)
            "expected_move": {
                "dte": safe_int(l14.get("em_dte")),
                "1sd": safe_float(l14.get("em_1sd")),
                "2sd": safe_float(l14.get("em_2sd")),
                "1sd_pct": safe_float(l14.get("em_1sd_pct")),
                "2sd_pct": safe_float(l14.get("em_2sd_pct")),
                "upper_1sd": safe_float(l14.get("em_upper_1sd")),
                "lower_1sd": safe_float(l14.get("em_lower_1sd")),
                "upper_2sd": safe_float(l14.get("em_upper_2sd")),
                "lower_2sd": safe_float(l14.get("em_lower_2sd"))
            },
            # Expected Move - Multiple DTEs
            "expected_move_by_dte": {
                "7d_1sd_pct": safe_float(l14.get("em_7d_1sd_pct")),
                "14d_1sd_pct": safe_float(l14.get("em_14d_1sd_pct")),
                "30d_1sd_pct": safe_float(l14.get("em_30d_1sd_pct")),
                "45d_1sd_pct": safe_float(l14.get("em_45d_1sd_pct")),
                "60d_1sd_pct": safe_float(l14.get("em_60d_1sd_pct"))
            },
            # Distance to Thresholds
            "distances": {
                "to_extreme_high": safe_float(l14.get("distance_to_extreme_high")),
                "to_high": safe_float(l14.get("distance_to_high")),
                "to_low": safe_float(l14.get("distance_to_low")),
                "to_extreme_low": safe_float(l14.get("distance_to_extreme_low"))
            }
        }
    
    # =========================================================================
    # LAYER 15: GAMMA & MAX PAIN
    # =========================================================================
    def _extract_gamma_max_pain(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 15 - Gamma exposure and max pain"""
        l15 = layers.get("layer_15", {}) or {}
        
        max_pain = safe_float(l15.get("max_pain"))
        
        return {
            "success": safe_bool(l15.get("success")),
            # Max Pain
            "max_pain": {
                "price": max_pain,
                "total_loss": safe_float(l15.get("max_pain_total_loss")),
                "confidence": l15.get("max_pain_confidence"),
                "price_vs_max_pain": calculate_distance(current_price, max_pain),
                "above_max_pain": current_price > max_pain if max_pain else None,
                "below_max_pain": current_price < max_pain if max_pain else None,
                "distance_pct": safe_float(l15.get("distance_to_max_pain_pct")),
                "pin_probability_pct": safe_float(l15.get("pin_probability_pct"))
            },
            # Distance Zones
            "distance_zones": {
                "within_extreme_danger": safe_bool(l15.get("within_extreme_danger")),
                "within_danger": safe_bool(l15.get("within_danger")),
                "within_caution": safe_bool(l15.get("within_caution")),
                "within_safe": safe_bool(l15.get("within_safe")),
                "beyond_safe": safe_bool(l15.get("beyond_safe")),
                "thresholds": {
                    "extreme_danger_pct": safe_float(l15.get("threshold_extreme_danger_pct"), 0.5),
                    "danger_pct": safe_float(l15.get("threshold_danger_pct"), 1.0),
                    "caution_pct": safe_float(l15.get("threshold_caution_pct"), 2.0),
                    "safe_pct": safe_float(l15.get("threshold_safe_pct"), 5.0)
                }
            },
            # Open Interest
            "open_interest": {
                "total_call": safe_int(l15.get("total_call_oi")),
                "total_put": safe_int(l15.get("total_put_oi")),
                "total": safe_int(l15.get("total_oi")),
                "put_call_ratio": safe_float(l15.get("put_call_oi_ratio")),
                "strike_min": safe_float(l15.get("strike_min")),
                "strike_max": safe_float(l15.get("strike_max"))
            },
            # GEX (Gamma Exposure)
            "gex": {
                "total": safe_float(l15.get("gex_total")),
                "call": safe_float(l15.get("gex_call")),
                "put": safe_float(l15.get("gex_put")),
                "regime": l15.get("gex_regime"),  # POSITIVE/NEGATIVE/NEUTRAL
                "gamma_wall": safe_float(l15.get("gamma_wall")),
                "is_positive": safe_bool(l15.get("gex_is_positive")),
                "is_negative": safe_bool(l15.get("gex_is_negative")),
                "above_high_threshold": safe_bool(l15.get("gex_above_high_threshold")),
                "below_neg_high_threshold": safe_bool(l15.get("gex_below_neg_high_threshold")),
                "above_medium_threshold": safe_bool(l15.get("gex_above_medium_threshold")),
                "below_neg_medium_threshold": safe_bool(l15.get("gex_below_neg_medium_threshold")),
                "thresholds": {
                    "high": safe_float(l15.get("threshold_gex_high"), 1000000),
                    "medium": safe_float(l15.get("threshold_gex_medium"), 100000)
                }
            },
            # Expiration
            "expiration": {
                "date": l15.get("expiration"),
                "days_to_expiry": safe_int(l15.get("days_to_expiry")),
                "is_expiry_day": safe_bool(l15.get("is_expiry_day")),
                "is_expiry_week": safe_bool(l15.get("is_expiry_week")),
                "is_monthly": safe_bool(l15.get("is_monthly"))
            },
            "strikes_analyzed": safe_int(l15.get("strikes_analyzed"))
        }
    
    # =========================================================================
    # LAYER 16: PUT/CALL RATIO
    # =========================================================================
    def _extract_put_call_ratio(self, layers: Dict) -> Dict:
        """Extract Layer 16 - Put/Call ratio analysis"""
        l16 = layers.get("layer_16", {}) or {}
        
        return {
            "success": safe_bool(l16.get("success")),
            # Current PCR
            "pcr": {
                "current": safe_float(l16.get("pcr_current")),
                "ma_200": safe_float(l16.get("pcr_ma_200")),
                "stdev": safe_float(l16.get("pcr_stdev")),
                "upper_band": safe_float(l16.get("pcr_upper_band")),
                "lower_band": safe_float(l16.get("pcr_lower_band")),
                "z_score": safe_float(l16.get("z_score"))
            },
            # PCR vs Bands
            "band_position": {
                "above_upper": safe_bool(l16.get("pcr_above_upper_band")),
                "below_lower": safe_bool(l16.get("pcr_below_lower_band")),
                "within_bands": safe_bool(l16.get("pcr_within_bands")),
                "distance_from_ma": safe_float(l16.get("distance_from_ma")),
                "distance_from_upper": safe_float(l16.get("distance_from_upper")),
                "distance_from_lower": safe_float(l16.get("distance_from_lower"))
            },
            # Sentiment
            "sentiment": {
                "state": l16.get("sentiment_state"),  # EXTREME_FEAR/FEAR/NEUTRAL/GREED/EXTREME_GREED
                "is_extreme": safe_bool(l16.get("is_extreme_sentiment")),
                "above_extreme_fear": safe_bool(l16.get("above_extreme_fear")),
                "above_fear": safe_bool(l16.get("above_fear")),
                "in_neutral_zone": safe_bool(l16.get("in_neutral_zone")),
                "below_greed": safe_bool(l16.get("below_greed")),
                "below_extreme_greed": safe_bool(l16.get("below_extreme_greed")),
                "thresholds": {
                    "extreme_fear": safe_float(l16.get("threshold_extreme_fear"), 1.3),
                    "fear": safe_float(l16.get("threshold_fear"), 1.1),
                    "neutral_upper": safe_float(l16.get("threshold_neutral_upper"), 1.0),
                    "neutral_lower": safe_float(l16.get("threshold_neutral_lower"), 0.9),
                    "greed": safe_float(l16.get("threshold_greed"), 0.8),
                    "extreme_greed": safe_float(l16.get("threshold_extreme_greed"), 0.7)
                }
            },
            # Put vs Call
            "ratio_bias": {
                "more_puts_than_calls": safe_bool(l16.get("more_puts_than_calls")),
                "more_calls_than_puts": safe_bool(l16.get("more_calls_than_puts")),
                "balanced": safe_bool(l16.get("balanced"))
            },
            # Volume
            "volume": {
                "total_call": safe_int(l16.get("total_call_volume")),
                "total_put": safe_int(l16.get("total_put_volume")),
                "total": safe_int(l16.get("total_volume")),
                "call_pct": safe_float(l16.get("call_volume_pct")),
                "put_pct": safe_float(l16.get("put_volume_pct"))
            },
            # History
            "history": {
                "length": safe_int(l16.get("pcr_history_length")),
                "has_sufficient": safe_bool(l16.get("has_sufficient_history"))
            },
            # Distance to Thresholds
            "distances": {
                "to_extreme_fear": safe_float(l16.get("distance_to_extreme_fear")),
                "to_fear": safe_float(l16.get("distance_to_fear")),
                "to_greed": safe_float(l16.get("distance_to_greed")),
                "to_extreme_greed": safe_float(l16.get("distance_to_extreme_greed"))
            }
        }
    
    # =========================================================================
    # LAYER 17: GREEKS ANALYSIS
    # =========================================================================
    def _extract_greeks(self, layers: Dict, current_price: float) -> Dict:
        """Extract Layer 17 - Greeks analysis and strike selection"""
        l17 = layers.get("layer_17", {}) or {}
        
        return {
            # Best Strike
            "best_strike": {
                "strike": safe_float(l17.get("best_strike")),
                "type": l17.get("best_strike_type"),  # call/put
                "score": safe_float(l17.get("best_strike_score")),
                "delta": safe_float(l17.get("best_delta")),
                "gamma": safe_float(l17.get("best_gamma")),
                "theta": safe_float(l17.get("best_theta")),
                "vega": safe_float(l17.get("best_vega")),
                "iv": safe_float(l17.get("best_iv")),
                "dte": safe_int(l17.get("best_dte")),
                "expiry": l17.get("best_expiry")
            },
            # Best Strike Scores
            "scores": {
                "delta": safe_float(l17.get("best_delta_score")),
                "gamma": safe_float(l17.get("best_gamma_score")),
                "theta": safe_float(l17.get("best_theta_score")),
                "vega_iv": safe_float(l17.get("best_vega_iv_score"))
            },
            # Classifications
            "classifications": {
                "delta_is_atm": safe_bool(l17.get("best_delta_is_atm")),
                "delta_is_itm": safe_bool(l17.get("best_delta_is_itm")),
                "delta_is_otm": safe_bool(l17.get("best_delta_is_otm")),
                "gamma_is_high": safe_bool(l17.get("best_gamma_is_high")),
                "gamma_is_low": safe_bool(l17.get("best_gamma_is_low")),
                "theta_is_low": safe_bool(l17.get("best_theta_is_low")),
                "theta_is_high": safe_bool(l17.get("best_theta_is_high")),
                "vega_is_high": safe_bool(l17.get("best_vega_is_high")),
                "vega_is_low": safe_bool(l17.get("best_vega_is_low"))
            },
            # Greeks Ranges
            "ranges": {
                "delta_min": safe_float(l17.get("delta_min")),
                "delta_max": safe_float(l17.get("delta_max")),
                "delta_avg": safe_float(l17.get("delta_avg")),
                "gamma_peak_strike": safe_float(l17.get("gamma_peak_strike")),
                "theta_avg": safe_float(l17.get("theta_avg")),
                "vega_avg": safe_float(l17.get("vega_avg"))
            },
            # Strike Analysis
            "analysis": {
                "total_analyzed": safe_int(l17.get("total_strikes_analyzed")),
                "with_high_gamma": safe_int(l17.get("strikes_with_high_gamma")),
                "with_low_theta": safe_int(l17.get("strikes_with_low_theta")),
                "atm_strikes": safe_int(l17.get("strikes_atm")),
                "contracts_analyzed": safe_int(l17.get("contracts_analyzed"))
            },
            # IV Context
            "iv_context": {
                "rank": safe_float(l17.get("iv_rank")),
                "is_low": safe_bool(l17.get("iv_rank_is_low")),
                "dte": safe_int(l17.get("dte")),
                "dte_is_short": safe_bool(l17.get("dte_is_short"))
            }
        }
    
    # =========================================================================
    # DERIVED CALCULATIONS
    # =========================================================================
    def _calculate_derived_metrics(
        self,
        layers: Dict,
        current_price: float,
        mode: TradeMode
    ) -> Dict:
        """Calculate derived metrics that AI would need"""
        l1 = layers.get("layer_1", {}) or {}
        l5 = layers.get("layer_5", {}) or {}
        l6 = layers.get("layer_6", {}) or {}
        l9 = layers.get("layer_9", {}) or {}
        l12 = layers.get("layer_12", {}) or {}
        l14 = layers.get("layer_14", {}) or {}
        l15 = layers.get("layer_15", {}) or {}
        
        # Calculate bias summary
        bull_signals = 0
        bear_signals = 0
        
        # RSI bias
        rsi = safe_float(l1.get("rsi_14"))
        if rsi:
            if rsi < 30: bull_signals += 1
            elif rsi > 70: bear_signals += 1
        
        # SuperTrend bias
        if safe_bool(l5.get("supertrend_bullish")): bull_signals += 1
        if safe_bool(l5.get("supertrend_bearish")): bear_signals += 1
        
        # Structure bias
        if safe_bool(l6.get("bos_bull_detected")): bull_signals += 1
        if safe_bool(l6.get("bos_bear_detected")): bear_signals += 1
        if safe_bool(l6.get("choch_bull_detected")): bull_signals += 1
        if safe_bool(l6.get("choch_bear_detected")): bear_signals += 1
        
        # VWAP bias
        if safe_bool(l12.get("price_above_vwap")): bull_signals += 1
        if safe_bool(l12.get("price_below_vwap")): bear_signals += 1
        
        # EMA bias
        if safe_bool(l6.get("is_above_trend_ema")): bull_signals += 1
        else: bear_signals += 1
        
        # MTF bias
        mtf_bull = safe_int(l9.get("bull_count"), 0)
        mtf_bear = safe_int(l9.get("bear_count"), 0)
        if mtf_bull > mtf_bear: bull_signals += 1
        elif mtf_bear > mtf_bull: bear_signals += 1
        
        # Calculate risk zones
        max_pain = safe_float(l15.get("max_pain"))
        max_pain_distance = None
        max_pain_zone = None
        if max_pain and current_price:
            max_pain_distance = abs((current_price - max_pain) / max_pain * 100)
            if max_pain_distance < 0.5:
                max_pain_zone = "EXTREME_DANGER"
            elif max_pain_distance < 1.0:
                max_pain_zone = "DANGER"
            elif max_pain_distance < 2.0:
                max_pain_zone = "CAUTION"
            elif max_pain_distance < 5.0:
                max_pain_zone = "SAFE"
            else:
                max_pain_zone = "BEYOND_SAFE"
        
        return {
            # Bias Summary
            "bias_summary": {
                "bullish_signals": bull_signals,
                "bearish_signals": bear_signals,
                "net_bias": bull_signals - bear_signals,
                "bias_direction": "bullish" if bull_signals > bear_signals else "bearish" if bear_signals > bull_signals else "neutral"
            },
            # Key Levels Summary
            "key_levels": {
                "pivot_high": safe_float(l6.get("last_pivot_high")),
                "pivot_low": safe_float(l6.get("last_pivot_low")),
                "vwap": safe_float(l12.get("vwap")),
                "trend_ema": safe_float(l6.get("trend_ema")),
                "supertrend": safe_float(l5.get("supertrend_value")),
                "max_pain": max_pain
            },
            # Position Context
            "position_context": {
                "above_vwap": safe_bool(l12.get("price_above_vwap")),
                "above_trend_ema": safe_bool(l6.get("is_above_trend_ema")),
                "supertrend_direction": "bullish" if safe_bool(l5.get("supertrend_bullish")) else "bearish" if safe_bool(l5.get("supertrend_bearish")) else "neutral",
                "mtf_alignment_pct": safe_float(l9.get("alignment_pct"))
            },
            # IV Context for Options
            "options_context": {
                "iv_rank": safe_float(l14.get("iv_rank")),
                "iv_state": l14.get("iv_state"),
                "max_pain_zone": max_pain_zone,
                "max_pain_distance_pct": round(max_pain_distance, 2) if max_pain_distance else None
            },
            # Mode-Specific Info
            "mode_context": {
                "mode": mode.value if isinstance(mode, TradeMode) else str(mode),
                "recommended_dte_min": MODE_CONFIG[mode]["dte_min"] if isinstance(mode, TradeMode) else 7,
                "recommended_dte_max": MODE_CONFIG[mode]["dte_max"] if isinstance(mode, TradeMode) else 45,
                "ideal_delta_range": MODE_CONFIG[mode]["ideal_delta_range"] if isinstance(mode, TradeMode) else (0.50, 0.70)
            }
        }
    
    # =========================================================================
    # DATA QUALITY ASSESSMENT
    # =========================================================================
    def _assess_data_quality(self, layers: Dict) -> Dict:
        """Assess completeness and quality of layer data"""
        total_layers = 17
        layers_with_data = 0
        layers_missing = []
        layer_status = {}
        warnings = []
        
        for i in range(1, 18):
            layer_key = f"layer_{i}"
            layer_data = layers.get(layer_key, {})
            
            if layer_data and len(layer_data) > 0:
                layers_with_data += 1
                layer_status[layer_key] = True
            else:
                layers_missing.append(layer_key)
                layer_status[layer_key] = False
                warnings.append(f"Missing data: {layer_key}")
        
        # Critical data checks
        l1 = layers.get("layer_1", {})
        if l1 and l1.get("rsi_14") is None:
            warnings.append("Critical: RSI data missing")
        
        l5 = layers.get("layer_5", {})
        if l5 and l5.get("supertrend_value") is None:
            warnings.append("Critical: SuperTrend data missing")
        
        l6 = layers.get("layer_6", {})
        if l6 and l6.get("last_pivot_high") is None:
            warnings.append("Warning: Pivot data missing")
        
        l12 = layers.get("layer_12", {})
        if l12 and l12.get("vwap") is None:
            warnings.append("Warning: VWAP data missing")
        
        # Options data check
        l14 = layers.get("layer_14", {})
        l15 = layers.get("layer_15", {})
        if not l14 or l14.get("iv_rank") is None:
            warnings.append("Warning: IV analysis data missing")
        if not l15 or l15.get("max_pain") is None:
            warnings.append("Warning: Max pain data missing")
        
        completeness_pct = (layers_with_data / total_layers) * 100
        
        return {
            "total_layers": total_layers,
            "layers_with_data": layers_with_data,
            "layers_missing": len(layers_missing),
            "missing_layer_list": layers_missing,
            "completeness_pct": round(completeness_pct, 2),
            "layer_status": layer_status,
            "warnings": warnings,
            "is_usable": completeness_pct >= 50,  # At least half the data
            "is_complete": completeness_pct >= 90,  # Almost complete
            "is_optimal": completeness_pct == 100  # All data present
        }
    
    # =========================================================================
    # CLASSIFICATION HELPERS
    # =========================================================================
    def _classify_rsi_zone(self, rsi: float) -> Optional[str]:
        """Classify RSI into zones"""
        if rsi is None:
            return None
        if rsi <= 20: return "extremely_oversold"
        if rsi <= 30: return "oversold"
        if rsi <= 40: return "bearish"
        if rsi <= 60: return "neutral"
        if rsi <= 70: return "bullish"
        if rsi <= 80: return "overbought"
        return "extremely_overbought"
    
    def _classify_stoch_zone(self, stoch_k: float) -> Optional[str]:
        """Classify Stochastic into zones"""
        if stoch_k is None:
            return None
        if stoch_k <= 20: return "oversold"
        if stoch_k >= 80: return "overbought"
        return "neutral"
    
    def _classify_adx_strength(self, adx: float) -> Optional[str]:
        """Classify ADX trend strength"""
        if adx is None:
            return None
        if adx < 20: return "no_trend"
        if adx < 25: return "weak"
        if adx < 50: return "trending"
        if adx < 75: return "strong"
        return "very_strong"
    
    def _classify_volume(self, vol_ratio: float) -> Optional[str]:
        """Classify volume level"""
        if vol_ratio is None:
            return None
        if vol_ratio < 0.5: return "very_low"
        if vol_ratio < 0.8: return "low"
        if vol_ratio < 1.2: return "normal"
        if vol_ratio < 2.0: return "high"
        return "very_high"
    
    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================
    def to_dict(self, result: Dict) -> Dict:
        """Convert result to clean dictionary (handles numpy types)"""
        return json.loads(json.dumps(result, default=numpy_safe_serializer))
    
    def to_json(self, result: Dict, indent: int = 2) -> str:
        """Convert result to JSON string"""
        return json.dumps(result, indent=indent, default=numpy_safe_serializer)
    
    def to_compact_json(self, result: Dict) -> str:
        """Convert result to compact JSON string (no indentation)"""
        return json.dumps(result, default=numpy_safe_serializer)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

Layer18BrainV3 = Layer18PureDataAggregator
Layer18DataAggregator = Layer18PureDataAggregator
Layer18MasterAggregator = Layer18PureDataAggregator


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage"""
    
    aggregator = Layer18PureDataAggregator()
    
    # Example layer results (minimal data for testing)
    example_layers = {
        "layer_1": {
            "rsi_14": 48.38,
            "rsi_7": 52.10,
            "rsi_prev": 47.20,
            "macd_line": 1.25,
            "macd_signal_line": 1.10,
            "macd_histogram": 0.15,
            "macd_histogram_prev": 0.08,
            "macd_histogram_rising": True,
            "stoch_k": 55.2,
            "stoch_d": 52.8,
            "cmf": 0.12,
            "adx": 28.5,
            "plus_di": 25.3,
            "minus_di": 18.7
        },
        "layer_2": {
            "volume_ratio": 1.15,
            "obv_slope": 44496618.5,
            "current_volume": 15000000,
            "avg_volume_20": 13043478
        },
        "layer_5": {
            "supertrend_value": 665.50,
            "supertrend_bullish": True,
            "supertrend_bearish": False,
            "atr": 8.25,
            "atr_percent": 1.22,
            "volume_confirmed": True
        },
        "layer_6": {
            "last_pivot_high": 689.70,
            "last_pivot_low": 652.84,
            "is_above_trend_ema": True,
            "trend_ema": 666.04,
            "bos_bull_detected": True,
            "choch_bull_detected": False
        },
        "layer_9": {
            "alignment_pct": 75.0,
            "bull_count": 3,
            "bear_count": 2,
            "current_st_bullish": True
        },
        "layer_12": {
            "vwap": 668.50,
            "price_above_vwap": True,
            "stdev_distance": 0.35
        },
        "layer_14": {
            "iv_rank": 35.0,
            "iv_state": "NORMAL"
        },
        "layer_15": {
            "max_pain": 670.00,
            "success": True
        }
    }
    
    # Run analysis
    result = aggregator.analyze(
        ticker="SPY",
        layer_results=example_layers,
        current_price=675.02,
        mode=TradeMode.SWING
    )
    
    # Output
    print("=" * 80)
    print("TRADEPILOT LAYER 18 - PURE DATA AGGREGATOR v6.0")
    print("=" * 80)
    print(f"\nTicker: {result['meta']['ticker']}")
    print(f"Price: ${result['meta']['current_price']}")
    print(f"Mode: {result['meta']['mode']}")
    print(f"Timeframe: {result['meta']['mode_config']['timeframe']}")
    print(f"DTE Range: {result['meta']['mode_config']['dte_min']}-{result['meta']['mode_config']['dte_max']}")
    print(f"\nData Quality: {result['data_quality']['completeness_pct']}% complete")
    print(f"Layers Available: {result['data_quality']['layers_with_data']}/17")
    print(f"\nBias: {result['derived']['bias_summary']['bias_direction'].upper()}")
    print(f"  Bullish Signals: {result['derived']['bias_summary']['bullish_signals']}")
    print(f"  Bearish Signals: {result['derived']['bias_summary']['bearish_signals']}")
    print(f"\nKey Levels:")
    print(f"  Pivot High: {result['derived']['key_levels']['pivot_high']}")
    print(f"  Pivot Low: {result['derived']['key_levels']['pivot_low']}")
    print(f"  VWAP: {result['derived']['key_levels']['vwap']}")
    print(f"  SuperTrend: {result['derived']['key_levels']['supertrend']}")
    print(f"  Max Pain: {result['derived']['key_levels']['max_pain']}")
    print("\n" + "=" * 80)
    print("AI DECISION REQUIRED - Pure data output, no recommendations")
    print("=" * 80)
