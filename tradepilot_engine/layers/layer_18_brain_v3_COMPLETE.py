"""
Layer 18: Data Aggregator for AI Decision Making
================================================================

PURPOSE:
Organizes all 17 layers of data into structured categories for AI interpretation.
NO playbooks, NO scoring, NO pass/fail - just organized facts.

The AI (Claude/ChatGPT) receives this organized data and makes the trade decision
using its reasoning capabilities.

Author: TradePilot MCP Server
Version: 4.0 - Pure Data Aggregator
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from enum import Enum

class TradeMode(Enum):
    SCALP = "SCALP"
    SWING = "SWING"

@dataclass
class DataAggregatorResult:
    """Complete organized data output"""
    ticker: str
    timeframe: str
    current_price: float
    timestamp: str
    
    price_context: Dict[str, Any]
    momentum: Dict[str, Any]
    volume: Dict[str, Any]
    structure: Dict[str, Any]
    vwap: Dict[str, Any]
    support_resistance: Dict[str, Any]
    options: Dict[str, Any]
    volatility: Dict[str, Any]
    confirmations: Dict[str, Any]
    
    data_quality: Dict[str, Any]
    raw_layers: Dict[str, Any] = field(default_factory=dict)


class Layer18DataAggregator:
    """
    Pure Data Aggregator - No Playbooks, No Scoring
    
    Organizes layer data into logical categories for AI consumption.
    AI makes the trade decision based on the organized facts.
    """
    
    def __init__(self):
        pass
    
    def analyze(self, ticker: str, layer_results: Dict[str, Any],
                current_price: float, mode = None) -> DataAggregatorResult:
        
        price_context = self._extract_price_context(layer_results, current_price)
        momentum = self._extract_momentum(layer_results)
        volume = self._extract_volume(layer_results)
        structure = self._extract_structure(layer_results)
        vwap = self._extract_vwap(layer_results)
        support_resistance = self._extract_support_resistance(layer_results)
        options = self._extract_options(layer_results)
        volatility = self._extract_volatility(layer_results)
        confirmations = self._extract_confirmations(layer_results)
        data_quality = self._calculate_data_quality(layer_results)
        
        return DataAggregatorResult(
            ticker=ticker,
            timeframe=str(mode) if mode else "day",
            current_price=current_price,
            timestamp=datetime.now().isoformat(),
            price_context=price_context,
            momentum=momentum,
            volume=volume,
            structure=structure,
            vwap=vwap,
            support_resistance=support_resistance,
            options=options,
            volatility=volatility,
            confirmations=confirmations,
            data_quality=data_quality,
            raw_layers=layer_results,
        )
    
    def _extract_price_context(self, layers: Dict, current_price: float) -> Dict[str, Any]:
        l5 = layers.get("layer_5", {})
        l6 = layers.get("layer_6", {})
        l12 = layers.get("layer_12", {})
        
        return {
            "current_price": current_price,
            "trend_ema": l6.get("trend_ema"),
            "price_vs_trend_ema": l6.get("price_vs_trend_ema"),
            "price_vs_trend_ema_pct": l6.get("price_vs_trend_ema_pct"),
            "above_trend_ema": l6.get("is_above_trend_ema"),
            "vwap": l12.get("vwap"),
            "price_vs_vwap": l12.get("price_vs_vwap"),
            "price_vs_vwap_pct": l12.get("price_vs_vwap_pct"),
            "above_vwap": l12.get("price_above_vwap"),
            "atr": l5.get("atr"),
            "atr_pct": l5.get("atr_percent"),
        }
    
    def _extract_momentum(self, layers: Dict) -> Dict[str, Any]:
        l1 = layers.get("layer_1", {})
        l3 = layers.get("layer_3", {})
        l5 = layers.get("layer_5", {})
        l6 = layers.get("layer_6", {})
        
        return {
            "rsi_14": l1.get("rsi_14"),
            "rsi_zone": self._get_rsi_zone(l1.get("rsi_14")),
            "macd_line": l1.get("macd_line"),
            "macd_signal": l1.get("macd_signal_line"),
            "macd_histogram": l1.get("macd_histogram"),
            "macd_histogram_rising": l1.get("macd_histogram_rising"),
            "stoch_k": l1.get("stoch_k"),
            "stoch_d": l1.get("stoch_d"),
            "adx": l5.get("adx") or l1.get("adx"),
            "adx_interpretation": self._interpret_adx(l5.get("adx") or l1.get("adx")),
            "bullish_divergences": l3.get("total_bullish_divergences", 0),
            "bearish_divergences": l3.get("total_bearish_divergences", 0),
            "rsi_bullish_div": l3.get("rsi_total_bullish", 0),
            "rsi_bearish_div": l3.get("rsi_total_bearish", 0),
        }
    
    def _extract_volume(self, layers: Dict) -> Dict[str, Any]:
        l2 = layers.get("layer_2", {})
        l4 = layers.get("layer_4", {})
        l5 = layers.get("layer_5", {})
        l6 = layers.get("layer_6", {})
        
        return {
            "volume_ratio": l2.get("volume_ratio"),
            "volume_trend": "rising" if l2.get("obv_slope", 0) > 0 else "falling" if l2.get("obv_slope", 0) < 0 else "flat",
            "volume_interpretation": self._interpret_volume_ratio(l2.get("volume_ratio")),
            "obv": l2.get("obv"),
            "obv_slope": l2.get("obv_slope"),
            "obv_trend": "rising" if l2.get("obv_slope", 0) > 0 else "falling" if l2.get("obv_slope", 0) < 0 else "flat",
            "cvd": l4.get("cvd"),
            "cvd_trend": "rising" if (l4.get("cvd", 0) or 0) > (l4.get("cvd_prev", 0) or 0) else "falling" if (l4.get("cvd", 0) or 0) < (l4.get("cvd_prev", 0) or 0) else "flat",
            "buying_volume_pct": l4.get("buying_volume_pct"),
            "selling_volume_pct": l4.get("selling_volume_pct"),
            "volume_pressure": "buying" if l4.get("buying_volume_pct", 50) > 55 else "selling" if l4.get("selling_volume_pct", 50) > 55 else "neutral",
            "trend_volume_confirmed": l5.get("volume_confirmed"),
        }
    
    def _extract_structure(self, layers: Dict) -> Dict[str, Any]:
        l6 = layers.get("layer_6", {})
        l7 = layers.get("layer_7", {})
        
        return {
            # Pivot trend analysis (from fixed Layer 6)
            "recent_pivot_highs": l6.get("recent_pivot_highs", []),
            "recent_pivot_lows": l6.get("recent_pivot_lows", []),
            "consecutive_higher_highs": l6.get("consecutive_higher_highs", 0),
            "consecutive_higher_lows": l6.get("consecutive_higher_lows", 0),
            "consecutive_lower_highs": l6.get("consecutive_lower_highs", 0),
            "consecutive_lower_lows": l6.get("consecutive_lower_lows", 0),
            "highs_ascending": l6.get("highs_ascending"),
            "lows_ascending": l6.get("lows_ascending"),
            "highs_descending": l6.get("highs_descending"),
            "lows_descending": l6.get("lows_descending"),
            "price_above_last_pivot_high": l6.get("price_above_last_pivot_high"),
            "price_below_last_pivot_low": l6.get("price_below_last_pivot_low"),
            "price_above_all_recent_lows": l6.get("price_above_all_recent_lows"),
            "price_below_all_recent_highs": l6.get("price_below_all_recent_highs"),
            "price_position_in_range_pct": l6.get("price_position_in_range_pct"),
            "distance_to_last_pivot_high_pct": l6.get("distance_to_last_pivot_high_pct"),
            "distance_to_last_pivot_low_pct": l6.get("distance_to_last_pivot_low_pct"),
            # CHoCH
            "choch_bull_detected": l6.get("choch_bull_detected"),
            "choch_bear_detected": l6.get("choch_bear_detected"),
            # BOS
            "bos_bull_detected": l6.get("bos_bull_detected"),
            "bos_bear_detected": l6.get("bos_bear_detected"),
            "current_bias": l6.get("current_trend"),
            # Order Blocks
            "ob_bull_detected": l6.get("ob_bull_detected"),
            "ob_bear_detected": l6.get("ob_bear_detected"),
            # FVG
            "fvg_bull_detected": l6.get("fvg_bull_detected"),
            "fvg_bear_detected": l6.get("fvg_bear_detected"),
            # Liquidity
            "bullish_grab_detected": l7.get("bullish_grab_detected"),
            "bearish_grab_detected": l7.get("bearish_grab_detected"),
            "bull_grab_type": l7.get("bull_grab_type"),
            "bear_grab_type": l7.get("bear_grab_type"),
        }
    
    def _extract_vwap(self, layers: Dict) -> Dict[str, Any]:
        l12 = layers.get("layer_12", {})
        
        return {
            "vwap": l12.get("vwap"),
            "upper_band_1": l12.get("upper_1sd"),
            "lower_band_1": l12.get("lower_1sd"),
            "price_vs_vwap_pct": l12.get("price_vs_vwap_pct"),
            "price_above_vwap": l12.get("price_above_vwap"),
            "crossed_above_vwap": l12.get("crossed_above_vwap"),
            "crossed_below_vwap": l12.get("crossed_below_vwap"),
            "vwap_slope": l12.get("vwap_slope"),
        }
    
    def _extract_support_resistance(self, layers: Dict) -> Dict[str, Any]:
        l11 = layers.get("layer_11", {})
        l13 = layers.get("layer_13", {})
        
        return {
            "nearest_support": l11.get("nearest_support"),
            "nearest_resistance": l11.get("nearest_resistance"),
            "distance_to_support_pct": l11.get("distance_to_support_pct"),
            "distance_to_resistance_pct": l11.get("distance_to_resistance_pct"),
            "support_levels": l11.get("support_levels", []),
            "resistance_levels": l11.get("resistance_levels", []),
            "poc": l13.get("poc_price"),
            "value_area_high": l13.get("vah_price"),
            "value_area_low": l13.get("val_price"),
            "in_value_area": l13.get("in_value_area"),
        }
    
    def _extract_options(self, layers: Dict) -> Dict[str, Any]:
        l14 = layers.get("layer_14", {})
        l15 = layers.get("layer_15", {})
        l16 = layers.get("layer_16", {})
        l17 = layers.get("layer_17", {})
        
        return {
            "iv_current": l14.get("hv_current"),
            "iv_rank": l14.get("iv_rank"),
            "iv_percentile": l14.get("iv_percentile"),
            "iv_interpretation": self._interpret_iv_rank(l14.get("iv_rank")),
            "max_pain": l15.get("max_pain"),
            "distance_to_max_pain_pct": l15.get("distance_to_max_pain_pct"),
            "pcr_current": l16.get("pcr_current"),
            "pcr_interpretation": self._interpret_pcr(l16.get("pcr_current")),
            "total_gex": l15.get("gex_total"),
            "best_strike": l17.get("best_strike"),
            "best_delta": l17.get("best_delta"),
            "best_dte": l17.get("best_dte"),
        }
    
    def _extract_volatility(self, layers: Dict) -> Dict[str, Any]:
        l8 = layers.get("layer_8", {})
        l5 = layers.get("layer_5", {})
        l6 = layers.get("layer_6", {})
        
        return {
            "atr": l5.get("atr"),
            "atr_pct": l8.get("atrp"),
            "volatility_regime": l8.get("percentile_bucket"),
            "volatility_percentile": l8.get("percentile_rank"),
            "volatility_expanding": l8.get("volatility_expanding"),
            "volatility_contracting": l8.get("volatility_contracting"),
            "is_above_p80": l8.get("is_above_p80"),
            "is_below_p20": l8.get("is_below_p20"),
        }
    
    def _extract_confirmations(self, layers: Dict) -> Dict[str, Any]:
        l5 = layers.get("layer_5", {})
        l6 = layers.get("layer_6", {})
        l9 = layers.get("layer_9", {})
        l10 = layers.get("layer_10", {})
        
        return {
            "supertrend_bullish": l5.get("supertrend_bullish"),
            "supertrend_bearish": l5.get("supertrend_bearish"),
            "supertrend_value": l5.get("supertrend_value"),
            "mtf_alignment_pct": l9.get("alignment_pct"),
            "mtf_bull_count": l9.get("bull_count"),
            "mtf_bear_count": l9.get("bear_count"),
            "mtf_dominant_trend": self._get_mtf_dominant(l9),
            "current_tf_st_bullish": l9.get("current_st_bullish"),
            "bullish_patterns": l10.get("total_bullish_patterns", 0),
            "bearish_patterns": l10.get("total_bearish_patterns", 0),
            "bullish_pattern_names": l10.get("bullish_patterns", []),
            "bearish_pattern_names": l10.get("bearish_patterns", []),
        }
    
    def _calculate_data_quality(self, layers: Dict) -> Dict[str, Any]:
        layers_with_data = []
        layers_missing = []
        
        for i in range(1, 18):
            layer_key = f"layer_{i}"
            layer_data = layers.get(layer_key, {})
            
            if layer_data and any(v is not None for v in layer_data.values()):
                layers_with_data.append(layer_key)
            else:
                layers_missing.append(layer_key)
        
        return {
            "total_layers": 17,
            "layers_with_data": len(layers_with_data),
            "layers_missing": len(layers_missing),
            "missing_layer_list": layers_missing,
            "data_completeness_pct": round(len(layers_with_data) / 17 * 100, 1),
        }
    
    # Helper methods
    def _get_rsi_zone(self, rsi: Optional[float]) -> str:
        if rsi is None: return "unknown"
        if rsi >= 70: return "overbought"
        elif rsi >= 60: return "bullish"
        elif rsi >= 40: return "neutral"
        elif rsi >= 30: return "bearish"
        else: return "oversold"
    
    def _interpret_adx(self, adx: Optional[float]) -> str:
        if adx is None: return "unknown"
        if adx >= 50: return "very_strong_trend"
        elif adx >= 35: return "strong_trend"
        elif adx >= 25: return "trending"
        elif adx >= 15: return "weak_trend"
        else: return "no_trend"
    
    def _interpret_volume_ratio(self, ratio: Optional[float]) -> str:
        if ratio is None: return "unknown"
        if ratio >= 2.0: return "very_high"
        elif ratio >= 1.5: return "high"
        elif ratio >= 1.0: return "normal"
        elif ratio >= 0.5: return "low"
        else: return "very_low"
    
    def _interpret_iv_rank(self, iv_rank: Optional[float]) -> str:
        if iv_rank is None: return "unknown"
        if iv_rank >= 80: return "very_high_iv_crush_risk"
        elif iv_rank >= 60: return "elevated"
        elif iv_rank >= 40: return "normal"
        elif iv_rank >= 20: return "low"
        else: return "very_low_cheap_options"
    
    def _interpret_pcr(self, pcr: Optional[float]) -> str:
        if pcr is None: return "unknown"
        if pcr >= 1.5: return "very_bearish_sentiment"
        elif pcr >= 1.0: return "bearish_sentiment"
        elif pcr >= 0.7: return "neutral"
        elif pcr >= 0.5: return "bullish_sentiment"
        else: return "very_bullish_sentiment"
    
    def _get_mtf_dominant(self, l9: Dict) -> str:
        bull = l9.get("bull_count", 0)
        bear = l9.get("bear_count", 0)
        if bull > bear: return "bullish"
        elif bear > bull: return "bearish"
        else: return "mixed"
    
    # Output methods
    def to_dict(self, result: DataAggregatorResult) -> Dict:
        return {
            "ticker": result.ticker,
            "timeframe": result.timeframe,
            "current_price": result.current_price,
            "timestamp": result.timestamp,
            "price_context": result.price_context,
            "momentum": result.momentum,
            "volume": result.volume,
            "structure": result.structure,
            "vwap": result.vwap,
            "support_resistance": result.support_resistance,
            "options": result.options,
            "volatility": result.volatility,
            "confirmations": result.confirmations,
            "data_quality": result.data_quality,
        }
    
    def to_json(self, result: DataAggregatorResult, include_raw: bool = False) -> str:
        data = self.to_dict(result)
        if include_raw:
            data["raw_layers"] = result.raw_layers
        return json.dumps(data, indent=2, default=str)
    
    def to_human_readable(self, result: DataAggregatorResult) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("TRADEPILOT LAYER 18 - DATA AGGREGATOR")
        lines.append("=" * 80)
        lines.append(f"Ticker: {result.ticker} @ ${result.current_price:.2f}")
        lines.append(f"Timeframe: {result.timeframe}")
        lines.append("")
        
        # Structure summary
        s = result.structure
        lines.append("MARKET STRUCTURE:")
        lines.append(f"  Consecutive Higher Highs: {s.get('consecutive_higher_highs', 0)}")
        lines.append(f"  Consecutive Higher Lows: {s.get('consecutive_higher_lows', 0)}")
        lines.append(f"  Price Position in Range: {s.get('price_position_in_range_pct', 'N/A')}%")
        lines.append("")
        
        # Confirmations
        c = result.confirmations
        st = "BULLISH" if c.get("supertrend_bullish") else "BEARISH" if c.get("supertrend_bearish") else "N/A"
        lines.append("CONFIRMATIONS:")
        lines.append(f"  SuperTrend: {st}")
        lines.append(f"  MTF Alignment: {c.get('mtf_alignment_pct', 'N/A')}% ({c.get('mtf_dominant_trend', 'N/A')})")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("DATA READY FOR AI ANALYSIS")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Backward compatibility alias
Layer18BrainV3 = Layer18DataAggregator
