"""
Layer 16: Put/Call Ratio Analysis (Raw Data Output)
Pine Script logic preserved - calculates PCR, bands, sentiment
Outputs RAW PCR data only - no signals or recommendations
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math


class Layer16PutCallRatio:
    """
    Professional Put/Call Ratio analyzer
    
    Core Logic (Pine Script exact):
    - SMA(200) of P/C Ratio
    - STDEV(200) of P/C Ratio
    - Upper Band = MA + (1.5 × SD)
    - Lower Band = MA - (1.5 × SD)
    
    Theory:
    - High P/C Ratio (>1.3) = Extreme fear
    - Low P/C Ratio (<0.7) = Extreme greed
    """
    
    def __init__(self):
        """Initialize Layer 16 Put/Call Ratio analyzer"""
        
        # Core Settings (Pine Script)
        self.lookback = 200
        self.std_multiplier = 1.5
        
        # Sentiment Thresholds
        self.extreme_fear_level = 1.3
        self.fear_level = 1.1
        self.neutral_upper = 1.0
        self.neutral_lower = 0.9
        self.greed_level = 0.8
        self.extreme_greed_level = 0.7
        
    def analyze(self,
                options_data: Dict,
                pcr_history: Optional[List[float]] = None) -> Dict:
        """
        Complete Put/Call Ratio analysis
        
        Args:
            options_data: Polygon.io options chain data
            pcr_history: Optional pre-calculated P/C ratio history (200+ values)
            
        Returns:
            Dict with RAW PCR analysis data
        """
        
        # Calculate current P/C Ratio
        current_pcr = self._calculate_pcr_from_options(options_data)
        
        if current_pcr is None:
            return self._empty_result("Unable to calculate P/C Ratio")
        
        # Get volume data
        volume_data = self._get_volume_data(options_data)
        
        # If no history, return simple analysis
        if pcr_history is None or len(pcr_history) < self.lookback:
            return self._simple_analysis(current_pcr, volume_data, pcr_history)
        
        # Add current to history
        pcr_series = pcr_history + [current_pcr]
        
        # Calculate bands (exact Pine Script)
        bands = self._calculate_bands(pcr_series)
        
        # Return RAW DATA ONLY - no signals or recommendations
        return {
            "success": True,
            
            # Current PCR Data
            "pcr_current": round(current_pcr, 4),
            "pcr_ma_200": round(bands['ma'], 4) if bands['ma'] else None,
            "pcr_stdev": round(bands['stdev'], 4) if bands['stdev'] else None,
            "pcr_upper_band": round(bands['upper_band'], 4) if bands['upper_band'] else None,
            "pcr_lower_band": round(bands['lower_band'], 4) if bands['lower_band'] else None,
            
            # PCR vs Bands (raw facts)
            "pcr_above_upper_band": current_pcr > bands['upper_band'] if bands['upper_band'] else None,
            "pcr_below_lower_band": current_pcr < bands['lower_band'] if bands['lower_band'] else None,
            "pcr_within_bands": bands['lower_band'] <= current_pcr <= bands['upper_band'] if bands['lower_band'] and bands['upper_band'] else None,
            
            # Distance from Bands
            "distance_from_ma": round(bands['distance_from_ma'], 4) if bands.get('distance_from_ma') else None,
            "distance_from_upper": round(current_pcr - bands['upper_band'], 4) if bands['upper_band'] else None,
            "distance_from_lower": round(current_pcr - bands['lower_band'], 4) if bands['lower_band'] else None,
            "z_score": round(bands['z_score'], 2) if bands.get('z_score') else None,
            
            # Sentiment Threshold Comparisons (raw booleans)
            "above_extreme_fear": current_pcr >= self.extreme_fear_level,
            "above_fear": current_pcr >= self.fear_level,
            "in_neutral_zone": self.neutral_lower <= current_pcr <= self.neutral_upper,
            "below_greed": current_pcr <= self.greed_level,
            "below_extreme_greed": current_pcr <= self.extreme_greed_level,
            
            # Sentiment State (raw label, no recommendation)
            "sentiment_state": self._get_sentiment_label(current_pcr),
            "is_extreme_sentiment": current_pcr >= self.extreme_fear_level or current_pcr <= self.extreme_greed_level,
            
            # Put vs Call Comparison
            "more_puts_than_calls": current_pcr > 1.0,
            "more_calls_than_puts": current_pcr < 1.0,
            "balanced": 0.95 <= current_pcr <= 1.05,
            
            # Volume Data
            "total_call_volume": volume_data['call_volume'],
            "total_put_volume": volume_data['put_volume'],
            "total_volume": volume_data['total_volume'],
            "call_volume_pct": round(volume_data['call_volume'] / volume_data['total_volume'] * 100, 2) if volume_data['total_volume'] > 0 else 0,
            "put_volume_pct": round(volume_data['put_volume'] / volume_data['total_volume'] * 100, 2) if volume_data['total_volume'] > 0 else 0,
            
            # Historical Context
            "pcr_history_length": len(pcr_history) if pcr_history else 0,
            "has_sufficient_history": len(pcr_history) >= self.lookback if pcr_history else False,
            
            # Thresholds (for reference)
            "threshold_extreme_fear": self.extreme_fear_level,
            "threshold_fear": self.fear_level,
            "threshold_neutral_upper": self.neutral_upper,
            "threshold_neutral_lower": self.neutral_lower,
            "threshold_greed": self.greed_level,
            "threshold_extreme_greed": self.extreme_greed_level,
            "band_std_multiplier": self.std_multiplier,
            "band_lookback": self.lookback,
            
            # Distance to Thresholds
            "distance_to_extreme_fear": round(self.extreme_fear_level - current_pcr, 4),
            "distance_to_fear": round(self.fear_level - current_pcr, 4),
            "distance_to_greed": round(current_pcr - self.greed_level, 4),
            "distance_to_extreme_greed": round(current_pcr - self.extreme_greed_level, 4),
            
            # Timestamp
            "timestamp": datetime.now().isoformat()
        }
    
    # ==================== P/C RATIO CALCULATION ====================
    
    def _calculate_pcr_from_options(self, options_data: Dict) -> Optional[float]:
        """Calculate Put/Call Ratio from Polygon.io options data"""
        if not options_data or "results" not in options_data:
            return None
        
        results = options_data["results"]
        if not results or len(results) == 0:
            return None
        
        total_call_volume = 0
        total_put_volume = 0
        
        for contract in results:
            contract_type = self._extract_contract_type(contract)
            if not contract_type:
                continue
            
            volume = self._extract_volume(contract)
            if volume is None or volume <= 0:
                continue
            
            if contract_type == "call":
                total_call_volume += volume
            else:
                total_put_volume += volume
        
        if total_call_volume == 0:
            return None
        
        return total_put_volume / total_call_volume
    
    def _get_volume_data(self, options_data: Dict) -> Dict:
        """Get volume breakdown from options data"""
        if not options_data or "results" not in options_data:
            return {'call_volume': 0, 'put_volume': 0, 'total_volume': 0}
        
        results = options_data["results"]
        total_call_volume = 0
        total_put_volume = 0
        
        for contract in results:
            contract_type = self._extract_contract_type(contract)
            if not contract_type:
                continue
            
            volume = self._extract_volume(contract)
            if volume is None or volume <= 0:
                continue
            
            if contract_type == "call":
                total_call_volume += volume
            else:
                total_put_volume += volume
        
        return {
            'call_volume': total_call_volume,
            'put_volume': total_put_volume,
            'total_volume': total_call_volume + total_put_volume
        }
    
    def _extract_contract_type(self, contract: Dict) -> Optional[str]:
        """Extract contract type (call/put)"""
        if "details" in contract and "contract_type" in contract["details"]:
            return contract["details"]["contract_type"].lower()
        elif "contract_type" in contract:
            return contract["contract_type"].lower()
        return None
    
    def _extract_volume(self, contract: Dict) -> Optional[int]:
        """Extract volume from contract"""
        return contract.get("volume", contract.get("day", {}).get("volume", 0))
    
    # ==================== BAND CALCULATIONS (EXACT PINE SCRIPT) ====================
    
    def _calculate_bands(self, pcr_series: List[float]) -> Dict:
        """Calculate bands exactly as Pine Script"""
        data = pcr_series[-self.lookback:] if len(pcr_series) >= self.lookback else pcr_series
        
        if len(data) == 0:
            return {
                'ma': None, 'stdev': None, 'upper_band': None, 'lower_band': None,
                'distance_from_ma': None, 'z_score': None
            }
        
        current_pcr = pcr_series[-1]
        
        # Calculate SMA (exact)
        ma = sum(data) / len(data)
        
        # Calculate STDEV (exact)
        variance = sum((x - ma) ** 2 for x in data) / len(data)
        sd = math.sqrt(variance)
        
        # Calculate bands (exact)
        upper_band = ma + (self.std_multiplier * sd)
        lower_band = ma - (self.std_multiplier * sd)
        
        # Calculate metrics
        distance_from_ma = abs(current_pcr - ma)
        z_score = (current_pcr - ma) / sd if sd > 0 else 0
        
        return {
            'ma': ma,
            'stdev': sd,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'distance_from_ma': distance_from_ma,
            'z_score': z_score
        }
    
    # ==================== SENTIMENT LABEL (raw, no recommendation) ====================
    
    def _get_sentiment_label(self, pcr: float) -> str:
        """Get sentiment label without recommendation"""
        if pcr >= self.extreme_fear_level:
            return "EXTREME_FEAR"
        elif pcr >= self.fear_level:
            return "FEAR"
        elif pcr >= self.neutral_lower:
            return "NEUTRAL"
        elif pcr >= self.greed_level:
            return "GREED"
        else:
            return "EXTREME_GREED"
    
    # ==================== SIMPLE ANALYSIS (NO HISTORY) ====================
    
    def _simple_analysis(self, pcr: float, volume_data: Dict, pcr_history: Optional[List[float]]) -> Dict:
        """Simple analysis when insufficient history"""
        return {
            "success": True,
            
            # Current PCR Data
            "pcr_current": round(pcr, 4),
            "pcr_ma_200": None,
            "pcr_stdev": None,
            "pcr_upper_band": None,
            "pcr_lower_band": None,
            
            # PCR vs Bands (not available)
            "pcr_above_upper_band": None,
            "pcr_below_lower_band": None,
            "pcr_within_bands": None,
            
            # Distance from Bands (not available)
            "distance_from_ma": None,
            "distance_from_upper": None,
            "distance_from_lower": None,
            "z_score": None,
            
            # Sentiment Threshold Comparisons
            "above_extreme_fear": pcr >= self.extreme_fear_level,
            "above_fear": pcr >= self.fear_level,
            "in_neutral_zone": self.neutral_lower <= pcr <= self.neutral_upper,
            "below_greed": pcr <= self.greed_level,
            "below_extreme_greed": pcr <= self.extreme_greed_level,
            
            # Sentiment State
            "sentiment_state": self._get_sentiment_label(pcr),
            "is_extreme_sentiment": pcr >= self.extreme_fear_level or pcr <= self.extreme_greed_level,
            
            # Put vs Call Comparison
            "more_puts_than_calls": pcr > 1.0,
            "more_calls_than_puts": pcr < 1.0,
            "balanced": 0.95 <= pcr <= 1.05,
            
            # Volume Data
            "total_call_volume": volume_data['call_volume'],
            "total_put_volume": volume_data['put_volume'],
            "total_volume": volume_data['total_volume'],
            "call_volume_pct": round(volume_data['call_volume'] / volume_data['total_volume'] * 100, 2) if volume_data['total_volume'] > 0 else 0,
            "put_volume_pct": round(volume_data['put_volume'] / volume_data['total_volume'] * 100, 2) if volume_data['total_volume'] > 0 else 0,
            
            # Historical Context
            "pcr_history_length": len(pcr_history) if pcr_history else 0,
            "has_sufficient_history": False,
            
            # Thresholds
            "threshold_extreme_fear": self.extreme_fear_level,
            "threshold_fear": self.fear_level,
            "threshold_neutral_upper": self.neutral_upper,
            "threshold_neutral_lower": self.neutral_lower,
            "threshold_greed": self.greed_level,
            "threshold_extreme_greed": self.extreme_greed_level,
            "band_std_multiplier": self.std_multiplier,
            "band_lookback": self.lookback,
            
            # Distance to Thresholds
            "distance_to_extreme_fear": round(self.extreme_fear_level - pcr, 4),
            "distance_to_fear": round(self.fear_level - pcr, 4),
            "distance_to_greed": round(pcr - self.greed_level, 4),
            "distance_to_extreme_greed": round(pcr - self.extreme_greed_level, 4),
            
            # Timestamp
            "timestamp": datetime.now().isoformat()
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "success": False,
            "error": reason,
            "pcr_current": None,
            "pcr_ma_200": None,
            "pcr_stdev": None,
            "pcr_upper_band": None,
            "pcr_lower_band": None,
            "pcr_above_upper_band": None,
            "pcr_below_lower_band": None,
            "pcr_within_bands": None,
            "distance_from_ma": None,
            "distance_from_upper": None,
            "distance_from_lower": None,
            "z_score": None,
            "above_extreme_fear": None,
            "above_fear": None,
            "in_neutral_zone": None,
            "below_greed": None,
            "below_extreme_greed": None,
            "sentiment_state": "UNKNOWN",
            "is_extreme_sentiment": None,
            "more_puts_than_calls": None,
            "more_calls_than_puts": None,
            "balanced": None,
            "total_call_volume": 0,
            "total_put_volume": 0,
            "total_volume": 0,
            "call_volume_pct": 0,
            "put_volume_pct": 0,
            "pcr_history_length": 0,
            "has_sufficient_history": False,
            "threshold_extreme_fear": self.extreme_fear_level,
            "threshold_fear": self.fear_level,
            "threshold_neutral_upper": self.neutral_upper,
            "threshold_neutral_lower": self.neutral_lower,
            "threshold_greed": self.greed_level,
            "threshold_extreme_greed": self.extreme_greed_level,
            "band_std_multiplier": self.std_multiplier,
            "band_lookback": self.lookback,
            "distance_to_extreme_fear": None,
            "distance_to_fear": None,
            "distance_to_greed": None,
            "distance_to_extreme_greed": None
        }
