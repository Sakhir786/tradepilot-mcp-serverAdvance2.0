"""
Layer 14: IV Analysis Engine (Raw Data Output)
Complete IV system with HV, IV Rank, IV Percentile, Expected Move
Outputs RAW IV data only - no signals or strategy recommendations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Layer14IVAnalysis:
    """
    Professional IV Analysis System.
    
    Components:
    1. HV Calculation (Historical Volatility)
    2. IV Rank (current position in 52-week range)
    3. IV Percentile (% of days with lower IV)
    4. Expected Move calculation
    5. State thresholds (raw comparisons)
    """
    
    def __init__(self):
        # Core Settings
        self.iv_length = 252  # 1 year lookback
        self.hv_window = 20  # HV calculation window
        self.smooth_length = 5  # IV smoothing
        
        # Threshold levels (for raw comparison only)
        self.extreme_high_level = 80
        self.high_level = 60
        self.low_level = 40
        self.extreme_low_level = 20
        
        # Expected Move Settings
        self.dte_options = [7, 14, 30, 45, 60]  # Multiple DTE calculations
        
    def analyze(self, df: pd.DataFrame, dte: int = 30) -> Dict:
        """
        Complete IV analysis
        
        Args:
            df: OHLCV DataFrame
            dte: Days to expiration for expected move calc
            
        Returns:
            Dict with RAW IV analysis data
        """
        if len(df) < self.iv_length:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        current_price = df['close'].iloc[-1]
        
        # Calculate HV
        hv_analysis = self._calculate_hv(df)
        
        # Calculate IV Rank
        iv_rank_data = self._calculate_iv_rank(hv_analysis)
        
        # Calculate IV Percentile
        iv_percentile_data = self._calculate_iv_percentile(hv_analysis)
        
        # Calculate Expected Move for multiple DTEs
        expected_moves = self._calculate_expected_moves(current_price, hv_analysis['smoothed'])
        
        # Calculate expected move for specified DTE
        expected_move_dte = self._calculate_expected_move(current_price, hv_analysis['smoothed'], dte)
        
        # Raw threshold comparisons
        iv_rank = iv_rank_data['value']
        
        # HV trend analysis
        hv_trend = self._analyze_hv_trend(hv_analysis)
        
        # Return RAW DATA ONLY - no signals or recommendations
        return {
            # HV Data
            "hv_current": round(hv_analysis['current'], 2) if hv_analysis['current'] else None,
            "hv_smoothed": round(hv_analysis['smoothed'], 2) if hv_analysis['smoothed'] else None,
            "hv_high_52w": round(hv_analysis['high_52w'], 2) if hv_analysis['high_52w'] else None,
            "hv_low_52w": round(hv_analysis['low_52w'], 2) if hv_analysis['low_52w'] else None,
            "hv_range_52w": round(hv_analysis['high_52w'] - hv_analysis['low_52w'], 2) if hv_analysis['high_52w'] and hv_analysis['low_52w'] else None,
            
            # IV Rank Data
            "iv_rank": round(iv_rank, 2) if iv_rank else None,
            "iv_rank_valid": iv_rank_data['valid'],
            
            # IV Percentile Data
            "iv_percentile": round(iv_percentile_data['value'], 2) if iv_percentile_data['value'] else None,
            "iv_percentile_valid": iv_percentile_data['valid'],
            
            # Threshold Comparisons (raw boolean facts)
            "iv_above_extreme_high": iv_rank >= self.extreme_high_level if iv_rank else None,
            "iv_above_high": iv_rank >= self.high_level if iv_rank else None,
            "iv_below_low": iv_rank <= self.low_level if iv_rank else None,
            "iv_below_extreme_low": iv_rank <= self.extreme_low_level if iv_rank else None,
            "iv_in_normal_range": self.low_level < iv_rank < self.high_level if iv_rank else None,
            
            # State Classification (raw, no recommendation)
            "iv_state": self._get_state_label(iv_rank),
            
            # Expected Move - Specified DTE
            "em_dte": dte,
            "em_1sd": round(expected_move_dte['1sd'], 2) if expected_move_dte else None,
            "em_2sd": round(expected_move_dte['2sd'], 2) if expected_move_dte else None,
            "em_1sd_pct": round(expected_move_dte['percent_1sd'], 2) if expected_move_dte else None,
            "em_2sd_pct": round(expected_move_dte['percent_2sd'], 2) if expected_move_dte else None,
            "em_upper_1sd": round(expected_move_dte['range_1sd']['upper'], 2) if expected_move_dte else None,
            "em_lower_1sd": round(expected_move_dte['range_1sd']['lower'], 2) if expected_move_dte else None,
            "em_upper_2sd": round(expected_move_dte['range_2sd']['upper'], 2) if expected_move_dte else None,
            "em_lower_2sd": round(expected_move_dte['range_2sd']['lower'], 2) if expected_move_dte else None,
            
            # Expected Moves - Multiple DTEs
            "em_7d_1sd_pct": round(expected_moves.get(7, {}).get('percent_1sd', 0), 2),
            "em_14d_1sd_pct": round(expected_moves.get(14, {}).get('percent_1sd', 0), 2),
            "em_30d_1sd_pct": round(expected_moves.get(30, {}).get('percent_1sd', 0), 2),
            "em_45d_1sd_pct": round(expected_moves.get(45, {}).get('percent_1sd', 0), 2),
            "em_60d_1sd_pct": round(expected_moves.get(60, {}).get('percent_1sd', 0), 2),
            
            # HV Trend Data
            "hv_rising": hv_trend['rising'],
            "hv_falling": hv_trend['falling'],
            "hv_stable": hv_trend['stable'],
            "hv_5d_change": round(hv_trend['change_5d'], 2) if hv_trend['change_5d'] else None,
            "hv_10d_change": round(hv_trend['change_10d'], 2) if hv_trend['change_10d'] else None,
            "hv_vs_avg": round(hv_trend['vs_avg'], 2) if hv_trend['vs_avg'] else None,
            
            # Threshold Levels (for reference)
            "threshold_extreme_high": self.extreme_high_level,
            "threshold_high": self.high_level,
            "threshold_low": self.low_level,
            "threshold_extreme_low": self.extreme_low_level,
            
            # Distance to Thresholds
            "distance_to_extreme_high": round(self.extreme_high_level - iv_rank, 2) if iv_rank else None,
            "distance_to_high": round(self.high_level - iv_rank, 2) if iv_rank else None,
            "distance_to_low": round(iv_rank - self.low_level, 2) if iv_rank else None,
            "distance_to_extreme_low": round(iv_rank - self.extreme_low_level, 2) if iv_rank else None,
            
            # Price Context
            "current_price": round(current_price, 2),
            
            # Timestamp
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== HV CALCULATION ====================
    
    def _calculate_hv(self, df: pd.DataFrame) -> Dict:
        """Calculate Historical Volatility"""
        if len(df) < self.hv_window:
            return {'current': None, 'smoothed': None, 'high_52w': None, 'low_52w': None, 'series': None}
        
        # Log returns
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate HV
        hv_series = log_returns.rolling(self.hv_window).std() * np.sqrt(252) * 100
        
        # Smooth HV
        hv_smoothed = hv_series.rolling(self.smooth_length).mean()
        
        # Get 52-week high/low
        lookback_data = hv_smoothed.iloc[-self.iv_length:] if len(hv_smoothed) >= self.iv_length else hv_smoothed
        lookback_data = lookback_data.dropna()
        
        return {
            'current': hv_series.iloc[-1] if not pd.isna(hv_series.iloc[-1]) else None,
            'smoothed': hv_smoothed.iloc[-1] if not pd.isna(hv_smoothed.iloc[-1]) else None,
            'high_52w': lookback_data.max() if len(lookback_data) > 0 else None,
            'low_52w': lookback_data.min() if len(lookback_data) > 0 else None,
            'series': hv_smoothed
        }
    
    # ==================== IV RANK ====================
    
    def _calculate_iv_rank(self, hv_analysis: Dict) -> Dict:
        """Calculate IV Rank"""
        current = hv_analysis['smoothed']
        high_52w = hv_analysis['high_52w']
        low_52w = hv_analysis['low_52w']
        
        if current is None or high_52w is None or low_52w is None:
            return {'value': None, 'valid': False}
        
        if high_52w != low_52w:
            iv_rank = ((current - low_52w) / (high_52w - low_52w)) * 100
        else:
            iv_rank = 50.0
        
        return {
            'value': iv_rank,
            'valid': True
        }
    
    # ==================== IV PERCENTILE ====================
    
    def _calculate_iv_percentile(self, hv_analysis: Dict) -> Dict:
        """Calculate IV Percentile"""
        hv_series = hv_analysis.get('series')
        current = hv_analysis['smoothed']
        
        if hv_series is None or current is None:
            return {'value': None, 'valid': False}
        
        lookback_series = hv_series.iloc[-self.iv_length:] if len(hv_series) >= self.iv_length else hv_series
        lookback_series = lookback_series.dropna()
        
        if len(lookback_series) == 0:
            return {'value': None, 'valid': False}
        
        count = (lookback_series < current).sum()
        total = len(lookback_series)
        
        percentile = (count / total) * 100
        
        return {
            'value': percentile,
            'valid': True
        }
    
    # ==================== STATE LABEL (raw, no recommendation) ====================
    
    def _get_state_label(self, iv_rank: float) -> str:
        """Get state label without recommendation"""
        if iv_rank is None:
            return "UNKNOWN"
        
        if iv_rank >= self.extreme_high_level:
            return "EXTREME_HIGH"
        elif iv_rank >= self.high_level:
            return "HIGH"
        elif iv_rank >= self.low_level:
            return "NORMAL"
        elif iv_rank >= self.extreme_low_level:
            return "LOW"
        else:
            return "EXTREME_LOW"
    
    # ==================== EXPECTED MOVE ====================
    
    def _calculate_expected_move(self, price: float, iv: float, dte: int) -> Dict:
        """Calculate expected move for specified DTE"""
        if iv is None or price is None or dte <= 0:
            return None
        
        # 1 Standard Deviation
        expected_move_1sd = price * (iv / 100) * np.sqrt(dte / 365)
        expected_move_2sd = expected_move_1sd * 2
        
        return {
            'dte': dte,
            '1sd': expected_move_1sd,
            '2sd': expected_move_2sd,
            'range_1sd': {
                'lower': price - expected_move_1sd,
                'upper': price + expected_move_1sd
            },
            'range_2sd': {
                'lower': price - expected_move_2sd,
                'upper': price + expected_move_2sd
            },
            'percent_1sd': (expected_move_1sd / price) * 100,
            'percent_2sd': (expected_move_2sd / price) * 100
        }
    
    def _calculate_expected_moves(self, price: float, iv: float) -> Dict:
        """Calculate expected moves for multiple DTEs"""
        expected_moves = {}
        
        for dte in self.dte_options:
            em = self._calculate_expected_move(price, iv, dte)
            if em:
                expected_moves[dte] = em
        
        return expected_moves
    
    # ==================== HV TREND ANALYSIS ====================
    
    def _analyze_hv_trend(self, hv_analysis: Dict) -> Dict:
        """Analyze HV trend"""
        hv_series = hv_analysis.get('series')
        
        if hv_series is None or len(hv_series) < 20:
            return {
                'rising': None, 'falling': None, 'stable': None,
                'change_5d': None, 'change_10d': None, 'vs_avg': None
            }
        
        current = hv_series.iloc[-1]
        hv_5d_ago = hv_series.iloc[-5] if len(hv_series) >= 5 else None
        hv_10d_ago = hv_series.iloc[-10] if len(hv_series) >= 10 else None
        hv_avg_20 = hv_series.iloc[-20:].mean() if len(hv_series) >= 20 else None
        
        # Calculate changes
        change_5d = None
        change_10d = None
        
        if hv_5d_ago and not pd.isna(hv_5d_ago) and hv_5d_ago != 0:
            change_5d = ((current - hv_5d_ago) / hv_5d_ago) * 100
        
        if hv_10d_ago and not pd.isna(hv_10d_ago) and hv_10d_ago != 0:
            change_10d = ((current - hv_10d_ago) / hv_10d_ago) * 100
        
        # Calculate vs average
        vs_avg = None
        if hv_avg_20 and not pd.isna(hv_avg_20) and hv_avg_20 != 0:
            vs_avg = ((current - hv_avg_20) / hv_avg_20) * 100
        
        # Determine trend direction
        rising = change_5d > 5 if change_5d else None
        falling = change_5d < -5 if change_5d else None
        stable = -5 <= change_5d <= 5 if change_5d else None
        
        return {
            'rising': rising,
            'falling': falling,
            'stable': stable,
            'change_5d': change_5d,
            'change_10d': change_10d,
            'vs_avg': vs_avg
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "hv_current": None, "hv_smoothed": None,
            "hv_high_52w": None, "hv_low_52w": None, "hv_range_52w": None,
            "iv_rank": None, "iv_rank_valid": False,
            "iv_percentile": None, "iv_percentile_valid": False,
            "iv_above_extreme_high": None, "iv_above_high": None,
            "iv_below_low": None, "iv_below_extreme_low": None,
            "iv_in_normal_range": None, "iv_state": "UNKNOWN",
            "em_dte": None, "em_1sd": None, "em_2sd": None,
            "em_1sd_pct": None, "em_2sd_pct": None,
            "em_upper_1sd": None, "em_lower_1sd": None,
            "em_upper_2sd": None, "em_lower_2sd": None,
            "em_7d_1sd_pct": 0, "em_14d_1sd_pct": 0, "em_30d_1sd_pct": 0,
            "em_45d_1sd_pct": 0, "em_60d_1sd_pct": 0,
            "hv_rising": None, "hv_falling": None, "hv_stable": None,
            "hv_5d_change": None, "hv_10d_change": None, "hv_vs_avg": None,
            "threshold_extreme_high": self.extreme_high_level,
            "threshold_high": self.high_level,
            "threshold_low": self.low_level,
            "threshold_extreme_low": self.extreme_low_level,
            "distance_to_extreme_high": None, "distance_to_high": None,
            "distance_to_low": None, "distance_to_extreme_low": None,
            "current_price": None, "error": reason
        }
