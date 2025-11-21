"""
Layer 12: VWAP Analysis Engine (Raw Data Output)
Anchored VWAP Trading System with Standard Deviation Bands
Outputs RAW VWAP data only - no signals
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime

class Layer12VWAPAnalysis:
    """
    Professional VWAP Trading System.
    
    Features:
    - Anchored VWAP (custom start date)
    - Standard deviation bands (±1σ, ±2σ)
    - Volume filtering
    - Rejection counting (bounces at VWAP)
    - Zone acceptance tracking (upper/lower)
    - VWAP slope calculation
    """
    
    def __init__(self):
        # Anchor Settings
        self.anchor_year = 2024
        self.anchor_month = 11
        self.anchor_day = 1
        
        # Filter Settings
        self.min_volume_mult = 1.5
        self.min_rejections = 3
        self.min_acceptance_bars = 5
        self.vwap_touch_threshold = 0.003
        
        # Slope Settings
        self.slope_lookback = 10
        self.vwap_slope_bull_threshold = 0.002
        self.vwap_slope_bear_threshold = -0.002
        
        # Other Settings
        self.volume_sma_period = 20
        self.rejection_min_gap_bars = 5
        
        # State Variables
        self.sum_pv = 0.0
        self.sum_v = 0.0
        self.sum_sq = 0.0
        self.bars = 0
        self.rejection_count = 0
        self.last_rejection_bar = 0
        self.current_zone = "NEUTRAL"
        self.bars_in_zone = 0
        self.anchor_datetime = None
        
    def analyze(self, df: pd.DataFrame, anchor_date: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Run VWAP analysis
        
        Args:
            df: DataFrame with OHLCV data (must have DateTimeIndex)
            anchor_date: Optional (year, month, day) tuple to override default
            
        Returns:
            Dict with RAW VWAP levels, bands, and statistics
        """
        if len(df) < 20:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        if anchor_date:
            self.anchor_year, self.anchor_month, self.anchor_day = anchor_date
        
        self._reset_state()
        
        # Calculate VWAP and bands
        vwap_data = self._calculate_vwap(df)
        
        # Calculate volume data
        volume_data = self._calculate_volume_filter(df)
        
        # Calculate rejection count
        rejection_data = self._calculate_rejections(df, vwap_data)
        
        # Calculate zone acceptance
        zone_data = self._calculate_zone_acceptance(df, vwap_data)
        
        # Calculate VWAP slope
        slope_data = self._calculate_vwap_slope(vwap_data)
        
        # Calculate position relative to bands
        position_data = self._calculate_position(df, vwap_data)
        
        # Calculate crossover data (raw facts)
        crossover_data = self._calculate_crossovers(df, vwap_data)
        
        current_price = df['close'].iloc[-1]
        vwap_val = vwap_data['vwap'].iloc[-1] if len(vwap_data['vwap']) > 0 else None
        
        # Return RAW DATA ONLY - no signals
        return {
            # VWAP Core Data
            "vwap": round(vwap_val, 2) if vwap_val and not pd.isna(vwap_val) else None,
            "stdev": round(vwap_data['stdev'].iloc[-1], 4) if len(vwap_data['stdev']) > 0 and not pd.isna(vwap_data['stdev'].iloc[-1]) else None,
            
            # Band Levels
            "upper_1sd": round(vwap_data['upper_1sd'].iloc[-1], 2) if len(vwap_data['upper_1sd']) > 0 and not pd.isna(vwap_data['upper_1sd'].iloc[-1]) else None,
            "lower_1sd": round(vwap_data['lower_1sd'].iloc[-1], 2) if len(vwap_data['lower_1sd']) > 0 and not pd.isna(vwap_data['lower_1sd'].iloc[-1]) else None,
            "upper_2sd": round(vwap_data['upper_2sd'].iloc[-1], 2) if len(vwap_data['upper_2sd']) > 0 and not pd.isna(vwap_data['upper_2sd'].iloc[-1]) else None,
            "lower_2sd": round(vwap_data['lower_2sd'].iloc[-1], 2) if len(vwap_data['lower_2sd']) > 0 and not pd.isna(vwap_data['lower_2sd'].iloc[-1]) else None,
            
            # Price vs VWAP
            "price_vs_vwap": round(current_price - vwap_val, 2) if vwap_val and not pd.isna(vwap_val) else None,
            "price_vs_vwap_pct": round((current_price - vwap_val) / vwap_val * 100, 2) if vwap_val and not pd.isna(vwap_val) and vwap_val != 0 else None,
            "price_above_vwap": current_price > vwap_val if vwap_val and not pd.isna(vwap_val) else None,
            "price_below_vwap": current_price < vwap_val if vwap_val and not pd.isna(vwap_val) else None,
            
            # Position Data
            "stdev_distance": round(position_data['stdev_distance'], 2),
            "at_upper_1sd": position_data['stdev_distance'] >= 1.0,
            "at_lower_1sd": position_data['stdev_distance'] <= -1.0,
            "at_upper_2sd": position_data['at_upper_2sd'],
            "at_lower_2sd": position_data['at_lower_2sd'],
            "between_bands": -1.0 < position_data['stdev_distance'] < 1.0,
            
            # Volume Data
            "current_volume": round(volume_data['current'], 0),
            "avg_volume_20": round(volume_data['avg_20'], 0),
            "volume_ratio": round(volume_data['current'] / volume_data['avg_20'], 2) if volume_data['avg_20'] > 0 else 1.0,
            "is_high_volume": volume_data['is_high_volume'],
            
            # Rejection Data
            "rejection_count": rejection_data['count'],
            "is_strong_level": rejection_data['is_strong'],
            
            # Zone Data
            "current_zone": zone_data['current'],
            "bars_in_zone": zone_data['bars_in_zone'],
            "accepted_above": zone_data['accepted_above'],
            "accepted_below": zone_data['accepted_below'],
            
            # Slope Data
            "vwap_slope": round(slope_data['value'], 6),
            "vwap_slope_pct": round(slope_data['value'] * 100, 4),
            "slope_above_bull_threshold": slope_data['bullish'],
            "slope_below_bear_threshold": slope_data['bearish'],
            "slope_neutral": not slope_data['bullish'] and not slope_data['bearish'],
            
            # Crossover Data (raw facts)
            "crossed_above_vwap": crossover_data['crossed_above_vwap'],
            "crossed_below_vwap": crossover_data['crossed_below_vwap'],
            "crossed_above_lower_1sd": crossover_data['crossed_above_lower_1sd'],
            "crossed_below_upper_1sd": crossover_data['crossed_below_upper_1sd'],
            "crossed_above_upper_2sd": crossover_data['crossed_above_upper_2sd'],
            "crossed_below_lower_2sd": crossover_data['crossed_below_lower_2sd'],
            
            # Distance to Bands
            "distance_to_upper_1sd": round(vwap_data['upper_1sd'].iloc[-1] - current_price, 2) if len(vwap_data['upper_1sd']) > 0 and not pd.isna(vwap_data['upper_1sd'].iloc[-1]) else None,
            "distance_to_lower_1sd": round(current_price - vwap_data['lower_1sd'].iloc[-1], 2) if len(vwap_data['lower_1sd']) > 0 and not pd.isna(vwap_data['lower_1sd'].iloc[-1]) else None,
            "distance_to_upper_2sd": round(vwap_data['upper_2sd'].iloc[-1] - current_price, 2) if len(vwap_data['upper_2sd']) > 0 and not pd.isna(vwap_data['upper_2sd'].iloc[-1]) else None,
            "distance_to_lower_2sd": round(current_price - vwap_data['lower_2sd'].iloc[-1], 2) if len(vwap_data['lower_2sd']) > 0 and not pd.isna(vwap_data['lower_2sd'].iloc[-1]) else None,
            
            # Price Context
            "current_price": round(current_price, 2),
            
            # Timestamp
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    def _reset_state(self):
        """Reset all state variables"""
        self.sum_pv = 0.0
        self.sum_v = 0.0
        self.sum_sq = 0.0
        self.bars = 0
        self.rejection_count = 0
        self.last_rejection_bar = 0
        self.current_zone = "NEUTRAL"
        self.bars_in_zone = 0
        self.anchor_datetime = pd.Timestamp(year=self.anchor_year, 
                                            month=self.anchor_month, 
                                            day=self.anchor_day)
    
    # ==================== VWAP CALCULATION ====================
    
    def _calculate_vwap(self, df: pd.DataFrame) -> Dict:
        """Calculate Anchored VWAP with standard deviation bands"""
        if not isinstance(df.index, pd.DatetimeIndex):
            anchor_idx = 0
        else:
            anchor_idx = df.index.searchsorted(self.anchor_datetime)
            if anchor_idx >= len(df):
                anchor_idx = 0
        
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        
        vwap_values = np.full(len(df), np.nan)
        stdev_values = np.full(len(df), np.nan)
        
        for i in range(anchor_idx, len(df)):
            if i == anchor_idx:
                self.sum_pv = hlc3.iloc[i] * df['volume'].iloc[i]
                self.sum_v = df['volume'].iloc[i]
                self.sum_sq = 0.0
                self.bars = 1
            else:
                self.sum_pv += hlc3.iloc[i] * df['volume'].iloc[i]
                self.sum_v += df['volume'].iloc[i]
                self.bars += 1
            
            if self.sum_v > 0:
                vwap = self.sum_pv / self.sum_v
                vwap_values[i] = vwap
                
                diff = hlc3.iloc[i] - vwap
                
                if i == anchor_idx:
                    self.sum_sq = diff * diff
                else:
                    self.sum_sq += diff * diff
                
                if self.bars > 1:
                    variance = self.sum_sq / self.bars
                    stdev = np.sqrt(variance)
                    stdev_values[i] = stdev
        
        vwap_series = pd.Series(vwap_values, index=df.index)
        stdev_series = pd.Series(stdev_values, index=df.index)
        
        upper_1sd = vwap_series + stdev_series
        lower_1sd = vwap_series - stdev_series
        upper_2sd = vwap_series + (2 * stdev_series)
        lower_2sd = vwap_series - (2 * stdev_series)
        
        return {
            'vwap': vwap_series,
            'stdev': stdev_series,
            'upper_1sd': upper_1sd,
            'lower_1sd': lower_1sd,
            'upper_2sd': upper_2sd,
            'lower_2sd': lower_2sd
        }
    
    # ==================== VOLUME FILTER ====================
    
    def _calculate_volume_filter(self, df: pd.DataFrame) -> Dict:
        """Calculate volume filter"""
        avg_volume = df['volume'].rolling(window=self.volume_sma_period).mean()
        current_volume = df['volume'].iloc[-1]
        avg_volume_current = avg_volume.iloc[-1]
        
        is_high_volume = current_volume > (avg_volume_current * self.min_volume_mult)
        
        return {
            'current': float(current_volume),
            'avg_20': float(avg_volume_current),
            'is_high_volume': bool(is_high_volume)
        }
    
    # ==================== REJECTION COUNTING ====================
    
    def _calculate_rejections(self, df: pd.DataFrame, vwap_data: Dict) -> Dict:
        """Count rejections at VWAP level"""
        vwap = vwap_data['vwap']
        rejection_count = 0
        last_rejection_bar = 0
        
        for i in range(len(df)):
            if pd.isna(vwap.iloc[i]):
                continue
            
            close = df['close'].iloc[i]
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            vwap_val = vwap.iloc[i]
            
            distance = abs(close - vwap_val) / vwap_val if vwap_val != 0 else 0
            touching = distance < self.vwap_touch_threshold
            
            is_rejection = ((high > vwap_val and close < vwap_val) or 
                          (low < vwap_val and close > vwap_val))
            
            if touching and is_rejection and (i - last_rejection_bar) > self.rejection_min_gap_bars:
                rejection_count += 1
                last_rejection_bar = i
        
        is_strong = rejection_count >= self.min_rejections
        
        return {
            'count': rejection_count,
            'is_strong': is_strong
        }
    
    # ==================== ZONE ACCEPTANCE ====================
    
    def _calculate_zone_acceptance(self, df: pd.DataFrame, vwap_data: Dict) -> Dict:
        """Calculate zone acceptance"""
        vwap = vwap_data['vwap']
        stdev = vwap_data['stdev']
        
        current_zone = "NEUTRAL"
        bars_in_zone = 0
        
        for i in range(len(df)):
            if pd.isna(vwap.iloc[i]) or pd.isna(stdev.iloc[i]):
                continue
            
            close = df['close'].iloc[i]
            vwap_val = vwap.iloc[i]
            stdev_val = stdev.iloc[i]
            
            if close > vwap_val + stdev_val:
                if current_zone == "UPPER":
                    bars_in_zone += 1
                else:
                    current_zone = "UPPER"
                    bars_in_zone = 1
            elif close < vwap_val - stdev_val:
                if current_zone == "LOWER":
                    bars_in_zone += 1
                else:
                    current_zone = "LOWER"
                    bars_in_zone = 1
            else:
                current_zone = "NEUTRAL"
                bars_in_zone = 0
        
        accepted_above = current_zone == "UPPER" and bars_in_zone >= self.min_acceptance_bars
        accepted_below = current_zone == "LOWER" and bars_in_zone >= self.min_acceptance_bars
        
        return {
            'current': current_zone,
            'bars_in_zone': bars_in_zone,
            'accepted_above': accepted_above,
            'accepted_below': accepted_below
        }
    
    # ==================== VWAP SLOPE ====================
    
    def _calculate_vwap_slope(self, vwap_data: Dict) -> Dict:
        """Calculate VWAP slope"""
        vwap = vwap_data['vwap']
        
        if len(vwap) < self.slope_lookback + 1:
            return {
                'value': 0.0,
                'bullish': False,
                'bearish': False
            }
        
        vwap_current = vwap.iloc[-1]
        vwap_past = vwap.iloc[-(self.slope_lookback + 1)]
        
        if pd.isna(vwap_current) or pd.isna(vwap_past) or vwap_past == 0:
            slope = 0.0
        else:
            slope = (vwap_current - vwap_past) / vwap_past
        
        bullish = slope > self.vwap_slope_bull_threshold
        bearish = slope < self.vwap_slope_bear_threshold
        
        return {
            'value': float(slope),
            'bullish': bool(bullish),
            'bearish': bool(bearish)
        }
    
    # ==================== POSITION CALCULATION ====================
    
    def _calculate_position(self, df: pd.DataFrame, vwap_data: Dict) -> Dict:
        """Calculate position relative to bands"""
        close = df['close'].iloc[-1]
        vwap = vwap_data['vwap'].iloc[-1]
        stdev = vwap_data['stdev'].iloc[-1]
        
        if pd.isna(vwap) or pd.isna(stdev) or stdev == 0:
            return {
                'stdev_distance': 0.0,
                'at_upper_2sd': False,
                'at_lower_2sd': False
            }
        
        stdev_distance = (close - vwap) / stdev
        at_upper_2sd = stdev_distance >= 2.0
        at_lower_2sd = stdev_distance <= -2.0
        
        return {
            'stdev_distance': float(stdev_distance),
            'at_upper_2sd': bool(at_upper_2sd),
            'at_lower_2sd': bool(at_lower_2sd)
        }
    
    # ==================== CROSSOVER CALCULATION ====================
    
    def _calculate_crossovers(self, df: pd.DataFrame, vwap_data: Dict) -> Dict:
        """Calculate crossover data - raw facts only"""
        if len(df) < 2:
            return {
                'crossed_above_vwap': False,
                'crossed_below_vwap': False,
                'crossed_above_lower_1sd': False,
                'crossed_below_upper_1sd': False,
                'crossed_above_upper_2sd': False,
                'crossed_below_lower_2sd': False
            }
        
        close = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        
        vwap = vwap_data['vwap'].iloc[-1]
        vwap_prev = vwap_data['vwap'].iloc[-2]
        lower_1sd = vwap_data['lower_1sd'].iloc[-1]
        lower_1sd_prev = vwap_data['lower_1sd'].iloc[-2]
        upper_1sd = vwap_data['upper_1sd'].iloc[-1]
        upper_1sd_prev = vwap_data['upper_1sd'].iloc[-2]
        lower_2sd = vwap_data['lower_2sd'].iloc[-1]
        lower_2sd_prev = vwap_data['lower_2sd'].iloc[-2]
        upper_2sd = vwap_data['upper_2sd'].iloc[-1]
        upper_2sd_prev = vwap_data['upper_2sd'].iloc[-2]
        
        # Handle NaN values
        def safe_cross_above(curr, prev, level_curr, level_prev):
            if pd.isna(curr) or pd.isna(prev) or pd.isna(level_curr) or pd.isna(level_prev):
                return False
            return curr > level_curr and prev <= level_prev
        
        def safe_cross_below(curr, prev, level_curr, level_prev):
            if pd.isna(curr) or pd.isna(prev) or pd.isna(level_curr) or pd.isna(level_prev):
                return False
            return curr < level_curr and prev >= level_prev
        
        return {
            'crossed_above_vwap': safe_cross_above(close, close_prev, vwap, vwap_prev),
            'crossed_below_vwap': safe_cross_below(close, close_prev, vwap, vwap_prev),
            'crossed_above_lower_1sd': safe_cross_above(close, close_prev, lower_1sd, lower_1sd_prev),
            'crossed_below_upper_1sd': safe_cross_below(close, close_prev, upper_1sd, upper_1sd_prev),
            'crossed_above_upper_2sd': safe_cross_above(close, close_prev, upper_2sd, upper_2sd_prev),
            'crossed_below_lower_2sd': safe_cross_below(close, close_prev, lower_2sd, lower_2sd_prev)
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "vwap": None, "stdev": None,
            "upper_1sd": None, "lower_1sd": None,
            "upper_2sd": None, "lower_2sd": None,
            "price_vs_vwap": None, "price_vs_vwap_pct": None,
            "price_above_vwap": None, "price_below_vwap": None,
            "stdev_distance": 0, "at_upper_1sd": False, "at_lower_1sd": False,
            "at_upper_2sd": False, "at_lower_2sd": False, "between_bands": False,
            "current_volume": 0, "avg_volume_20": 0, "volume_ratio": 1.0, "is_high_volume": False,
            "rejection_count": 0, "is_strong_level": False,
            "current_zone": "NEUTRAL", "bars_in_zone": 0,
            "accepted_above": False, "accepted_below": False,
            "vwap_slope": 0, "vwap_slope_pct": 0,
            "slope_above_bull_threshold": False, "slope_below_bear_threshold": False, "slope_neutral": True,
            "crossed_above_vwap": False, "crossed_below_vwap": False,
            "crossed_above_lower_1sd": False, "crossed_below_upper_1sd": False,
            "crossed_above_upper_2sd": False, "crossed_below_lower_2sd": False,
            "distance_to_upper_1sd": None, "distance_to_lower_1sd": None,
            "distance_to_upper_2sd": None, "distance_to_lower_2sd": None,
            "current_price": None, "error": reason
        }
