"""
Layer 12: VWAP Analysis Engine
Anchored VWAP Trading System with Standard Deviation Bands
Converted from Pine Script - Logic unchanged
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
    - Volume filtering (>1.5x average)
    - Rejection counting (bounces at VWAP)
    - Zone acceptance tracking (upper/lower)
    - VWAP slope bias (bullish/bearish)
    - Mean reversion signals (at ±2SD)
    - Breakout signals (acceptance + continuation)
    """
    
    def __init__(self):
        # ==================== Anchor Settings ====================
        self.anchor_year = 2024
        self.anchor_month = 11
        self.anchor_day = 1
        
        # ==================== Filter Settings ====================
        self.min_volume_mult = 1.5  # Minimum volume multiple
        self.min_rejections = 3  # Minimum rejections for strong level
        self.min_acceptance_bars = 5  # Minimum bars for zone acceptance
        self.vwap_touch_threshold = 0.003  # 0.3% threshold for VWAP touch
        
        # ==================== Slope Settings ====================
        self.slope_lookback = 10  # Bars to calculate slope
        self.vwap_slope_bull_threshold = 0.002  # 0.2% bullish threshold
        self.vwap_slope_bear_threshold = -0.002  # -0.2% bearish threshold
        
        # ==================== Other Settings ====================
        self.volume_sma_period = 20  # Volume average period
        self.rejection_min_gap_bars = 5  # Minimum bars between rejections
        
        # ==================== State Variables ====================
        self.sum_pv = 0.0  # Cumulative price * volume
        self.sum_v = 0.0  # Cumulative volume
        self.sum_sq = 0.0  # Cumulative squared differences
        self.bars = 0  # Bar count since anchor
        self.rejection_count = 0  # Total rejections
        self.last_rejection_bar = 0  # Last rejection bar index
        self.current_zone = "NEUTRAL"  # Current zone
        self.bars_in_zone = 0  # Bars in current zone
        self.anchor_datetime = None
        
    def analyze(self, df: pd.DataFrame, anchor_date: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Run VWAP analysis
        
        Args:
            df: DataFrame with OHLCV data (must have DateTimeIndex)
            anchor_date: Optional (year, month, day) tuple to override default
            
        Returns:
            Dict with VWAP levels, bands, signals, and statistics
        """
        if len(df) < 20:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Set anchor date
        if anchor_date:
            self.anchor_year, self.anchor_month, self.anchor_day = anchor_date
        
        # Reset state
        self._reset_state()
        
        # Calculate VWAP and bands
        vwap_data = self._calculate_vwap(df)
        
        # Calculate volume filter
        volume_data = self._calculate_volume_filter(df)
        
        # Calculate rejection count
        rejection_data = self._calculate_rejections(df, vwap_data)
        
        # Calculate zone acceptance
        zone_data = self._calculate_zone_acceptance(df, vwap_data)
        
        # Calculate VWAP slope
        slope_data = self._calculate_vwap_slope(vwap_data)
        
        # Calculate position
        position_data = self._calculate_position(df, vwap_data)
        
        # Generate trading signals
        signals = self._generate_signals(df, vwap_data, volume_data, rejection_data, 
                                         zone_data, slope_data, position_data)
        
        return {
            "vwap": vwap_data['vwap'].iloc[-1] if len(vwap_data['vwap']) > 0 else None,
            "stdev": vwap_data['stdev'].iloc[-1] if len(vwap_data['stdev']) > 0 else None,
            "bands": {
                "upper_1sd": vwap_data['upper_1sd'].iloc[-1] if len(vwap_data['upper_1sd']) > 0 else None,
                "lower_1sd": vwap_data['lower_1sd'].iloc[-1] if len(vwap_data['lower_1sd']) > 0 else None,
                "upper_2sd": vwap_data['upper_2sd'].iloc[-1] if len(vwap_data['upper_2sd']) > 0 else None,
                "lower_2sd": vwap_data['lower_2sd'].iloc[-1] if len(vwap_data['lower_2sd']) > 0 else None
            },
            "volume": volume_data,
            "rejection_count": rejection_data['count'],
            "is_strong_level": rejection_data['is_strong'],
            "zone": zone_data,
            "vwap_slope": slope_data,
            "position": position_data,
            "signals": signals,
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
        """
        Calculate Anchored VWAP with standard deviation bands
        
        Formula (exact Pine Script):
        hlc3 = (high + low + close) / 3
        sum_pv = cumsum(hlc3 * volume) from anchor
        sum_v = cumsum(volume) from anchor
        vwap = sum_pv / sum_v
        
        variance = sum((hlc3 - vwap)^2) / bars
        stdev = sqrt(variance)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            # Fallback: use all data
            anchor_idx = 0
        else:
            # Find anchor date
            anchor_idx = df.index.searchsorted(self.anchor_datetime)
            if anchor_idx >= len(df):
                anchor_idx = 0
        
        # Calculate HLC3 (typical price)
        hlc3 = (df['high'] + df['low'] + df['close']) / 3
        
        # Initialize arrays
        vwap_values = np.full(len(df), np.nan)
        stdev_values = np.full(len(df), np.nan)
        
        # Calculate from anchor date forward
        for i in range(anchor_idx, len(df)):
            if i == anchor_idx:
                # Reset on first bar
                self.sum_pv = hlc3.iloc[i] * df['volume'].iloc[i]
                self.sum_v = df['volume'].iloc[i]
                self.sum_sq = 0.0
                self.bars = 1
            else:
                # Accumulate
                self.sum_pv += hlc3.iloc[i] * df['volume'].iloc[i]
                self.sum_v += df['volume'].iloc[i]
                self.bars += 1
            
            # Calculate VWAP
            if self.sum_v > 0:
                vwap = self.sum_pv / self.sum_v
                vwap_values[i] = vwap
                
                # Calculate variance
                diff = hlc3.iloc[i] - vwap
                
                if i == anchor_idx:
                    self.sum_sq = diff * diff
                else:
                    self.sum_sq += diff * diff
                
                if self.bars > 1:
                    variance = self.sum_sq / self.bars
                    stdev = np.sqrt(variance)
                    stdev_values[i] = stdev
        
        # Create series
        vwap_series = pd.Series(vwap_values, index=df.index)
        stdev_series = pd.Series(stdev_values, index=df.index)
        
        # Calculate bands
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
        """
        Calculate volume filter
        
        Formula (exact Pine Script):
        avg_volume = SMA(volume, 20)
        is_high_volume = volume > avg_volume * 1.5
        """
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
        """
        Count rejections at VWAP level
        
        Formula (exact Pine Script):
        distance_to_vwap = abs(close - vwap) / vwap
        touching_vwap = distance_to_vwap < 0.003
        
        is_rejection = (high > vwap AND close < vwap) OR 
                       (low < vwap AND close > vwap)
        
        If touching AND rejection AND gap > 5 bars: count++
        """
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
            
            # Distance to VWAP
            distance = abs(close - vwap_val) / vwap_val if vwap_val != 0 else 0
            touching = distance < self.vwap_touch_threshold
            
            # Rejection logic
            is_rejection = ((high > vwap_val and close < vwap_val) or 
                          (low < vwap_val and close > vwap_val))
            
            # Count rejection if conditions met
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
        """
        Calculate zone acceptance
        
        Formula (exact Pine Script):
        If close > vwap + stdev:
            zone = "UPPER", bars_in_zone++
        Elif close < vwap - stdev:
            zone = "LOWER", bars_in_zone++
        Else:
            zone = "NEUTRAL", bars_in_zone = 0
        
        accepted_above = zone == "UPPER" AND bars_in_zone >= 5
        accepted_below = zone == "LOWER" AND bars_in_zone >= 5
        """
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
            
            # Determine zone
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
        """
        Calculate VWAP slope bias
        
        Formula (exact Pine Script):
        vwap_slope = (vwap - vwap[10]) / vwap[10]
        vwap_bullish = vwap_slope > 0.002
        vwap_bearish = vwap_slope < -0.002
        """
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
        """
        Calculate position relative to bands
        
        Formula (exact Pine Script):
        stdev_distance = (close - vwap) / stdev
        at_upper_2sd = stdev_distance >= 2.0
        at_lower_2sd = stdev_distance <= -2.0
        """
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
    
    # ==================== SIGNAL GENERATION ====================
    
    def _generate_signals(self, df: pd.DataFrame, vwap_data: Dict, volume_data: Dict,
                         rejection_data: Dict, zone_data: Dict, slope_data: Dict,
                         position_data: Dict) -> Dict:
        """
        Generate trading signals
        
        Signals (exact Pine Script):
        
        Mean Reversion:
        - long_setup = at_lower_2sd AND high_volume AND bullish_slope AND strong_level
        - long_entry = long_setup AND crossover(close, vwap - stdev)
        - short_setup = at_upper_2sd AND high_volume AND bearish_slope AND strong_level
        - short_entry = short_setup AND crossunder(close, vwap + stdev)
        
        Breakout:
        - breakout_long = accepted_above AND crossover(close, vwap + 2*stdev) AND high_volume
        - breakout_short = accepted_below AND crossunder(close, vwap - 2*stdev) AND high_volume
        """
        if len(df) < 2:
            return self._empty_signals()
        
        # Current values
        close = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        
        vwap = vwap_data['vwap'].iloc[-1]
        lower_1sd = vwap_data['lower_1sd'].iloc[-1]
        upper_1sd = vwap_data['upper_1sd'].iloc[-1]
        lower_2sd = vwap_data['lower_2sd'].iloc[-1]
        upper_2sd = vwap_data['upper_2sd'].iloc[-1]
        
        # Crossover/crossunder detection
        cross_above_lower_band = close > lower_1sd and close_prev <= lower_1sd
        cross_below_upper_band = close < upper_1sd and close_prev >= upper_1sd
        cross_above_upper_2sd = close > upper_2sd and close_prev <= upper_2sd
        cross_below_lower_2sd = close < lower_2sd and close_prev >= lower_2sd
        
        # Mean reversion signals
        long_setup = (position_data['at_lower_2sd'] and 
                     volume_data['is_high_volume'] and 
                     slope_data['bullish'] and 
                     rejection_data['is_strong'])
        
        long_entry = long_setup and cross_above_lower_band
        
        short_setup = (position_data['at_upper_2sd'] and 
                      volume_data['is_high_volume'] and 
                      slope_data['bearish'] and 
                      rejection_data['is_strong'])
        
        short_entry = short_setup and cross_below_upper_band
        
        # Breakout signals
        breakout_long = (zone_data['accepted_above'] and 
                        cross_above_upper_2sd and 
                        volume_data['is_high_volume'])
        
        breakout_short = (zone_data['accepted_below'] and 
                         cross_below_lower_2sd and 
                         volume_data['is_high_volume'])
        
        return {
            'long_setup': bool(long_setup),
            'long_entry': bool(long_entry),
            'short_setup': bool(short_setup),
            'short_entry': bool(short_entry),
            'breakout_long': bool(breakout_long),
            'breakout_short': bool(breakout_short)
        }
    
    def _empty_signals(self) -> Dict:
        """Return empty signals"""
        return {
            'long_setup': False,
            'long_entry': False,
            'short_setup': False,
            'short_entry': False,
            'breakout_long': False,
            'breakout_short': False
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "error": reason,
            "vwap": None,
            "stdev": None,
            "bands": {
                "upper_1sd": None,
                "lower_1sd": None,
                "upper_2sd": None,
                "lower_2sd": None
            },
            "volume": {
                "current": 0,
                "avg_20": 0,
                "is_high_volume": False
            },
            "rejection_count": 0,
            "is_strong_level": False,
            "zone": {
                "current": "NEUTRAL",
                "bars_in_zone": 0,
                "accepted_above": False,
                "accepted_below": False
            },
            "vwap_slope": {
                "value": 0.0,
                "bullish": False,
                "bearish": False
            },
            "position": {
                "stdev_distance": 0.0,
                "at_upper_2sd": False,
                "at_lower_2sd": False
            },
            "signals": self._empty_signals()
        }
