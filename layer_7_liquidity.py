"""
Layer 7: Liquidity Engine
Liquidity Sweeps, Stop Hunts, and Grab Detection
Converted from 3 Pine Script files - Logic unchanged
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Layer7Liquidity:
    """
    Professional Liquidity analysis with sweep detection and stop hunt identification.
    
    Features:
    - Pivot-based liquidity sweep detection (LuxAlgo method)
    - ICT liquidity concepts (buy/sell side)
    - Stop hunt detection with strength scoring (0-100%)
    - Multiple detection modes (wicks, outbreaks, both)
    - State tracking (broken, mitigated, taken, swept)
    - Volume confirmation with spike detection
    - Trend filters (EMA, ADX)
    - Round number detection
    - Equal highs/lows detection
    - Success rate tracking
    - Quality scoring for all patterns
    """
    
    def __init__(self):
        # LuxAlgo Settings
        self.swing_length = 5
        self.detection_mode = 'both'  # 'wicks', 'outbreaks', 'both'
        self.extend_zones = True
        self.max_bars = 300
        
        # ICT Settings
        self.volume_mult = 1.5
        self.trend_length = 50
        
        # Stop Hunt Settings
        self.lookback_bars = 7
        self.min_retrace = 0.2  # 20%
        self.use_close_confirmation = True
        self.use_volume = False
        self.vol_length = 20
        self.use_trend_filter = False
        self.ema_length = 50
        self.use_adx_filter = False
        self.adx_length = 14
        self.adx_threshold = 25
        self.use_round_numbers = True
        self.round_increment = 5.0
        self.detect_equal_levels = True
        self.equal_tolerance = 0.15  # 0.15%
        
        # State tracking
        self.pivot_highs = []
        self.pivot_lows = []
        self.sweep_zones = []
        self.liquidity_grabs = []
        
        # Statistics
        self.total_bull_sweeps = 0
        self.total_bear_sweeps = 0
        self.successful_bull_grabs = 0
        self.successful_bear_grabs = 0
        self.total_bull_grabs = 0
        self.total_bear_grabs = 0
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run complete liquidity analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with liquidity sweeps, stop hunts, and statistics
        """
        if len(df) < 50:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Calculate indicators
        trend_ema = df['close'].ewm(span=self.trend_length, adjust=False).mean()
        is_uptrend = df['close'].iloc[-1] > trend_ema.iloc[-1]
        avg_volume = df['volume'].rolling(window=self.vol_length).mean()
        
        # Calculate ADX
        adx_data = self._calculate_adx(df, self.adx_length)
        
        # Detect pivots
        pivot_data = self._detect_pivots(df)
        
        # LuxAlgo liquidity sweeps
        luxalgo_sweeps = self._detect_luxalgo_sweeps(
            df, pivot_data, avg_volume
        )
        
        # ICT liquidity concepts
        ict_liquidity = self._detect_ict_liquidity(
            df, pivot_data, avg_volume, is_uptrend
        )
        
        # Stop hunt detection
        stop_hunts = self._detect_stop_hunts(
            df, pivot_data, avg_volume, trend_ema, is_uptrend, adx_data
        )
        
        # Calculate statistics
        stats = self._calculate_statistics(stop_hunts)
        
        return {
            "luxalgo_sweeps": luxalgo_sweeps,
            "ict_liquidity": ict_liquidity,
            "stop_hunts": stop_hunts,
            "statistics": stats,
            "trend_ema": float(trend_ema.iloc[-1]),
            "is_uptrend": is_uptrend,
            "adx": float(adx_data['adx'].iloc[-1]) if len(adx_data['adx']) > 0 else 0,
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== PIVOT DETECTION ====================
    
    def _detect_pivots(self, df: pd.DataFrame) -> Dict:
        """
        Detect pivot highs and lows
        
        Pine Script logic:
        - ph = ta.pivothigh(len, len)
        - pl = ta.pivotlow(len, len)
        """
        high = df['high'].values
        low = df['low'].values
        
        pivot_highs = []
        pivot_lows = []
        pivot_high_indices = []
        pivot_low_indices = []
        
        # Detect pivot highs
        for i in range(self.swing_length, len(df) - self.swing_length):
            is_pivot_high = True
            center_high = high[i]
            
            for j in range(1, self.swing_length + 1):
                if high[i - j] >= center_high or high[i + j] >= center_high:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                pivot_highs.append(center_high)
                pivot_high_indices.append(i)
        
        # Detect pivot lows
        for i in range(self.swing_length, len(df) - self.swing_length):
            is_pivot_low = True
            center_low = low[i]
            
            for j in range(1, self.swing_length + 1):
                if low[i - j] <= center_low or low[i + j] <= center_low:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                pivot_lows.append(center_low)
                pivot_low_indices.append(i)
        
        return {
            "pivot_highs": pivot_highs,
            "pivot_lows": pivot_lows,
            "pivot_high_indices": pivot_high_indices,
            "pivot_low_indices": pivot_low_indices,
            "last_ph": pivot_highs[-1] if len(pivot_highs) > 0 else None,
            "last_pl": pivot_lows[-1] if len(pivot_lows) > 0 else None,
            "last_ph_index": pivot_high_indices[-1] if len(pivot_high_indices) > 0 else None,
            "last_pl_index": pivot_low_indices[-1] if len(pivot_low_indices) > 0 else None
        }
    
    # ==================== LUXALGO LIQUIDITY SWEEPS ====================
    
    def _detect_luxalgo_sweeps(self, df: pd.DataFrame, pivot_data: Dict,
                                avg_volume: pd.Series) -> Dict:
        """
        Detect liquidity sweeps using LuxAlgo methodology
        
        Pine Script logic:
        - 3 modes: Only Wicks, Only Outbreaks & Retest, Both
        - States: broken, mitigated, taken, wick
        - Wick detection: high > pivot but close < pivot
        - Outbreak detection: close > pivot (broken), then close < pivot (taken)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        current_index = len(df) - 1
        
        bull_sweep_detected = False
        bear_sweep_detected = False
        bull_sweep_type = ""
        bear_sweep_type = ""
        bull_sweep_zone = {"top": 0.0, "bottom": 0.0}
        bear_sweep_zone = {"top": 0.0, "bottom": 0.0}
        
        last_ph = pivot_data['last_ph']
        last_pl = pivot_data['last_pl']
        
        # Bearish sweep detection (sweep above pivot high)
        if last_ph is not None:
            # Wick detection (wicks or both mode)
            if self.detection_mode in ['wicks', 'both']:
                if high[-1] > last_ph and close[-1] < last_ph:
                    bear_sweep_detected = True
                    bear_sweep_type = "WICK"
                    bear_sweep_zone = {
                        "top": high[-1],
                        "bottom": last_ph
                    }
                    self.total_bear_sweeps += 1
            
            # Outbreak detection (outbreaks or both mode)
            if self.detection_mode in ['outbreaks', 'both']:
                # Check if previously broken and now retesting
                if close[-2] > last_ph and close[-1] < last_ph:
                    if high[-1] > last_ph:
                        bear_sweep_detected = True
                        bear_sweep_type = "OUTBREAK_RETEST"
                        bear_sweep_zone = {
                            "top": high[-1],
                            "bottom": last_ph
                        }
                        self.total_bear_sweeps += 1
        
        # Bullish sweep detection (sweep below pivot low)
        if last_pl is not None:
            # Wick detection (wicks or both mode)
            if self.detection_mode in ['wicks', 'both']:
                if low[-1] < last_pl and close[-1] > last_pl:
                    bull_sweep_detected = True
                    bull_sweep_type = "WICK"
                    bull_sweep_zone = {
                        "top": last_pl,
                        "bottom": low[-1]
                    }
                    self.total_bull_sweeps += 1
            
            # Outbreak detection (outbreaks or both mode)
            if self.detection_mode in ['outbreaks', 'both']:
                # Check if previously broken and now retesting
                if close[-2] < last_pl and close[-1] > last_pl:
                    if low[-1] < last_pl:
                        bull_sweep_detected = True
                        bull_sweep_type = "OUTBREAK_RETEST"
                        bull_sweep_zone = {
                            "top": last_pl,
                            "bottom": low[-1]
                        }
                        self.total_bull_sweeps += 1
        
        return {
            "bull_sweep_detected": bull_sweep_detected,
            "bear_sweep_detected": bear_sweep_detected,
            "bull_sweep_type": bull_sweep_type,
            "bear_sweep_type": bear_sweep_type,
            "bull_sweep_zone": bull_sweep_zone,
            "bear_sweep_zone": bear_sweep_zone,
            "total_bull_sweeps": self.total_bull_sweeps,
            "total_bear_sweeps": self.total_bear_sweeps
        }
    
    # ==================== ICT LIQUIDITY CONCEPTS ====================
    
    def _detect_ict_liquidity(self, df: pd.DataFrame, pivot_data: Dict,
                              avg_volume: pd.Series, is_uptrend: bool) -> Dict:
        """
        Detect ICT-style liquidity sweeps
        
        Pine Script logic:
        - Buy liquidity: close > pivot_high AND volume > avg * mult
        - Sell liquidity: close < pivot_low AND volume > avg * mult
        """
        close = df['close'].values
        volume = df['volume'].values
        
        last_ph = pivot_data['last_ph']
        last_pl = pivot_data['last_pl']
        last_ph_index = pivot_data['last_ph_index']
        last_pl_index = pivot_data['last_pl_index']
        
        buy_liq_detected = False
        sell_liq_detected = False
        buy_liq_level = 0.0
        sell_liq_level = 0.0
        
        avg_vol_val = avg_volume.iloc[-1]
        
        # Buy-side liquidity sweep (sweep above pivot high)
        if last_ph is not None:
            if close[-1] > last_ph and volume[-1] > avg_vol_val * self.volume_mult:
                buy_liq_detected = True
                buy_liq_level = last_ph
        
        # Sell-side liquidity sweep (sweep below pivot low)
        if last_pl is not None:
            if close[-1] < last_pl and volume[-1] > avg_vol_val * self.volume_mult:
                sell_liq_detected = True
                sell_liq_level = last_pl
        
        return {
            "buy_liq_detected": buy_liq_detected,
            "sell_liq_detected": sell_liq_detected,
            "buy_liq_level": float(buy_liq_level),
            "sell_liq_level": float(sell_liq_level)
        }
    
    # ==================== STOP HUNT DETECTION ====================
    
    def _detect_stop_hunts(self, df: pd.DataFrame, pivot_data: Dict,
                          avg_volume: pd.Series, trend_ema: pd.Series,
                          is_uptrend: bool, adx_data: Dict) -> Dict:
        """
        Detect liquidity grabs (stop hunts) with strength scoring
        
        Pine Script logic:
        - Lookback for previous high/low
        - Detect wick above/below with retrace confirmation
        - Multiple filters: volume, EMA trend, ADX
        - Round number detection
        - Equal highs/lows detection
        - Strength calculation: 0-100%
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        # Get previous high/low
        prev_high = np.max(high[-self.lookback_bars-1:-1])
        prev_low = np.min(low[-self.lookback_bars-1:-1])
        
        # Current values
        current_high = high[-1]
        current_low = low[-1]
        current_close = close[-1]
        current_volume = volume[-1]
        
        # Filters
        avg_vol_val = avg_volume.iloc[-1]
        volume_spike = current_volume > avg_vol_val * self.volume_mult
        volume_condition = volume_spike if self.use_volume else True
        
        trend_up = current_close > trend_ema.iloc[-1]
        trend_down = current_close < trend_ema.iloc[-1]
        trend_condition_bull = trend_up if self.use_trend_filter else True
        trend_condition_bear = trend_down if self.use_trend_filter else True
        
        adx_val = adx_data['adx'].iloc[-1] if len(adx_data['adx']) > 0 else 0
        adx_condition = adx_val > self.adx_threshold if self.use_adx_filter else True
        
        # Bearish grab detection
        bearish_grab = False
        bear_retrace = 0.0
        bear_level = 0.0
        bear_strength = 0.0
        bear_type = ""
        bear_is_round = False
        bear_is_equal = False
        
        if current_high > prev_high:
            bear_level = prev_high
            wick_above = current_high - prev_high
            
            if self.use_close_confirmation:
                retrace_amount = prev_high - current_close
                bear_retrace = min(retrace_amount / wick_above, 1.0) if wick_above > 0 else 0.0
                
                if (bear_retrace >= self.min_retrace and 
                    volume_condition and adx_condition and trend_condition_bear):
                    bearish_grab = True
            else:
                if volume_condition and adx_condition and trend_condition_bear:
                    bearish_grab = True
                    bear_retrace = 0.5
            
            if bearish_grab:
                # Round number detection
                bear_is_round = self._is_round_number(bear_level)
                
                # Equal level detection
                bear_is_equal, equal_count = self._detect_equal_level(
                    df, bear_level, True, current_index=len(df)-1
                )
                
                # Calculate strength
                bear_strength = self._calculate_strength(
                    bear_retrace, volume_condition, bear_is_round,
                    bear_is_equal, trend_condition_bear
                )
                
                # Determine type
                if bear_is_equal:
                    bear_type = "EQUAL_HIGH"
                elif bear_is_round:
                    bear_type = "ROUND_NUMBER"
                else:
                    bear_type = "RESISTANCE"
                
                self.total_bear_grabs += 1
        
        # Bullish grab detection
        bullish_grab = False
        bull_retrace = 0.0
        bull_level = 0.0
        bull_strength = 0.0
        bull_type = ""
        bull_is_round = False
        bull_is_equal = False
        
        if current_low < prev_low:
            bull_level = prev_low
            wick_below = prev_low - current_low
            
            if self.use_close_confirmation:
                retrace_amount = current_close - prev_low
                bull_retrace = min(retrace_amount / wick_below, 1.0) if wick_below > 0 else 0.0
                
                if (bull_retrace >= self.min_retrace and 
                    volume_condition and adx_condition and trend_condition_bull):
                    bullish_grab = True
            else:
                if volume_condition and adx_condition and trend_condition_bull:
                    bullish_grab = True
                    bull_retrace = 0.5
            
            if bullish_grab:
                # Round number detection
                bull_is_round = self._is_round_number(bull_level)
                
                # Equal level detection
                bull_is_equal, equal_count = self._detect_equal_level(
                    df, bull_level, False, current_index=len(df)-1
                )
                
                # Calculate strength
                bull_strength = self._calculate_strength(
                    bull_retrace, volume_condition, bull_is_round,
                    bull_is_equal, trend_condition_bull
                )
                
                # Determine type
                if bull_is_equal:
                    bull_type = "EQUAL_LOW"
                elif bull_is_round:
                    bull_type = "ROUND_NUMBER"
                else:
                    bull_type = "SUPPORT"
                
                self.total_bull_grabs += 1
        
        return {
            "bullish_grab": bullish_grab,
            "bearish_grab": bearish_grab,
            "bull_level": float(bull_level),
            "bear_level": float(bear_level),
            "bull_strength": bull_strength,
            "bear_strength": bear_strength,
            "bull_retrace": bull_retrace,
            "bear_retrace": bear_retrace,
            "bull_type": bull_type,
            "bear_type": bear_type,
            "bull_is_round": bull_is_round,
            "bear_is_round": bear_is_round,
            "bull_is_equal": bull_is_equal,
            "bear_is_equal": bear_is_equal,
            "total_bull_grabs": self.total_bull_grabs,
            "total_bear_grabs": self.total_bear_grabs
        }
    
    # ==================== HELPER FUNCTIONS ====================
    
    def _is_round_number(self, price: float) -> bool:
        """
        Check if price is near a round number
        
        Pine Script logic:
        - remainder = price % roundIncrement
        - tolerance = roundIncrement * 0.1
        - is_round = remainder < tolerance OR remainder > (increment - tolerance)
        """
        if not self.use_round_numbers:
            return False
        
        remainder = price % self.round_increment
        tolerance = self.round_increment * 0.1
        
        return remainder < tolerance or remainder > (self.round_increment - tolerance)
    
    def _detect_equal_level(self, df: pd.DataFrame, current_level: float,
                           is_high: bool, current_index: int) -> Tuple[bool, int]:
        """
        Detect if current level matches previous highs/lows (equal levels)
        
        Pine Script logic:
        - Look back 50 bars
        - Count matches within tolerance
        - Equal if count >= 2
        """
        if not self.detect_equal_levels:
            return False, 0
        
        count = 0
        tolerance = current_level * (self.equal_tolerance / 100)
        
        high = df['high'].values
        low = df['low'].values
        
        lookback = min(50, current_index)
        
        for i in range(1, lookback + 1):
            idx = current_index - i
            if idx < 0:
                break
            
            if is_high:
                if abs(high[idx] - current_level) < tolerance:
                    count += 1
            else:
                if abs(low[idx] - current_level) < tolerance:
                    count += 1
        
        return count >= 2, count
    
    def _calculate_strength(self, retrace_pct: float, vol_confirmed: bool,
                           is_round: bool, is_equal: bool,
                           trend_aligned: bool) -> float:
        """
        Calculate grab strength (0-100%)
        
        Pine Script formula:
        - Base strength: retrace_pct * 30
        - Volume confirmed: +25
        - Round number: +15
        - Equal level: +20
        - Trend aligned: +10
        - Max: 100
        """
        strength = 0.0
        
        strength += retrace_pct * 30
        strength += 25 if vol_confirmed else 0
        strength += 15 if is_round else 0
        strength += 20 if is_equal else 0
        strength += 10 if trend_aligned else 0
        
        return min(strength, 100)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> Dict:
        """
        Calculate ADX (Average Directional Index)
        
        Pine Script: ta.dmi(length, length)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Calculate +DM and -DM
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smooth with RMA (Wilder's smoothing)
        atr = np.zeros(len(df))
        smooth_plus_dm = np.zeros(len(df))
        smooth_minus_dm = np.zeros(len(df))
        
        atr[period-1] = np.mean(tr[:period])
        smooth_plus_dm[period-1] = np.mean(plus_dm[:period])
        smooth_minus_dm[period-1] = np.mean(minus_dm[:period])
        
        for i in range(period, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            smooth_plus_dm[i] = (smooth_plus_dm[i-1] * (period - 1) + plus_dm[i]) / period
            smooth_minus_dm[i] = (smooth_minus_dm[i-1] * (period - 1) + minus_dm[i]) / period
        
        # Calculate +DI and -DI
        plus_di = 100 * smooth_plus_dm / atr
        minus_di = 100 * smooth_minus_dm / atr
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.nan_to_num(dx)
        
        # Calculate ADX (smoothed DX)
        adx = np.zeros(len(df))
        adx[period*2-2] = np.mean(dx[period-1:period*2-1])
        
        for i in range(period*2-1, len(df)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return {
            'plus_di': pd.Series(plus_di, index=df.index),
            'minus_di': pd.Series(minus_di, index=df.index),
            'adx': pd.Series(adx, index=df.index)
        }
    
    def _calculate_statistics(self, stop_hunts: Dict) -> Dict:
        """Calculate success rates and statistics"""
        total_grabs = self.total_bull_grabs + self.total_bear_grabs
        
        bull_success_rate = 0.0
        bear_success_rate = 0.0
        
        if self.total_bull_grabs > 0:
            bull_success_rate = (self.successful_bull_grabs / self.total_bull_grabs) * 100
        
        if self.total_bear_grabs > 0:
            bear_success_rate = (self.successful_bear_grabs / self.total_bear_grabs) * 100
        
        overall_success_rate = 0.0
        if total_grabs > 0:
            total_successful = self.successful_bull_grabs + self.successful_bear_grabs
            overall_success_rate = (total_successful / total_grabs) * 100
        
        return {
            "total_bull_grabs": self.total_bull_grabs,
            "total_bear_grabs": self.total_bear_grabs,
            "total_grabs": total_grabs,
            "successful_bull_grabs": self.successful_bull_grabs,
            "successful_bear_grabs": self.successful_bear_grabs,
            "bull_success_rate": bull_success_rate,
            "bear_success_rate": bear_success_rate,
            "overall_success_rate": overall_success_rate
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "error": reason,
            "luxalgo_sweeps": {
                "bull_sweep_detected": False,
                "bear_sweep_detected": False,
                "total_bull_sweeps": 0,
                "total_bear_sweeps": 0
            },
            "ict_liquidity": {
                "buy_liq_detected": False,
                "sell_liq_detected": False
            },
            "stop_hunts": {
                "bullish_grab": False,
                "bearish_grab": False,
                "total_bull_grabs": 0,
                "total_bear_grabs": 0
            },
            "statistics": {
                "total_grabs": 0,
                "overall_success_rate": 0
            }
        }
