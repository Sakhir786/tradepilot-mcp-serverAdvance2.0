"""
Layer 9: Multi-Timeframe Confirmation Engine
MTF SuperTrend Alignment with Weighted Scoring
Converted from Pine Script - Logic unchanged
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Layer9Confirmation:
    """
    Professional Multi-Timeframe Confirmation system.
    
    Features:
    - Multi-timeframe SuperTrend alignment (5 timeframes)
    - Weighted scoring system (configurable)
    - SuperTrend direction per timeframe
    - ADX calculation per timeframe
    - Alignment score (0-100%)
    - Confidence adjustments (+20 to -20)
    - Bull/bear counting
    - ADX filtering (optional)
    """
    
    def __init__(self):
        # Timeframe Settings
        self.timeframes = {
            '5': '5min',
            '15': '15min', 
            '60': '1h',
            '240': '4h',
            'D': '1D'
        }
        
        # Default Weights (%)
        self.weights = {
            '5': 25,
            '15': 25,
            '60': 25,
            '240': 15,
            'D': 10
        }
        
        # Display Settings
        self.show_timeframes = {
            '5': True,
            '15': True,
            '60': True,
            '240': True,
            'D': True
        }
        
        # SuperTrend Settings
        self.st_factor = 3.0
        self.st_period = 10
        
        # ADX Settings
        self.adx_length = 14
        self.adx_threshold = 20
        self.use_adx_filter = False
        
        # Current timeframe (will be set during analysis)
        self.current_tf = None
        self.current_st_dir = None
    
    def analyze(self, df: pd.DataFrame, current_timeframe: str = '5') -> Dict:
        """
        Run multi-timeframe confirmation analysis
        
        Args:
            df: DataFrame with OHLCV data
            current_timeframe: Current timeframe ('5', '15', '60', '240', 'D')
            
        Returns:
            Dict with MTF alignment data
        """
        if len(df) < 50:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        self.current_tf = current_timeframe
        
        # Calculate SuperTrend and ADX for current timeframe
        current_st_line, current_st_dir = self._calculate_supertrend(df)
        current_adx = self._calculate_adx(df)
        self.current_st_dir = current_st_dir.iloc[-1]
        
        # Get multi-timeframe data
        mtf_data = self._get_mtf_data(df)
        
        # Calculate alignment score
        alignment_result = self._calculate_alignment(mtf_data)
        
        # Calculate confidence adjustment
        conf_adj = self._calculate_confidence_adjustment(alignment_result['alignment_score'])
        
        return {
            "current_timeframe": current_timeframe,
            "current_st_direction": int(self.current_st_dir),
            "current_adx": float(current_adx.iloc[-1]),
            "mtf_data": mtf_data,
            "alignment_score": alignment_result['alignment_score'],
            "bull_count": alignment_result['bull_count'],
            "bear_count": alignment_result['bear_count'],
            "confidence_adjustment": conf_adj,
            "signal": self._generate_signal(alignment_result['alignment_score']),
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== SUPERTREND CALCULATION ====================
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SuperTrend indicator
        
        Pine Script: ta.supertrend(factor, period)
        Returns: [line, direction]
        Direction: < 0 = bullish, >= 0 = bearish
        
        Python representation:
        direction: 1 = bullish, -1 = bearish
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate ATR
        atr = self._calculate_atr(df, self.st_period)
        
        # Calculate basic bands
        hl2 = (high + low) / 2
        
        upperband = hl2 + (self.st_factor * atr)
        lowerband = hl2 - (self.st_factor * atr)
        
        # Initialize arrays
        final_upperband = np.zeros(len(df))
        final_lowerband = np.zeros(len(df))
        supertrend = np.zeros(len(df))
        direction = np.zeros(len(df))
        
        # First values
        final_upperband[0] = upperband[0]
        final_lowerband[0] = lowerband[0]
        supertrend[0] = final_upperband[0]
        direction[0] = -1  # Start bearish
        
        # Calculate SuperTrend
        for i in range(1, len(df)):
            # Upper band
            if upperband[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
                final_upperband[i] = upperband[i]
            else:
                final_upperband[i] = final_upperband[i-1]
            
            # Lower band
            if lowerband[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]
            
            # SuperTrend line and direction
            if supertrend[i-1] == final_upperband[i-1] and close[i] <= final_upperband[i]:
                supertrend[i] = final_upperband[i]
                direction[i] = -1  # Bearish
            elif supertrend[i-1] == final_upperband[i-1] and close[i] > final_upperband[i]:
                supertrend[i] = final_lowerband[i]
                direction[i] = 1  # Bullish
            elif supertrend[i-1] == final_lowerband[i-1] and close[i] >= final_lowerband[i]:
                supertrend[i] = final_lowerband[i]
                direction[i] = 1  # Bullish
            elif supertrend[i-1] == final_lowerband[i-1] and close[i] < final_lowerband[i]:
                supertrend[i] = final_upperband[i]
                direction[i] = -1  # Bearish
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
        
        st_line = pd.Series(supertrend, index=df.index)
        st_dir = pd.Series(direction, index=df.index)
        
        return st_line, st_dir
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> np.ndarray:
        """Calculate Average True Range"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.zeros(len(df))
        atr[0] = tr[0]
        
        for i in range(1, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    # ==================== ADX CALCULATION ====================
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate ADX (Average Directional Index)
        
        Pine Script: ta.dmi(length, length)
        Returns: [+DI, -DI, ADX]
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        period = self.adx_length
        
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
        
        return pd.Series(adx, index=df.index)
    
    # ==================== MULTI-TIMEFRAME DATA ====================
    
    def _get_mtf_data(self, df: pd.DataFrame) -> Dict:
        """
        Get SuperTrend direction and ADX for all timeframes
        
        Pine Script equivalent:
        st_dir_5m = request.security(syminfo.tickerid, tf_5m, f_supertrend_direction())
        adx_5m = request.security(syminfo.tickerid, tf_5m, f_adx())
        """
        mtf_data = {}
        
        for tf_key, tf_name in self.timeframes.items():
            if not self.show_timeframes[tf_key]:
                continue
            
            # Resample data to target timeframe
            resampled_df = self._resample_timeframe(df, tf_name)
            
            if resampled_df is None or len(resampled_df) < 50:
                mtf_data[tf_key] = {
                    'direction': None,
                    'adx': None,
                    'enabled': False
                }
                continue
            
            # Calculate SuperTrend and ADX
            st_line, st_dir = self._calculate_supertrend(resampled_df)
            adx = self._calculate_adx(resampled_df)
            
            mtf_data[tf_key] = {
                'direction': int(st_dir.iloc[-1]),
                'adx': float(adx.iloc[-1]),
                'enabled': True
            }
        
        return mtf_data
    
    def _resample_timeframe(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Resample DataFrame to target timeframe
        
        Timeframe format: '5min', '15min', '1h', '4h', '1D'
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        
        try:
            # Resample OHLCV data
            resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
        except Exception as e:
            return None
    
    # ==================== ALIGNMENT CALCULATION ====================
    
    def _calculate_alignment(self, mtf_data: Dict) -> Dict:
        """
        Calculate weighted alignment score
        
        Pine Script logic:
        - For each timeframe: if direction matches current, add weight to weighted_sum
        - alignment_score = (weighted_sum / total_weight) * 100
        - Count bulls and bears
        """
        weighted_sum = 0.0
        total_weight = 0.0
        bull_count = 0
        bear_count = 0
        
        for tf_key in self.timeframes.keys():
            if tf_key not in mtf_data or not mtf_data[tf_key]['enabled']:
                continue
            
            direction = mtf_data[tf_key]['direction']
            
            if direction is None:
                continue
            
            # Add to weighted sum if direction matches current timeframe
            if direction == self.current_st_dir:
                weighted_sum += self.weights[tf_key]
            
            total_weight += self.weights[tf_key]
            
            # Count bulls and bears
            if direction == 1:
                bull_count += 1
            else:
                bear_count += 1
        
        # Calculate alignment score
        alignment_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 0
        
        return {
            'alignment_score': alignment_score,
            'bull_count': bull_count,
            'bear_count': bear_count,
            'weighted_sum': weighted_sum,
            'total_weight': total_weight
        }
    
    # ==================== CONFIDENCE ADJUSTMENT ====================
    
    def _calculate_confidence_adjustment(self, alignment_score: float) -> int:
        """
        Calculate confidence adjustment based on alignment score
        
        Pine Script formula:
        ≥80%: +20
        ≥60%: +10
        ≥40%: 0
        ≥20%: -10
        <20%: -20
        """
        if alignment_score >= 80:
            return 20
        elif alignment_score >= 60:
            return 10
        elif alignment_score >= 40:
            return 0
        elif alignment_score >= 20:
            return -10
        else:
            return -20
    
    def _generate_signal(self, alignment_score: float) -> str:
        """Generate trading signal based on alignment score"""
        if alignment_score >= 80:
            signal = "STRONG_ALIGN"
        elif alignment_score >= 60:
            signal = "GOOD_ALIGN"
        elif alignment_score >= 40:
            signal = "MODERATE_ALIGN"
        elif alignment_score >= 20:
            signal = "WEAK_ALIGN"
        else:
            signal = "NO_ALIGN"
        
        return signal
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "error": reason,
            "current_timeframe": None,
            "current_st_direction": 0,
            "current_adx": 0,
            "mtf_data": {},
            "alignment_score": 0,
            "bull_count": 0,
            "bear_count": 0,
            "confidence_adjustment": 0,
            "signal": "NO_ALIGN"
        }
