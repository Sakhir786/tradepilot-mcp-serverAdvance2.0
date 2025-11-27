"""
Layer 9: Multi-Timeframe Confirmation Engine (Raw Data Output)
MTF SuperTrend Alignment with Weighted Scoring
Outputs RAW MTF data only - no scores, no signals
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
    - SuperTrend direction per timeframe
    - ADX calculation per timeframe
    - Bull/bear counting
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
            Dict with RAW MTF alignment data
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
        
        # Calculate raw alignment metrics
        alignment_metrics = self._calculate_raw_alignment(mtf_data)
        
        # Return RAW DATA ONLY - no scores, no signals
        return {
            # Current Timeframe Data
            "current_timeframe": current_timeframe,
            "current_st_direction": int(self.current_st_dir),
            "current_st_bullish": self.current_st_dir == 1,
            "current_st_bearish": self.current_st_dir == -1,
            "current_adx": round(float(current_adx.iloc[-1]), 2),
            "current_st_line": round(float(current_st_line.iloc[-1]), 2),
            
            # Per-Timeframe Data
            "tf_5min_direction": mtf_data.get('5', {}).get('direction'),
            "tf_5min_bullish": mtf_data.get('5', {}).get('direction') == 1,
            "tf_5min_adx": mtf_data.get('5', {}).get('adx'),
            "tf_5min_weight": self.weights['5'],
            
            "tf_15min_direction": mtf_data.get('15', {}).get('direction'),
            "tf_15min_bullish": mtf_data.get('15', {}).get('direction') == 1,
            "tf_15min_adx": mtf_data.get('15', {}).get('adx'),
            "tf_15min_weight": self.weights['15'],
            
            "tf_1h_direction": mtf_data.get('60', {}).get('direction'),
            "tf_1h_bullish": mtf_data.get('60', {}).get('direction') == 1,
            "tf_1h_adx": mtf_data.get('60', {}).get('adx'),
            "tf_1h_weight": self.weights['60'],
            
            "tf_4h_direction": mtf_data.get('240', {}).get('direction'),
            "tf_4h_bullish": mtf_data.get('240', {}).get('direction') == 1,
            "tf_4h_adx": mtf_data.get('240', {}).get('adx'),
            "tf_4h_weight": self.weights['240'],
            
            "tf_1d_direction": mtf_data.get('D', {}).get('direction'),
            "tf_1d_bullish": mtf_data.get('D', {}).get('direction') == 1,
            "tf_1d_adx": mtf_data.get('D', {}).get('adx'),
            "tf_1d_weight": self.weights['D'],
            
            # Alignment Metrics (raw counts)
            "bull_count": alignment_metrics['bull_count'],
            "bear_count": alignment_metrics['bear_count'],
            "total_timeframes": alignment_metrics['total_timeframes'],
            "aligned_with_current_count": alignment_metrics['aligned_count'],
            "not_aligned_count": alignment_metrics['not_aligned_count'],
            
            # Weighted Alignment (raw calculation, no interpretation)
            "weighted_aligned": alignment_metrics['weighted_aligned'],
            "weighted_not_aligned": alignment_metrics['weighted_not_aligned'],
            "total_weight": alignment_metrics['total_weight'],
            "alignment_pct": alignment_metrics['alignment_pct'],
            
            # Higher Timeframe Context
            "htf_bullish_count": alignment_metrics['htf_bullish_count'],
            "htf_bearish_count": alignment_metrics['htf_bearish_count'],
            "htf_aligned": alignment_metrics['htf_aligned'],
            
            # Price Context
            "current_price": round(df["close"].iloc[-1], 2),
            
            # Timestamp
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== SUPERTREND CALCULATION ====================
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate SuperTrend indicator"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = self._calculate_atr(df, self.st_period)
        
        hl2 = (high + low) / 2
        
        upperband = hl2 + (self.st_factor * atr)
        lowerband = hl2 - (self.st_factor * atr)
        
        final_upperband = np.zeros(len(df))
        final_lowerband = np.zeros(len(df))
        supertrend = np.zeros(len(df))
        direction = np.zeros(len(df))
        
        final_upperband[0] = upperband[0]
        final_lowerband[0] = lowerband[0]
        supertrend[0] = final_upperband[0]
        direction[0] = -1
        
        for i in range(1, len(df)):
            if upperband[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
                final_upperband[i] = upperband[i]
            else:
                final_upperband[i] = final_upperband[i-1]
            
            if lowerband[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]
            
            if supertrend[i-1] == final_upperband[i-1] and close[i] <= final_upperband[i]:
                supertrend[i] = final_upperband[i]
                direction[i] = -1
            elif supertrend[i-1] == final_upperband[i-1] and close[i] > final_upperband[i]:
                supertrend[i] = final_lowerband[i]
                direction[i] = 1
            elif supertrend[i-1] == final_lowerband[i-1] and close[i] >= final_lowerband[i]:
                supertrend[i] = final_lowerband[i]
                direction[i] = 1
            elif supertrend[i-1] == final_lowerband[i-1] and close[i] < final_lowerband[i]:
                supertrend[i] = final_upperband[i]
                direction[i] = -1
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
        """Calculate ADX (Average Directional Index)"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        period = self.adx_length
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
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
        
        plus_di = np.divide(100 * smooth_plus_dm, atr, out=np.zeros_like(atr), where=atr!=0)
        minus_di = np.divide(100 * smooth_minus_dm, atr, out=np.zeros_like(atr), where=atr!=0)
        
        dx = np.divide(100 * np.abs(plus_di - minus_di), (plus_di + minus_di), out=np.zeros_like(plus_di), where=(plus_di + minus_di)!=0)
        dx = np.nan_to_num(dx)
        
        adx = np.zeros(len(df))
        adx[period*2-2] = np.mean(dx[period-1:period*2-1])
        
        for i in range(period*2-1, len(df)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return pd.Series(adx, index=df.index)
    
    # ==================== MULTI-TIMEFRAME DATA ====================
    
    def _get_mtf_data(self, df: pd.DataFrame) -> Dict:
        """Get SuperTrend direction and ADX for all timeframes"""
        mtf_data = {}
        
        for tf_key, tf_name in self.timeframes.items():
            if not self.show_timeframes[tf_key]:
                continue
            
            resampled_df = self._resample_timeframe(df, tf_name)
            
            if resampled_df is None or len(resampled_df) < 50:
                mtf_data[tf_key] = {
                    'direction': None,
                    'adx': None,
                    'enabled': False
                }
                continue
            
            st_line, st_dir = self._calculate_supertrend(resampled_df)
            adx = self._calculate_adx(resampled_df)
            
            mtf_data[tf_key] = {
                'direction': int(st_dir.iloc[-1]),
                'adx': round(float(adx.iloc[-1]), 2),
                'enabled': True
            }
        
        return mtf_data
    
    def _resample_timeframe(self, df: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """Resample DataFrame to target timeframe"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return None
        
        try:
            resampled = df.resample(timeframe).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
        except Exception:
            return None
    
    # ==================== RAW ALIGNMENT CALCULATION ====================
    
    def _calculate_raw_alignment(self, mtf_data: Dict) -> Dict:
        """Calculate raw alignment metrics - NO interpretation"""
        bull_count = 0
        bear_count = 0
        aligned_count = 0
        not_aligned_count = 0
        weighted_aligned = 0.0
        weighted_not_aligned = 0.0
        total_weight = 0.0
        total_timeframes = 0
        
        # Higher timeframe tracking (60, 240, D)
        htf_bullish_count = 0
        htf_bearish_count = 0
        
        for tf_key in self.timeframes.keys():
            if tf_key not in mtf_data or not mtf_data[tf_key].get('enabled', False):
                continue
            
            direction = mtf_data[tf_key]['direction']
            
            if direction is None:
                continue
            
            total_timeframes += 1
            weight = self.weights[tf_key]
            total_weight += weight
            
            # Count bulls and bears
            if direction == 1:
                bull_count += 1
            else:
                bear_count += 1
            
            # Count alignment with current
            if direction == self.current_st_dir:
                aligned_count += 1
                weighted_aligned += weight
            else:
                not_aligned_count += 1
                weighted_not_aligned += weight
            
            # Track higher timeframes
            if tf_key in ['60', '240', 'D']:
                if direction == 1:
                    htf_bullish_count += 1
                else:
                    htf_bearish_count += 1
        
        # Calculate alignment percentage
        alignment_pct = (weighted_aligned / total_weight * 100) if total_weight > 0 else 0
        
        # HTF alignment check
        htf_aligned = False
        if self.current_st_dir == 1:
            htf_aligned = htf_bullish_count >= 2
        else:
            htf_aligned = htf_bearish_count >= 2
        
        return {
            'bull_count': bull_count,
            'bear_count': bear_count,
            'aligned_count': aligned_count,
            'not_aligned_count': not_aligned_count,
            'weighted_aligned': round(weighted_aligned, 2),
            'weighted_not_aligned': round(weighted_not_aligned, 2),
            'total_weight': round(total_weight, 2),
            'total_timeframes': total_timeframes,
            'alignment_pct': round(alignment_pct, 2),
            'htf_bullish_count': htf_bullish_count,
            'htf_bearish_count': htf_bearish_count,
            'htf_aligned': htf_aligned
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "current_timeframe": None,
            "current_st_direction": None,
            "current_st_bullish": None,
            "current_st_bearish": None,
            "current_adx": None,
            "current_st_line": None,
            "tf_5min_direction": None,
            "tf_5min_bullish": None,
            "tf_5min_adx": None,
            "tf_5min_weight": None,
            "tf_15min_direction": None,
            "tf_15min_bullish": None,
            "tf_15min_adx": None,
            "tf_15min_weight": None,
            "tf_1h_direction": None,
            "tf_1h_bullish": None,
            "tf_1h_adx": None,
            "tf_1h_weight": None,
            "tf_4h_direction": None,
            "tf_4h_bullish": None,
            "tf_4h_adx": None,
            "tf_4h_weight": None,
            "tf_1d_direction": None,
            "tf_1d_bullish": None,
            "tf_1d_adx": None,
            "tf_1d_weight": None,
            "bull_count": 0,
            "bear_count": 0,
            "total_timeframes": 0,
            "aligned_with_current_count": 0,
            "not_aligned_count": 0,
            "weighted_aligned": 0,
            "weighted_not_aligned": 0,
            "total_weight": 0,
            "alignment_pct": 0,
            "htf_bullish_count": 0,
            "htf_bearish_count": 0,
            "htf_aligned": False,
            "current_price": None,
            "error": reason
        }
