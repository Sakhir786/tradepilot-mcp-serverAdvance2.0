"""
Layer 3: Divergence Engine (Raw Data Output)
MACD and RSI divergence detection (Regular + Hidden)
Outputs RAW divergence data only - no scores, no signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class Layer3Divergence:
    """
    Professional divergence detection using MACD and RSI.
    
    Features:
    - MACD Divergence (3 timeframes)
    - RSI Divergence (pivot-based)
    - Regular and Hidden divergences
    - Multi-timeframe analysis
    """
    
    def __init__(self):
        # MACD Multi-Timeframe Settings
        self.macd_configs = {
            'tf1': {'fast': 12, 'slow': 26, 'label': '1h'},
            'tf2': {'fast': 48, 'slow': 104, 'label': '4h'},
            'tf3': {'fast': 288, 'slow': 624, 'label': '1D'}
        }
        
        # RSI Divergence Settings
        self.rsi_period = 14
        self.pivot_lookback_left = 5
        self.pivot_lookback_right = 5
        self.range_lower = 5
        self.range_upper = 60
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run complete divergence analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with RAW MACD and RSI divergence data
        """
        if len(df) < 100:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Run MACD divergence detection
        macd_results = self._analyze_macd_divergences(df)
        
        # Run RSI divergence detection
        rsi_results = self._analyze_rsi_divergences(df)
        
        # Aggregate raw counts (no interpretation)
        summary = self._create_raw_summary(macd_results, rsi_results)
        
        # Return RAW DATA ONLY - no scores, no signals
        return {
            # MACD Divergence Data
            "macd_divergences": macd_results,
            
            # RSI Divergence Data
            "rsi_divergences": rsi_results,
            
            # Raw Counts Summary
            "divergence_counts": summary,
            
            # Latest Indicator Values (for context)
            "latest_rsi": rsi_results.get('latest_rsi'),
            "latest_macd_1h": macd_results.get('tf1', {}).get('latest_macd'),
            "latest_macd_4h": macd_results.get('tf2', {}).get('latest_macd'),
            "latest_macd_1d": macd_results.get('tf3', {}).get('latest_macd'),
            
            # Current Price Context
            "current_price": round(df["close"].iloc[-1], 2),
            
            # Timestamp
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== MACD DIVERGENCE DETECTION ====================
    
    def _analyze_macd_divergences(self, df: pd.DataFrame) -> Dict:
        """Detect MACD divergences across multiple timeframes"""
        results = {}
        
        for tf_key, config in self.macd_configs.items():
            results[tf_key] = self._detect_macd_divergence_single_tf(
                df, 
                config['fast'], 
                config['slow'],
                config['label']
            )
        
        return results
    
    def _detect_macd_divergence_single_tf(self, df: pd.DataFrame, fast: int, slow: int, label: str) -> Dict:
        """Detect MACD divergences for a single timeframe"""
        # Calculate MACD
        close = df['close'].values
        fast_ema = self._ema(close, fast)
        slow_ema = self._ema(close, slow)
        macd = fast_ema - slow_ema
        
        # Detect pivots
        indy_top = self._find_top_pivots(macd)
        indy_bot = self._find_bottom_pivots(macd)
        
        # Detect divergences
        divergences = {
            'regular_bearish': [],
            'hidden_bearish': [],
            'regular_bullish': [],
            'hidden_bullish': []
        }
        
        high = df['high'].values
        low = df['low'].values
        
        # Check last 20 bars for divergences
        for i in range(max(10, len(df) - 20), len(df)):
            if i < 5:  # Need at least 5 bars for lookback
                continue
            
            # BEARISH DIVERGENCES (at tops)
            if indy_top[i]:
                high_prev, high_price = self._get_previous_pivot_values(
                    indy_top, macd, high, i
                )
                
                if high_prev is not None:
                    # Regular Bearish: Price HH, MACD LH
                    if high[i-2] > high_price and macd[i-2] < high_prev:
                        divergences['regular_bearish'].append({
                            'index': i,
                            'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                            'price': float(high[i-2]),
                            'macd': float(macd[i-2]),
                            'type': 'Regular Bearish',
                            'timeframe': label
                        })
                    
                    # Hidden Bearish: Price LH, MACD HH
                    if high[i-2] < high_price and macd[i-2] > high_prev:
                        divergences['hidden_bearish'].append({
                            'index': i,
                            'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                            'price': float(high[i-2]),
                            'macd': float(macd[i-2]),
                            'type': 'Hidden Bearish',
                            'timeframe': label
                        })
            
            # BULLISH DIVERGENCES (at bottoms)
            if indy_bot[i]:
                low_prev, low_price = self._get_previous_pivot_values(
                    indy_bot, macd, low, i
                )
                
                if low_prev is not None:
                    # Regular Bullish: Price LL, MACD HL
                    if low[i-2] < low_price and macd[i-2] > low_prev:
                        divergences['regular_bullish'].append({
                            'index': i,
                            'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                            'price': float(low[i-2]),
                            'macd': float(macd[i-2]),
                            'type': 'Regular Bullish',
                            'timeframe': label
                        })
                    
                    # Hidden Bullish: Price HL, MACD LL
                    if low[i-2] > low_price and macd[i-2] < low_prev:
                        divergences['hidden_bullish'].append({
                            'index': i,
                            'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                            'price': float(low[i-2]),
                            'macd': float(macd[i-2]),
                            'type': 'Hidden Bullish',
                            'timeframe': label
                        })
        
        return {
            'timeframe': label,
            'fast_length': fast,
            'slow_length': slow,
            'divergences': divergences,
            'latest_macd': float(macd[-1]) if len(macd) > 0 else None
        }
    
    def _find_top_pivots(self, series: np.ndarray) -> np.ndarray:
        """
        Find top pivots (peaks) in series
        Logic: series[i-4] < series[i-2] and series[i-3] < series[i-2] 
               and series[i-2] > series[i-1] and series[i-2] > series[i]
        """
        pivots = np.zeros(len(series), dtype=bool)
        
        for i in range(4, len(series)):
            if (series[i-4] < series[i-2] and 
                series[i-3] < series[i-2] and 
                series[i-2] > series[i-1] and 
                series[i-2] > series[i]):
                pivots[i] = True
        
        return pivots
    
    def _find_bottom_pivots(self, series: np.ndarray) -> np.ndarray:
        """
        Find bottom pivots (troughs) in series
        Logic: series[i-4] > series[i-2] and series[i-3] > series[i-2] 
               and series[i-2] < series[i-1] and series[i-2] < series[i]
        """
        pivots = np.zeros(len(series), dtype=bool)
        
        for i in range(4, len(series)):
            if (series[i-4] > series[i-2] and 
                series[i-3] > series[i-2] and 
                series[i-2] < series[i-1] and 
                series[i-2] < series[i]):
                pivots[i] = True
        
        return pivots
    
    def _get_previous_pivot_values(self, pivots: np.ndarray, macd: np.ndarray, 
                                   price: np.ndarray, current_idx: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get the previous pivot's MACD and price values (valuewhen logic)
        """
        # Find previous pivot before current index
        for i in range(current_idx - 1, -1, -1):
            if pivots[i] and i >= 2:
                return macd[i-2], price[i-2]
        
        return None, None
    
    # ==================== RSI DIVERGENCE DETECTION ====================
    
    def _analyze_rsi_divergences(self, df: pd.DataFrame) -> Dict:
        """Detect RSI divergences using pivot analysis"""
        # Calculate RSI
        rsi = self._calculate_rsi(df['close'].values, self.rsi_period)
        
        # Find pivot highs and lows
        pivot_highs = self._find_pivots_high(rsi, self.pivot_lookback_left, self.pivot_lookback_right)
        pivot_lows = self._find_pivots_low(rsi, self.pivot_lookback_left, self.pivot_lookback_right)
        
        # Detect divergences
        divergences = {
            'regular_bullish': [],
            'hidden_bullish': [],
            'regular_bearish': [],
            'hidden_bearish': []
        }
        
        high = df['high'].values
        low = df['low'].values
        
        # Check for divergences at each pivot
        for i in range(len(df)):
            # BULLISH DIVERGENCES (at pivot lows)
            if pivot_lows[i]:
                prev_pivot_idx = self._find_previous_pivot(pivot_lows, i, self.range_lower, self.range_upper)
                
                if prev_pivot_idx is not None:
                    rsi_idx = i - self.pivot_lookback_right
                    prev_rsi_idx = prev_pivot_idx - self.pivot_lookback_right
                    
                    if rsi_idx >= 0 and prev_rsi_idx >= 0:
                        # Regular Bullish: Price LL, RSI HL
                        if low[rsi_idx] < low[prev_rsi_idx] and rsi[rsi_idx] > rsi[prev_rsi_idx]:
                            divergences['regular_bullish'].append({
                                'index': i,
                                'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                                'price': float(low[rsi_idx]),
                                'rsi': float(rsi[rsi_idx]),
                                'type': 'Regular Bullish'
                            })
                        
                        # Hidden Bullish: Price HL, RSI LL
                        if low[rsi_idx] > low[prev_rsi_idx] and rsi[rsi_idx] < rsi[prev_rsi_idx]:
                            divergences['hidden_bullish'].append({
                                'index': i,
                                'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                                'price': float(low[rsi_idx]),
                                'rsi': float(rsi[rsi_idx]),
                                'type': 'Hidden Bullish'
                            })
            
            # BEARISH DIVERGENCES (at pivot highs)
            if pivot_highs[i]:
                prev_pivot_idx = self._find_previous_pivot(pivot_highs, i, self.range_lower, self.range_upper)
                
                if prev_pivot_idx is not None:
                    rsi_idx = i - self.pivot_lookback_right
                    prev_rsi_idx = prev_pivot_idx - self.pivot_lookback_right
                    
                    if rsi_idx >= 0 and prev_rsi_idx >= 0:
                        # Regular Bearish: Price HH, RSI LH
                        if high[rsi_idx] > high[prev_rsi_idx] and rsi[rsi_idx] < rsi[prev_rsi_idx]:
                            divergences['regular_bearish'].append({
                                'index': i,
                                'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                                'price': float(high[rsi_idx]),
                                'rsi': float(rsi[rsi_idx]),
                                'type': 'Regular Bearish'
                            })
                        
                        # Hidden Bearish: Price LH, RSI HH
                        if high[rsi_idx] < high[prev_rsi_idx] and rsi[rsi_idx] > rsi[prev_rsi_idx]:
                            divergences['hidden_bearish'].append({
                                'index': i,
                                'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i,
                                'price': float(high[rsi_idx]),
                                'rsi': float(rsi[rsi_idx]),
                                'type': 'Hidden Bearish'
                            })
        
        return {
            'rsi_period': self.rsi_period,
            'pivot_lookback_left': self.pivot_lookback_left,
            'pivot_lookback_right': self.pivot_lookback_right,
            'divergences': divergences,
            'latest_rsi': float(rsi[-1]) if len(rsi) > 0 else None
        }
    
    def _find_pivots_high(self, series: np.ndarray, left: int, right: int) -> np.ndarray:
        """Find pivot highs (like ta.pivothigh in Pine Script)"""
        pivots = np.zeros(len(series), dtype=bool)
        
        for i in range(left, len(series) - right):
            is_pivot = True
            center_val = series[i]
            
            # Check left side
            for j in range(1, left + 1):
                if series[i - j] >= center_val:
                    is_pivot = False
                    break
            
            # Check right side
            if is_pivot:
                for j in range(1, right + 1):
                    if series[i + j] > center_val:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots[i + right] = True  # Offset by right lookback
        
        return pivots
    
    def _find_pivots_low(self, series: np.ndarray, left: int, right: int) -> np.ndarray:
        """Find pivot lows (like ta.pivotlow in Pine Script)"""
        pivots = np.zeros(len(series), dtype=bool)
        
        for i in range(left, len(series) - right):
            is_pivot = True
            center_val = series[i]
            
            # Check left side
            for j in range(1, left + 1):
                if series[i - j] <= center_val:
                    is_pivot = False
                    break
            
            # Check right side
            if is_pivot:
                for j in range(1, right + 1):
                    if series[i + j] < center_val:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots[i + right] = True  # Offset by right lookback
        
        return pivots
    
    def _find_previous_pivot(self, pivots: np.ndarray, current_idx: int, 
                            range_lower: int, range_upper: int) -> Optional[int]:
        """Find previous pivot within valid range (ta.barssince logic)"""
        for i in range(current_idx - 1, max(0, current_idx - range_upper - 1), -1):
            if pivots[i]:
                bars_since = current_idx - i
                if range_lower <= bars_since <= range_upper:
                    return i
        return None
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def _ema(self, series: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(series) < period:
            return np.full(len(series), np.nan)
        
        ema = np.zeros(len(series))
        ema[0] = series[0]
        multiplier = 2.0 / (period + 1)
        
        for i in range(1, len(series)):
            ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))
        
        # Initial averages
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        # Smoothed averages
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        # Calculate RSI
        rsi = np.zeros(len(prices))
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _create_raw_summary(self, macd_results: Dict, rsi_results: Dict) -> Dict:
        """Create RAW summary of divergence counts - NO interpretation"""
        
        # Count MACD divergences by type
        macd_regular_bearish = 0
        macd_hidden_bearish = 0
        macd_regular_bullish = 0
        macd_hidden_bullish = 0
        
        for tf_result in macd_results.values():
            divs = tf_result['divergences']
            macd_regular_bearish += len(divs['regular_bearish'])
            macd_hidden_bearish += len(divs['hidden_bearish'])
            macd_regular_bullish += len(divs['regular_bullish'])
            macd_hidden_bullish += len(divs['hidden_bullish'])
        
        # Count RSI divergences
        rsi_divs = rsi_results['divergences']
        rsi_regular_bearish = len(rsi_divs['regular_bearish'])
        rsi_hidden_bearish = len(rsi_divs['hidden_bearish'])
        rsi_regular_bullish = len(rsi_divs['regular_bullish'])
        rsi_hidden_bullish = len(rsi_divs['hidden_bullish'])
        
        # Return RAW COUNTS ONLY - no signal, no confidence
        return {
            # MACD Counts
            "macd_regular_bearish_count": macd_regular_bearish,
            "macd_hidden_bearish_count": macd_hidden_bearish,
            "macd_regular_bullish_count": macd_regular_bullish,
            "macd_hidden_bullish_count": macd_hidden_bullish,
            "macd_total_bearish": macd_regular_bearish + macd_hidden_bearish,
            "macd_total_bullish": macd_regular_bullish + macd_hidden_bullish,
            
            # RSI Counts
            "rsi_regular_bearish_count": rsi_regular_bearish,
            "rsi_hidden_bearish_count": rsi_hidden_bearish,
            "rsi_regular_bullish_count": rsi_regular_bullish,
            "rsi_hidden_bullish_count": rsi_hidden_bullish,
            "rsi_total_bearish": rsi_regular_bearish + rsi_hidden_bearish,
            "rsi_total_bullish": rsi_regular_bullish + rsi_hidden_bullish,
            
            # Combined Totals
            "total_bearish_divergences": (macd_regular_bearish + macd_hidden_bearish + 
                                          rsi_regular_bearish + rsi_hidden_bearish),
            "total_bullish_divergences": (macd_regular_bullish + macd_hidden_bullish + 
                                          rsi_regular_bullish + rsi_hidden_bullish),
            "total_regular_divergences": (macd_regular_bearish + macd_regular_bullish + 
                                          rsi_regular_bearish + rsi_regular_bullish),
            "total_hidden_divergences": (macd_hidden_bearish + macd_hidden_bullish + 
                                         rsi_hidden_bearish + rsi_hidden_bullish)
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "macd_divergences": {},
            "rsi_divergences": {"divergences": {}},
            "divergence_counts": {
                "macd_regular_bearish_count": 0,
                "macd_hidden_bearish_count": 0,
                "macd_regular_bullish_count": 0,
                "macd_hidden_bullish_count": 0,
                "macd_total_bearish": 0,
                "macd_total_bullish": 0,
                "rsi_regular_bearish_count": 0,
                "rsi_hidden_bearish_count": 0,
                "rsi_regular_bullish_count": 0,
                "rsi_hidden_bullish_count": 0,
                "rsi_total_bearish": 0,
                "rsi_total_bullish": 0,
                "total_bearish_divergences": 0,
                "total_bullish_divergences": 0,
                "total_regular_divergences": 0,
                "total_hidden_divergences": 0
            },
            "latest_rsi": None,
            "latest_macd_1h": None,
            "latest_macd_4h": None,
            "latest_macd_1d": None,
            "current_price": None,
            "error": reason
        }
