"""
Layer 6: Market Structure Engine
CHoCH, BOS, Order Blocks, Fair Value Gaps, and Liquidity Detection
Converted from Pine Script - Logic unchanged
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class Layer6Structure:
    """
    Professional Market Structure analysis with Smart Money Concepts.
    
    Features:
    - CHoCH (Change of Character) detection with quality scoring
    - BOS (Break of Structure) detection with quality scoring
    - Order Block identification with mitigation tracking
    - Fair Value Gap (FVG) detection with quality scoring
    - Liquidity Sweep detection
    - Market bias classification (TRENDING/REVERSING/RANGING)
    - Volume delta analysis
    - Quality scoring for all patterns
    """
    
    def __init__(self):
        # Core Settings
        self.pivot_length = 10
        self.max_patterns = 15
        self.min_quality = 60
        self.volume_mult = 1.5
        self.trend_length = 50
        
        # Order Block Settings
        self.ob_length = 10
        self.max_ob = 3
        
        # FVG Settings
        self.max_fvg = 3
        self.fvg_threshold = 0.0  # Auto threshold
        
        # Liquidity Settings
        self.max_liq = 2
        
        # State tracking
        self.trend = 0  # 1=bullish, -1=bearish, 0=neutral
        self.last_ph = None
        self.last_pl = None
        
        # Counters
        self.total_choch_bull = 0
        self.total_choch_bear = 0
        self.total_bos_bull = 0
        self.total_bos_bear = 0
        self.total_ob_bull = 0
        self.total_ob_bear = 0
        self.total_fvg_bull = 0
        self.total_fvg_bear = 0
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run complete market structure analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with CHoCH, BOS, OB, FVG, and liquidity metrics
        """
        if len(df) < 50:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Calculate trend EMA
        trend_ema = df['close'].ewm(span=self.trend_length, adjust=False).mean()
        is_uptrend = df['close'].iloc[-1] > trend_ema.iloc[-1]
        
        # Calculate average volume
        avg_volume = df['volume'].rolling(window=20).mean()
        
        # Calculate ATR
        atr_val = self._calculate_atr(df, 14)
        
        # Detect pivots
        pivot_data = self._detect_pivots(df)
        
        # Detect CHoCH and BOS
        structure_data = self._detect_choch_bos(
            df, pivot_data, is_uptrend, avg_volume, atr_val
        )
        
        # Detect Order Blocks
        ob_data = self._detect_order_blocks(
            df, pivot_data, is_uptrend, avg_volume, atr_val
        )
        
        # Detect Fair Value Gaps
        fvg_data = self._detect_fvg(df, is_uptrend, avg_volume, atr_val)
        
        # Detect Liquidity Sweeps
        liq_data = self._detect_liquidity(df, pivot_data, avg_volume)
        
        # Calculate market bias
        bias_data = self._calculate_market_bias(structure_data, ob_data)
        
        return {
            "choch_bos": structure_data,
            "order_blocks": ob_data,
            "fvg": fvg_data,
            "liquidity": liq_data,
            "bias": bias_data,
            "trend_ema": float(trend_ema.iloc[-1]),
            "is_uptrend": is_uptrend,
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== PIVOT DETECTION ====================
    
    def _detect_pivots(self, df: pd.DataFrame) -> Dict:
        """
        Detect pivot highs and lows
        
        Pine Script logic:
        - pivotH = ta.pivothigh(len, len)
        - pivotL = ta.pivotlow(len, len)
        """
        high = df['high'].values
        low = df['low'].values
        
        pivot_highs = []
        pivot_lows = []
        pivot_high_indices = []
        pivot_low_indices = []
        
        # Detect pivot highs
        for i in range(self.pivot_length, len(df) - self.pivot_length):
            is_pivot_high = True
            center_high = high[i]
            
            # Check left side
            for j in range(1, self.pivot_length + 1):
                if high[i - j] >= center_high:
                    is_pivot_high = False
                    break
            
            # Check right side
            if is_pivot_high:
                for j in range(1, self.pivot_length + 1):
                    if high[i + j] >= center_high:
                        is_pivot_high = False
                        break
            
            if is_pivot_high:
                pivot_highs.append(center_high)
                pivot_high_indices.append(i)
        
        # Detect pivot lows
        for i in range(self.pivot_length, len(df) - self.pivot_length):
            is_pivot_low = True
            center_low = low[i]
            
            # Check left side
            for j in range(1, self.pivot_length + 1):
                if low[i - j] <= center_low:
                    is_pivot_low = False
                    break
            
            # Check right side
            if is_pivot_low:
                for j in range(1, self.pivot_length + 1):
                    if low[i + j] <= center_low:
                        is_pivot_low = False
                        break
            
            if is_pivot_low:
                pivot_lows.append(center_low)
                pivot_low_indices.append(i)
        
        # Get last pivots
        last_ph = pivot_highs[-1] if len(pivot_highs) > 0 else None
        last_pl = pivot_lows[-1] if len(pivot_lows) > 0 else None
        last_ph_index = pivot_high_indices[-1] if len(pivot_high_indices) > 0 else None
        last_pl_index = pivot_low_indices[-1] if len(pivot_low_indices) > 0 else None
        
        # Update trend based on pivots
        if last_ph is not None and self.last_ph is not None:
            if last_ph > self.last_ph:
                self.trend = 1
        
        if last_pl is not None and self.last_pl is not None:
            if last_pl < self.last_pl:
                self.trend = -1
        
        self.last_ph = last_ph
        self.last_pl = last_pl
        
        return {
            "pivot_highs": pivot_highs,
            "pivot_lows": pivot_lows,
            "pivot_high_indices": pivot_high_indices,
            "pivot_low_indices": pivot_low_indices,
            "last_ph": last_ph,
            "last_pl": last_pl,
            "last_ph_index": last_ph_index,
            "last_pl_index": last_pl_index
        }
    
    # ==================== CHOCH & BOS DETECTION ====================
    
    def _detect_choch_bos(self, df: pd.DataFrame, pivot_data: Dict,
                          is_uptrend: bool, avg_volume: pd.Series,
                          atr_val: np.ndarray) -> Dict:
        """
        Detect Change of Character (CHoCH) and Break of Structure (BOS)
        
        Pine Script logic:
        - CHoCH: Trend reversal (trend changes sign)
        - BOS: Trend continuation (trend same sign but stronger)
        - Quality scoring based on volume delta, volume ratio, trend alignment, ATR
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_ = df['open'].values
        volume = df['volume'].values
        
        last_ph = pivot_data['last_ph']
        last_pl = pivot_data['last_pl']
        last_ph_index = pivot_data['last_ph_index']
        last_pl_index = pivot_data['last_pl_index']
        
        current_index = len(df) - 1
        
        choch_bull_detected = False
        choch_bear_detected = False
        bos_bull_detected = False
        bos_bear_detected = False
        
        bull_quality = 0
        bear_quality = 0
        
        bull_delta = 0.0
        bear_delta = 0.0
        
        # Check for bullish CHoCH/BOS (crossover pivot high)
        if last_ph is not None and last_ph_index is not None:
            if high[-1] > last_ph:
                # Calculate metrics from pivot to current
                l = low[-1]
                delta = 0.0
                max_vol = 0.0
                
                bars_back = current_index - last_ph_index
                for i in range(min(bars_back, len(df))):
                    idx = -(i + 1)
                    l = min(l, low[idx])
                    delta += volume[idx] if close[idx] > open_[idx] else -volume[idx]
                    max_vol = max(max_vol, volume[idx])
                
                # Quality scoring (EXACT Pine Script logic)
                quality = 50.0
                abs_delta = abs(delta)
                
                if abs_delta > 1000000:
                    quality += 15
                elif abs_delta > 500000:
                    quality += 10
                
                avg_vol_val = avg_volume.iloc[-1]
                if max_vol > avg_vol_val * 2.0:
                    quality += 15
                elif max_vol > avg_vol_val * self.volume_mult:
                    quality += 10
                
                if delta > 0:
                    quality += 10
                
                if is_uptrend:
                    quality += 10
                
                if (high[-1] - last_ph) > atr_val[-1] * 0.5:
                    quality += 5
                
                quality = min(quality, 100)
                
                # Determine if CHoCH or BOS
                is_choch = self.trend <= 0
                is_bos = self.trend > 0
                
                if quality >= self.min_quality:
                    if is_choch:
                        choch_bull_detected = True
                        self.total_choch_bull += 1
                    else:
                        bos_bull_detected = True
                        self.total_bos_bull += 1
                    
                    bull_quality = quality
                    bull_delta = delta
                
                # Update trend
                self.trend = 1
        
        # Check for bearish CHoCH/BOS (crossunder pivot low)
        if last_pl is not None and last_pl_index is not None:
            if low[-1] < last_pl:
                # Calculate metrics from pivot to current
                h = high[-1]
                delta = 0.0
                max_vol = 0.0
                
                bars_back = current_index - last_pl_index
                for i in range(min(bars_back, len(df))):
                    idx = -(i + 1)
                    h = max(h, high[idx])
                    delta += volume[idx] if close[idx] > open_[idx] else -volume[idx]
                    max_vol = max(max_vol, volume[idx])
                
                # Quality scoring
                quality = 50.0
                abs_delta = abs(delta)
                
                if abs_delta > 1000000:
                    quality += 15
                elif abs_delta > 500000:
                    quality += 10
                
                avg_vol_val = avg_volume.iloc[-1]
                if max_vol > avg_vol_val * 2.0:
                    quality += 15
                elif max_vol > avg_vol_val * self.volume_mult:
                    quality += 10
                
                if delta < 0:
                    quality += 10
                
                if not is_uptrend:
                    quality += 10
                
                if (last_pl - low[-1]) > atr_val[-1] * 0.5:
                    quality += 5
                
                quality = min(quality, 100)
                
                # Determine if CHoCH or BOS
                is_choch = self.trend >= 0
                is_bos = self.trend < 0
                
                if quality >= self.min_quality:
                    if is_choch:
                        choch_bear_detected = True
                        self.total_choch_bear += 1
                    else:
                        bos_bear_detected = True
                        self.total_bos_bear += 1
                    
                    bear_quality = quality
                    bear_delta = delta
                
                # Update trend
                self.trend = -1
        
        return {
            "choch_bull_detected": choch_bull_detected,
            "choch_bear_detected": choch_bear_detected,
            "bos_bull_detected": bos_bull_detected,
            "bos_bear_detected": bos_bear_detected,
            "bull_quality": bull_quality,
            "bear_quality": bear_quality,
            "bull_delta": bull_delta,
            "bear_delta": bear_delta,
            "total_choch_bull": self.total_choch_bull,
            "total_choch_bear": self.total_choch_bear,
            "total_bos_bull": self.total_bos_bull,
            "total_bos_bear": self.total_bos_bear,
            "current_trend": self.trend
        }
    
    # ==================== ORDER BLOCK DETECTION ====================
    
    def _detect_order_blocks(self, df: pd.DataFrame, pivot_data: Dict,
                             is_uptrend: bool, avg_volume: pd.Series,
                             atr_val: np.ndarray) -> Dict:
        """
        Detect Order Blocks (last candle before strong move)
        
        Pine Script logic:
        - Bullish OB: Last down candle before up sequence
        - Bearish OB: Last up candle before down sequence
        - Quality scoring based on volume and trend
        """
        close = df['close'].values
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        last_ph = pivot_data['last_ph']
        last_pl = pivot_data['last_pl']
        
        ob_bull_detected = False
        ob_bear_detected = False
        ob_bull_quality = 0
        ob_bear_quality = 0
        ob_bull_top = 0.0
        ob_bull_btm = 0.0
        ob_bear_top = 0.0
        ob_bear_btm = 0.0
        
        # Detect bullish order block (crossover pivot high)
        if last_ph is not None and close[-1] > last_ph:
            # Get last candle data
            ob_high = high[-2]
            ob_low = low[-2]
            
            delta = 0.0
            max_vol = 0.0
            
            for i in range(min(10, len(df))):
                idx = -(i + 1)
                delta += volume[idx] if close[idx] > open_[idx] else -volume[idx]
                max_vol = max(max_vol, volume[idx])
            
            # Quality scoring
            quality = 50.0
            if abs(delta) > 1000000:
                quality += 15
            elif abs(delta) > 500000:
                quality += 10
            
            avg_vol_val = avg_volume.iloc[-1]
            if max_vol > avg_vol_val * 2.0:
                quality += 15
            elif max_vol > avg_vol_val * self.volume_mult:
                quality += 10
            
            if delta > 0:
                quality += 10
            
            if is_uptrend:
                quality += 10
            
            quality = min(quality, 100)
            
            if quality >= self.min_quality:
                ob_bull_detected = True
                ob_bull_quality = quality
                ob_bull_top = ob_high
                ob_bull_btm = ob_low
                self.total_ob_bull += 1
        
        # Detect bearish order block (crossunder pivot low)
        if last_pl is not None and close[-1] < last_pl:
            # Get last candle data
            ob_high = high[-2]
            ob_low = low[-2]
            
            delta = 0.0
            max_vol = 0.0
            
            for i in range(min(10, len(df))):
                idx = -(i + 1)
                delta += volume[idx] if close[idx] > open_[idx] else -volume[idx]
                max_vol = max(max_vol, volume[idx])
            
            # Quality scoring
            quality = 50.0
            if abs(delta) > 1000000:
                quality += 15
            elif abs(delta) > 500000:
                quality += 10
            
            avg_vol_val = avg_volume.iloc[-1]
            if max_vol > avg_vol_val * 2.0:
                quality += 15
            elif max_vol > avg_vol_val * self.volume_mult:
                quality += 10
            
            if delta < 0:
                quality += 10
            
            if not is_uptrend:
                quality += 10
            
            quality = min(quality, 100)
            
            if quality >= self.min_quality:
                ob_bear_detected = True
                ob_bear_quality = quality
                ob_bear_top = ob_high
                ob_bear_btm = ob_low
                self.total_ob_bear += 1
        
        return {
            "ob_bull_detected": ob_bull_detected,
            "ob_bear_detected": ob_bear_detected,
            "ob_bull_quality": ob_bull_quality,
            "ob_bear_quality": ob_bear_quality,
            "ob_bull_top": float(ob_bull_top),
            "ob_bull_btm": float(ob_bull_btm),
            "ob_bear_top": float(ob_bear_top),
            "ob_bear_btm": float(ob_bear_btm),
            "total_ob_bull": self.total_ob_bull,
            "total_ob_bear": self.total_ob_bear
        }
    
    # ==================== FAIR VALUE GAP DETECTION ====================
    
    def _detect_fvg(self, df: pd.DataFrame, is_uptrend: bool,
                    avg_volume: pd.Series, atr_val: np.ndarray) -> Dict:
        """
        Detect Fair Value Gaps
        
        Pine Script logic:
        - Bullish FVG: low > high[2] and close[1] > high[2]
        - Bearish FVG: high < low[2] and close[1] < low[2]
        - Threshold % for minimum gap size
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        if len(df) < 3:
            return {
                "fvg_bull_detected": False,
                "fvg_bear_detected": False,
                "fvg_bull_quality": 0,
                "fvg_bear_quality": 0,
                "total_fvg_bull": self.total_fvg_bull,
                "total_fvg_bear": self.total_fvg_bear
            }
        
        # Calculate threshold (auto or manual)
        if self.fvg_threshold == 0:
            # Auto threshold: average (high-low)/low
            threshold = np.mean((high - low) / low)
        else:
            threshold = self.fvg_threshold / 100
        
        fvg_bull_detected = False
        fvg_bear_detected = False
        fvg_bull_quality = 0
        fvg_bear_quality = 0
        fvg_bull_top = 0.0
        fvg_bull_btm = 0.0
        fvg_bear_top = 0.0
        fvg_bear_btm = 0.0
        
        # Bullish FVG detection
        bullish_fvg = low[-1] > high[-3] and close[-2] > high[-3]
        if bullish_fvg:
            gap_size = low[-1] - high[-3]
            if (gap_size / high[-3]) > threshold:
                fvg_top = low[-1]
                fvg_btm = high[-3]
                
                delta_vol = volume[-1] - volume[-2]
                
                # Quality scoring
                quality = 50.0
                avg_vol_val = avg_volume.iloc[-1]
                
                if delta_vol > avg_vol_val * 0.5:
                    quality += 20
                elif delta_vol > 0:
                    quality += 10
                
                if gap_size > atr_val[-1] * 0.3:
                    quality += 15
                elif gap_size > atr_val[-1] * 0.1:
                    quality += 10
                
                if is_uptrend:
                    quality += 15
                
                quality = min(quality, 100)
                
                if quality >= self.min_quality:
                    fvg_bull_detected = True
                    fvg_bull_quality = quality
                    fvg_bull_top = fvg_top
                    fvg_bull_btm = fvg_btm
                    self.total_fvg_bull += 1
        
        # Bearish FVG detection
        bearish_fvg = high[-1] < low[-3] and close[-2] < low[-3]
        if bearish_fvg:
            gap_size = low[-3] - high[-1]
            if (gap_size / high[-1]) > threshold:
                fvg_top = low[-3]
                fvg_btm = high[-1]
                
                delta_vol = volume[-1] - volume[-2]
                
                # Quality scoring
                quality = 50.0
                avg_vol_val = avg_volume.iloc[-1]
                
                if delta_vol > avg_vol_val * 0.5:
                    quality += 20
                elif delta_vol > 0:
                    quality += 10
                
                if gap_size > atr_val[-1] * 0.3:
                    quality += 15
                elif gap_size > atr_val[-1] * 0.1:
                    quality += 10
                
                if not is_uptrend:
                    quality += 15
                
                quality = min(quality, 100)
                
                if quality >= self.min_quality:
                    fvg_bear_detected = True
                    fvg_bear_quality = quality
                    fvg_bear_top = fvg_top
                    fvg_bear_btm = fvg_btm
                    self.total_fvg_bear += 1
        
        return {
            "fvg_bull_detected": fvg_bull_detected,
            "fvg_bear_detected": fvg_bear_detected,
            "fvg_bull_quality": fvg_bull_quality,
            "fvg_bear_quality": fvg_bear_quality,
            "fvg_bull_top": float(fvg_bull_top),
            "fvg_bull_btm": float(fvg_bull_btm),
            "fvg_bear_top": float(fvg_bear_top),
            "fvg_bear_btm": float(fvg_bear_btm),
            "total_fvg_bull": self.total_fvg_bull,
            "total_fvg_bear": self.total_fvg_bear
        }
    
    # ==================== LIQUIDITY DETECTION ====================
    
    def _detect_liquidity(self, df: pd.DataFrame, pivot_data: Dict,
                         avg_volume: pd.Series) -> Dict:
        """
        Detect liquidity sweeps (pivot highs/lows taken out with volume)
        
        Pine Script logic:
        - Buy liquidity: close > pivot high with volume > avg * mult
        - Sell liquidity: close < pivot low with volume > avg * mult
        """
        close = df['close'].values
        volume = df['volume'].values
        
        last_ph = pivot_data['last_ph']
        last_pl = pivot_data['last_pl']
        
        liq_buy_detected = False
        liq_sell_detected = False
        
        # Check buy liquidity (sweep pivot high)
        if last_ph is not None:
            avg_vol_val = avg_volume.iloc[-1]
            if close[-1] > last_ph and volume[-1] > avg_vol_val * self.volume_mult:
                liq_buy_detected = True
        
        # Check sell liquidity (sweep pivot low)
        if last_pl is not None:
            avg_vol_val = avg_volume.iloc[-1]
            if close[-1] < last_pl and volume[-1] > avg_vol_val * self.volume_mult:
                liq_sell_detected = True
        
        return {
            "liq_buy_detected": liq_buy_detected,
            "liq_sell_detected": liq_sell_detected,
            "liq_buy_level": float(last_ph) if last_ph is not None else 0.0,
            "liq_sell_level": float(last_pl) if last_pl is not None else 0.0
        }
    
    # ==================== MARKET BIAS CALCULATION ====================
    
    def _calculate_market_bias(self, structure_data: Dict, ob_data: Dict) -> Dict:
        """
        Calculate overall market bias
        
        Pine Script logic:
        - TRENDING: BOS > CHoCH * 2
        - REVERSING: CHoCH > BOS * 2
        - RANGING: Neither condition
        """
        total_choch = structure_data['total_choch_bull'] + structure_data['total_choch_bear']
        total_bos = structure_data['total_bos_bull'] + structure_data['total_bos_bear']
        
        if total_bos > total_choch * 2:
            market_state = "TRENDING"
        elif total_choch > total_bos * 2:
            market_state = "REVERSING"
        else:
            market_state = "RANGING"
        
        # Determine directional bias
        total_bull = structure_data['total_choch_bull'] + structure_data['total_bos_bull']
        total_bear = structure_data['total_choch_bear'] + structure_data['total_bos_bear']
        
        if total_bull > total_bear * 1.5:
            directional_bias = "BULLISH"
        elif total_bear > total_bull * 1.5:
            directional_bias = "BEARISH"
        else:
            directional_bias = "NEUTRAL"
        
        # Overall signal
        if market_state == "TRENDING" and directional_bias == "BULLISH":
            signal = "STRONG_BUY"
        elif market_state == "TRENDING" and directional_bias == "BEARISH":
            signal = "STRONG_SELL"
        elif directional_bias == "BULLISH":
            signal = "BUY"
        elif directional_bias == "BEARISH":
            signal = "SELL"
        else:
            signal = "NEUTRAL"
        
        return {
            "market_state": market_state,
            "directional_bias": directional_bias,
            "signal": signal,
            "total_choch": total_choch,
            "total_bos": total_bos,
            "choch_bos_ratio": total_choch / total_bos if total_bos > 0 else 0,
            "bull_bear_ratio": total_bull / total_bear if total_bear > 0 else 0
        }
    
    # ==================== UTILITY FUNCTIONS ====================
    
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
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "error": reason,
            "choch_bos": {
                "choch_bull_detected": False,
                "choch_bear_detected": False,
                "total_choch_bull": 0,
                "total_choch_bear": 0,
                "total_bos_bull": 0,
                "total_bos_bear": 0
            },
            "order_blocks": {
                "ob_bull_detected": False,
                "ob_bear_detected": False,
                "total_ob_bull": 0,
                "total_ob_bear": 0
            },
            "fvg": {
                "fvg_bull_detected": False,
                "fvg_bear_detected": False,
                "total_fvg_bull": 0,
                "total_fvg_bear": 0
            },
            "liquidity": {
                "liq_buy_detected": False,
                "liq_sell_detected": False
            },
            "bias": {
                "market_state": "RANGING",
                "directional_bias": "NEUTRAL",
                "signal": "NEUTRAL"
            }
        }
