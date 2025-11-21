"""
Layer 5: Trend Engine - SuperTrend Pro (Raw Data Output)
Professional SuperTrend with comprehensive metrics
Outputs RAW indicator values only - no scores, no signals
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

class Layer5Trend:
    """
    Professional SuperTrend analysis with comprehensive metrics.
    
    Features:
    - Adaptive ATR multiplier based on volatility
    - Market regime detection (ADX/DMI)
    - Whipsaw detection
    - Volume confirmation
    - Higher timeframe alignment
    - RSI momentum
    - Time of day context
    - Signal persistence tracking
    """
    
    def __init__(self):
        # Core Settings
        self.atr_length = 10
        self.base_multiplier = 2.5
        self.use_adaptive = True
        
        # Confirmation Settings
        self.min_adx = 20
        self.min_vol_ratio = 1.2
        
        # State tracking for whipsaw detection
        self.flip_count = 0
        self.last_flip_bar = 0
        self.bars_in_trend = 0
        self.last_direction = None
    
    def analyze(self, df: pd.DataFrame, current_timeframe: str = '5') -> Dict:
        """
        Run complete SuperTrend Pro analysis
        
        Args:
            df: DataFrame with OHLCV data
            current_timeframe: Current timeframe ('1', '5', '15', '60', '240', 'D')
            
        Returns:
            Dict with RAW SuperTrend metrics
        """
        if len(df) < 50:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Calculate volatility and adaptive multiplier
        volatility_data = self._calculate_volatility(df)
        
        # Calculate SuperTrend
        supertrend_data = self._calculate_supertrend(df, volatility_data['adaptive_multiplier'])
        
        # Market regime detection
        regime_data = self._detect_market_regime(df)
        
        # Whipsaw detection
        whipsaw_data = self._detect_whipsaw(supertrend_data['direction_array'], len(df))
        
        # Volume analysis
        volume_data = self._analyze_volume(df)
        
        # Higher timeframe alignment
        htf_data = self._check_htf_alignment(df, current_timeframe, supertrend_data)
        
        # RSI momentum
        momentum_data = self._calculate_momentum(df, supertrend_data)
        
        # Time of day context
        tod_data = self._time_of_day_filter(df)
        
        # Persistence tracking
        persistence_data = self._track_persistence(supertrend_data['direction_array'])
        
        # Return RAW DATA ONLY - no scores, no signals
        return {
            # SuperTrend Core Data
            "supertrend_value": supertrend_data["value"],
            "supertrend_direction": supertrend_data["direction"],
            "supertrend_bullish": supertrend_data["bullish"],
            "supertrend_bearish": supertrend_data["bearish"],
            "trend_changed": supertrend_data["trend_changed"],
            "raw_buy_signal": supertrend_data["raw_buy_signal"],
            "raw_sell_signal": supertrend_data["raw_sell_signal"],
            
            # Volatility Data
            "atr": volatility_data["current_atr"],
            "atr_percent": volatility_data["atr_percent"],
            "volatility_high": volatility_data["volatility_high"],
            "volatility_low": volatility_data["volatility_low"],
            "volatility_normal": volatility_data["volatility_normal"],
            "adaptive_multiplier": volatility_data["adaptive_multiplier"],
            
            # Market Regime Data (ADX/DMI)
            "adx": regime_data["adx"],
            "di_plus": regime_data["di_plus"],
            "di_minus": regime_data["di_minus"],
            "trending": regime_data["trending"],
            "weak_trend": regime_data["weak_trend"],
            "choppy": regime_data["choppy"],
            
            # Whipsaw Data
            "whipsaw_mode": whipsaw_data["whipsaw_mode"],
            "flip_count": whipsaw_data["flip_count"],
            "bars_since_flip": whipsaw_data["bars_since_flip"],
            
            # Volume Data
            "volume_ratio": volume_data["vol_ratio"],
            "avg_volume": volume_data["avg_volume"],
            "current_volume": volume_data["current_volume"],
            "volume_confirmed": volume_data["volume_confirmed"],
            
            # HTF Alignment Data
            "htf_timeframe": htf_data["htf_timeframe"],
            "htf_bullish": htf_data["htf_bullish"],
            "htf_bearish": htf_data["htf_bearish"],
            "htf_aligned": htf_data["htf_aligned"],
            
            # RSI Momentum Data
            "rsi": momentum_data["rsi"],
            "rsi_aligned": momentum_data["rsi_aligned"],
            
            # Time of Day Data
            "is_first_30min": tod_data["is_first_risk"],
            "is_lunch_hours": tod_data["is_lunch"],
            "is_close_risk": tod_data["is_close_risk"],
            "is_optimal_hours": tod_data["is_best_hours"],
            
            # Persistence Data
            "bars_in_trend": persistence_data["bars_in_trend"],
            
            # Price Context
            "current_price": round(df["close"].iloc[-1], 2),
            
            # Timestamp
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== VOLATILITY ANALYSIS ====================
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict:
        """Calculate volatility metrics and adaptive multiplier"""
        atr = self._calculate_atr(df, self.atr_length)
        close = df['close'].values
        
        atr_percent = (atr / close) * 100
        current_atr_percent = atr_percent[-1]
        
        volatility_high = current_atr_percent > 2.0
        volatility_low = current_atr_percent < 0.8
        volatility_normal = not volatility_high and not volatility_low
        
        if self.use_adaptive:
            if volatility_low:
                adaptive_multiplier = self.base_multiplier * 0.8
            elif volatility_high:
                adaptive_multiplier = self.base_multiplier * 1.4
            else:
                adaptive_multiplier = self.base_multiplier
        else:
            adaptive_multiplier = self.base_multiplier
        
        return {
            "atr_percent": round(float(current_atr_percent), 4),
            "volatility_high": volatility_high,
            "volatility_low": volatility_low,
            "volatility_normal": volatility_normal,
            "adaptive_multiplier": round(adaptive_multiplier, 2),
            "current_atr": round(float(atr[-1]), 4)
        }
    
    # ==================== SUPERTREND CALCULATION ====================
    
    def _calculate_supertrend(self, df: pd.DataFrame, multiplier: float) -> Dict:
        """Calculate SuperTrend indicator"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        atr = self._calculate_atr(df, self.atr_length)
        
        # Calculate basic bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        # Initialize arrays
        final_upper = np.zeros(len(df))
        final_lower = np.zeros(len(df))
        supertrend = np.zeros(len(df))
        direction = np.ones(len(df))  # 1 = bearish, -1 = bullish
        
        # First values
        final_upper[0] = basic_upper[0]
        final_lower[0] = basic_lower[0]
        supertrend[0] = basic_upper[0]
        direction[0] = 1
        
        # Calculate SuperTrend
        for i in range(1, len(df)):
            # Upper band
            if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i-1]
            
            # Lower band
            if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i-1]
            
            # SuperTrend direction
            if supertrend[i-1] == final_upper[i-1]:
                if close[i] <= final_upper[i]:
                    supertrend[i] = final_upper[i]
                    direction[i] = 1  # Bearish
                else:
                    supertrend[i] = final_lower[i]
                    direction[i] = -1  # Bullish
            else:
                if close[i] >= final_lower[i]:
                    supertrend[i] = final_lower[i]
                    direction[i] = -1  # Bullish
                else:
                    supertrend[i] = final_upper[i]
                    direction[i] = 1  # Bearish
        
        bullish = direction[-1] < 0
        bearish = direction[-1] > 0
        
        # Detect trend change
        trend_changed = False
        if len(direction) > 1:
            trend_changed = direction[-1] != direction[-2]
        
        raw_buy_signal = trend_changed and bullish
        raw_sell_signal = trend_changed and bearish
        
        return {
            "value": round(float(supertrend[-1]), 2),
            "direction": int(direction[-1]),
            "bullish": bullish,
            "bearish": bearish,
            "trend_changed": trend_changed,
            "raw_buy_signal": raw_buy_signal,
            "raw_sell_signal": raw_sell_signal,
            "direction_array": direction
        }
    
    # ==================== MARKET REGIME DETECTION ====================
    
    def _detect_market_regime(self, df: pd.DataFrame) -> Dict:
        """Detect market regime using ADX/DMI"""
        di_plus, di_minus, adx = self._calculate_dmi(df, 14)
        
        current_adx = adx[-1]
        
        trending = current_adx > 25
        weak_trend = current_adx >= 20 and current_adx <= 25
        choppy = current_adx < 20
        
        return {
            "adx": round(float(current_adx), 2),
            "di_plus": round(float(di_plus[-1]), 2),
            "di_minus": round(float(di_minus[-1]), 2),
            "trending": trending,
            "weak_trend": weak_trend,
            "choppy": choppy
        }
    
    # ==================== WHIPSAW DETECTION ====================
    
    def _detect_whipsaw(self, direction_array: np.ndarray, current_bar: int) -> Dict:
        """Detect whipsaw conditions (rapid trend flips)"""
        trend_changed = False
        if len(direction_array) > 1:
            trend_changed = direction_array[-1] != direction_array[-2]
        
        if trend_changed:
            if current_bar - self.last_flip_bar < 10:
                self.flip_count += 1
            else:
                self.flip_count = 1
            self.last_flip_bar = current_bar
        
        whipsaw_mode = self.flip_count >= 3
        
        # Reset if too long since last flip
        if current_bar - self.last_flip_bar > 20:
            self.flip_count = 0
        
        return {
            "whipsaw_mode": whipsaw_mode,
            "flip_count": self.flip_count,
            "bars_since_flip": current_bar - self.last_flip_bar
        }
    
    # ==================== VOLUME ANALYSIS ====================
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume confirmation"""
        volume = df['volume'].values
        avg_volume = np.convolve(volume, np.ones(20)/20, mode='valid')
        
        # Pad beginning with expanding mean
        for i in range(min(19, len(volume))):
            avg_volume = np.insert(avg_volume, 0, np.mean(volume[:i+1]))
        
        vol_ratio = volume[-1] / avg_volume[-1] if avg_volume[-1] > 0 else 1.0
        volume_confirmed = vol_ratio >= self.min_vol_ratio
        
        return {
            "vol_ratio": round(float(vol_ratio), 2),
            "avg_volume": round(float(avg_volume[-1]), 0),
            "current_volume": round(float(volume[-1]), 0),
            "volume_confirmed": volume_confirmed
        }
    
    # ==================== HIGHER TIMEFRAME ALIGNMENT ====================
    
    def _check_htf_alignment(self, df: pd.DataFrame, current_tf: str, 
                            supertrend_data: Dict) -> Dict:
        """Check higher timeframe alignment"""
        tf_map = {
            '1': '5',
            '5': '15',
            '15': '60',
            '60': '240',
            '240': 'D',
            'D': 'W'
        }
        
        htf_tf = tf_map.get(current_tf, 'D')
        
        close = df['close'].values
        ma_50 = np.convolve(close, np.ones(50)/50, mode='valid')
        
        if len(ma_50) > 0:
            htf_bullish = close[-1] > ma_50[-1]
            htf_bearish = close[-1] < ma_50[-1]
        else:
            htf_bullish = supertrend_data['bullish']
            htf_bearish = supertrend_data['bearish']
        
        htf_aligned = (
            (supertrend_data['bullish'] and htf_bullish) or 
            (supertrend_data['bearish'] and htf_bearish)
        )
        
        return {
            "htf_timeframe": htf_tf,
            "htf_bullish": htf_bullish,
            "htf_bearish": htf_bearish,
            "htf_aligned": htf_aligned
        }
    
    # ==================== RSI MOMENTUM ====================
    
    def _calculate_momentum(self, df: pd.DataFrame, supertrend_data: Dict) -> Dict:
        """Calculate RSI momentum alignment"""
        rsi = self._calculate_rsi(df['close'].values, 14)
        current_rsi = rsi[-1]
        
        rsi_aligned = (
            (supertrend_data['bullish'] and current_rsi > 50) or
            (supertrend_data['bearish'] and current_rsi < 50)
        )
        
        return {
            "rsi": round(float(current_rsi), 2),
            "rsi_aligned": rsi_aligned
        }
    
    # ==================== TIME OF DAY FILTER ====================
    
    def _time_of_day_filter(self, df: pd.DataFrame) -> Dict:
        """Get time of day context"""
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            last_time = df.index[-1]
            current_hour = last_time.hour
            current_minute = last_time.minute
        else:
            return {
                "is_first_risk": False,
                "is_lunch": False,
                "is_close_risk": False,
                "is_best_hours": True
            }
        
        current_time = current_hour * 60 + current_minute
        
        open_time = 9 * 60 + 30  # 9:30
        first_risk_end = 10 * 60  # 10:00
        lunch_start = 11 * 60 + 30  # 11:30
        lunch_end = 13 * 60 + 30  # 13:30
        close_start = 15 * 60 + 30  # 15:30
        
        is_first_risk = current_time >= open_time and current_time < first_risk_end
        is_lunch = current_time >= lunch_start and current_time < lunch_end
        is_close_risk = current_time >= close_start
        is_best_hours = not is_first_risk and not is_lunch and not is_close_risk
        
        return {
            "is_first_risk": is_first_risk,
            "is_lunch": is_lunch,
            "is_close_risk": is_close_risk,
            "is_best_hours": is_best_hours
        }
    
    # ==================== PERSISTENCE TRACKING ====================
    
    def _track_persistence(self, direction_array: np.ndarray) -> Dict:
        """Track signal persistence (bars in trend)"""
        if self.last_direction is None:
            self.bars_in_trend = 0
        elif direction_array[-1] == self.last_direction:
            self.bars_in_trend += 1
        else:
            self.bars_in_trend = 0
        
        self.last_direction = direction_array[-1]
        
        return {
            "bars_in_trend": self.bars_in_trend
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
    
    def _calculate_dmi(self, df: pd.DataFrame, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate DMI (Directional Movement Index) and ADX"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        atr = self._calculate_atr(df, period)
        
        plus_di = np.zeros(len(df))
        minus_di = np.zeros(len(df))
        
        plus_di[0] = 0
        minus_di[0] = 0
        
        for i in range(1, len(df)):
            if i < period:
                plus_di[i] = 0
                minus_di[i] = 0
            else:
                smoothed_plus_dm = np.sum(plus_dm[i-period+1:i+1])
                smoothed_minus_dm = np.sum(minus_dm[i-period+1:i+1])
                
                plus_di[i] = 100 * smoothed_plus_dm / (atr[i] * period) if atr[i] > 0 else 0
                minus_di[i] = 100 * smoothed_minus_dm / (atr[i] * period) if atr[i] > 0 else 0
        
        dx = np.zeros(len(df))
        for i in range(len(df)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(df))
        adx[period] = np.mean(dx[:period+1])
        
        for i in range(period + 1, len(df)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return plus_di, minus_di, adx
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(prices))
        avg_loss = np.zeros(len(prices))
        
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        rsi = np.zeros(len(prices))
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "supertrend_value": None,
            "supertrend_direction": None,
            "supertrend_bullish": None,
            "supertrend_bearish": None,
            "trend_changed": None,
            "raw_buy_signal": None,
            "raw_sell_signal": None,
            "atr": None,
            "atr_percent": None,
            "volatility_high": None,
            "volatility_low": None,
            "volatility_normal": None,
            "adaptive_multiplier": None,
            "adx": None,
            "di_plus": None,
            "di_minus": None,
            "trending": None,
            "weak_trend": None,
            "choppy": None,
            "whipsaw_mode": None,
            "flip_count": None,
            "bars_since_flip": None,
            "volume_ratio": None,
            "avg_volume": None,
            "current_volume": None,
            "volume_confirmed": None,
            "htf_timeframe": None,
            "htf_bullish": None,
            "htf_bearish": None,
            "htf_aligned": None,
            "rsi": None,
            "rsi_aligned": None,
            "is_first_30min": None,
            "is_lunch_hours": None,
            "is_close_risk": None,
            "is_optimal_hours": None,
            "bars_in_trend": None,
            "current_price": None,
            "error": reason
        }
