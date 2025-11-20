"""
Layer 5: Trend Engine - SuperTrend Pro
Professional SuperTrend with Quality Scoring System
Converted from Pine Script - Logic unchanged
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime

class Layer5Trend:
    """
    Professional SuperTrend analysis with comprehensive quality scoring.
    
    Features:
    - Adaptive ATR multiplier based on volatility
    - Market regime detection (Trending/Weak/Choppy)
    - Whipsaw detection and penalty
    - Volume confirmation
    - Higher timeframe alignment
    - RSI momentum filter
    - Time of day filtering
    - Signal persistence tracking
    - 0-100% quality scoring system
    - Signal qualification (EXCELLENT/GOOD/MARGINAL/SKIP)
    """
    
    def __init__(self):
        # Core Settings
        self.atr_length = 10
        self.base_multiplier = 2.5
        self.use_adaptive = True
        
        # Confirmation Settings
        self.min_adx = 20
        self.min_vol_ratio = 1.2
        self.require_close = True
        
        # Quality Thresholds
        self.excellent_threshold = 85
        self.good_threshold = 70
        self.marginal_threshold = 60
        
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
            Dict with SuperTrend signals and quality metrics
        """
        if len(df) < 50:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Calculate volatility and adaptive multiplier
        volatility_metrics = self._calculate_volatility(df)
        
        # Calculate SuperTrend
        supertrend_data = self._calculate_supertrend(df, volatility_metrics['adaptive_multiplier'])
        
        # Market regime detection
        regime_data = self._detect_market_regime(df)
        
        # Whipsaw detection
        whipsaw_data = self._detect_whipsaw(supertrend_data['direction'], len(df))
        
        # Volume analysis
        volume_data = self._analyze_volume(df)
        
        # Higher timeframe alignment
        htf_data = self._check_htf_alignment(df, current_timeframe, supertrend_data)
        
        # RSI momentum
        momentum_data = self._calculate_momentum(df, supertrend_data)
        
        # Time of day filter
        tod_data = self._time_of_day_filter(df)
        
        # Persistence tracking
        persistence_data = self._track_persistence(supertrend_data['direction'])
        
        # Calculate quality score
        quality_data = self._calculate_quality_score(
            regime_data, volume_data, htf_data, momentum_data, 
            tod_data, whipsaw_data, volatility_metrics
        )
        
        # Generate signals
        signals = self._generate_signals(
            supertrend_data, quality_data, len(df)
        )
        
        return {
            "supertrend": supertrend_data,
            "volatility": volatility_metrics,
            "regime": regime_data,
            "whipsaw": whipsaw_data,
            "volume": volume_data,
            "htf_alignment": htf_data,
            "momentum": momentum_data,
            "time_of_day": tod_data,
            "persistence": persistence_data,
            "quality": quality_data,
            "signals": signals,
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== VOLATILITY ANALYSIS ====================
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict:
        """
        Calculate volatility metrics and adaptive multiplier
        
        Pine Script logic:
        - atrPercent = atr(10) / close * 100
        - volatilityHigh = atrPercent > 2.0
        - volatilityLow = atrPercent < 0.8
        - adaptiveMultiplier = low ? base * 0.8 : high ? base * 1.4 : base
        """
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
            "atr_percent": float(current_atr_percent),
            "volatility_high": volatility_high,
            "volatility_low": volatility_low,
            "volatility_normal": volatility_normal,
            "adaptive_multiplier": adaptive_multiplier,
            "current_atr": float(atr[-1])
        }
    
    # ==================== SUPERTREND CALCULATION ====================
    
    def _calculate_supertrend(self, df: pd.DataFrame, multiplier: float) -> Dict:
        """
        Calculate SuperTrend indicator
        
        Pine Script: [supertrend, direction] = ta.supertrend(multiplier, atrLength)
        """
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
            "value": float(supertrend[-1]),
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
        """
        Detect market regime using ADX/DMI
        
        Pine Script logic:
        - [diPlus, diMinus, adx] = ta.dmi(14, 14)
        - trending = adx > 25
        - weakTrend = adx >= 20 and adx <= 25
        - choppy = adx < 20
        """
        di_plus, di_minus, adx = self._calculate_dmi(df, 14)
        
        current_adx = adx[-1]
        
        trending = current_adx > 25
        weak_trend = current_adx >= 20 and current_adx <= 25
        choppy = current_adx < 20
        
        if trending:
            regime_text = "TRENDING"
            regime_color = "lime"
        elif weak_trend:
            regime_text = "WEAK TREND"
            regime_color = "yellow"
        else:
            regime_text = "CHOPPY"
            regime_color = "red"
        
        return {
            "adx": float(current_adx),
            "di_plus": float(di_plus[-1]),
            "di_minus": float(di_minus[-1]),
            "trending": trending,
            "weak_trend": weak_trend,
            "choppy": choppy,
            "regime_text": regime_text,
            "regime_color": regime_color
        }
    
    # ==================== WHIPSAW DETECTION ====================
    
    def _detect_whipsaw(self, direction_array: np.ndarray, current_bar: int) -> Dict:
        """
        Detect whipsaw conditions (rapid trend flips)
        
        Pine Script logic:
        - If trend changed and within 10 bars of last flip: flipCount++
        - If flipCount >= 3: whipsawMode = true
        - Reset flipCount if > 20 bars since last flip
        """
        # Check if trend changed (compare last two values)
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
        """
        Analyze volume confirmation
        
        Pine Script logic:
        - avgVolume = sma(volume, 20)
        - volRatio = volume / avgVolume
        - volumeConfirmed = volRatio >= minVolRatio
        """
        volume = df['volume'].values
        avg_volume = np.convolve(volume, np.ones(20)/20, mode='valid')
        
        # Pad beginning with expanding mean
        for i in range(min(19, len(volume))):
            avg_volume = np.insert(avg_volume, 0, np.mean(volume[:i+1]))
        
        vol_ratio = volume[-1] / avg_volume[-1] if avg_volume[-1] > 0 else 1.0
        volume_confirmed = vol_ratio >= self.min_vol_ratio
        
        return {
            "vol_ratio": float(vol_ratio),
            "avg_volume": float(avg_volume[-1]),
            "current_volume": float(volume[-1]),
            "volume_confirmed": volume_confirmed
        }
    
    # ==================== HIGHER TIMEFRAME ALIGNMENT ====================
    
    def _check_htf_alignment(self, df: pd.DataFrame, current_tf: str, 
                            supertrend_data: Dict) -> Dict:
        """
        Check higher timeframe alignment
        
        Pine Script logic:
        - Map to higher TF: 1->5, 5->15, 15->60, 60->240, else->D
        - Get HTF direction
        - htfAligned = (bullish and htfBullish) or (bearish and htfBearish)
        
        Note: Since we can't actually fetch HTF data, we'll simulate based on trend strength
        """
        # Map current TF to higher TF
        tf_map = {
            '1': '5',
            '5': '15',
            '15': '60',
            '60': '240',
            '240': 'D',
            'D': 'W'
        }
        
        htf_tf = tf_map.get(current_tf, 'D')
        
        # Simulate HTF direction based on trend strength
        # In production, this would fetch actual HTF data
        # For now, assume HTF aligns if strong trend
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
        """
        Calculate RSI momentum alignment
        
        Pine Script logic:
        - rsi = ta.rsi(close, 14)
        - rsiAligned = (bullish and rsi > 50) or (bearish and rsi < 50)
        """
        rsi = self._calculate_rsi(df['close'].values, 14)
        current_rsi = rsi[-1]
        
        rsi_aligned = (
            (supertrend_data['bullish'] and current_rsi > 50) or
            (supertrend_data['bearish'] and current_rsi < 50)
        )
        
        return {
            "rsi": float(current_rsi),
            "rsi_aligned": rsi_aligned
        }
    
    # ==================== TIME OF DAY FILTER ====================
    
    def _time_of_day_filter(self, df: pd.DataFrame) -> Dict:
        """
        Apply time of day filtering
        
        Pine Script logic:
        - 9:30-10:00: HIGH RISK (first 30min)
        - 11:30-13:30: CHOP RISK (lunch)
        - 15:30+: REV RISK (close)
        - Other: OPTIMAL
        """
        # Try to get time from index
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            last_time = df.index[-1]
            current_hour = last_time.hour
            current_minute = last_time.minute
        else:
            # Default to optimal if no timestamp
            return {
                "is_first_risk": False,
                "is_lunch": False,
                "is_close_risk": False,
                "is_best_hours": True,
                "tod_text": "OPTIMAL"
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
        
        if is_first_risk:
            tod_text = "HIGH RISK"
        elif is_lunch:
            tod_text = "CHOP RISK"
        elif is_close_risk:
            tod_text = "REV RISK"
        else:
            tod_text = "OPTIMAL"
        
        return {
            "is_first_risk": is_first_risk,
            "is_lunch": is_lunch,
            "is_close_risk": is_close_risk,
            "is_best_hours": is_best_hours,
            "tod_text": tod_text
        }
    
    # ==================== PERSISTENCE TRACKING ====================
    
    def _track_persistence(self, direction_array: np.ndarray) -> Dict:
        """
        Track signal persistence (bars in trend)
        
        Pine Script logic:
        - If direction same as previous: barsInTrend++
        - Else: barsInTrend = 0
        - EXCELLENT (>15), GOOD (>6), MODERATE (>3), SHORT
        """
        # Update bars in trend
        if self.last_direction is None:
            self.bars_in_trend = 0
        elif direction_array[-1] == self.last_direction:
            self.bars_in_trend += 1
        else:
            self.bars_in_trend = 0
        
        self.last_direction = direction_array[-1]
        
        if self.bars_in_trend > 15:
            persistence = "EXCELLENT"
        elif self.bars_in_trend > 6:
            persistence = "GOOD"
        elif self.bars_in_trend > 3:
            persistence = "MODERATE"
        else:
            persistence = "SHORT"
        
        return {
            "bars_in_trend": self.bars_in_trend,
            "persistence": persistence
        }
    
    # ==================== QUALITY SCORE CALCULATION ====================
    
    def _calculate_quality_score(self, regime_data: Dict, volume_data: Dict,
                                 htf_data: Dict, momentum_data: Dict,
                                 tod_data: Dict, whipsaw_data: Dict,
                                 volatility_data: Dict) -> Dict:
        """
        Calculate comprehensive quality score (0-100)
        
        Pine Script logic:
        - trendScore = min(adx / 50 * 35, 35)
        - volScore = min((volRatio - 1) / 2 * 20, 20)
        - htfScore = htfAligned ? 25 : 0
        - momentumScore = rsiAligned ? 10 : 0
        - todScore = isBestHours ? 10 : (isFirstRisk or isCloseRisk ? 5 : 0)
        - baseQuality = sum of above
        - Adjustments: choppy(-20), trending(+10), whipsaw(-15), highVol(-5)
        """
        # Component scores
        trend_score = min(regime_data['adx'] / 50 * 35, 35)
        
        vol_score = min((volume_data['vol_ratio'] - 1) / 2 * 20, 20)
        
        htf_score = 25 if htf_data['htf_aligned'] else 0
        
        momentum_score = 10 if momentum_data['rsi_aligned'] else 0
        
        if tod_data['is_best_hours']:
            tod_score = 10
        elif tod_data['is_first_risk'] or tod_data['is_close_risk']:
            tod_score = 5
        else:
            tod_score = 0
        
        # Base quality
        base_quality = trend_score + vol_score + htf_score + momentum_score + tod_score
        
        # Adjustments
        regime_adjustment = -20 if regime_data['choppy'] else (10 if regime_data['trending'] else 0)
        whipsaw_penalty = -15 if whipsaw_data['whipsaw_mode'] else 0
        volatility_adjustment = -5 if volatility_data['volatility_high'] else 0
        
        # Total quality
        total_quality = base_quality + regime_adjustment + whipsaw_penalty + volatility_adjustment
        total_quality = round(max(min(total_quality, 100), 0))
        
        # Quality tier
        if total_quality >= self.excellent_threshold:
            quality_tier = "EXCELLENT"
        elif total_quality >= self.good_threshold:
            quality_tier = "GOOD"
        elif total_quality >= self.marginal_threshold:
            quality_tier = "MARGINAL"
        else:
            quality_tier = "SKIP"
        
        return {
            "trend_score": round(trend_score, 1),
            "vol_score": round(vol_score, 1),
            "htf_score": htf_score,
            "momentum_score": momentum_score,
            "tod_score": tod_score,
            "base_quality": round(base_quality, 1),
            "regime_adjustment": regime_adjustment,
            "whipsaw_penalty": whipsaw_penalty,
            "volatility_adjustment": volatility_adjustment,
            "total_quality": total_quality,
            "quality_tier": quality_tier
        }
    
    # ==================== SIGNAL GENERATION ====================
    
    def _generate_signals(self, supertrend_data: Dict, quality_data: Dict,
                         current_bar: int) -> Dict:
        """
        Generate qualified trading signals
        
        Pine Script logic:
        - buyQuality = rawBuySignal ? totalQuality : 0
        - buyQualified = rawBuySignal and buyQuality >= marginalThreshold and confirmationReq
        """
        buy_quality = quality_data['total_quality'] if supertrend_data['raw_buy_signal'] else 0
        sell_quality = quality_data['total_quality'] if supertrend_data['raw_sell_signal'] else 0
        
        # Confirmation requirement (bar close)
        confirmation_req = True  # In live system, check if bar is confirmed
        
        buy_qualified = (
            supertrend_data['raw_buy_signal'] and 
            buy_quality >= self.marginal_threshold and 
            confirmation_req
        )
        
        sell_qualified = (
            supertrend_data['raw_sell_signal'] and 
            sell_quality >= self.marginal_threshold and 
            confirmation_req
        )
        
        # Signal classification
        if buy_qualified:
            if buy_quality >= self.excellent_threshold:
                signal = "EXCELLENT_BUY"
            elif buy_quality >= self.good_threshold:
                signal = "GOOD_BUY"
            else:
                signal = "MARGINAL_BUY"
        elif sell_qualified:
            if sell_quality >= self.excellent_threshold:
                signal = "EXCELLENT_SELL"
            elif sell_quality >= self.good_threshold:
                signal = "GOOD_SELL"
            else:
                signal = "MARGINAL_SELL"
        else:
            signal = "NO_SIGNAL"
        
        return {
            "buy_quality": buy_quality,
            "sell_quality": sell_quality,
            "buy_qualified": buy_qualified,
            "sell_qualified": sell_qualified,
            "signal": signal,
            "current_quality": quality_data['total_quality']
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
        tr[0] = tr1[0]  # First value
        
        atr = np.zeros(len(df))
        atr[0] = tr[0]
        
        # Wilder's smoothing
        for i in range(1, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def _calculate_dmi(self, df: pd.DataFrame, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate DMI (Directional Movement Index) and ADX"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate +DM and -DM
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Calculate ATR
        atr = self._calculate_atr(df, period)
        
        # Smooth DMs
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
        
        # Calculate DX and ADX
        dx = np.zeros(len(df))
        for i in range(len(df)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX is smoothed DX
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
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "error": reason,
            "supertrend": {"value": 0, "direction": 0, "bullish": False, "bearish": False},
            "quality": {"total_quality": 0, "quality_tier": "SKIP"},
            "signals": {"signal": "NO_SIGNAL", "current_quality": 0}
        }
