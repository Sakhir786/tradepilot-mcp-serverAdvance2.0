"""
Layer 4: Volume Strength Engine
Cumulative Volume Delta (CVD) and Ease of Movement (EOM)
Converted from Pine Script - Logic unchanged
"""
import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime

class Layer4VolumeStrength:
    """
    Professional volume strength analysis using CVD and EOM.
    
    Features:
    - Cumulative Volume Delta with buying/selling pressure
    - Ease of Movement for price efficiency
    - Volume strength wave analysis
    - EMA smoothing for cumulative volumes
    """
    
    def __init__(self):
        # CVD Settings
        self.cvd_cumulation_length = 14
        
        # EOM Settings
        self.eom_length = 14
        self.eom_divisor = 10000
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run complete volume strength analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with CVD and EOM metrics
        """
        if len(df) < self.cvd_cumulation_length:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Validate volume data exists
        if df['volume'].sum() == 0:
            return self._empty_result("No volume data available")
        
        # Calculate CVD metrics
        cvd_results = self._calculate_cvd(df)
        
        # Calculate EOM
        eom_results = self._calculate_eom(df)
        
        # Create summary
        summary = self._create_summary(cvd_results, eom_results, df)
        
        return {
            "cvd": cvd_results,
            "eom": eom_results,
            "summary": summary,
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== CUMULATIVE VOLUME DELTA (CVD) ====================
    
    def _calculate_cvd(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Cumulative Volume Delta with buying/selling pressure
        
        Pine Script Logic:
        1. Calculate upper_wick, lower_wick, spread, body_length
        2. Calculate percentages of each component
        3. Distribute volume to buying/selling based on candle type
        4. Apply EMA smoothing
        5. Calculate CVD as difference
        """
        close = df['close'].values
        open_ = df['open'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate wicks and spread (EXACT Pine Script logic)
        upper_wick = np.where(close > open_, high - close, high - open_)
        lower_wick = np.where(close > open_, open_ - low, close - low)
        spread = high - low
        
        # Avoid division by zero
        spread = np.where(spread == 0, 0.0001, spread)
        
        # Calculate body length
        body_length = spread - (upper_wick + lower_wick)
        
        # Calculate percentages
        percent_upper_wick = upper_wick / spread
        percent_lower_wick = lower_wick / spread
        percent_body_length = body_length / spread
        
        # Calculate buying and selling volume (EXACT Pine Script logic)
        # Bullish candle (close > open): body + half wicks go to buying
        # Bearish candle (close < open): body + half wicks go to selling
        # The other side gets only half the wicks
        
        buying_volume = np.where(
            close > open_,
            (percent_body_length + (percent_upper_wick + percent_lower_wick) / 2) * volume,
            ((percent_upper_wick + percent_lower_wick) / 2) * volume
        )
        
        selling_volume = np.where(
            close < open_,
            (percent_body_length + (percent_upper_wick + percent_lower_wick) / 2) * volume,
            ((percent_upper_wick + percent_lower_wick) / 2) * volume
        )
        
        # Apply EMA smoothing (cumulation_length = 14)
        cumulative_buying_volume = self._ema(buying_volume, self.cvd_cumulation_length)
        cumulative_selling_volume = self._ema(selling_volume, self.cvd_cumulation_length)
        
        # Calculate volume strength wave (max of buying or selling)
        volume_strength_wave = np.where(
            cumulative_buying_volume > cumulative_selling_volume,
            cumulative_buying_volume,
            cumulative_selling_volume
        )
        
        # EMA of volume strength wave
        ema_volume_strength_wave = self._ema(volume_strength_wave, self.cvd_cumulation_length)
        
        # Calculate CVD (Cumulative Volume Delta)
        cumulative_volume_delta = cumulative_buying_volume - cumulative_selling_volume
        
        # Determine bias
        latest_cvd = cumulative_volume_delta[-1]
        bias = "BULLISH" if latest_cvd > 0 else "BEARISH" if latest_cvd < 0 else "NEUTRAL"
        
        # Get fill color logic
        fill_color = "GREEN" if cumulative_buying_volume[-1] > cumulative_selling_volume[-1] else \
                     "RED" if cumulative_buying_volume[-1] < cumulative_selling_volume[-1] else \
                     "YELLOW"
        
        return {
            "cumulative_buying_volume": float(cumulative_buying_volume[-1]),
            "cumulative_selling_volume": float(cumulative_selling_volume[-1]),
            "cvd": float(cumulative_volume_delta[-1]),
            "volume_strength_wave": float(volume_strength_wave[-1]),
            "ema_volume_strength_wave": float(ema_volume_strength_wave[-1]),
            "bias": bias,
            "fill_color": fill_color,
            "cumulation_length": self.cvd_cumulation_length,
            "latest_buying_volume": float(buying_volume[-1]),
            "latest_selling_volume": float(selling_volume[-1])
        }
    
    # ==================== EASE OF MOVEMENT (EOM) ====================
    
    def _calculate_eom(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Ease of Movement indicator
        
        Pine Script Formula:
        eom = sma(divisor * change(hl2) * (high - low) / volume, length)
        
        Where:
        - hl2 = (high + low) / 2
        - change(hl2) = hl2[current] - hl2[previous]
        - divisor = 10000 (default)
        - length = 14 (default)
        """
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate hl2 (midpoint)
        hl2 = (high + low) / 2
        
        # Calculate change in hl2
        hl2_change = np.diff(hl2, prepend=hl2[0])  # First value has 0 change
        
        # Calculate distance moved (high - low)
        distance = high - low
        
        # Avoid division by zero
        volume_safe = np.where(volume == 0, 0.0001, volume)
        
        # Calculate raw EOM values
        # Formula: divisor * change(hl2) * (high - low) / volume
        raw_eom = self.eom_divisor * hl2_change * distance / volume_safe
        
        # Apply SMA smoothing
        eom_values = self._sma(raw_eom, self.eom_length)
        
        # Current EOM value
        current_eom = eom_values[-1]
        
        # Determine state
        if current_eom > 0:
            state = "POSITIVE"
            meaning = "Price rising easily (low volume needed)"
        elif current_eom < 0:
            state = "NEGATIVE"
            meaning = "Price falling easily (low volume needed)"
        else:
            state = "NEUTRAL"
            meaning = "No clear movement efficiency"
        
        return {
            "value": float(current_eom),
            "state": state,
            "meaning": meaning,
            "length": self.eom_length,
            "divisor": self.eom_divisor,
            "latest_hl2_change": float(hl2_change[-1]),
            "latest_distance": float(distance[-1])
        }
    
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
    
    def _sma(self, series: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if len(series) < period:
            # For first values, use expanding mean
            sma = np.zeros(len(series))
            for i in range(len(series)):
                sma[i] = np.mean(series[:i+1])
            return sma
        
        sma = np.zeros(len(series))
        
        # First SMA value
        sma[:period-1] = np.nan
        sma[period-1] = np.mean(series[:period])
        
        # Subsequent values
        for i in range(period, len(series)):
            sma[i] = np.mean(series[i-period+1:i+1])
        
        return sma
    
    def _create_summary(self, cvd_results: Dict, eom_results: Dict, df: pd.DataFrame) -> Dict:
        """Create summary of volume strength analysis"""
        
        # CVD Signal
        cvd_value = cvd_results['cvd']
        cvd_strength = abs(cvd_value)
        
        if cvd_results['bias'] == "BULLISH":
            cvd_signal = "STRONG_BUY" if cvd_strength > cvd_results['cumulative_buying_volume'] * 0.2 else "BUY"
        elif cvd_results['bias'] == "BEARISH":
            cvd_signal = "STRONG_SELL" if cvd_strength > cvd_results['cumulative_selling_volume'] * 0.2 else "SELL"
        else:
            cvd_signal = "NEUTRAL"
        
        # EOM Signal
        eom_value = eom_results['value']
        if eom_value > 1000:
            eom_signal = "STRONG_BUY"
        elif eom_value > 0:
            eom_signal = "BUY"
        elif eom_value < -1000:
            eom_signal = "STRONG_SELL"
        elif eom_value < 0:
            eom_signal = "SELL"
        else:
            eom_signal = "NEUTRAL"
        
        # Combined signal
        signals = [cvd_signal, eom_signal]
        buy_count = signals.count("STRONG_BUY") * 2 + signals.count("BUY")
        sell_count = signals.count("STRONG_SELL") * 2 + signals.count("SELL")
        
        if buy_count > sell_count * 1.5:
            overall_signal = "STRONG_BUY"
        elif buy_count > sell_count:
            overall_signal = "BUY"
        elif sell_count > buy_count * 1.5:
            overall_signal = "STRONG_SELL"
        elif sell_count > buy_count:
            overall_signal = "SELL"
        else:
            overall_signal = "NEUTRAL"
        
        # Calculate RVOL for context
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        rvol = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return {
            "cvd_signal": cvd_signal,
            "eom_signal": eom_signal,
            "overall_signal": overall_signal,
            "cvd_bias": cvd_results['bias'],
            "eom_state": eom_results['state'],
            "rvol": round(float(rvol), 2),
            "volume_quality": "HIGH" if rvol >= 1.5 else "NORMAL" if rvol >= 1.0 else "LOW",
            "buying_pressure": round((cvd_results['cumulative_buying_volume'] / 
                                    (cvd_results['cumulative_buying_volume'] + 
                                     cvd_results['cumulative_selling_volume'])) * 100, 2),
            "confidence": min(100, int(abs(cvd_value) / 1000) + int(abs(eom_value) / 100))
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "cvd": {
                "cumulative_buying_volume": 0,
                "cumulative_selling_volume": 0,
                "cvd": 0,
                "volume_strength_wave": 0,
                "ema_volume_strength_wave": 0,
                "bias": "NEUTRAL"
            },
            "eom": {
                "value": 0,
                "state": "NEUTRAL",
                "meaning": "No data"
            },
            "summary": {
                "overall_signal": "NEUTRAL",
                "confidence": 0
            },
            "error": reason
        }
