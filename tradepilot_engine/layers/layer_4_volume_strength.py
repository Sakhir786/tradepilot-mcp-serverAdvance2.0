"""
Layer 4: Volume Strength Engine (Raw Data Output)
Cumulative Volume Delta (CVD) and Ease of Movement (EOM)
Outputs RAW indicator values only - no scores, no signals
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
            Dict with RAW CVD and EOM values
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
        
        # Calculate volume context
        volume_context = self._calculate_volume_context(df)
        
        # Return RAW DATA ONLY - no scores, no signals
        return {
            # CVD Data
            "cvd": cvd_results["cvd"],
            "cvd_prev": cvd_results["cvd_prev"],
            "cumulative_buying_volume": cvd_results["cumulative_buying_volume"],
            "cumulative_selling_volume": cvd_results["cumulative_selling_volume"],
            "volume_strength_wave": cvd_results["volume_strength_wave"],
            "ema_volume_strength_wave": cvd_results["ema_volume_strength_wave"],
            "latest_buying_volume": cvd_results["latest_buying_volume"],
            "latest_selling_volume": cvd_results["latest_selling_volume"],
            "buying_volume_pct": cvd_results["buying_volume_pct"],
            "selling_volume_pct": cvd_results["selling_volume_pct"],
            
            # EOM Data
            "eom": eom_results["eom"],
            "eom_prev": eom_results["eom_prev"],
            "eom_hl2_change": eom_results["hl2_change"],
            "eom_distance": eom_results["distance"],
            
            # Volume Context
            "current_volume": volume_context["current_volume"],
            "avg_volume_20": volume_context["avg_volume_20"],
            "volume_ratio": volume_context["volume_ratio"],
            "volume_change_5bar_pct": volume_context["volume_change_5bar_pct"],
            
            # Price Context
            "current_price": round(df["close"].iloc[-1], 2),
            
            # Timestamp
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
        
        # Calculate buying/selling percentages
        total_volume = cumulative_buying_volume[-1] + cumulative_selling_volume[-1]
        buying_pct = (cumulative_buying_volume[-1] / total_volume * 100) if total_volume > 0 else 50
        selling_pct = (cumulative_selling_volume[-1] / total_volume * 100) if total_volume > 0 else 50
        
        return {
            "cumulative_buying_volume": round(float(cumulative_buying_volume[-1]), 2),
            "cumulative_selling_volume": round(float(cumulative_selling_volume[-1]), 2),
            "cvd": round(float(cumulative_volume_delta[-1]), 2),
            "cvd_prev": round(float(cumulative_volume_delta[-2]), 2) if len(cumulative_volume_delta) > 1 else None,
            "volume_strength_wave": round(float(volume_strength_wave[-1]), 2),
            "ema_volume_strength_wave": round(float(ema_volume_strength_wave[-1]), 2),
            "latest_buying_volume": round(float(buying_volume[-1]), 2),
            "latest_selling_volume": round(float(selling_volume[-1]), 2),
            "buying_volume_pct": round(buying_pct, 2),
            "selling_volume_pct": round(selling_pct, 2)
        }
    
    # ==================== EASE OF MOVEMENT (EOM) ====================
    
    def _calculate_eom(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Ease of Movement indicator
        
        Pine Script Formula:
        eom = sma(divisor * change(hl2) * (high - low) / volume, length)
        """
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate hl2 (midpoint)
        hl2 = (high + low) / 2
        
        # Calculate change in hl2
        hl2_change = np.diff(hl2, prepend=hl2[0])
        
        # Calculate distance moved (high - low)
        distance = high - low
        
        # Avoid division by zero
        volume_safe = np.where(volume == 0, 0.0001, volume)
        
        # Calculate raw EOM values
        raw_eom = self.eom_divisor * hl2_change * distance / volume_safe
        
        # Apply SMA smoothing
        eom_values = self._sma(raw_eom, self.eom_length)
        
        return {
            "eom": round(float(eom_values[-1]), 4),
            "eom_prev": round(float(eom_values[-2]), 4) if len(eom_values) > 1 else None,
            "hl2_change": round(float(hl2_change[-1]), 4),
            "distance": round(float(distance[-1]), 4)
        }
    
    # ==================== VOLUME CONTEXT ====================
    
    def _calculate_volume_context(self, df: pd.DataFrame) -> Dict:
        """Calculate additional volume context"""
        volume = df['volume']
        
        # Average volume
        avg_volume_20 = volume.rolling(window=20).mean().iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Volume ratio (RVOL)
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        # Volume change over 5 bars
        vol_5_ago = volume.iloc[-5] if len(volume) >= 5 else volume.iloc[0]
        volume_change_5bar_pct = ((current_volume - vol_5_ago) / vol_5_ago * 100) if vol_5_ago > 0 else 0
        
        return {
            "current_volume": round(float(current_volume), 0),
            "avg_volume_20": round(float(avg_volume_20), 0),
            "volume_ratio": round(float(volume_ratio), 2),
            "volume_change_5bar_pct": round(float(volume_change_5bar_pct), 2)
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
            sma = np.zeros(len(series))
            for i in range(len(series)):
                sma[i] = np.mean(series[:i+1])
            return sma
        
        sma = np.zeros(len(series))
        sma[:period-1] = np.nan
        sma[period-1] = np.mean(series[:period])
        
        for i in range(period, len(series)):
            sma[i] = np.mean(series[i-period+1:i+1])
        
        return sma
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "cvd": None,
            "cvd_prev": None,
            "cumulative_buying_volume": None,
            "cumulative_selling_volume": None,
            "volume_strength_wave": None,
            "ema_volume_strength_wave": None,
            "latest_buying_volume": None,
            "latest_selling_volume": None,
            "buying_volume_pct": None,
            "selling_volume_pct": None,
            "eom": None,
            "eom_prev": None,
            "eom_hl2_change": None,
            "eom_distance": None,
            "current_volume": None,
            "avg_volume_20": None,
            "volume_ratio": None,
            "volume_change_5bar_pct": None,
            "current_price": None,
            "error": reason
        }
