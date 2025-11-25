"""
Layer 2: Volume Engine (Raw Data Output)
Combines OBV, A/D Line, and CMF with divergence detection
Outputs RAW indicator values only - no scores, no signals
"""
import pandas as pd
import numpy as np
from typing import Dict

class Layer2Volume:
    """Volume analysis with OBV, A/D Line, CMF and divergence detection"""
    
    def __init__(self):
        self.obv_ma_length = 14
        self.ad_ma_length = 14
        self.cmf_length = 20
        self.vol_sma_length = 20
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run full volume analysis
        
        Args:
            df: OHLCV DataFrame with basic features
            
        Returns:
            Dictionary with RAW volume indicator values
        """
        df = df.copy()
        
        # Calculate OBV
        obv = self._calculate_obv(df)
        obv_ma = obv.rolling(window=self.obv_ma_length).mean()
        obv_slope = self._calculate_slope(obv, 5)
        df["obv"] = obv
        df["obv_ma"] = obv_ma
        
        # Calculate A/D Line
        ad_line = self._calculate_ad_line(df)
        ad_ma = ad_line.rolling(window=self.ad_ma_length).mean()
        ad_slope = self._calculate_slope(ad_line, 5)
        df["ad_line"] = ad_line
        df["ad_ma"] = ad_ma
        
        # Calculate CMF
        cmf = self._calculate_cmf(df, self.cmf_length)
        df["cmf"] = cmf
        
        # Volume analysis
        avg_vol = df["volume"].rolling(window=self.vol_sma_length).mean()
        vol_ratio = df["volume"].iloc[-1] / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1
        
        # Volume trend (5-bar comparison)
        vol_5_ago = df["volume"].iloc[-5] if len(df) >= 5 else df["volume"].iloc[0]
        vol_change_5 = ((df["volume"].iloc[-1] - vol_5_ago) / vol_5_ago * 100) if vol_5_ago > 0 else 0
        
        # Price vs Volume divergence detection (raw)
        price_slope = self._calculate_slope(df["close"], 5)
        
        # Return RAW DATA ONLY - no scores, no signals
        return {
            # OBV Data
            "obv": round(obv.iloc[-1], 0),
            "obv_ma": round(obv_ma.iloc[-1], 0),
            "obv_prev": round(obv.iloc[-2], 0) if len(obv) > 1 else None,
            "obv_slope": round(obv_slope, 2),
            "obv_vs_ma": round(obv.iloc[-1] - obv_ma.iloc[-1], 0),
            
            # A/D Line Data
            "ad_line": round(ad_line.iloc[-1], 0),
            "ad_ma": round(ad_ma.iloc[-1], 0),
            "ad_prev": round(ad_line.iloc[-2], 0) if len(ad_line) > 1 else None,
            "ad_slope": round(ad_slope, 2),
            "ad_vs_ma": round(ad_line.iloc[-1] - ad_ma.iloc[-1], 0),
            
            # CMF Data
            "cmf": round(cmf.iloc[-1], 4),
            "cmf_prev": round(cmf.iloc[-2], 4) if len(cmf) > 1 else None,
            
            # Volume Data
            "current_volume": round(df["volume"].iloc[-1], 0),
            "avg_volume_20": round(avg_vol.iloc[-1], 0),
            "volume_ratio": round(vol_ratio, 2),
            "volume_change_5bar_pct": round(vol_change_5, 2),
            
            # Price Slope (for divergence context)
            "price_slope": round(price_slope, 4),
            
            # Divergence Detection (raw boolean)
            "price_rising_volume_falling": price_slope > 0 and obv_slope < 0,
            "price_falling_volume_rising": price_slope < 0 and obv_slope > 0,
            
            # Current Price Context
            "current_price": round(df["close"].iloc[-1], 2)
        }
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        direction = np.sign(df["close"].diff())
        obv = (direction * df["volume"]).cumsum()
        return obv
    
    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mf_multiplier = mf_multiplier.fillna(0)
        mf_volume = mf_multiplier * df["volume"]
        ad_line = mf_volume.cumsum()
        return ad_line
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mf_multiplier = mf_multiplier.fillna(0)
        mf_volume = mf_multiplier * df["volume"]
        cmf = mf_volume.rolling(window=period).sum() / df["volume"].rolling(window=period).sum()
        return cmf
    
    def _calculate_slope(self, series: pd.Series, period: int) -> float:
        """Calculate slope of a series"""
        if len(series) < period:
            return 0
        recent = series.iloc[-period:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
