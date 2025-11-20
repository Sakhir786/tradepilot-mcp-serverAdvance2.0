"""
Layer 2: Volume Engine
Combines OBV, A/D Line, and CMF with divergence detection
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
        self.cmf_threshold = 0.05
        self.vol_sma_length = 20
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Run full volume analysis"""
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
        
        # Calculate volume flow score
        obv_strength = self._calculate_strength(obv_slope)
        ad_strength = self._calculate_strength(ad_slope)
        cmf_strength = cmf.iloc[-1] * 100 if not pd.isna(cmf.iloc[-1]) else 0
        
        volume_flow_score = (obv_strength + ad_strength + cmf_strength) / 3
        
        # Signal generation
        signal = self._generate_signal(
            volume_flow_score, obv_slope, ad_slope, cmf.iloc[-1]
        )
        
        return {
            "volume_flow_score": round(volume_flow_score, 2),
            "obv": round(obv.iloc[-1], 0),
            "obv_slope": round(obv_slope, 2),
            "obv_trend": "RISING" if obv_slope > 0 else "FALLING",
            "ad_line": round(ad_line.iloc[-1], 0),
            "ad_slope": round(ad_slope, 2),
            "ad_trend": "ACCUMULATION" if ad_slope > 0 else "DISTRIBUTION",
            "cmf": round(cmf.iloc[-1], 4),
            "volume_ratio": round(vol_ratio, 2),
            "avg_volume": round(avg_vol.iloc[-1], 0),
            "current_volume": round(df["volume"].iloc[-1], 0),
            "signal": signal
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
    
    def _calculate_strength(self, slope: float) -> float:
        """Convert slope to strength score"""
        # Normalize slope to -100 to +100 range
        return np.clip(slope / 100, -100, 100)
    
    def _generate_signal(self, flow_score: float, obv_slope: float, ad_slope: float, cmf: float) -> str:
        """Generate trading signal"""
        if flow_score > 50 and obv_slope > 0 and ad_slope > 0 and cmf > self.cmf_threshold:
            return "STRONG_BUY"
        elif flow_score > 20 and obv_slope > 0:
            return "BUY"
        elif flow_score < -50 and obv_slope < 0 and ad_slope < 0 and cmf < -self.cmf_threshold:
            return "STRONG_SELL"
        elif flow_score < -20 and obv_slope < 0:
            return "SELL"
        else:
            return "NEUTRAL"
