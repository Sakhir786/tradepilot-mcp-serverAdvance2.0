"""
Layer 8: Volatility Regime Engine
ATR-based volatility classification
"""
import pandas as pd
import numpy as np
from typing import Dict

class Layer8VolatilityRegime:
    """Volatility regime classification"""
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Run volatility analysis"""
        df = df.copy()
        
        # Calculate ATRP (ATR as percentage)
        atr = df["true_range"].rolling(window=14).mean()
        atrp = (atr / df["close"]) * 100
        atrp_smoothed = atrp.rolling(window=5).mean()
        
        # Calculate percentiles
        atrp_values = atrp_smoothed.dropna().iloc[-100:]
        if len(atrp_values) > 0:
            p20 = np.percentile(atrp_values, 20)
            p40 = np.percentile(atrp_values, 40)
            p60 = np.percentile(atrp_values, 60)
            p80 = np.percentile(atrp_values, 80)
            
            current_atrp = atrp_smoothed.iloc[-1]
            
            if current_atrp <= p20:
                regime = "LOW"
            elif current_atrp <= p40:
                regime = "NORMAL-LOW"
            elif current_atrp <= p60:
                regime = "NORMAL"
            elif current_atrp <= p80:
                regime = "ELEVATED"
            else:
                regime = "EXTREME"
        else:
            regime = "NORMAL"
            current_atrp = 0
        
        return {
            "regime": regime,
            "atrp": round(current_atrp, 4) if not pd.isna(current_atrp) else 0,
            "atr": round(atr.iloc[-1], 4),
            "signal": "NEUTRAL"
        }
