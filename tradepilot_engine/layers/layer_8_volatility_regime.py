"""
Layer 8: Volatility Regime Engine (Raw Data Output)
ATR-based volatility classification
Outputs RAW volatility data only - no signals
"""
import pandas as pd
import numpy as np
from typing import Dict

class Layer8VolatilityRegime:
    """Volatility regime classification"""
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run volatility analysis
        
        Args:
            df: DataFrame with OHLCV data and true_range
            
        Returns:
            Dict with RAW volatility data
        """
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
            prev_atrp = atrp_smoothed.iloc[-2] if len(atrp_smoothed) > 1 else None
            
            # Calculate which percentile bucket
            if current_atrp <= p20:
                percentile_bucket = "0-20"
            elif current_atrp <= p40:
                percentile_bucket = "20-40"
            elif current_atrp <= p60:
                percentile_bucket = "40-60"
            elif current_atrp <= p80:
                percentile_bucket = "60-80"
            else:
                percentile_bucket = "80-100"
            
            # Calculate exact percentile rank
            percentile_rank = (atrp_values < current_atrp).sum() / len(atrp_values) * 100
            
        else:
            current_atrp = 0
            prev_atrp = None
            p20 = 0
            p40 = 0
            p60 = 0
            p80 = 0
            percentile_bucket = "40-60"
            percentile_rank = 50
        
        # Calculate ATR change
        atr_current = atr.iloc[-1]
        atr_prev = atr.iloc[-2] if len(atr) > 1 else atr_current
        atr_change_pct = ((atr_current - atr_prev) / atr_prev * 100) if atr_prev > 0 else 0
        
        # Calculate volatility trend (5-bar)
        if len(atrp_smoothed) >= 5:
            atrp_5_ago = atrp_smoothed.iloc[-5]
            atrp_trend = current_atrp - atrp_5_ago
        else:
            atrp_trend = 0
        
        # Return RAW DATA ONLY - no regime classification, no signals
        return {
            # ATR Data
            "atr": round(atr_current, 4),
            "atr_prev": round(atr_prev, 4),
            "atr_change_pct": round(atr_change_pct, 4),
            
            # ATRP Data (ATR as percentage of price)
            "atrp": round(current_atrp, 4) if not pd.isna(current_atrp) else 0,
            "atrp_prev": round(prev_atrp, 4) if prev_atrp is not None and not pd.isna(prev_atrp) else None,
            "atrp_smoothed": round(current_atrp, 4) if not pd.isna(current_atrp) else 0,
            "atrp_trend_5bar": round(atrp_trend, 4),
            
            # Percentile Data
            "percentile_rank": round(percentile_rank, 2),
            "percentile_bucket": percentile_bucket,
            "p20_threshold": round(p20, 4),
            "p40_threshold": round(p40, 4),
            "p60_threshold": round(p60, 4),
            "p80_threshold": round(p80, 4),
            
            # Volatility State Booleans (raw facts, not interpretation)
            "is_below_p20": current_atrp <= p20,
            "is_below_p40": current_atrp <= p40,
            "is_above_p60": current_atrp > p60,
            "is_above_p80": current_atrp > p80,
            "volatility_expanding": atrp_trend > 0,
            "volatility_contracting": atrp_trend < 0,
            
            # Price Context
            "current_price": round(df["close"].iloc[-1], 2)
        }
