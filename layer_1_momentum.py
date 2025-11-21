"""
Layer 1: Momentum Engine (Raw Data Output)
Combines RSI, MACD, Stochastic, CMF, ADX, and Ichimoku
Outputs RAW indicator values only - no scores, no signals
"""
import pandas as pd
import numpy as np
from typing import Dict

class Layer1Momentum:
    """Momentum analysis combining multiple oscillators and trend indicators"""
    
    def __init__(self):
        # Default parameters (matching Pine Script)
        self.rsi_length = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_length = 14
        self.stoch_smooth = 3
        self.cmf_length = 20
        self.adx_length = 14
        self.ichimoku_conv = 9
        self.ichimoku_base = 26
        self.ichimoku_span = 52
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run full momentum analysis
        
        Args:
            df: OHLCV DataFrame with basic features
            
        Returns:
            Dictionary with RAW momentum indicator values
        """
        df = df.copy()
        
        # Calculate RSI
        rsi = self._calculate_rsi(df, self.rsi_length)
        rsi_7 = self._calculate_rsi(df, 7)  # Additional timeframe
        df["rsi"] = rsi
        
        # Calculate MACD
        macd_line, signal_line, macd_hist = self._calculate_macd(
            df, self.macd_fast, self.macd_slow, self.macd_signal
        )
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_hist
        
        # MACD trend detection (raw)
        macd_hist_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0
        macd_hist_current = macd_hist.iloc[-1]
        
        # Calculate Stochastic
        k, d = self._calculate_stochastic(df, self.stoch_length, self.stoch_smooth)
        df["stoch_k"] = k
        df["stoch_d"] = d
        
        # Calculate CMF
        cmf = self._calculate_cmf(df, self.cmf_length)
        df["cmf"] = cmf
        
        # Calculate ADX and DMI
        adx, plus_di, minus_di = self._calculate_adx(df, self.adx_length)
        df["adx"] = adx
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        
        # Calculate Ichimoku
        conv_line, base_line, lead1, lead2 = self._calculate_ichimoku(
            df, self.ichimoku_conv, self.ichimoku_base, self.ichimoku_span
        )
        df["ichimoku_conv"] = conv_line
        df["ichimoku_base"] = base_line
        df["ichimoku_lead1"] = lead1
        df["ichimoku_lead2"] = lead2
        
        # Current price for context
        current_price = df["close"].iloc[-1]
        
        # Return RAW DATA ONLY - no scores, no signals
        return {
            # RSI Data
            "rsi_14": round(rsi.iloc[-1], 2),
            "rsi_7": round(rsi_7.iloc[-1], 2),
            "rsi_prev": round(rsi.iloc[-2], 2) if len(rsi) > 1 else None,
            
            # MACD Data
            "macd_line": round(macd_line.iloc[-1], 4),
            "macd_signal_line": round(signal_line.iloc[-1], 4),
            "macd_histogram": round(macd_hist.iloc[-1], 4),
            "macd_histogram_prev": round(macd_hist_prev, 4),
            "macd_histogram_rising": macd_hist_current > macd_hist_prev,
            
            # Stochastic Data
            "stoch_k": round(k.iloc[-1], 2),
            "stoch_d": round(d.iloc[-1], 2),
            "stoch_k_prev": round(k.iloc[-2], 2) if len(k) > 1 else None,
            
            # CMF Data
            "cmf": round(cmf.iloc[-1], 4),
            "cmf_prev": round(cmf.iloc[-2], 4) if len(cmf) > 1 else None,
            
            # ADX/DMI Data
            "adx": round(adx.iloc[-1], 2),
            "plus_di": round(plus_di.iloc[-1], 2),
            "minus_di": round(minus_di.iloc[-1], 2),
            "di_diff": round(plus_di.iloc[-1] - minus_di.iloc[-1], 2),
            
            # Ichimoku Data
            "ichimoku_conv": round(conv_line.iloc[-1], 2),
            "ichimoku_base": round(base_line.iloc[-1], 2),
            "ichimoku_lead1": round(lead1.iloc[-1], 2),
            "ichimoku_lead2": round(lead2.iloc[-1], 2),
            "price_vs_cloud_top": round(current_price - max(lead1.iloc[-1], lead2.iloc[-1]), 2),
            "price_vs_cloud_bottom": round(current_price - min(lead1.iloc[-1], lead2.iloc[-1]), 2),
            
            # Price Context
            "current_price": round(current_price, 2)
        }
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int, slow: int, signal: int):
        """Calculate MACD"""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        return macd_line, signal_line, macd_hist
    
    def _calculate_stochastic(self, df: pd.DataFrame, length: int, smooth: int):
        """Calculate Stochastic Oscillator"""
        lowest_low = df["low"].rolling(window=length).min()
        highest_high = df["high"].rolling(window=length).max()
        
        k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        k = k.rolling(window=smooth).mean()
        d = k.rolling(window=smooth).mean()
        
        return k, d
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
        mf_multiplier = mf_multiplier.fillna(0)
        
        mf_volume = mf_multiplier * df["volume"]
        cmf = mf_volume.rolling(window=period).sum() / df["volume"].rolling(window=period).sum()
        
        return cmf
    
    def _calculate_adx(self, df: pd.DataFrame, period: int):
        """Calculate ADX and DMI"""
        up_move = df["high"].diff()
        down_move = -df["low"].diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr = df["true_range"]
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    def _calculate_ichimoku(self, df: pd.DataFrame, conv: int, base: int, span: int):
        """Calculate Ichimoku Cloud components"""
        conv_line = (df["high"].rolling(window=conv).max() + df["low"].rolling(window=conv).min()) / 2
        base_line = (df["high"].rolling(window=base).max() + df["low"].rolling(window=base).min()) / 2
        
        lead1 = (conv_line + base_line) / 2
        lead2 = (df["high"].rolling(window=span).max() + df["low"].rolling(window=span).min()) / 2
        
        return conv_line, base_line, lead1, lead2
