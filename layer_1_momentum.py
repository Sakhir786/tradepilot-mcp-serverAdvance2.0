"""
Layer 1: Momentum Engine
Combines RSI Divergence, MACD, Stochastic, CMF, ADX, and Ichimoku
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
            Dictionary with momentum analysis results
        """
        df = df.copy()
        
        # Calculate RSI
        rsi = self._calculate_rsi(df, self.rsi_length)
        df["rsi"] = rsi
        
        # Calculate MACD
        macd_line, signal_line, macd_hist = self._calculate_macd(
            df, self.macd_fast, self.macd_slow, self.macd_signal
        )
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_hist
        
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
        
        # Calculate momentum scores
        rsi_momentum = self._calc_rsi_momentum(rsi.iloc[-1])
        macd_momentum = self._calc_macd_momentum(df["macd_hist"])
        stoch_momentum = self._calc_stoch_momentum(k.iloc[-1])
        trend_momentum = self._calc_trend_momentum(adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1])
        cmf_momentum = cmf.iloc[-1] * 100 if not pd.isna(cmf.iloc[-1]) else 0
        
        # Combined momentum score (-100 to +100)
        momentum_score = (rsi_momentum + macd_momentum + stoch_momentum + trend_momentum + cmf_momentum) / 5
        
        # Trend classification
        trend_strength = self._classify_trend_strength(adx.iloc[-1])
        trend_direction = "BULLISH" if plus_di.iloc[-1] > minus_di.iloc[-1] else "BEARISH"
        
        # Cloud trend
        cloud_trend = self._classify_cloud_trend(
            df["close"].iloc[-1], lead1.iloc[-1], lead2.iloc[-1]
        )
        
        # Signal generation
        signal = self._generate_signal(
            momentum_score, trend_direction, adx.iloc[-1], cloud_trend
        )
        
        return {
            "momentum_score": round(momentum_score, 2),
            "rsi": round(rsi.iloc[-1], 2),
            "rsi_momentum": round(rsi_momentum, 2),
            "macd": round(macd_line.iloc[-1], 4),
            "macd_signal": round(signal_line.iloc[-1], 4),
            "macd_hist": round(macd_hist.iloc[-1], 4),
            "macd_momentum": round(macd_momentum, 2),
            "stochastic_k": round(k.iloc[-1], 2),
            "stochastic_d": round(d.iloc[-1], 2),
            "stoch_momentum": round(stoch_momentum, 2),
            "cmf": round(cmf.iloc[-1], 4),
            "cmf_momentum": round(cmf_momentum, 2),
            "adx": round(adx.iloc[-1], 2),
            "plus_di": round(plus_di.iloc[-1], 2),
            "minus_di": round(minus_di.iloc[-1], 2),
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "trend_momentum": round(trend_momentum, 2),
            "ichimoku_conv": round(conv_line.iloc[-1], 2),
            "ichimoku_base": round(base_line.iloc[-1], 2),
            "cloud_trend": cloud_trend,
            "signal": signal
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
    
    def _calc_rsi_momentum(self, rsi: float) -> float:
        """Convert RSI to momentum score"""
        if rsi > 50:
            return (rsi - 50) / 50 * 100
        else:
            return -(50 - rsi) / 50 * 100
    
    def _calc_macd_momentum(self, macd_hist: pd.Series) -> float:
        """Convert MACD histogram to momentum score"""
        current_hist = macd_hist.iloc[-1]
        hist_abs = abs(macd_hist)
        max_hist = hist_abs.rolling(window=100).max().iloc[-1]
        
        if max_hist == 0:
            return 0
        
        if current_hist > 0:
            return 100 * (current_hist / max_hist)
        else:
            return -100 * (abs(current_hist) / max_hist)
    
    def _calc_stoch_momentum(self, k: float) -> float:
        """Convert Stochastic to momentum score"""
        return (k - 50) / 50 * 100
    
    def _calc_trend_momentum(self, adx: float, plus_di: float, minus_di: float) -> float:
        """Convert ADX/DMI to momentum score"""
        if plus_di > minus_di:
            return (adx / 100) * 100
        else:
            return -(adx / 100) * 100
    
    def _classify_trend_strength(self, adx: float) -> str:
        """Classify trend strength based on ADX"""
        if adx > 40:
            return "STRONG"
        elif adx > 25:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _classify_cloud_trend(self, price: float, lead1: float, lead2: float) -> str:
        """Classify Ichimoku cloud trend"""
        if price > lead1 and price > lead2:
            return "BULLISH"
        elif price < lead1 and price < lead2:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _generate_signal(self, momentum_score: float, trend_dir: str, adx: float, cloud_trend: str) -> str:
        """Generate trading signal"""
        if momentum_score > 50 and trend_dir == "BULLISH" and adx > 25 and cloud_trend == "BULLISH":
            return "STRONG_BUY"
        elif momentum_score > 20 and trend_dir == "BULLISH":
            return "BUY"
        elif momentum_score < -50 and trend_dir == "BEARISH" and adx > 25 and cloud_trend == "BEARISH":
            return "STRONG_SELL"
        elif momentum_score < -20 and trend_dir == "BEARISH":
            return "SELL"
        else:
            return "NEUTRAL"
