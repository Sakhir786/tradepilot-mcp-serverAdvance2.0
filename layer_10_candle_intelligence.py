"""
Layer 10: Candle Intelligence Engine (Raw Data Output)
Advanced candlestick pattern detection
Outputs RAW pattern data only - no signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class Layer10CandleIntelligence:
    """Advanced candle pattern intelligence - raw data output"""
    
    def __init__(self):
        """Initialize candle intelligence analyzer"""
        self.name = "Layer 10: Candle Intelligence"
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run comprehensive candle pattern analysis
        
        Args:
            df: DataFrame with OHLCV + calculated columns
            
        Returns:
            Dict with RAW pattern detection results
        """
        if len(df) < 50:
            return self._empty_result("Insufficient data (need 50+ bars)")
        
        df = df.copy()
        
        # Calculate required columns if missing
        if 'body' not in df.columns:
            df['body'] = abs(df['close'] - df['open'])
        if 'range' not in df.columns:
            df['range'] = df['high'] - df['low']
        if 'upper_wick' not in df.columns:
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        if 'lower_wick' not in df.columns:
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Run all pattern detection systems
        three_white_soldiers = self._detect_three_white_soldiers(df)
        inside_bar = self._detect_inside_bar(df)
        candlestick_patterns = self._detect_candlestick_patterns(df)
        star_patterns = self._detect_star_patterns(df)
        
        # Count patterns
        bullish_patterns = [p for p in candlestick_patterns['patterns'] if p in 
                          ["Hammer", "Inverted Hammer", "Morning Star", "Bullish Engulfing", 
                           "Bullish Harami", "Piercing Line", "Bullish Kicker"]]
        bearish_patterns = [p for p in candlestick_patterns['patterns'] if p in 
                          ["Hanging Man", "Shooting Star", "Evening Star", "Bearish Engulfing",
                           "Bearish Harami", "Dark Cloud", "Bearish Kicker"]]
        
        # Return RAW DATA ONLY - no signals
        return {
            # Three White Soldiers Data
            "tws_detected": three_white_soldiers["signal"],
            "tws_quality": three_white_soldiers["quality_score"],
            "tws_high_quality": three_white_soldiers["high_quality"],
            "tws_pattern_base": three_white_soldiers["pattern_base"],
            "tws_volume_strong": three_white_soldiers["volume_strong"],
            "tws_after_downtrend": three_white_soldiers["after_downtrend"],
            "tws_near_support": three_white_soldiers["near_support"],
            "tws_entry": three_white_soldiers["entry_price"],
            "tws_stop": three_white_soldiers["stop_loss"],
            "tws_target1": three_white_soldiers["target1"],
            "tws_target2": three_white_soldiers["target2"],
            
            # Inside Bar Data
            "ib_detected": inside_bar["inside_bar_signal"],
            "ib_bullish_breakout": inside_bar["bullish_breakout"],
            "ib_bearish_breakout": inside_bar["bearish_breakout"],
            "ib_quality": inside_bar["quality_score"],
            "ib_mother_high": inside_bar["mother_high"],
            "ib_mother_low": inside_bar["mother_low"],
            "ib_inside_ratio_pct": inside_bar["inside_ratio"],
            "ib_bull_entry": inside_bar["bull_entry"],
            "ib_bull_stop": inside_bar["bull_stop"],
            "ib_bull_target1": inside_bar["bull_target1"],
            "ib_bull_target2": inside_bar["bull_target2"],
            "ib_bear_entry": inside_bar["bear_entry"],
            "ib_bear_stop": inside_bar["bear_stop"],
            "ib_bear_target1": inside_bar["bear_target1"],
            "ib_bear_target2": inside_bar["bear_target2"],
            
            # Candlestick Patterns Data
            "patterns_detected": candlestick_patterns["patterns"],
            "bullish_patterns": bullish_patterns,
            "bearish_patterns": bearish_patterns,
            "bullish_pattern_count": candlestick_patterns["bullish_count"],
            "bearish_pattern_count": candlestick_patterns["bearish_count"],
            "neutral_pattern_count": candlestick_patterns["neutral_count"],
            "doji_detected": "Doji" in candlestick_patterns["patterns"],
            "hammer_detected": "Hammer" in candlestick_patterns["patterns"],
            "inverted_hammer_detected": "Inverted Hammer" in candlestick_patterns["patterns"],
            "hanging_man_detected": "Hanging Man" in candlestick_patterns["patterns"],
            "shooting_star_detected": "Shooting Star" in candlestick_patterns["patterns"],
            "morning_star_detected": "Morning Star" in candlestick_patterns["patterns"],
            "evening_star_detected": "Evening Star" in candlestick_patterns["patterns"],
            "bullish_engulfing_detected": "Bullish Engulfing" in candlestick_patterns["patterns"],
            "bearish_engulfing_detected": "Bearish Engulfing" in candlestick_patterns["patterns"],
            "bullish_harami_detected": "Bullish Harami" in candlestick_patterns["patterns"],
            "bearish_harami_detected": "Bearish Harami" in candlestick_patterns["patterns"],
            "piercing_line_detected": "Piercing Line" in candlestick_patterns["patterns"],
            "dark_cloud_detected": "Dark Cloud" in candlestick_patterns["patterns"],
            "bullish_kicker_detected": "Bullish Kicker" in candlestick_patterns["patterns"],
            "bearish_kicker_detected": "Bearish Kicker" in candlestick_patterns["patterns"],
            "patterns_bull_quality": candlestick_patterns["bull_quality"],
            "patterns_bear_quality": candlestick_patterns["bear_quality"],
            
            # Star Patterns Data
            "morning_star_pro_detected": star_patterns["bullish_morning_star"],
            "evening_star_pro_detected": star_patterns["bearish_evening_star"],
            "morning_star_quality": star_patterns["bullish_quality"],
            "evening_star_quality": star_patterns["bearish_quality"],
            "morning_star_high_quality": star_patterns["bullish_high_quality"],
            "evening_star_high_quality": star_patterns["bearish_high_quality"],
            "morning_star_entry": star_patterns["bullish_entry"],
            "morning_star_stop": star_patterns["bullish_stop"],
            "morning_star_target1": star_patterns["bullish_target1"],
            "morning_star_target2": star_patterns["bullish_target2"],
            "evening_star_entry": star_patterns["bearish_entry"],
            "evening_star_stop": star_patterns["bearish_stop"],
            "evening_star_target1": star_patterns["bearish_target1"],
            "evening_star_target2": star_patterns["bearish_target2"],
            
            # Summary Counts (raw facts)
            "total_bullish_patterns": (candlestick_patterns["bullish_count"] + 
                                       (1 if three_white_soldiers["signal"] else 0) +
                                       (1 if inside_bar["bullish_breakout"] else 0) +
                                       (1 if star_patterns["bullish_morning_star"] else 0)),
            "total_bearish_patterns": (candlestick_patterns["bearish_count"] +
                                       (1 if inside_bar["bearish_breakout"] else 0) +
                                       (1 if star_patterns["bearish_evening_star"] else 0)),
            "total_patterns_detected": len(candlestick_patterns["patterns"]),
            
            # Candle Context
            "current_candle_bullish": df['close'].iloc[-1] > df['open'].iloc[-1],
            "current_candle_bearish": df['close'].iloc[-1] < df['open'].iloc[-1],
            "current_body_size": round(df['body'].iloc[-1], 4),
            "current_range": round(df['range'].iloc[-1], 4),
            "current_upper_wick": round(df['upper_wick'].iloc[-1], 4),
            "current_lower_wick": round(df['lower_wick'].iloc[-1], 4),
            "avg_body": round(df['body'].rolling(window=14).mean().iloc[-1], 4),
            "body_vs_avg_ratio": round(df['body'].iloc[-1] / df['body'].rolling(window=14).mean().iloc[-1], 2),
            
            # Price Context
            "current_price": round(df["close"].iloc[-1], 2)
        }
    
    def _detect_three_white_soldiers(self, df: pd.DataFrame) -> Dict:
        """Three White Soldiers pattern detection - logic preserved from Pine Script"""
        if len(df) < 20:
            return {"signal": False, "quality_score": 0, "high_quality": False,
                    "pattern_base": False, "volume_strong": False, "after_downtrend": False,
                    "near_support": False, "entry_price": 0, "stop_loss": 0,
                    "target1": 0, "target2": 0, "risk_reward": 0}
        
        # Settings
        min_body_size = 0.4
        max_shadow_size = 0.30
        volume_increase = 1.2
        trend_lookback = 10
        trend_ma_length = 50
        support_lookback = 20
        support_proximity = 0.05
        min_quality_score = 30
        risk_reward_ratio = 2.5
        
        avg_body = df['body'].ewm(span=14, adjust=False).mean()
        
        # Get last 3 candles
        candle1_body = abs(df['close'].iloc[-3] - df['open'].iloc[-3])
        candle1_high = df['high'].iloc[-3]
        candle1_low = df['low'].iloc[-3]
        candle1_open = df['open'].iloc[-3]
        candle1_close = df['close'].iloc[-3]
        candle1_bullish = df['close'].iloc[-3] > df['open'].iloc[-3]
        candle1_upper_shadow = candle1_high - max(candle1_open, candle1_close)
        
        candle2_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
        candle2_high = df['high'].iloc[-2]
        candle2_low = df['low'].iloc[-2]
        candle2_open = df['open'].iloc[-2]
        candle2_close = df['close'].iloc[-2]
        candle2_bullish = df['close'].iloc[-2] > df['open'].iloc[-2]
        candle2_upper_shadow = candle2_high - max(candle2_open, candle2_close)
        
        candle3_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        candle3_high = df['high'].iloc[-1]
        candle3_low = df['low'].iloc[-1]
        candle3_open = df['open'].iloc[-1]
        candle3_close = df['close'].iloc[-1]
        candle3_bullish = df['close'].iloc[-1] > df['open'].iloc[-1]
        candle3_upper_shadow = candle3_high - max(candle3_open, candle3_close)
        
        # Bodies long enough
        candle1_long = candle1_body >= (avg_body.iloc[-3] * min_body_size)
        candle2_long = candle2_body >= (avg_body.iloc[-2] * min_body_size)
        candle3_long = candle3_body >= (avg_body.iloc[-1] * min_body_size)
        bodies_long_enough = candle1_long and candle2_long and candle3_long
        
        # Shadows small enough
        shadow1_ok = candle1_upper_shadow <= (candle1_body * max_shadow_size)
        shadow2_ok = candle2_upper_shadow <= (candle2_body * max_shadow_size)
        shadow3_ok = candle3_upper_shadow <= (candle3_body * max_shadow_size)
        shadows_small_enough = shadow1_ok and shadow2_ok and shadow3_ok
        
        # Volume check
        avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_strong = df['volume'].iloc[-1] >= (avg_vol * volume_increase)
        
        # Trend check
        trend_ma = df['close'].rolling(window=trend_ma_length).mean()
        price_was_below = df['close'].iloc[-trend_lookback] < trend_ma.iloc[-trend_lookback]
        lower_lows = df['low'].iloc[-3] < df['low'].iloc[-trend_lookback:-3].min()
        is_after_downtrend = price_was_below and lower_lows
        
        # Support check
        support_level = df['low'].iloc[-support_lookback:].min()
        distance_to_support = (candle1_low - support_level) / candle1_low
        is_near_support = distance_to_support <= support_proximity
        
        # Quality score
        safe_candle1_body = max(candle1_body, 0.0001)
        safe_candle2_body = max(candle2_body, 0.0001)
        safe_candle3_body = max(candle3_body, 0.0001)
        
        shadow_ratio_1 = candle1_upper_shadow / safe_candle1_body
        shadow_ratio_2 = candle2_upper_shadow / safe_candle2_body
        shadow_ratio_3 = candle3_upper_shadow / safe_candle3_body
        avg_shadow_ratio = (shadow_ratio_1 + shadow_ratio_2 + shadow_ratio_3) / 3.0
        shadow_quality = 1.0 - min(avg_shadow_ratio, 1.0)
        
        body_diff_12 = abs(candle1_body - candle2_body)
        body_diff_23 = abs(candle2_body - candle3_body)
        total_body = candle1_body + candle2_body + candle3_body
        body_consistency = 1.0 - ((body_diff_12 + body_diff_23) / total_body) if total_body > 0 else 0.0
        
        quality_score = 20.0 + 25.0 + 25.0 + 15.0 + (shadow_quality * 10.0) + (body_consistency * 5.0)
        
        # Pattern base conditions
        all_bullish = candle1_bullish and candle2_bullish and candle3_bullish
        progressive_closes = candle2_close > candle1_close and candle3_close > candle2_close
        opens_in_body_2 = candle2_open > candle1_open and candle2_open < candle1_close
        opens_in_body_3 = candle3_open > candle2_open and candle3_open < candle2_close
        
        pattern_base = (all_bullish and progressive_closes and opens_in_body_2 and 
                       opens_in_body_3 and bodies_long_enough and shadows_small_enough)
        
        three_white_soldiers = pattern_base and quality_score >= min_quality_score
        
        # Entry/Stop/Targets
        entry_price = candle3_close
        stop_loss = min(candle1_low, candle2_low, candle3_low)
        risk_amount = entry_price - stop_loss
        target1 = entry_price + (risk_amount * risk_reward_ratio)
        target2 = entry_price + (risk_amount * risk_reward_ratio * 1.5)
        
        return {
            "signal": bool(three_white_soldiers),
            "quality_score": round(quality_score, 2),
            "high_quality": bool(three_white_soldiers and quality_score >= 80),
            "pattern_base": bool(pattern_base),
            "volume_strong": bool(volume_strong),
            "after_downtrend": bool(is_after_downtrend),
            "near_support": bool(is_near_support),
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target1": round(target1, 2),
            "target2": round(target2, 2),
            "risk_reward": risk_reward_ratio
        }
    
    def _detect_inside_bar(self, df: pd.DataFrame) -> Dict:
        """Inside Bar pattern detection - logic preserved from Pine Script"""
        if len(df) < 20:
            return {
                "inside_bar_signal": False, "bullish_breakout": False, "bearish_breakout": False,
                "quality_score": 0, "mother_high": 0, "mother_low": 0, "inside_ratio": 0,
                "bull_entry": 0, "bull_stop": 0, "bull_target1": 0, "bull_target2": 0,
                "bear_entry": 0, "bear_stop": 0, "bear_target1": 0, "bear_target2": 0
            }
        
        # Settings
        min_mother_size = 1.5
        max_inside_ratio = 0.75
        min_quality = 50
        long_target1_mult = 1.0
        long_target2_mult = 1.5
        short_target1_mult = 1.0
        short_target2_mult = 1.5
        
        avg_range = df['range'].rolling(window=14).mean()
        
        # Current bar
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_range = df['range'].iloc[-1]
        
        # Mother bar
        mother_high = df['high'].iloc[-2]
        mother_low = df['low'].iloc[-2]
        mother_range = df['range'].iloc[-2]
        
        # Inside bar definition
        is_inside = current_high < mother_high and current_low > mother_low
        mother_is_large = mother_range >= (avg_range.iloc[-2] * min_mother_size)
        inside_ratio = current_range / max(mother_range, 0.0001)
        inside_is_tight = inside_ratio <= max_inside_ratio
        
        base_pattern = is_inside and mother_is_large and inside_is_tight
        
        # Quality score
        quality_score = 0.0
        if base_pattern:
            quality_score += 30.0
        
        mother_size_ratio = mother_range / avg_range.iloc[-2]
        if mother_size_ratio >= 2.0:
            quality_score += 20.0
        elif mother_size_ratio >= 1.5:
            quality_score += 15.0
        elif mother_size_ratio >= 1.0:
            quality_score += 10.0
        
        if inside_ratio < 0.4:
            quality_score += 15.0
        elif inside_ratio < 0.6:
            quality_score += 10.0
        elif inside_ratio < 0.75:
            quality_score += 5.0
        
        quality_score += 15.0 + 10.0 + 10.0  # Volume, Trend, S/R
        
        inside_bar_signal = base_pattern and quality_score >= min_quality
        
        # Breakout detection
        was_inside_bar = False
        if len(df) >= 3:
            prev_current_high = df['high'].iloc[-2]
            prev_current_low = df['low'].iloc[-2]
            prev_mother_high = df['high'].iloc[-3]
            prev_mother_low = df['low'].iloc[-3]
            was_inside_bar = prev_current_high < prev_mother_high and prev_current_low > prev_mother_low
        
        bullish_breakout = was_inside_bar and current_close > mother_high
        bearish_breakout = was_inside_bar and current_close < mother_low
        
        # Entry/Stop/Targets
        bull_entry = mother_high
        bull_stop = mother_low
        bull_risk = bull_entry - bull_stop
        bull_target1 = bull_entry + (bull_risk * long_target1_mult)
        bull_target2 = bull_entry + (bull_risk * long_target2_mult)
        
        bear_entry = mother_low
        bear_stop = mother_high
        bear_risk = bear_stop - bear_entry
        bear_target1 = bear_entry - (bear_risk * short_target1_mult)
        bear_target2 = bear_entry - (bear_risk * short_target2_mult)
        
        return {
            "inside_bar_signal": bool(inside_bar_signal),
            "bullish_breakout": bool(bullish_breakout),
            "bearish_breakout": bool(bearish_breakout),
            "quality_score": round(quality_score, 2),
            "mother_high": round(mother_high, 2),
            "mother_low": round(mother_low, 2),
            "inside_ratio": round(inside_ratio * 100, 2),
            "bull_entry": round(bull_entry, 2),
            "bull_stop": round(bull_stop, 2),
            "bull_target1": round(bull_target1, 2),
            "bull_target2": round(bull_target2, 2),
            "bear_entry": round(bear_entry, 2),
            "bear_stop": round(bear_stop, 2),
            "bear_target1": round(bear_target1, 2),
            "bear_target2": round(bear_target2, 2)
        }
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """15 candlestick patterns detection - logic preserved from Pine Script"""
        if len(df) < 3:
            return {"bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
                    "patterns": [], "bull_quality": 0, "bear_quality": 0}
        
        # Settings
        doji_size = 0.05
        body_avg_period = 14
        min_quality = 50
        
        # Current candle
        c_open = df['open'].iloc[-1]
        c_high = df['high'].iloc[-1]
        c_low = df['low'].iloc[-1]
        c_close = df['close'].iloc[-1]
        c_body = abs(c_close - c_open)
        c_range = c_high - c_low
        c_upper_wick = c_high - max(c_open, c_close)
        c_lower_wick = min(c_open, c_close) - c_low
        
        # Previous candles
        p1_open = df['open'].iloc[-2]
        p1_high = df['high'].iloc[-2]
        p1_low = df['low'].iloc[-2]
        p1_close = df['close'].iloc[-2]
        p1_body = abs(p1_close - p1_open)
        
        p2_open = df['open'].iloc[-3]
        p2_close = df['close'].iloc[-3]
        p2_body = abs(p2_close - p2_open)
        
        avg_body = df['body'].rolling(window=body_avg_period).mean().iloc[-1]
        
        is_bullish = c_close > c_open
        is_bearish = c_close < c_open
        p1_bullish = p1_close > p1_open
        p1_bearish = p1_close < p1_open
        p2_bullish = p2_close > p2_open
        p2_bearish = p2_close < p2_open
        
        # Pattern detection
        doji = c_body <= (c_range * doji_size)
        
        hammer = (c_range > 3 * c_body and 
                 (c_close - c_low) / (c_range + 0.001) > 0.6 and 
                 (c_open - c_low) / (c_range + 0.001) > 0.6)
        
        inverted_hammer = (c_range > 3 * c_body and 
                         (c_high - c_close) / (c_range + 0.001) > 0.6 and 
                         (c_high - c_open) / (c_range + 0.001) > 0.6)
        
        hanging_man = (c_range > 4 * c_body and 
                      (c_close - c_low) / (c_range + 0.001) >= 0.75 and 
                      (c_open - c_low) / (c_range + 0.001) >= 0.75 and 
                      p1_high < c_open and df['high'].iloc[-3] < c_open)
        
        shooting_star = (p1_bullish and c_open > p1_close and 
                        c_upper_wick >= abs(c_open - c_close) * 3 and 
                        c_lower_wick <= abs(c_open - c_close))
        
        morning_star = (p2_bearish and 
                       max(p1_open, p1_close) < p2_close and 
                       c_open > max(p1_open, p1_close) and is_bullish)
        
        evening_star = (p2_bullish and 
                       min(p1_open, p1_close) > p2_close and 
                       c_open < min(p1_open, p1_close) and is_bearish)
        
        bullish_engulfing = (p1_bearish and is_bullish and 
                           c_close >= p1_open and p1_close >= c_open and c_body > p1_body)
        
        bearish_engulfing = (p1_bullish and is_bearish and 
                           c_open >= p1_close and p1_open >= c_close and c_body > p1_body)
        
        bullish_harami = (p1_bearish and is_bullish and 
                        c_close <= p1_open and p1_close <= c_open and c_body < p1_body)
        
        bearish_harami = (p1_bullish and is_bearish and 
                        c_open <= p1_close and p1_open <= c_close and c_body < p1_body)
        
        piercing_line = (p1_bearish and c_open < p1_low and 
                        c_close > p1_close + (p1_body / 2) and c_close < p1_open)
        
        dark_cloud = (p1_bullish and ((p1_close + p1_open) / 2) > c_close and 
                     is_bearish and c_open > p1_close and c_close > p1_open and 
                     (c_body / (c_range + 0.001)) > 0.6)
        
        bullish_kicker = p1_bearish and c_open >= p1_open and is_bullish
        bearish_kicker = p1_bullish and c_open <= p1_open and is_bearish
        
        # Quality scoring
        quality_bullish = 50.0 + 20.0 + 20.0 + (10.0 if c_body >= avg_body else 0)
        quality_bearish = 50.0 + 20.0 + 20.0 + (10.0 if c_body >= avg_body else 0)
        quality_neutral = 60.0
        
        # Build patterns list
        patterns_detected = []
        
        if hammer and quality_bullish >= min_quality:
            patterns_detected.append("Hammer")
        if inverted_hammer and quality_bullish >= min_quality:
            patterns_detected.append("Inverted Hammer")
        if morning_star and quality_bullish >= min_quality:
            patterns_detected.append("Morning Star")
        if bullish_engulfing and quality_bullish >= min_quality:
            patterns_detected.append("Bullish Engulfing")
        if bullish_harami and quality_bullish >= min_quality:
            patterns_detected.append("Bullish Harami")
        if piercing_line and quality_bullish >= min_quality:
            patterns_detected.append("Piercing Line")
        if bullish_kicker and quality_bullish >= min_quality:
            patterns_detected.append("Bullish Kicker")
        
        if hanging_man and quality_bearish >= min_quality:
            patterns_detected.append("Hanging Man")
        if shooting_star and quality_bearish >= min_quality:
            patterns_detected.append("Shooting Star")
        if evening_star and quality_bearish >= min_quality:
            patterns_detected.append("Evening Star")
        if bearish_engulfing and quality_bearish >= min_quality:
            patterns_detected.append("Bearish Engulfing")
        if bearish_harami and quality_bearish >= min_quality:
            patterns_detected.append("Bearish Harami")
        if dark_cloud and quality_bearish >= min_quality:
            patterns_detected.append("Dark Cloud")
        if bearish_kicker and quality_bearish >= min_quality:
            patterns_detected.append("Bearish Kicker")
        
        if doji and quality_neutral >= min_quality:
            patterns_detected.append("Doji")
        
        bullish_patterns = ["Hammer", "Inverted Hammer", "Morning Star", "Bullish Engulfing", 
                           "Bullish Harami", "Piercing Line", "Bullish Kicker"]
        bearish_patterns_list = ["Hanging Man", "Shooting Star", "Evening Star", "Bearish Engulfing",
                           "Bearish Harami", "Dark Cloud", "Bearish Kicker"]
        
        bullish_count = sum(1 for p in patterns_detected if p in bullish_patterns)
        bearish_count = sum(1 for p in patterns_detected if p in bearish_patterns_list)
        neutral_count = 1 if "Doji" in patterns_detected else 0
        
        return {
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "patterns": patterns_detected,
            "bull_quality": round(quality_bullish, 2),
            "bear_quality": round(quality_bearish, 2)
        }
    
    def _detect_star_patterns(self, df: pd.DataFrame) -> Dict:
        """Morning Star and Evening Star PRO detection - logic preserved from Pine Script"""
        if len(df) < 50:
            return {
                "bullish_morning_star": False, "bearish_evening_star": False,
                "bullish_quality": 0, "bearish_quality": 0,
                "bullish_high_quality": False, "bearish_high_quality": False,
                "bullish_entry": 0, "bullish_stop": 0, "bullish_target1": 0, "bullish_target2": 0,
                "bearish_entry": 0, "bearish_stop": 0, "bearish_target1": 0, "bearish_target2": 0
            }
        
        # Settings
        body_ratio = 0.333
        volume_multiplier = 1.2
        trend_ma_length = 50
        lookback_bars = 20
        min_quality_score = 60
        risk_reward_ratio = 2.0
        
        # Candle data
        candle1_open = df['open'].iloc[-3]
        candle1_close = df['close'].iloc[-3]
        candle1_high = df['high'].iloc[-3]
        candle1_low = df['low'].iloc[-3]
        
        candle2_open = df['open'].iloc[-2]
        candle2_close = df['close'].iloc[-2]
        candle2_high = df['high'].iloc[-2]
        candle2_low = df['low'].iloc[-2]
        
        candle3_open = df['open'].iloc[-1]
        candle3_close = df['close'].iloc[-1]
        candle3_high = df['high'].iloc[-1]
        candle3_low = df['low'].iloc[-1]
        
        # Helper functions
        avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        trend_ma = df['close'].rolling(window=trend_ma_length).mean().iloc[-1]
        highest_level = df['high'].iloc[-lookback_bars:].max()
        lowest_level = df['low'].iloc[-lookback_bars:].min()
        
        bullish_volume_ok = df['volume'].iloc[-1] > (avg_vol * volume_multiplier)
        bullish_trend_ok = df['close'].iloc[-1] < trend_ma
        distance_to_support = (df['close'].iloc[-1] - lowest_level) / df['close'].iloc[-1]
        bullish_sr_ok = distance_to_support < 0.02
        
        bearish_volume_ok = df['volume'].iloc[-1] > (avg_vol * volume_multiplier)
        bearish_trend_ok = df['close'].iloc[-1] > trend_ma
        distance_to_resistance = (highest_level - df['close'].iloc[-1]) / df['close'].iloc[-1]
        bearish_sr_ok = distance_to_resistance < 0.02
        
        # BULLISH MORNING STAR
        bullish_candle1_body = candle1_open - candle1_close
        bullish_candle2_body = abs(candle2_close - candle2_open)
        bullish_candle2_wick = candle2_high - max(candle2_open, candle2_close)
        bullish_candle3_body = candle3_close - candle3_open
        
        bullish_pattern_base = (
            (candle1_close < candle1_open) and
            (bullish_candle2_body <= (body_ratio * bullish_candle1_body)) and
            (bullish_candle2_wick > bullish_candle2_body) and
            (candle3_close > candle3_open) and
            (bullish_candle2_body <= (body_ratio * bullish_candle3_body)) and
            (candle3_close > candle1_open)
        )
        
        bullish_body_ratio = bullish_candle2_body / max(bullish_candle1_body, bullish_candle3_body) if max(bullish_candle1_body, bullish_candle3_body) > 0 else 1
        
        bullish_quality = 30.0
        if bullish_volume_ok:
            bullish_quality += 25.0
        if bullish_trend_ok:
            bullish_quality += 20.0
        if bullish_sr_ok:
            bullish_quality += 15.0
        if bullish_body_ratio < 0.2:
            bullish_quality += 10.0
        elif bullish_body_ratio < 0.3:
            bullish_quality += 5.0
        
        bullish_star = (bullish_pattern_base and bullish_volume_ok and 
                       bullish_trend_ok and bullish_sr_ok and 
                       bullish_quality >= min_quality_score)
        
        # BEARISH EVENING STAR
        bearish_candle1_body = candle1_close - candle1_open
        bearish_candle2_body = abs(candle2_open - candle2_close)
        bearish_candle2_wick = max(candle2_open, candle2_close) - candle2_low
        bearish_candle3_body = candle3_open - candle3_close
        
        bearish_pattern_base = (
            (candle1_close > candle1_open) and
            (bearish_candle2_body <= (body_ratio * bearish_candle1_body)) and
            (bearish_candle2_wick > bearish_candle2_body) and
            (candle3_close < candle3_open) and
            (bearish_candle2_body <= (body_ratio * bearish_candle3_body)) and
            (candle3_close < candle1_open)
        )
        
        bearish_body_ratio = bearish_candle2_body / max(bearish_candle1_body, bearish_candle3_body) if max(bearish_candle1_body, bearish_candle3_body) > 0 else 1
        
        bearish_quality = 30.0
        if bearish_volume_ok:
            bearish_quality += 25.0
        if bearish_trend_ok:
            bearish_quality += 20.0
        if bearish_sr_ok:
            bearish_quality += 15.0
        if bearish_body_ratio < 0.2:
            bearish_quality += 10.0
        elif bearish_body_ratio < 0.3:
            bearish_quality += 5.0
        
        bearish_star = (bearish_pattern_base and bearish_volume_ok and
                       bearish_trend_ok and bearish_sr_ok and
                       bearish_quality >= min_quality_score)
        
        # Risk/Reward calculations
        bullish_entry = candle3_close
        bullish_stop = min(candle1_low, candle2_low, candle3_low)
        bullish_risk = bullish_entry - bullish_stop
        bullish_target1 = bullish_entry + (bullish_risk * risk_reward_ratio)
        bullish_target2 = bullish_entry + (bullish_risk * risk_reward_ratio * 1.5)
        
        bearish_entry = candle3_close
        bearish_stop = max(candle1_high, candle2_high, candle3_high)
        bearish_risk = bearish_stop - bearish_entry
        bearish_target1 = bearish_entry - (bearish_risk * risk_reward_ratio)
        bearish_target2 = bearish_entry - (bearish_risk * risk_reward_ratio * 1.5)
        
        return {
            "bullish_morning_star": bool(bullish_star),
            "bearish_evening_star": bool(bearish_star),
            "bullish_quality": round(bullish_quality, 2),
            "bearish_quality": round(bearish_quality, 2),
            "bullish_high_quality": bool(bullish_star and bullish_quality >= 80),
            "bearish_high_quality": bool(bearish_star and bearish_quality >= 80),
            "bullish_entry": round(bullish_entry, 2),
            "bullish_stop": round(bullish_stop, 2),
            "bullish_target1": round(bullish_target1, 2),
            "bullish_target2": round(bullish_target2, 2),
            "bearish_entry": round(bearish_entry, 2),
            "bearish_stop": round(bearish_stop, 2),
            "bearish_target1": round(bearish_target1, 2),
            "bearish_target2": round(bearish_target2, 2)
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "tws_detected": False, "tws_quality": 0, "tws_high_quality": False,
            "tws_pattern_base": False, "tws_volume_strong": False, "tws_after_downtrend": False,
            "tws_near_support": False, "tws_entry": 0, "tws_stop": 0, "tws_target1": 0, "tws_target2": 0,
            "ib_detected": False, "ib_bullish_breakout": False, "ib_bearish_breakout": False,
            "ib_quality": 0, "ib_mother_high": 0, "ib_mother_low": 0, "ib_inside_ratio_pct": 0,
            "ib_bull_entry": 0, "ib_bull_stop": 0, "ib_bull_target1": 0, "ib_bull_target2": 0,
            "ib_bear_entry": 0, "ib_bear_stop": 0, "ib_bear_target1": 0, "ib_bear_target2": 0,
            "patterns_detected": [], "bullish_patterns": [], "bearish_patterns": [],
            "bullish_pattern_count": 0, "bearish_pattern_count": 0, "neutral_pattern_count": 0,
            "doji_detected": False, "hammer_detected": False, "inverted_hammer_detected": False,
            "hanging_man_detected": False, "shooting_star_detected": False,
            "morning_star_detected": False, "evening_star_detected": False,
            "bullish_engulfing_detected": False, "bearish_engulfing_detected": False,
            "bullish_harami_detected": False, "bearish_harami_detected": False,
            "piercing_line_detected": False, "dark_cloud_detected": False,
            "bullish_kicker_detected": False, "bearish_kicker_detected": False,
            "patterns_bull_quality": 0, "patterns_bear_quality": 0,
            "morning_star_pro_detected": False, "evening_star_pro_detected": False,
            "morning_star_quality": 0, "evening_star_quality": 0,
            "morning_star_high_quality": False, "evening_star_high_quality": False,
            "morning_star_entry": 0, "morning_star_stop": 0, "morning_star_target1": 0, "morning_star_target2": 0,
            "evening_star_entry": 0, "evening_star_stop": 0, "evening_star_target1": 0, "evening_star_target2": 0,
            "total_bullish_patterns": 0, "total_bearish_patterns": 0, "total_patterns_detected": 0,
            "current_candle_bullish": None, "current_candle_bearish": None,
            "current_body_size": 0, "current_range": 0, "current_upper_wick": 0, "current_lower_wick": 0,
            "avg_body": 0, "body_vs_avg_ratio": 0, "current_price": None, "error": reason
        }
