"""
Layer 13: Volume Profile Analysis Engine
Institutional-grade Volume Profile with POC, Value Area, and trading signals
Converted from Pine Script - Logic unchanged
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Layer13VolumeProfile:
    """
    Professional Volume Profile trading system.
    
    Components:
    1. Volume Profile - Distribution of volume at price levels
    2. POC (Point of Control) - Price level with highest volume
    3. Value Area - 70% of volume (VAH/VAL boundaries)
    4. Volume Filter - High volume detection
    5. POC Touch/Rejection Detection - Institutional behavior
    6. Acceptance Tracking - Price acceptance zones
    7. Buy Pressure - Buying vs selling pressure at levels
    8. Trading Signals - 6 professional setups
    """
    
    def __init__(self):
        # ==================== Profile Settings ====================
        self.anchor_mode = "Session"  # "Session", "Week", "Month", "Last N Bars"
        self.custom_length = 100  # For "Last N Bars" mode
        self.num_levels = 24  # Number of price levels
        self.va_percent = 70  # Value area percent
        
        # ==================== Filter Settings ====================
        self.min_volume_mult = 2.0  # Minimum volume multiplier
        self.min_touches = 3  # Minimum touches for strong POC
        self.acceptance_bars = 3  # Bars for acceptance
        self.rejection_wick_pct = 60  # Rejection wick percent
        
        # ==================== Thresholds ====================
        self.poc_touch_threshold = 0.005  # 0.5%
        self.val_vah_touch_threshold = 0.003  # 0.3%
        self.touch_min_gap_bars = 5  # Minimum bars between touches
        self.volume_sma_period = 20
        self.buy_pressure_bull_threshold = 65
        self.buy_pressure_bear_threshold = 35
        
        # ==================== State Variables ====================
        self.anchor_bar = 0
        self.bars_since_anchor = 0
        self.poc_touches = 0
        self.last_poc_touch_bar = 0
        self.bars_at_poc = 0
        self.bars_above_vah = 0
        self.bars_below_val = 0
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Run complete volume profile analysis
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dict with profile, POC, value area, and signals
        """
        if len(df) < 20:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Determine anchor and range
        anchor_info = self._determine_anchor(df)
        range_length = anchor_info['range_length']
        
        if range_length < 1:
            return self._empty_result("Invalid range")
        
        # Calculate profile
        profile = self._calculate_volume_profile(df, range_length)
        
        if not profile['is_valid']:
            return self._empty_result("Profile calculation failed")
        
        # Calculate POC
        poc = self._calculate_poc(profile)
        
        # Calculate Value Area
        value_area = self._calculate_value_area(profile, poc)
        
        # Volume analysis
        volume_analysis = self._analyze_volume(df)
        
        # POC touch detection
        poc_touch_analysis = self._analyze_poc_touches(df, poc)
        
        # Rejection detection
        rejection_analysis = self._analyze_rejections(df, poc)
        
        # Acceptance tracking
        acceptance = self._track_acceptance(df, poc, value_area)
        
        # Buy pressure
        buy_pressure = self._calculate_buy_pressure(df, profile)
        
        # Position classification
        position = self._classify_position(df, value_area)
        
        # Trading signals
        signals = self._generate_signals(
            df, poc, value_area, volume_analysis,
            poc_touch_analysis, rejection_analysis,
            acceptance, buy_pressure
        )
        
        return {
            "anchor": anchor_info,
            "profile": profile,
            "poc": poc,
            "value_area": value_area,
            "volume": volume_analysis,
            "poc_touches": poc_touch_analysis,
            "rejections": rejection_analysis,
            "acceptance": acceptance,
            "buy_pressure": buy_pressure,
            "position": position,
            "signals": signals,
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== ANCHOR DETERMINATION ====================
    
    def _determine_anchor(self, df: pd.DataFrame) -> Dict:
        """Determine anchor point and range length"""
        if self.anchor_mode == "Last N Bars":
            range_length = min(self.custom_length, len(df))
            anchor_bar = len(df) - range_length
        else:
            # For time-based anchors, use full available data
            # In production, this would track session/week/month changes
            range_length = len(df)
            anchor_bar = 0
        
        return {
            'mode': self.anchor_mode,
            'anchor_bar': anchor_bar,
            'range_length': range_length,
            'bars_since_anchor': range_length
        }
    
    # ==================== VOLUME PROFILE CALCULATION ====================
    
    def _calculate_volume_profile(self, df: pd.DataFrame, range_length: int) -> Dict:
        """
        Calculate volume profile distribution
        
        Exact Pine Script logic:
        - Divide price range into num_levels
        - Distribute volume to each level
        - Track buy volume (close > open)
        """
        # Get range data
        range_data = df.iloc[-range_length:]
        
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        price_range = range_high - range_low
        
        if price_range <= 0:
            return {'is_valid': False}
        
        level_height = price_range / self.num_levels
        
        # Initialize arrays
        volume_at_price = np.zeros(self.num_levels)
        buy_volume_at_price = np.zeros(self.num_levels)
        
        # Distribute volume
        for idx in range(len(range_data)):
            close_price = range_data['close'].iloc[idx]
            open_price = range_data['open'].iloc[idx]
            vol = range_data['volume'].iloc[idx]
            
            # Determine price level
            price_level = int(np.floor((close_price - range_low) / level_height))
            price_level = max(0, min(self.num_levels - 1, price_level))
            
            # Add volume
            volume_at_price[price_level] += vol
            
            # Track buy volume
            if close_price > open_price:
                buy_volume_at_price[price_level] += vol
        
        return {
            'is_valid': True,
            'range_high': range_high,
            'range_low': range_low,
            'price_range': price_range,
            'level_height': level_height,
            'volume_at_price': volume_at_price,
            'buy_volume_at_price': buy_volume_at_price,
            'total_volume': volume_at_price.sum(),
            'levels': self.num_levels
        }
    
    # ==================== POC CALCULATION ====================
    
    def _calculate_poc(self, profile: Dict) -> Dict:
        """
        Calculate Point of Control (price with max volume)
        
        Exact Pine Script logic:
        - Find level with maximum volume
        - POC price = center of that level
        """
        if not profile['is_valid']:
            return {'price': None, 'level': None, 'volume': 0}
        
        volume_at_price = profile['volume_at_price']
        
        # Find max volume level
        max_volume = 0.0
        poc_level = 0
        
        for i in range(len(volume_at_price)):
            vol = volume_at_price[i]
            if vol > max_volume:
                max_volume = vol
                poc_level = i
        
        # Calculate POC price (center of level)
        level_height = profile['level_height']
        range_low = profile['range_low']
        poc_price = range_low + (poc_level * level_height) + (level_height / 2)
        
        return {
            'price': poc_price,
            'level': poc_level,
            'volume': max_volume,
            'volume_pct': (max_volume / profile['total_volume'] * 100) if profile['total_volume'] > 0 else 0
        }
    
    # ==================== VALUE AREA CALCULATION ====================
    
    def _calculate_value_area(self, profile: Dict, poc: Dict) -> Dict:
        """
        Calculate Value Area (70% of volume)
        
        Exact Pine Script logic:
        - Start at POC
        - Expand up/down based on higher volume
        - Stop when reaching va_percent of total volume
        """
        if not profile['is_valid'] or poc['level'] is None:
            return {'vah_price': None, 'val_price': None}
        
        volume_at_price = profile['volume_at_price']
        total_volume = profile['total_volume']
        
        if total_volume == 0:
            return {'vah_price': None, 'val_price': None}
        
        va_target = total_volume * (self.va_percent / 100)
        va_volume = volume_at_price[poc['level']]
        
        va_high_level = poc['level']
        va_low_level = poc['level']
        
        num_levels = len(volume_at_price)
        
        # Expand until reaching target
        iterations = 0
        max_iterations = num_levels * 2
        
        while va_volume < va_target and iterations < max_iterations:
            iterations += 1
            
            # Volume above
            vol_above = volume_at_price[va_high_level + 1] if va_high_level < num_levels - 1 else 0.0
            
            # Volume below
            vol_below = volume_at_price[va_low_level - 1] if va_low_level > 0 else 0.0
            
            # Expand to side with more volume
            if vol_above > vol_below and va_high_level < num_levels - 1:
                va_high_level += 1
                va_volume += vol_above
            elif vol_below > 0 and va_low_level > 0:
                va_low_level -= 1
                va_volume += vol_below
            elif vol_above > 0 and va_high_level < num_levels - 1:
                va_high_level += 1
                va_volume += vol_above
            else:
                break
        
        # Calculate prices
        level_height = profile['level_height']
        range_low = profile['range_low']
        
        vah_price = range_low + (va_high_level * level_height) + level_height
        val_price = range_low + (va_low_level * level_height)
        
        return {
            'vah_price': vah_price,
            'val_price': val_price,
            'vah_level': va_high_level,
            'val_level': va_low_level,
            'va_volume': va_volume,
            'va_volume_pct': (va_volume / total_volume * 100) if total_volume > 0 else 0
        }
    
    # ==================== VOLUME ANALYSIS ====================
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Analyze volume levels"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(self.volume_sma_period).mean().iloc[-1]
        
        is_high_volume = current_volume > avg_volume * self.min_volume_mult
        volume_spike = current_volume > avg_volume * (self.min_volume_mult * 1.5)
        
        return {
            'current': current_volume,
            'avg_20': avg_volume,
            'is_high_volume': is_high_volume,
            'volume_spike': volume_spike,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0
        }
    
    # ==================== POC TOUCH DETECTION ====================
    
    def _analyze_poc_touches(self, df: pd.DataFrame, poc: Dict) -> Dict:
        """
        Track POC touches
        
        Exact Pine Script logic:
        - Touch if close within 0.5% of POC
        - Minimum 5 bars between touches
        """
        if poc['price'] is None:
            return {'touching': False, 'count': 0, 'is_strong': False}
        
        close = df['close'].iloc[-1]
        poc_price = poc['price']
        
        distance_to_poc = abs(close - poc_price) / poc_price
        touching_poc = distance_to_poc < self.poc_touch_threshold
        
        # Update touch count (in production, maintain state)
        current_bar = len(df) - 1
        
        if touching_poc and (current_bar - self.last_poc_touch_bar) > self.touch_min_gap_bars:
            self.poc_touches += 1
            self.last_poc_touch_bar = current_bar
        
        poc_is_strong = self.poc_touches >= self.min_touches
        
        return {
            'touching': touching_poc,
            'count': self.poc_touches,
            'is_strong': poc_is_strong,
            'distance_pct': distance_to_poc * 100
        }
    
    # ==================== REJECTION DETECTION ====================
    
    def _analyze_rejections(self, df: pd.DataFrame, poc: Dict) -> Dict:
        """
        Detect rejections from POC
        
        Exact Pine Script logic:
        - Upper wick > 60% of candle range = rejection from above
        - Lower wick > 60% of candle range = rejection from below
        """
        last_bar = df.iloc[-1]
        
        high = last_bar['high']
        low = last_bar['low']
        open_price = last_bar['open']
        close = last_bar['close']
        
        candle_range = high - low
        
        if candle_range == 0:
            return {
                'rejection_from_above': False,
                'rejection_from_below': False,
                'poc_rejection_bull': False,
                'poc_rejection_bear': False
            }
        
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        rejection_from_above = (upper_wick / candle_range) * 100 > self.rejection_wick_pct
        rejection_from_below = (lower_wick / candle_range) * 100 > self.rejection_wick_pct
        
        # POC-specific rejections
        touching_poc = poc['price'] is not None and abs(close - poc['price']) / poc['price'] < self.poc_touch_threshold
        
        poc_rejection_bull = touching_poc and rejection_from_below and close > open_price
        poc_rejection_bear = touching_poc and rejection_from_above and close < open_price
        
        return {
            'rejection_from_above': rejection_from_above,
            'rejection_from_below': rejection_from_below,
            'poc_rejection_bull': poc_rejection_bull,
            'poc_rejection_bear': poc_rejection_bear,
            'upper_wick_pct': (upper_wick / candle_range * 100) if candle_range > 0 else 0,
            'lower_wick_pct': (lower_wick / candle_range * 100) if candle_range > 0 else 0
        }
    
    # ==================== ACCEPTANCE TRACKING ====================
    
    def _track_acceptance(self, df: pd.DataFrame, poc: Dict, value_area: Dict) -> Dict:
        """
        Track acceptance at key levels
        
        Exact Pine Script logic:
        - Increment counter if at level, reset if not
        - Acceptance = counter >= acceptance_bars
        """
        close = df['close'].iloc[-1]
        
        # POC acceptance
        touching_poc = poc['price'] is not None and abs(close - poc['price']) / poc['price'] < self.poc_touch_threshold
        
        if touching_poc:
            self.bars_at_poc += 1
        else:
            self.bars_at_poc = 0
        
        accepted_at_poc = self.bars_at_poc >= self.acceptance_bars
        
        # Above VAH acceptance
        if value_area['vah_price'] is not None and close > value_area['vah_price']:
            self.bars_above_vah += 1
        else:
            self.bars_above_vah = 0
        
        accepted_above_va = self.bars_above_vah >= self.acceptance_bars
        
        # Below VAL acceptance
        if value_area['val_price'] is not None and close < value_area['val_price']:
            self.bars_below_val += 1
        else:
            self.bars_below_val = 0
        
        accepted_below_va = self.bars_below_val >= self.acceptance_bars
        
        return {
            'bars_at_poc': self.bars_at_poc,
            'bars_above_vah': self.bars_above_vah,
            'bars_below_val': self.bars_below_val,
            'accepted_at_poc': accepted_at_poc,
            'accepted_above_va': accepted_above_va,
            'accepted_below_va': accepted_below_va
        }
    
    # ==================== BUY PRESSURE ====================
    
    def _calculate_buy_pressure(self, df: pd.DataFrame, profile: Dict) -> Dict:
        """
        Calculate buy pressure at current price level
        
        Exact Pine Script logic:
        - Buy pressure = (buy_volume / total_volume) * 100
        - Strong buying > 65%, Strong selling < 35%
        """
        if not profile['is_valid']:
            return {'buy_pressure': 50.0, 'strong_buying': False, 'strong_selling': False}
        
        close = df['close'].iloc[-1]
        range_low = profile['range_low']
        level_height = profile['level_height']
        
        # Determine current level
        current_level = int(np.floor((close - range_low) / level_height))
        current_level = max(0, min(self.num_levels - 1, current_level))
        
        level_buy_vol = profile['buy_volume_at_price'][current_level]
        level_total_vol = profile['volume_at_price'][current_level]
        
        buy_pressure = (level_buy_vol / level_total_vol * 100) if level_total_vol > 0 else 50.0
        
        strong_buying = buy_pressure > self.buy_pressure_bull_threshold
        strong_selling = buy_pressure < self.buy_pressure_bear_threshold
        
        return {
            'current_level': current_level,
            'buy_pressure': buy_pressure,
            'strong_buying': strong_buying,
            'strong_selling': strong_selling,
            'level_volume': level_total_vol,
            'level_buy_volume': level_buy_vol
        }
    
    # ==================== POSITION CLASSIFICATION ====================
    
    def _classify_position(self, df: pd.DataFrame, value_area: Dict) -> Dict:
        """Classify price position relative to value area"""
        close = df['close'].iloc[-1]
        
        if value_area['val_price'] is None or value_area['vah_price'] is None:
            return {
                'in_value_area': False,
                'above_value_area': False,
                'below_value_area': False,
                'location': 'UNKNOWN'
            }
        
        val_price = value_area['val_price']
        vah_price = value_area['vah_price']
        
        in_value_area = close >= val_price and close <= vah_price
        above_value_area = close > vah_price
        below_value_area = close < val_price
        
        if above_value_area:
            location = "ABOVE VA"
        elif below_value_area:
            location = "BELOW VA"
        elif in_value_area:
            location = "IN VA"
        else:
            location = "UNKNOWN"
        
        return {
            'in_value_area': in_value_area,
            'above_value_area': above_value_area,
            'below_value_area': below_value_area,
            'location': location,
            'distance_to_val': close - val_price,
            'distance_to_vah': close - vah_price
        }
    
    # ==================== SIGNAL GENERATION ====================
    
    def _generate_signals(self, df: pd.DataFrame, poc: Dict, value_area: Dict,
                         volume: Dict, poc_touches: Dict, rejections: Dict,
                         acceptance: Dict, buy_pressure: Dict) -> Dict:
        """
        Generate trading signals
        
        Exact Pine Script logic:
        6 signal types:
        1. Long POC bounce
        2. Long VAL support
        3. Long breakout
        4. Short POC rejection
        5. Short VAH resistance
        6. Short breakdown
        """
        close = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-2] if len(df) >= 2 else close
        
        # Crossover/Crossunder detection
        cross_above_vah = (value_area['vah_price'] is not None and
                          close > value_area['vah_price'] and
                          close_prev <= value_area['vah_price'])
        
        cross_below_val = (value_area['val_price'] is not None and
                          close < value_area['val_price'] and
                          close_prev >= value_area['val_price'])
        
        # ==================== LONG SIGNALS ====================
        
        # 1. POC Bounce
        long_poc_bounce = (rejections['poc_rejection_bull'] and
                          volume['is_high_volume'] and
                          poc_touches['is_strong'] and
                          buy_pressure['strong_buying'])
        
        # 2. VAL Support
        long_val_support = False
        if value_area['val_price'] is not None:
            distance_to_val = abs(close - value_area['val_price']) / value_area['val_price']
            long_val_support = (distance_to_val < self.val_vah_touch_threshold and
                               rejections['rejection_from_below'] and
                               volume['is_high_volume'] and
                               buy_pressure['strong_buying'])
        
        # 3. Breakout
        long_breakout = (acceptance['accepted_above_va'] and
                        cross_above_vah and
                        volume['volume_spike'])
        
        # ==================== SHORT SIGNALS ====================
        
        # 4. POC Rejection
        short_poc_rejection = (rejections['poc_rejection_bear'] and
                              volume['is_high_volume'] and
                              poc_touches['is_strong'] and
                              buy_pressure['strong_selling'])
        
        # 5. VAH Resistance
        short_vah_resistance = False
        if value_area['vah_price'] is not None:
            distance_to_vah = abs(close - value_area['vah_price']) / value_area['vah_price']
            short_vah_resistance = (distance_to_vah < self.val_vah_touch_threshold and
                                   rejections['rejection_from_above'] and
                                   volume['is_high_volume'] and
                                   buy_pressure['strong_selling'])
        
        # 6. Breakdown
        short_breakdown = (acceptance['accepted_below_va'] and
                          cross_below_val and
                          volume['volume_spike'])
        
        # ==================== COMBINED SIGNALS ====================
        
        long_signal = long_poc_bounce or long_val_support or long_breakout
        short_signal = short_poc_rejection or short_vah_resistance or short_breakdown
        
        # Signal type
        signal_type = None
        if long_poc_bounce:
            signal_type = "LONG_POC_BOUNCE"
        elif long_val_support:
            signal_type = "LONG_VAL_SUPPORT"
        elif long_breakout:
            signal_type = "LONG_BREAKOUT"
        elif short_poc_rejection:
            signal_type = "SHORT_POC_REJECTION"
        elif short_vah_resistance:
            signal_type = "SHORT_VAH_RESISTANCE"
        elif short_breakdown:
            signal_type = "SHORT_BREAKDOWN"
        
        return {
            'long_poc_bounce': long_poc_bounce,
            'long_val_support': long_val_support,
            'long_breakout': long_breakout,
            'short_poc_rejection': short_poc_rejection,
            'short_vah_resistance': short_vah_resistance,
            'short_breakdown': short_breakdown,
            'long_signal': long_signal,
            'short_signal': short_signal,
            'signal_type': signal_type
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "error": reason,
            "anchor": {},
            "profile": {'is_valid': False},
            "poc": {'price': None},
            "value_area": {'vah_price': None, 'val_price': None},
            "volume": {},
            "poc_touches": {'touching': False, 'count': 0},
            "rejections": {},
            "acceptance": {},
            "buy_pressure": {},
            "position": {},
            "signals": {'long_signal': False, 'short_signal': False}
        }
