"""
Layer 13: Volume Profile Analysis Engine (Raw Data Output)
Institutional-grade Volume Profile with POC, Value Area
Outputs RAW volume profile data only - no signals
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
    """
    
    def __init__(self):
        # Profile Settings
        self.anchor_mode = "Session"
        self.custom_length = 100
        self.num_levels = 24
        self.va_percent = 70
        
        # Filter Settings
        self.min_volume_mult = 2.0
        self.min_touches = 3
        self.acceptance_bars = 3
        self.rejection_wick_pct = 60
        
        # Thresholds
        self.poc_touch_threshold = 0.005
        self.val_vah_touch_threshold = 0.003
        self.touch_min_gap_bars = 5
        self.volume_sma_period = 20
        self.buy_pressure_bull_threshold = 65
        self.buy_pressure_bear_threshold = 35
        
        # State Variables
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
            Dict with RAW profile, POC, value area data
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
        
        # Crossover data (raw facts)
        crossover_data = self._calculate_crossovers(df, value_area)
        
        current_price = df['close'].iloc[-1]
        
        # Return RAW DATA ONLY - no signals
        return {
            # POC Data
            "poc_price": round(poc['price'], 2) if poc['price'] else None,
            "poc_level": poc['level'],
            "poc_volume": round(poc['volume'], 0) if poc['volume'] else 0,
            "poc_volume_pct": round(poc['volume_pct'], 2),
            
            # Value Area Data
            "vah_price": round(value_area['vah_price'], 2) if value_area['vah_price'] else None,
            "val_price": round(value_area['val_price'], 2) if value_area['val_price'] else None,
            "va_volume_pct": round(value_area.get('va_volume_pct', 0), 2),
            "va_width": round(value_area['vah_price'] - value_area['val_price'], 2) if value_area['vah_price'] and value_area['val_price'] else None,
            
            # Profile Range Data
            "profile_high": round(profile['range_high'], 2),
            "profile_low": round(profile['range_low'], 2),
            "profile_range": round(profile['price_range'], 2),
            "profile_levels": profile['levels'],
            "total_volume": round(profile['total_volume'], 0),
            
            # Price vs POC
            "price_vs_poc": round(current_price - poc['price'], 2) if poc['price'] else None,
            "price_vs_poc_pct": round((current_price - poc['price']) / poc['price'] * 100, 2) if poc['price'] and poc['price'] != 0 else None,
            "price_above_poc": current_price > poc['price'] if poc['price'] else None,
            "price_below_poc": current_price < poc['price'] if poc['price'] else None,
            
            # Price vs Value Area
            "price_vs_vah": round(current_price - value_area['vah_price'], 2) if value_area['vah_price'] else None,
            "price_vs_val": round(current_price - value_area['val_price'], 2) if value_area['val_price'] else None,
            "in_value_area": position['in_value_area'],
            "above_value_area": position['above_value_area'],
            "below_value_area": position['below_value_area'],
            "position_location": position['location'],
            
            # POC Touch Data
            "touching_poc": poc_touch_analysis['touching'],
            "poc_touch_count": poc_touch_analysis['count'],
            "poc_is_strong": poc_touch_analysis['is_strong'],
            "distance_to_poc_pct": round(poc_touch_analysis['distance_pct'], 2),
            
            # VAH/VAL Touch Data
            "touching_vah": self._is_touching_level(current_price, value_area['vah_price'], self.val_vah_touch_threshold),
            "touching_val": self._is_touching_level(current_price, value_area['val_price'], self.val_vah_touch_threshold),
            
            # Rejection Data
            "rejection_from_above": rejection_analysis['rejection_from_above'],
            "rejection_from_below": rejection_analysis['rejection_from_below'],
            "poc_rejection_bull": rejection_analysis['poc_rejection_bull'],
            "poc_rejection_bear": rejection_analysis['poc_rejection_bear'],
            "upper_wick_pct": round(rejection_analysis['upper_wick_pct'], 2),
            "lower_wick_pct": round(rejection_analysis['lower_wick_pct'], 2),
            
            # Acceptance Data
            "bars_at_poc": acceptance['bars_at_poc'],
            "bars_above_vah": acceptance['bars_above_vah'],
            "bars_below_val": acceptance['bars_below_val'],
            "accepted_at_poc": acceptance['accepted_at_poc'],
            "accepted_above_va": acceptance['accepted_above_va'],
            "accepted_below_va": acceptance['accepted_below_va'],
            
            # Buy Pressure Data
            "buy_pressure_pct": round(buy_pressure['buy_pressure'], 2),
            "strong_buying": buy_pressure['strong_buying'],
            "strong_selling": buy_pressure['strong_selling'],
            "current_level_volume": round(buy_pressure['level_volume'], 0),
            "current_level_buy_volume": round(buy_pressure['level_buy_volume'], 0),
            
            # Volume Data
            "current_volume": round(volume_analysis['current'], 0),
            "avg_volume_20": round(volume_analysis['avg_20'], 0),
            "volume_ratio": round(volume_analysis['volume_ratio'], 2),
            "is_high_volume": volume_analysis['is_high_volume'],
            "volume_spike": volume_analysis['volume_spike'],
            
            # Crossover Data (raw facts)
            "crossed_above_vah": crossover_data['crossed_above_vah'],
            "crossed_below_vah": crossover_data['crossed_below_vah'],
            "crossed_above_val": crossover_data['crossed_above_val'],
            "crossed_below_val": crossover_data['crossed_below_val'],
            "crossed_above_poc": crossover_data['crossed_above_poc'],
            "crossed_below_poc": crossover_data['crossed_below_poc'],
            
            # Anchor Info
            "anchor_mode": anchor_info['mode'],
            "bars_in_profile": anchor_info['range_length'],
            
            # Price Context
            "current_price": round(current_price, 2),
            
            # Timestamp
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== HELPER ====================
    
    def _is_touching_level(self, price: float, level: Optional[float], threshold: float) -> bool:
        """Check if price is touching a level"""
        if level is None or level == 0:
            return False
        return abs(price - level) / level < threshold
    
    # ==================== ANCHOR DETERMINATION ====================
    
    def _determine_anchor(self, df: pd.DataFrame) -> Dict:
        """Determine anchor point and range length"""
        if self.anchor_mode == "Last N Bars":
            range_length = min(self.custom_length, len(df))
            anchor_bar = len(df) - range_length
        else:
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
        """Calculate volume profile distribution"""
        range_data = df.iloc[-range_length:]
        
        range_high = range_data['high'].max()
        range_low = range_data['low'].min()
        price_range = range_high - range_low
        
        if price_range <= 0:
            return {'is_valid': False}
        
        level_height = price_range / self.num_levels
        
        volume_at_price = np.zeros(self.num_levels)
        buy_volume_at_price = np.zeros(self.num_levels)
        
        for idx in range(len(range_data)):
            close_price = range_data['close'].iloc[idx]
            open_price = range_data['open'].iloc[idx]
            vol = range_data['volume'].iloc[idx]
            
            price_level = int(np.floor((close_price - range_low) / level_height))
            price_level = max(0, min(self.num_levels - 1, price_level))
            
            volume_at_price[price_level] += vol
            
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
        """Calculate Point of Control (price with max volume)"""
        if not profile['is_valid']:
            return {'price': None, 'level': None, 'volume': 0, 'volume_pct': 0}
        
        volume_at_price = profile['volume_at_price']
        
        max_volume = 0.0
        poc_level = 0
        
        for i in range(len(volume_at_price)):
            vol = volume_at_price[i]
            if vol > max_volume:
                max_volume = vol
                poc_level = i
        
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
        """Calculate Value Area (70% of volume)"""
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
        
        iterations = 0
        max_iterations = num_levels * 2
        
        while va_volume < va_target and iterations < max_iterations:
            iterations += 1
            
            vol_above = volume_at_price[va_high_level + 1] if va_high_level < num_levels - 1 else 0.0
            vol_below = volume_at_price[va_low_level - 1] if va_low_level > 0 else 0.0
            
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
        """Track POC touches"""
        if poc['price'] is None:
            return {'touching': False, 'count': 0, 'is_strong': False, 'distance_pct': 100}
        
        close = df['close'].iloc[-1]
        poc_price = poc['price']
        
        distance_to_poc = abs(close - poc_price) / poc_price
        touching_poc = distance_to_poc < self.poc_touch_threshold
        
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
        """Detect rejections from POC"""
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
                'poc_rejection_bear': False,
                'upper_wick_pct': 0,
                'lower_wick_pct': 0
            }
        
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        rejection_from_above = (upper_wick / candle_range) * 100 > self.rejection_wick_pct
        rejection_from_below = (lower_wick / candle_range) * 100 > self.rejection_wick_pct
        
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
        """Track acceptance at key levels"""
        close = df['close'].iloc[-1]
        
        touching_poc = poc['price'] is not None and abs(close - poc['price']) / poc['price'] < self.poc_touch_threshold
        
        if touching_poc:
            self.bars_at_poc += 1
        else:
            self.bars_at_poc = 0
        
        accepted_at_poc = self.bars_at_poc >= self.acceptance_bars
        
        if value_area['vah_price'] is not None and close > value_area['vah_price']:
            self.bars_above_vah += 1
        else:
            self.bars_above_vah = 0
        
        accepted_above_va = self.bars_above_vah >= self.acceptance_bars
        
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
        """Calculate buy pressure at current price level"""
        if not profile['is_valid']:
            return {'buy_pressure': 50.0, 'strong_buying': False, 'strong_selling': False,
                    'level_volume': 0, 'level_buy_volume': 0}
        
        close = df['close'].iloc[-1]
        range_low = profile['range_low']
        level_height = profile['level_height']
        
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
    
    # ==================== CROSSOVER CALCULATION ====================
    
    def _calculate_crossovers(self, df: pd.DataFrame, value_area: Dict) -> Dict:
        """Calculate crossover data - raw facts only"""
        if len(df) < 2:
            return {
                'crossed_above_vah': False, 'crossed_below_vah': False,
                'crossed_above_val': False, 'crossed_below_val': False,
                'crossed_above_poc': False, 'crossed_below_poc': False
            }
        
        close = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        
        vah = value_area.get('vah_price')
        val = value_area.get('val_price')
        
        def safe_cross_above(curr, prev, level):
            if level is None:
                return False
            return curr > level and prev <= level
        
        def safe_cross_below(curr, prev, level):
            if level is None:
                return False
            return curr < level and prev >= level
        
        return {
            'crossed_above_vah': safe_cross_above(close, close_prev, vah),
            'crossed_below_vah': safe_cross_below(close, close_prev, vah),
            'crossed_above_val': safe_cross_above(close, close_prev, val),
            'crossed_below_val': safe_cross_below(close, close_prev, val),
            'crossed_above_poc': False,  # Would need POC passed in
            'crossed_below_poc': False
        }
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "poc_price": None, "poc_level": None, "poc_volume": 0, "poc_volume_pct": 0,
            "vah_price": None, "val_price": None, "va_volume_pct": 0, "va_width": None,
            "profile_high": None, "profile_low": None, "profile_range": None,
            "profile_levels": 0, "total_volume": 0,
            "price_vs_poc": None, "price_vs_poc_pct": None,
            "price_above_poc": None, "price_below_poc": None,
            "price_vs_vah": None, "price_vs_val": None,
            "in_value_area": False, "above_value_area": False, "below_value_area": False,
            "position_location": "UNKNOWN",
            "touching_poc": False, "poc_touch_count": 0, "poc_is_strong": False, "distance_to_poc_pct": 100,
            "touching_vah": False, "touching_val": False,
            "rejection_from_above": False, "rejection_from_below": False,
            "poc_rejection_bull": False, "poc_rejection_bear": False,
            "upper_wick_pct": 0, "lower_wick_pct": 0,
            "bars_at_poc": 0, "bars_above_vah": 0, "bars_below_val": 0,
            "accepted_at_poc": False, "accepted_above_va": False, "accepted_below_va": False,
            "buy_pressure_pct": 50, "strong_buying": False, "strong_selling": False,
            "current_level_volume": 0, "current_level_buy_volume": 0,
            "current_volume": 0, "avg_volume_20": 0, "volume_ratio": 1.0,
            "is_high_volume": False, "volume_spike": False,
            "crossed_above_vah": False, "crossed_below_vah": False,
            "crossed_above_val": False, "crossed_below_val": False,
            "crossed_above_poc": False, "crossed_below_poc": False,
            "anchor_mode": None, "bars_in_profile": 0,
            "current_price": None, "error": reason
        }
