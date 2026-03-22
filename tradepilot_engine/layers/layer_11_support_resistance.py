"""
Layer 11: Support/Resistance Engine (Raw Data Output)
Complete S/R system with 4 professional indicators
Outputs RAW S/R data only - no signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Layer11SupportResistance:
    """
    Professional Support/Resistance detection system.
    
    Components:
    1. Williams Fractals - Swing high/low detection
    2. Dynamic S/R Channels - Pivot-based channels with strength scoring
    3. Pivot Points - Daily/Weekly/Monthly PP, R1-R3, S1-S3
    4. MTF High/Low - PDH/PDL/PWH/PWL/PMH/PML/ATH/ATL
    """
    
    def __init__(self):
        # Williams Fractal Settings
        self.fractal_period = 2
        
        # S/R Channel Settings
        self.sr_pivot_period = 10
        self.sr_pivot_source = "High/Low"
        self.sr_channel_width_pct = 5
        self.sr_min_strength = 1
        self.sr_max_channels = 6
        self.sr_loopback = 290
        
        # Pivot Points Settings
        self.pivot_type = "Standard"
        
        # MTF Settings
        self.mtf_touch_tolerance = 3
        
    def analyze(self, df: pd.DataFrame, current_timeframe: str = '5min') -> Dict:
        """
        Run complete support/resistance analysis
        
        Args:
            df: DataFrame with OHLCV data (must have DateTimeIndex)
            current_timeframe: Current timeframe for resampling
            
        Returns:
            Dict with RAW S/R levels and data
        """
        if len(df) < 100:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        current_price = df['close'].iloc[-1]
        
        # Calculate all components
        fractals = self._calculate_fractals(df)
        sr_channels = self._calculate_sr_channels(df)
        pivots = self._calculate_pivot_points(df)
        mtf_levels = self._calculate_mtf_levels(df)
        
        # Confluence analysis (raw data)
        confluence = self._analyze_confluence(df, fractals, sr_channels, pivots, mtf_levels)
        
        # Get nearest levels
        nearest_support, nearest_resistance = self._find_nearest_levels(
            current_price, fractals, sr_channels, pivots, mtf_levels
        )
        
        # Return RAW DATA ONLY - no signals
        return {
            # Fractal Data
            "fractal_high_current": fractals['current_high'],
            "fractal_low_current": fractals['current_low'],
            "fractal_high_count": len(fractals['highs']),
            "fractal_low_count": len(fractals['lows']),
            "fractal_highs_recent": [f['price'] for f in fractals['highs'][-5:]],
            "fractal_lows_recent": [f['price'] for f in fractals['lows'][-5:]],
            
            # S/R Channel Data
            "sr_channel_count": len(sr_channels['channels']),
            "sr_strongest_top": sr_channels['strongest']['top'] if sr_channels['strongest'] else None,
            "sr_strongest_bottom": sr_channels['strongest']['bottom'] if sr_channels['strongest'] else None,
            "sr_strongest_strength": sr_channels['strongest']['strength'] if sr_channels['strongest'] else None,
            "sr_strongest_type": sr_channels['strongest']['type'] if sr_channels['strongest'] else None,
            "sr_resistance_broken": sr_channels['breaks']['resistance_broken'],
            "sr_support_broken": sr_channels['breaks']['support_broken'],
            "sr_channels": [{
                'top': ch['top'],
                'bottom': ch['bottom'],
                'strength': ch['strength'],
                'num_pivots': ch['num_pivots'],
                'touches': ch['touches'],
                'type': ch['type']
            } for ch in sr_channels['channels']],
            
            # Daily Pivot Points
            "daily_pp": pivots.get('daily', {}).get('PP'),
            "daily_r1": pivots.get('daily', {}).get('R1'),
            "daily_r2": pivots.get('daily', {}).get('R2'),
            "daily_r3": pivots.get('daily', {}).get('R3'),
            "daily_s1": pivots.get('daily', {}).get('S1'),
            "daily_s2": pivots.get('daily', {}).get('S2'),
            "daily_s3": pivots.get('daily', {}).get('S3'),
            
            # Weekly Pivot Points
            "weekly_pp": pivots.get('weekly', {}).get('PP'),
            "weekly_r1": pivots.get('weekly', {}).get('R1'),
            "weekly_r2": pivots.get('weekly', {}).get('R2'),
            "weekly_r3": pivots.get('weekly', {}).get('R3'),
            "weekly_s1": pivots.get('weekly', {}).get('S1'),
            "weekly_s2": pivots.get('weekly', {}).get('S2'),
            "weekly_s3": pivots.get('weekly', {}).get('S3'),
            
            # Monthly Pivot Points
            "monthly_pp": pivots.get('monthly', {}).get('PP'),
            "monthly_r1": pivots.get('monthly', {}).get('R1'),
            "monthly_r2": pivots.get('monthly', {}).get('R2'),
            "monthly_r3": pivots.get('monthly', {}).get('R3'),
            "monthly_s1": pivots.get('monthly', {}).get('S1'),
            "monthly_s2": pivots.get('monthly', {}).get('S2'),
            "monthly_s3": pivots.get('monthly', {}).get('S3'),
            
            # MTF Levels
            "pdh": mtf_levels.get('PDH'),
            "pdl": mtf_levels.get('PDL'),
            "pwh": mtf_levels.get('PWH'),
            "pwl": mtf_levels.get('PWL'),
            "pmh": mtf_levels.get('PMH'),
            "pml": mtf_levels.get('PML'),
            "ath": mtf_levels.get('ATH'),
            "atl": mtf_levels.get('ATL'),
            
            # MTF Touches
            "pdh_touch": mtf_levels.get('touches', {}).get('PDH', False),
            "pdl_touch": mtf_levels.get('touches', {}).get('PDL', False),
            "pwh_touch": mtf_levels.get('touches', {}).get('PWH', False),
            "pwl_touch": mtf_levels.get('touches', {}).get('PWL', False),
            "pmh_touch": mtf_levels.get('touches', {}).get('PMH', False),
            "pml_touch": mtf_levels.get('touches', {}).get('PML', False),
            
            # MTF Breaks
            "pdh_break": mtf_levels.get('breaks', {}).get('pdh_break', False),
            "pdl_break": mtf_levels.get('breaks', {}).get('pdl_break', False),
            "pwh_break": mtf_levels.get('breaks', {}).get('pwh_break', False),
            "pwl_break": mtf_levels.get('breaks', {}).get('pwl_break', False),
            
            # Confluence Data (raw, no signal interpretation)
            "confluence_zone_count": len(confluence),
            "confluence_zones": [{
                'price': z['price'],
                'num_levels': z['num_levels'],
                'levels': z['levels'],
                'sources': z['sources'],
                'distance_from_price': z['distance_from_price'],
                'is_above_price': z['price'] > current_price,
                'is_below_price': z['price'] < current_price
            } for z in confluence[:5]],
            
            # Nearest Levels
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "distance_to_support": round(current_price - nearest_support, 2) if nearest_support else None,
            "distance_to_resistance": round(nearest_resistance - current_price, 2) if nearest_resistance else None,
            "distance_to_support_pct": round((current_price - nearest_support) / current_price * 100, 2) if nearest_support else None,
            "distance_to_resistance_pct": round((nearest_resistance - current_price) / current_price * 100, 2) if nearest_resistance else None,
            
            # Price Context
            "price_above_daily_pp": current_price > pivots.get('daily', {}).get('PP', 0) if pivots.get('daily', {}).get('PP') else None,
            "price_above_weekly_pp": current_price > pivots.get('weekly', {}).get('PP', 0) if pivots.get('weekly', {}).get('PP') else None,
            "price_above_pdh": current_price > mtf_levels.get('PDH', 0) if mtf_levels.get('PDH') else None,
            "price_below_pdl": current_price < mtf_levels.get('PDL', float('inf')) if mtf_levels.get('PDL') else None,
            "current_price": round(current_price, 2),
            
            # Timestamp
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== WILLIAMS FRACTALS ====================
    
    def _calculate_fractals(self, df: pd.DataFrame) -> Dict:
        """Calculate Williams Fractals (exact Pine Script logic)"""
        n = self.fractal_period
        high = df['high'].values
        low = df['low'].values
        
        fractal_highs = []
        fractal_lows = []
        
        for i in range(n, len(df) - n):
            # UP FRACTAL (Resistance)
            upflag_down_frontier = True
            upflag_up_frontier_0 = True
            upflag_up_frontier_1 = True
            upflag_up_frontier_2 = True
            upflag_up_frontier_3 = True
            upflag_up_frontier_4 = True
            
            for j in range(1, n + 1):
                if high[i - j] >= high[i]:
                    upflag_down_frontier = False
                    break
            
            if upflag_down_frontier:
                for j in range(1, n + 1):
                    if high[i + j] >= high[i]:
                        upflag_up_frontier_0 = False
                        break
                
                if high[i + 1] <= high[i]:
                    for j in range(1, n + 1):
                        if high[i + j + 1] >= high[i]:
                            upflag_up_frontier_1 = False
                            break
                else:
                    upflag_up_frontier_1 = False
                
                if high[i + 1] <= high[i] and high[i + 2] <= high[i]:
                    for j in range(1, n + 1):
                        if high[i + j + 2] >= high[i]:
                            upflag_up_frontier_2 = False
                            break
                else:
                    upflag_up_frontier_2 = False
                
                if (high[i + 1] <= high[i] and high[i + 2] <= high[i] and 
                    high[i + 3] <= high[i]):
                    for j in range(1, n + 1):
                        if high[i + j + 3] >= high[i]:
                            upflag_up_frontier_3 = False
                            break
                else:
                    upflag_up_frontier_3 = False
                
                if (high[i + 1] <= high[i] and high[i + 2] <= high[i] and 
                    high[i + 3] <= high[i] and high[i + 4] <= high[i]):
                    for j in range(1, n + 1):
                        if high[i + j + 4] >= high[i]:
                            upflag_up_frontier_4 = False
                            break
                else:
                    upflag_up_frontier_4 = False
                
                flag_up_frontier = (upflag_up_frontier_0 or upflag_up_frontier_1 or 
                                   upflag_up_frontier_2 or upflag_up_frontier_3 or 
                                   upflag_up_frontier_4)
                
                if flag_up_frontier:
                    fractal_highs.append({
                        'index': i,
                        'price': high[i],
                        'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    })
            
            # DOWN FRACTAL (Support)
            downflag_down_frontier = True
            downflag_up_frontier_0 = True
            downflag_up_frontier_1 = True
            downflag_up_frontier_2 = True
            downflag_up_frontier_3 = True
            downflag_up_frontier_4 = True
            
            for j in range(1, n + 1):
                if low[i - j] <= low[i]:
                    downflag_down_frontier = False
                    break
            
            if downflag_down_frontier:
                for j in range(1, n + 1):
                    if low[i + j] <= low[i]:
                        downflag_up_frontier_0 = False
                        break
                
                if low[i + 1] >= low[i]:
                    for j in range(1, n + 1):
                        if low[i + j + 1] <= low[i]:
                            downflag_up_frontier_1 = False
                            break
                else:
                    downflag_up_frontier_1 = False
                
                if low[i + 1] >= low[i] and low[i + 2] >= low[i]:
                    for j in range(1, n + 1):
                        if low[i + j + 2] <= low[i]:
                            downflag_up_frontier_2 = False
                            break
                else:
                    downflag_up_frontier_2 = False
                
                if (low[i + 1] >= low[i] and low[i + 2] >= low[i] and 
                    low[i + 3] >= low[i]):
                    for j in range(1, n + 1):
                        if low[i + j + 3] <= low[i]:
                            downflag_up_frontier_3 = False
                            break
                else:
                    downflag_up_frontier_3 = False
                
                if (low[i + 1] >= low[i] and low[i + 2] >= low[i] and 
                    low[i + 3] >= low[i] and low[i + 4] >= low[i]):
                    for j in range(1, n + 1):
                        if low[i + j + 4] <= low[i]:
                            downflag_up_frontier_4 = False
                            break
                else:
                    downflag_up_frontier_4 = False
                
                flag_down_frontier = (downflag_up_frontier_0 or downflag_up_frontier_1 or 
                                     downflag_up_frontier_2 or downflag_up_frontier_3 or 
                                     downflag_up_frontier_4)
                
                if flag_down_frontier:
                    fractal_lows.append({
                        'index': i,
                        'price': low[i],
                        'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    })
        
        return {
            'highs': fractal_highs[-20:] if len(fractal_highs) > 20 else fractal_highs,
            'lows': fractal_lows[-20:] if len(fractal_lows) > 20 else fractal_lows,
            'current_high': fractal_highs[-1]['price'] if fractal_highs else None,
            'current_low': fractal_lows[-1]['price'] if fractal_lows else None
        }
    
    # ==================== S/R CHANNELS ====================
    
    def _calculate_sr_channels(self, df: pd.DataFrame) -> Dict:
        """Calculate dynamic S/R channels (exact Pine Script logic)"""
        prd = self.sr_pivot_period
        
        if self.sr_pivot_source == "High/Low":
            src1 = df['high'].values
            src2 = df['low'].values
        else:
            src1 = np.maximum(df['close'].values, df['open'].values)
            src2 = np.minimum(df['close'].values, df['open'].values)
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(prd, len(df) - prd):
            is_pivot_high = True
            for j in range(1, prd + 1):
                if src1[i - j] >= src1[i] or src1[i + j] >= src1[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append({'bar': i, 'price': src1[i]})
            
            is_pivot_low = True
            for j in range(1, prd + 1):
                if src2[i - j] <= src2[i] or src2[i + j] <= src2[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append({'bar': i, 'price': src2[i]})
        
        all_pivots = []
        for ph in pivot_highs:
            all_pivots.append({'bar': ph['bar'], 'price': ph['price']})
        for pl in pivot_lows:
            all_pivots.append({'bar': pl['bar'], 'price': pl['price']})
        
        current_bar = len(df) - 1
        all_pivots = [p for p in all_pivots if current_bar - p['bar'] <= self.sr_loopback]
        
        if len(all_pivots) < 2:
            return {'channels': [], 'strongest': None, 'breaks': {'resistance_broken': False, 'support_broken': False}}
        
        highest_300 = df['high'].iloc[-300:].max() if len(df) >= 300 else df['high'].max()
        lowest_300 = df['low'].iloc[-300:].min() if len(df) >= 300 else df['low'].min()
        cwidth = (highest_300 - lowest_300) * self.sr_channel_width_pct / 100
        
        channels_data = []
        for pivot in all_pivots:
            lo = pivot['price']
            hi = pivot['price']
            numpp = 0
            
            for other_pivot in all_pivots:
                cpp = other_pivot['price']
                wdth = cpp - lo if cpp > hi else hi - cpp
                
                if wdth <= cwidth:
                    lo = min(lo, cpp)
                    hi = max(hi, cpp)
                    numpp += 20
            
            touches = 0
            for i in range(max(0, len(df) - self.sr_loopback), len(df)):
                if (df['high'].iloc[i] <= hi and df['high'].iloc[i] >= lo) or \
                   (df['low'].iloc[i] <= hi and df['low'].iloc[i] >= lo):
                    touches += 1
            
            total_strength = numpp + touches
            
            if total_strength >= self.sr_min_strength * 20:
                channels_data.append({
                    'top': round(hi, 2),
                    'bottom': round(lo, 2),
                    'strength': total_strength,
                    'num_pivots': numpp // 20,
                    'touches': touches
                })
        
        unique_channels = []
        for ch in channels_data:
            is_duplicate = False
            for uch in unique_channels:
                if abs(ch['top'] - uch['top']) < 0.01 and abs(ch['bottom'] - uch['bottom']) < 0.01:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_channels.append(ch)
        
        unique_channels.sort(key=lambda x: x['strength'], reverse=True)
        top_channels = unique_channels[:self.sr_max_channels]
        
        current_price = df['close'].iloc[-1]
        for ch in top_channels:
            if ch['top'] > current_price and ch['bottom'] > current_price:
                ch['type'] = 'resistance'
            elif ch['top'] < current_price and ch['bottom'] < current_price:
                ch['type'] = 'support'
            else:
                ch['type'] = 'inside'
        
        breaks = self._check_sr_breaks(df, top_channels)
        
        return {
            'channels': top_channels,
            'strongest': top_channels[0] if top_channels else None,
            'breaks': breaks
        }
    
    def _check_sr_breaks(self, df: pd.DataFrame, channels: List[Dict]) -> Dict:
        """Check if support/resistance was broken"""
        if len(df) < 2:
            return {'resistance_broken': False, 'support_broken': False}
        
        close = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        
        not_in_channel = True
        for ch in channels:
            if close <= ch['top'] and close >= ch['bottom']:
                not_in_channel = False
                break
        
        resistance_broken = False
        support_broken = False
        
        if not_in_channel:
            for ch in channels:
                if close_prev <= ch['top'] and close > ch['top']:
                    resistance_broken = True
                if close_prev >= ch['bottom'] and close < ch['bottom']:
                    support_broken = True
        
        return {
            'resistance_broken': resistance_broken,
            'support_broken': support_broken
        }
    
    # ==================== PIVOT POINTS ====================
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate Standard Pivot Points"""
        pivots = {}
        
        if isinstance(df.index, pd.DatetimeIndex):
            daily_df = df.resample('1D').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            if len(daily_df) >= 2:
                h = daily_df['high'].iloc[-2]
                l = daily_df['low'].iloc[-2]
                c = daily_df['close'].iloc[-2]
                pivots['daily'] = self._standard_pivots(h, l, c)
            
            weekly_df = df.resample('1W').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            if len(weekly_df) >= 2:
                h = weekly_df['high'].iloc[-2]
                l = weekly_df['low'].iloc[-2]
                c = weekly_df['close'].iloc[-2]
                pivots['weekly'] = self._standard_pivots(h, l, c)
            
            monthly_df = df.resample('1M').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            if len(monthly_df) >= 2:
                h = monthly_df['high'].iloc[-2]
                l = monthly_df['low'].iloc[-2]
                c = monthly_df['close'].iloc[-2]
                pivots['monthly'] = self._standard_pivots(h, l, c)
        
        return pivots
    
    def _standard_pivots(self, h: float, l: float, c: float) -> Dict:
        """Calculate standard pivot points"""
        pp = (h + l + c) / 3
        r1 = (2 * pp) - l
        s1 = (2 * pp) - h
        r2 = pp + (h - l)
        s2 = pp - (h - l)
        r3 = h + 2 * (pp - l)
        s3 = l - 2 * (h - pp)
        
        return {
            'PP': round(pp, 2),
            'R1': round(r1, 2), 'R2': round(r2, 2), 'R3': round(r3, 2),
            'S1': round(s1, 2), 'S2': round(s2, 2), 'S3': round(s3, 2)
        }
    
    # ==================== MTF HIGH/LOW ====================
    
    def _calculate_mtf_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Multi-Timeframe High/Low levels"""
        levels = {}
        
        if isinstance(df.index, pd.DatetimeIndex):
            daily_df = df.resample('1D').agg({
                'high': 'max',
                'low': 'min'
            }).dropna()
            
            if len(daily_df) >= 2:
                levels['PDH'] = round(daily_df['high'].iloc[-2], 2)
                levels['PDL'] = round(daily_df['low'].iloc[-2], 2)
            
            weekly_df = df.resample('1W').agg({
                'high': 'max',
                'low': 'min'
            }).dropna()
            
            if len(weekly_df) >= 2:
                levels['PWH'] = round(weekly_df['high'].iloc[-2], 2)
                levels['PWL'] = round(weekly_df['low'].iloc[-2], 2)
            
            monthly_df = df.resample('1M').agg({
                'high': 'max',
                'low': 'min'
            }).dropna()
            
            if len(monthly_df) >= 2:
                levels['PMH'] = round(monthly_df['high'].iloc[-2], 2)
                levels['PML'] = round(monthly_df['low'].iloc[-2], 2)
        
        levels['ATH'] = round(df['high'].max(), 2)
        levels['ATL'] = round(df['low'].min(), 2)
        
        levels['touches'] = self._detect_mtf_touches(df, levels)
        levels['breaks'] = self._detect_mtf_breaks(df, levels)
        
        return levels
    
    def _detect_mtf_touches(self, df: pd.DataFrame, levels: Dict) -> Dict:
        """Detect touches at MTF levels"""
        if len(df) < 1:
            return {}
        
        price_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        tolerance = price_range * 0.001 * self.mtf_touch_tolerance
        
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        
        touches = {}
        for level_name, level_price in levels.items():
            if level_name in ['touches', 'breaks']:
                continue
            if level_price is not None:
                touch = abs(high - level_price) <= tolerance or abs(low - level_price) <= tolerance
                touches[level_name] = touch
        
        return touches
    
    def _detect_mtf_breaks(self, df: pd.DataFrame, levels: Dict) -> Dict:
        """Detect breaks at MTF levels"""
        if len(df) < 2:
            return {}
        
        close = df['close'].iloc[-1]
        close_prev = df['close'].iloc[-2]
        
        breaks = {}
        
        if 'PDH' in levels and levels['PDH'] is not None:
            breaks['pdh_break'] = close > levels['PDH'] and close_prev <= levels['PDH']
        if 'PDL' in levels and levels['PDL'] is not None:
            breaks['pdl_break'] = close < levels['PDL'] and close_prev >= levels['PDL']
        if 'PWH' in levels and levels['PWH'] is not None:
            breaks['pwh_break'] = close > levels['PWH'] and close_prev <= levels['PWH']
        if 'PWL' in levels and levels['PWL'] is not None:
            breaks['pwl_break'] = close < levels['PWL'] and close_prev >= levels['PWL']
        
        return breaks
    
    # ==================== CONFLUENCE ANALYSIS ====================
    
    def _analyze_confluence(self, df: pd.DataFrame, fractals: Dict, 
                           sr_channels: Dict, pivots: Dict, mtf_levels: Dict) -> List[Dict]:
        """Find confluence zones where multiple S/R levels align - NO signal interpretation"""
        current_price = df['close'].iloc[-1]
        tolerance = (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.01
        
        all_levels = []
        
        if fractals['current_high']:
            all_levels.append({
                'price': fractals['current_high'],
                'type': 'Fractal High',
                'source': 'fractals'
            })
        if fractals['current_low']:
            all_levels.append({
                'price': fractals['current_low'],
                'type': 'Fractal Low',
                'source': 'fractals'
            })
        
        for ch in sr_channels.get('channels', []):
            all_levels.append({
                'price': ch['top'],
                'type': f"SR {ch['type'].title()} Top",
                'source': 'sr_channel',
                'strength': ch['strength']
            })
            all_levels.append({
                'price': ch['bottom'],
                'type': f"SR {ch['type'].title()} Bottom",
                'source': 'sr_channel',
                'strength': ch['strength']
            })
        
        for tf_name, tf_pivots in pivots.items():
            for level_name, level_price in tf_pivots.items():
                all_levels.append({
                    'price': level_price,
                    'type': f"{tf_name.title()} {level_name}",
                    'source': 'pivot'
                })
        
        for level_name, level_price in mtf_levels.items():
            if level_name not in ['touches', 'breaks'] and level_price is not None:
                all_levels.append({
                    'price': level_price,
                    'type': level_name,
                    'source': 'mtf'
                })
        
        confluence_zones = []
        processed_prices = set()
        
        for level in all_levels:
            price = level['price']
            
            if any(abs(price - p) <= tolerance for p in processed_prices):
                continue
            
            nearby = [l for l in all_levels if abs(l['price'] - price) <= tolerance]
            
            if len(nearby) >= 2:
                avg_price = sum(l['price'] for l in nearby) / len(nearby)
                
                confluence_zones.append({
                    'price': round(avg_price, 2),
                    'num_levels': len(nearby),
                    'levels': [l['type'] for l in nearby],
                    'sources': list(set(l['source'] for l in nearby)),
                    'distance_from_price': round(abs(avg_price - current_price), 2)
                })
                
                processed_prices.add(avg_price)
        
        confluence_zones.sort(key=lambda x: x['num_levels'], reverse=True)
        
        return confluence_zones[:10]
    
    # ==================== NEAREST LEVELS ====================
    
    def _find_nearest_levels(self, current_price: float, fractals: Dict,
                            sr_channels: Dict, pivots: Dict, mtf_levels: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Find nearest support and resistance levels"""
        all_levels = []
        
        if fractals['current_high']:
            all_levels.append(fractals['current_high'])
        if fractals['current_low']:
            all_levels.append(fractals['current_low'])
        
        for ch in sr_channels.get('channels', []):
            all_levels.append(ch['top'])
            all_levels.append(ch['bottom'])
        
        for tf_name, tf_pivots in pivots.items():
            for level_name, level_price in tf_pivots.items():
                all_levels.append(level_price)
        
        for level_name, level_price in mtf_levels.items():
            if level_name not in ['touches', 'breaks'] and level_price is not None:
                all_levels.append(level_price)
        
        supports = [l for l in all_levels if l < current_price]
        resistances = [l for l in all_levels if l > current_price]
        
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None
        
        return (round(nearest_support, 2) if nearest_support else None,
                round(nearest_resistance, 2) if nearest_resistance else None)
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "fractal_high_current": None, "fractal_low_current": None,
            "fractal_high_count": 0, "fractal_low_count": 0,
            "fractal_highs_recent": [], "fractal_lows_recent": [],
            "sr_channel_count": 0, "sr_strongest_top": None, "sr_strongest_bottom": None,
            "sr_strongest_strength": None, "sr_strongest_type": None,
            "sr_resistance_broken": False, "sr_support_broken": False, "sr_channels": [],
            "daily_pp": None, "daily_r1": None, "daily_r2": None, "daily_r3": None,
            "daily_s1": None, "daily_s2": None, "daily_s3": None,
            "weekly_pp": None, "weekly_r1": None, "weekly_r2": None, "weekly_r3": None,
            "weekly_s1": None, "weekly_s2": None, "weekly_s3": None,
            "monthly_pp": None, "monthly_r1": None, "monthly_r2": None, "monthly_r3": None,
            "monthly_s1": None, "monthly_s2": None, "monthly_s3": None,
            "pdh": None, "pdl": None, "pwh": None, "pwl": None,
            "pmh": None, "pml": None, "ath": None, "atl": None,
            "pdh_touch": False, "pdl_touch": False, "pwh_touch": False, "pwl_touch": False,
            "pmh_touch": False, "pml_touch": False,
            "pdh_break": False, "pdl_break": False, "pwh_break": False, "pwl_break": False,
            "confluence_zone_count": 0, "confluence_zones": [],
            "nearest_support": None, "nearest_resistance": None,
            "distance_to_support": None, "distance_to_resistance": None,
            "distance_to_support_pct": None, "distance_to_resistance_pct": None,
            "price_above_daily_pp": None, "price_above_weekly_pp": None,
            "price_above_pdh": None, "price_below_pdl": None,
            "current_price": None, "error": reason
        }
