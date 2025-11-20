"""
Layer 11: Support/Resistance Engine
Complete S/R system with 4 professional indicators
Converted from Pine Script - Logic unchanged
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
        # ==================== Williams Fractal Settings ====================
        self.fractal_period = 2  # Default n=2, minimum 2
        
        # ==================== S/R Channel Settings ====================
        self.sr_pivot_period = 10  # Pivot period (checks left & right)
        self.sr_pivot_source = "High/Low"  # or "Close/Open"
        self.sr_channel_width_pct = 5  # Maximum channel width %
        self.sr_min_strength = 1  # Minimum strength (1 = at least 2 pivots)
        self.sr_max_channels = 6  # Maximum number of S/R channels
        self.sr_loopback = 290  # Loopback period for pivot checking
        
        # ==================== Pivot Points Settings ====================
        self.pivot_type = "Standard"  # Standard formula
        
        # ==================== MTF Settings ====================
        self.mtf_touch_tolerance = 3  # Multiplier for mintick
        
    def analyze(self, df: pd.DataFrame, current_timeframe: str = '5min') -> Dict:
        """
        Run complete support/resistance analysis
        
        Args:
            df: DataFrame with OHLCV data (must have DateTimeIndex)
            current_timeframe: Current timeframe for resampling
            
        Returns:
            Dict with all S/R levels and signals
        """
        if len(df) < 100:
            return self._empty_result("Insufficient data")
        
        df = df.copy()
        
        # Calculate all components
        fractals = self._calculate_fractals(df)
        sr_channels = self._calculate_sr_channels(df)
        pivots = self._calculate_pivot_points(df)
        mtf_levels = self._calculate_mtf_levels(df)
        
        # Confluence analysis
        confluence = self._analyze_confluence(df, fractals, sr_channels, pivots, mtf_levels)
        
        return {
            "fractals": fractals,
            "sr_channels": sr_channels,
            "pivot_points": pivots,
            "mtf_levels": mtf_levels,
            "confluence": confluence,
            "timestamp": df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.now()
        }
    
    # ==================== WILLIAMS FRACTALS ====================
    
    def _calculate_fractals(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Williams Fractals (exact Pine Script logic)
        
        Up Fractal: high[n] is highest among n bars before & after
        Down Fractal: low[n] is lowest among n bars before & after
        
        Complex frontier checking allows 0-4 equal highs/lows
        """
        n = self.fractal_period
        high = df['high'].values
        low = df['low'].values
        
        fractal_highs = []
        fractal_lows = []
        
        # Need at least 2*n + 1 bars
        for i in range(n, len(df) - n):
            # ==================== UP FRACTAL (Resistance) ====================
            upflag_down_frontier = True
            upflag_up_frontier_0 = True
            upflag_up_frontier_1 = True
            upflag_up_frontier_2 = True
            upflag_up_frontier_3 = True
            upflag_up_frontier_4 = True
            
            # Check bars before
            for j in range(1, n + 1):
                if high[i - j] >= high[i]:
                    upflag_down_frontier = False
                    break
            
            # Check bars after (5 frontier conditions)
            if upflag_down_frontier:
                # Frontier 0: All after < high[n]
                for j in range(1, n + 1):
                    if high[i + j] >= high[i]:
                        upflag_up_frontier_0 = False
                        break
                
                # Frontier 1: high[n+1] <= high[n], rest < high[n]
                if high[i + 1] <= high[i]:
                    for j in range(1, n + 1):
                        if high[i + j + 1] >= high[i]:
                            upflag_up_frontier_1 = False
                            break
                else:
                    upflag_up_frontier_1 = False
                
                # Frontier 2: high[n+1], high[n+2] <= high[n], rest < high[n]
                if high[i + 1] <= high[i] and high[i + 2] <= high[i]:
                    for j in range(1, n + 1):
                        if high[i + j + 2] >= high[i]:
                            upflag_up_frontier_2 = False
                            break
                else:
                    upflag_up_frontier_2 = False
                
                # Frontier 3: high[n+1,2,3] <= high[n], rest < high[n]
                if (high[i + 1] <= high[i] and high[i + 2] <= high[i] and 
                    high[i + 3] <= high[i]):
                    for j in range(1, n + 1):
                        if high[i + j + 3] >= high[i]:
                            upflag_up_frontier_3 = False
                            break
                else:
                    upflag_up_frontier_3 = False
                
                # Frontier 4: high[n+1,2,3,4] <= high[n], rest < high[n]
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
            
            # ==================== DOWN FRACTAL (Support) ====================
            downflag_down_frontier = True
            downflag_up_frontier_0 = True
            downflag_up_frontier_1 = True
            downflag_up_frontier_2 = True
            downflag_up_frontier_3 = True
            downflag_up_frontier_4 = True
            
            # Check bars before
            for j in range(1, n + 1):
                if low[i - j] <= low[i]:
                    downflag_down_frontier = False
                    break
            
            # Check bars after (5 frontier conditions)
            if downflag_down_frontier:
                # Frontier 0: All after > low[n]
                for j in range(1, n + 1):
                    if low[i + j] <= low[i]:
                        downflag_up_frontier_0 = False
                        break
                
                # Frontier 1: low[n+1] >= low[n], rest > low[n]
                if low[i + 1] >= low[i]:
                    for j in range(1, n + 1):
                        if low[i + j + 1] <= low[i]:
                            downflag_up_frontier_1 = False
                            break
                else:
                    downflag_up_frontier_1 = False
                
                # Frontier 2: low[n+1], low[n+2] >= low[n], rest > low[n]
                if low[i + 1] >= low[i] and low[i + 2] >= low[i]:
                    for j in range(1, n + 1):
                        if low[i + j + 2] <= low[i]:
                            downflag_up_frontier_2 = False
                            break
                else:
                    downflag_up_frontier_2 = False
                
                # Frontier 3: low[n+1,2,3] >= low[n], rest > low[n]
                if (low[i + 1] >= low[i] and low[i + 2] >= low[i] and 
                    low[i + 3] >= low[i]):
                    for j in range(1, n + 1):
                        if low[i + j + 3] <= low[i]:
                            downflag_up_frontier_3 = False
                            break
                else:
                    downflag_up_frontier_3 = False
                
                # Frontier 4: low[n+1,2,3,4] >= low[n], rest > low[n]
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
            'highs': fractal_highs[-20:] if len(fractal_highs) > 20 else fractal_highs,  # Last 20
            'lows': fractal_lows[-20:] if len(fractal_lows) > 20 else fractal_lows,  # Last 20
            'current_high': fractal_highs[-1]['price'] if fractal_highs else None,
            'current_low': fractal_lows[-1]['price'] if fractal_lows else None
        }
    
    # ==================== S/R CHANNELS ====================
    
    def _calculate_sr_channels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate dynamic S/R channels (exact Pine Script logic)
        
        Process:
        1. Find pivot highs/lows
        2. Calculate max channel width (5% of 300-bar range)
        3. Group pivots into channels
        4. Score by strength (20 per pivot + touches)
        5. Return top 6 strongest channels
        """
        prd = self.sr_pivot_period
        
        # Get pivot source
        if self.sr_pivot_source == "High/Low":
            src1 = df['high'].values
            src2 = df['low'].values
        else:  # Close/Open
            src1 = np.maximum(df['close'].values, df['open'].values)
            src2 = np.minimum(df['close'].values, df['open'].values)
        
        # Find pivot highs/lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(prd, len(df) - prd):
            # Pivot high
            is_pivot_high = True
            for j in range(1, prd + 1):
                if src1[i - j] >= src1[i] or src1[i + j] >= src1[i]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append({'bar': i, 'price': src1[i]})
            
            # Pivot low
            is_pivot_low = True
            for j in range(1, prd + 1):
                if src2[i - j] <= src2[i] or src2[i + j] <= src2[i]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append({'bar': i, 'price': src2[i]})
        
        # Combine all pivots
        all_pivots = []
        for ph in pivot_highs:
            all_pivots.append({'bar': ph['bar'], 'price': ph['price']})
        for pl in pivot_lows:
            all_pivots.append({'bar': pl['bar'], 'price': pl['price']})
        
        # Keep only recent pivots (loopback period)
        current_bar = len(df) - 1
        all_pivots = [p for p in all_pivots if current_bar - p['bar'] <= self.sr_loopback]
        
        if len(all_pivots) < 2:
            return {'channels': [], 'strongest': None}
        
        # Calculate max channel width
        highest_300 = df['high'].iloc[-300:].max() if len(df) >= 300 else df['high'].max()
        lowest_300 = df['low'].iloc[-300:].min() if len(df) >= 300 else df['low'].min()
        cwidth = (highest_300 - lowest_300) * self.sr_channel_width_pct / 100
        
        # Create channels for each pivot
        channels_data = []
        for pivot in all_pivots:
            lo = pivot['price']
            hi = pivot['price']
            numpp = 0
            
            # Group pivots that fit within channel width
            for other_pivot in all_pivots:
                cpp = other_pivot['price']
                wdth = cpp - lo if cpp > hi else hi - cpp
                
                if wdth <= cwidth:
                    lo = min(lo, cpp)
                    hi = max(hi, cpp)
                    numpp += 20  # Each pivot adds 20 strength
            
            # Add touches (bars where high/low intersected channel)
            touches = 0
            for i in range(max(0, len(df) - self.sr_loopback), len(df)):
                if (df['high'].iloc[i] <= hi and df['high'].iloc[i] >= lo) or \
                   (df['low'].iloc[i] <= hi and df['low'].iloc[i] >= lo):
                    touches += 1
            
            total_strength = numpp + touches
            
            if total_strength >= self.sr_min_strength * 20:
                channels_data.append({
                    'top': hi,
                    'bottom': lo,
                    'strength': total_strength,
                    'num_pivots': numpp // 20,
                    'touches': touches
                })
        
        # Remove duplicate/overlapping channels
        unique_channels = []
        for ch in channels_data:
            is_duplicate = False
            for uch in unique_channels:
                if abs(ch['top'] - uch['top']) < 0.01 and abs(ch['bottom'] - uch['bottom']) < 0.01:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_channels.append(ch)
        
        # Sort by strength, keep top N
        unique_channels.sort(key=lambda x: x['strength'], reverse=True)
        top_channels = unique_channels[:self.sr_max_channels]
        
        # Classify each channel (resistance/support/inside)
        current_price = df['close'].iloc[-1]
        for ch in top_channels:
            if ch['top'] > current_price and ch['bottom'] > current_price:
                ch['type'] = 'resistance'
            elif ch['top'] < current_price and ch['bottom'] < current_price:
                ch['type'] = 'support'
            else:
                ch['type'] = 'inside'
        
        # Check for breaks
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
        
        # Check if price is NOT in any channel
        not_in_channel = True
        for ch in channels:
            if close <= ch['top'] and close >= ch['bottom']:
                not_in_channel = False
                break
        
        resistance_broken = False
        support_broken = False
        
        if not_in_channel:
            for ch in channels:
                # Resistance break
                if close_prev <= ch['top'] and close > ch['top']:
                    resistance_broken = True
                # Support break
                if close_prev >= ch['bottom'] and close < ch['bottom']:
                    support_broken = True
        
        return {
            'resistance_broken': resistance_broken,
            'support_broken': support_broken
        }
    
    # ==================== PIVOT POINTS ====================
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Standard Pivot Points (exact formula)
        
        PP = (H + L + C) / 3
        R1 = (2 * PP) - L
        S1 = (2 * PP) - H
        R2 = PP + (H - L)
        S2 = PP - (H - L)
        R3 = H + 2 * (PP - L)
        S3 = L - 2 * (H - PP)
        """
        pivots = {}
        
        # Daily Pivots
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
            
            # Weekly Pivots
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
            
            # Monthly Pivots
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
            'PP': pp,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
    
    # ==================== MTF HIGH/LOW ====================
    
    def _calculate_mtf_levels(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Multi-Timeframe High/Low levels
        
        Levels:
        - PDH/PDL: Previous Day High/Low
        - PWH/PWL: Previous Week High/Low
        - PMH/PML: Previous Month High/Low
        - ATH/ATL: All-Time High/Low
        """
        levels = {}
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Daily
            daily_df = df.resample('1D').agg({
                'high': 'max',
                'low': 'min'
            }).dropna()
            
            if len(daily_df) >= 2:
                levels['PDH'] = daily_df['high'].iloc[-2]
                levels['PDL'] = daily_df['low'].iloc[-2]
            
            # Weekly
            weekly_df = df.resample('1W').agg({
                'high': 'max',
                'low': 'min'
            }).dropna()
            
            if len(weekly_df) >= 2:
                levels['PWH'] = weekly_df['high'].iloc[-2]
                levels['PWL'] = weekly_df['low'].iloc[-2]
            
            # Monthly
            monthly_df = df.resample('1M').agg({
                'high': 'max',
                'low': 'min'
            }).dropna()
            
            if len(monthly_df) >= 2:
                levels['PMH'] = monthly_df['high'].iloc[-2]
                levels['PML'] = monthly_df['low'].iloc[-2]
        
        # All-Time High/Low
        levels['ATH'] = df['high'].max()
        levels['ATL'] = df['low'].min()
        
        # Touch Detection
        touches = self._detect_mtf_touches(df, levels)
        levels['touches'] = touches
        
        # Break Detection
        breaks = self._detect_mtf_breaks(df, levels)
        levels['breaks'] = breaks
        
        return levels
    
    def _detect_mtf_touches(self, df: pd.DataFrame, levels: Dict) -> Dict:
        """Detect touches at MTF levels"""
        if len(df) < 1:
            return {}
        
        # Calculate tolerance (3 * mintick approximation)
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
        
        # PDH/PDL breaks
        if 'PDH' in levels and levels['PDH'] is not None:
            breaks['pdh_break'] = close > levels['PDH'] and close_prev <= levels['PDH']
        if 'PDL' in levels and levels['PDL'] is not None:
            breaks['pdl_break'] = close < levels['PDL'] and close_prev >= levels['PDL']
        
        # PWH/PWL breaks
        if 'PWH' in levels and levels['PWH'] is not None:
            breaks['pwh_break'] = close > levels['PWH'] and close_prev <= levels['PWH']
        if 'PWL' in levels and levels['PWL'] is not None:
            breaks['pwl_break'] = close < levels['PWL'] and close_prev >= levels['PWL']
        
        return breaks
    
    # ==================== CONFLUENCE ANALYSIS ====================
    
    def _analyze_confluence(self, df: pd.DataFrame, fractals: Dict, 
                           sr_channels: Dict, pivots: Dict, mtf_levels: Dict) -> List[Dict]:
        """
        Find confluence zones where multiple S/R levels align
        
        Confluence = multiple S/R types at same price (within tolerance)
        """
        current_price = df['close'].iloc[-1]
        tolerance = (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.01  # 1% tolerance
        
        # Collect all levels
        all_levels = []
        
        # Fractal levels
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
        
        # S/R channel levels
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
        
        # Pivot levels
        for tf_name, tf_pivots in pivots.items():
            for level_name, level_price in tf_pivots.items():
                all_levels.append({
                    'price': level_price,
                    'type': f"{tf_name.title()} {level_name}",
                    'source': 'pivot'
                })
        
        # MTF levels
        for level_name, level_price in mtf_levels.items():
            if level_name not in ['touches', 'breaks'] and level_price is not None:
                all_levels.append({
                    'price': level_price,
                    'type': level_name,
                    'source': 'mtf'
                })
        
        # Find confluence zones
        confluence_zones = []
        processed_prices = set()
        
        for level in all_levels:
            price = level['price']
            
            # Skip if already processed
            if any(abs(price - p) <= tolerance for p in processed_prices):
                continue
            
            # Find all nearby levels
            nearby = [l for l in all_levels if abs(l['price'] - price) <= tolerance]
            
            if len(nearby) >= 2:  # At least 2 levels = confluence
                avg_price = sum(l['price'] for l in nearby) / len(nearby)
                
                confluence_zones.append({
                    'price': avg_price,
                    'num_levels': len(nearby),
                    'score': len(nearby) * 20,  # 20 points per level
                    'levels': [l['type'] for l in nearby],
                    'sources': list(set(l['source'] for l in nearby)),
                    'distance_from_price': abs(avg_price - current_price),
                    'signal': 'STRONG_SUPPORT' if avg_price < current_price else 'STRONG_RESISTANCE'
                })
                
                processed_prices.add(avg_price)
        
        # Sort by score
        confluence_zones.sort(key=lambda x: x['score'], reverse=True)
        
        return confluence_zones[:10]  # Top 10 confluence zones
    
    def _empty_result(self, reason: str) -> Dict:
        """Return empty result structure"""
        return {
            "error": reason,
            "fractals": {'highs': [], 'lows': []},
            "sr_channels": {'channels': []},
            "pivot_points": {},
            "mtf_levels": {},
            "confluence": []
        }
