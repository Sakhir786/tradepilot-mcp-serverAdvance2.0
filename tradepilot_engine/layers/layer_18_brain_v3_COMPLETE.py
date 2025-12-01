"""
================================================================================
Layer 18: Master Data Aggregator for AI Decision Making
================================================================================

PURPOSE:
Extract 100% of data from all 17 layers, organize into logical categories,
calculate derived values, track data quality, and output clean JSON for AI.

NO trading decisions, NO scoring, NO pass/fail - PURE DATA AGGREGATION.
The AI (Claude/ChatGPT) receives this data and makes the trade decision.

Author: TradePilot MCP Server
Version: 5.0 - Complete Data Extraction
================================================================================
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json


class TradeMode(Enum):
    SCALP = "SCALP"         # 5-min timeframe
    INTRADAY = "INTRADAY"   # 15-min timeframe
    SWING = "SWING"         # Daily timeframe
    LEAPS = "LEAPS"         # Daily timeframe, long-term


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PriceContext:
    """Complete price and trend context"""
    current_price: Optional[float] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    
    # Trend EMAs
    trend_ema_50: Optional[float] = None
    price_vs_trend_ema: Optional[float] = None
    price_vs_trend_ema_pct: Optional[float] = None
    is_above_trend_ema: Optional[bool] = None
    
    # ATR Context
    atr: Optional[float] = None
    atr_percent: Optional[float] = None
    
    # SuperTrend Context
    supertrend_value: Optional[float] = None
    supertrend_bullish: Optional[bool] = None
    supertrend_bearish: Optional[bool] = None
    price_vs_supertrend: Optional[float] = None


@dataclass
class MomentumData:
    """All momentum indicators from Layers 1, 3, 5"""
    # RSI Data
    rsi_14: Optional[float] = None
    rsi_7: Optional[float] = None
    rsi_prev: Optional[float] = None
    rsi_direction: Optional[str] = None  # rising/falling
    rsi_zone: Optional[str] = None       # oversold/bearish/neutral/bullish/overbought
    
    # MACD Data
    macd_line: Optional[float] = None
    macd_signal_line: Optional[float] = None
    macd_histogram: Optional[float] = None
    macd_histogram_prev: Optional[float] = None
    macd_histogram_rising: Optional[bool] = None
    macd_above_signal: Optional[bool] = None
    macd_bullish_cross: Optional[bool] = None
    macd_bearish_cross: Optional[bool] = None
    
    # Stochastic Data
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None
    stoch_k_prev: Optional[float] = None
    stoch_zone: Optional[str] = None  # oversold/neutral/overbought
    stoch_bullish_cross: Optional[bool] = None
    stoch_bearish_cross: Optional[bool] = None
    
    # CMF Data
    cmf: Optional[float] = None
    cmf_prev: Optional[float] = None
    cmf_positive: Optional[bool] = None
    cmf_direction: Optional[str] = None
    
    # ADX/DMI Data
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    di_diff: Optional[float] = None
    adx_trend_strength: Optional[str] = None  # no_trend/weak/trending/strong/very_strong
    di_bullish: Optional[bool] = None
    di_bearish: Optional[bool] = None
    
    # Ichimoku Data
    ichimoku_conv: Optional[float] = None
    ichimoku_base: Optional[float] = None
    ichimoku_lead1: Optional[float] = None
    ichimoku_lead2: Optional[float] = None
    price_vs_cloud_top: Optional[float] = None
    price_vs_cloud_bottom: Optional[float] = None
    price_above_cloud: Optional[bool] = None
    price_below_cloud: Optional[bool] = None
    price_in_cloud: Optional[bool] = None


@dataclass  
class VolumeData:
    """All volume data from Layers 2, 4"""
    # Basic Volume
    current_volume: Optional[float] = None
    avg_volume_20: Optional[float] = None
    volume_ratio: Optional[float] = None
    volume_change_5bar_pct: Optional[float] = None
    volume_interpretation: Optional[str] = None  # very_low/low/normal/high/very_high
    
    # OBV Data
    obv: Optional[float] = None
    obv_ma: Optional[float] = None
    obv_prev: Optional[float] = None
    obv_slope: Optional[float] = None
    obv_vs_ma: Optional[float] = None
    obv_trend: Optional[str] = None  # rising/flat/falling
    
    # A/D Line Data
    ad_line: Optional[float] = None
    ad_ma: Optional[float] = None
    ad_prev: Optional[float] = None
    ad_slope: Optional[float] = None
    ad_vs_ma: Optional[float] = None
    ad_trend: Optional[str] = None
    
    # CVD Data (Layer 4)
    cvd: Optional[float] = None
    cvd_prev: Optional[float] = None
    cvd_trend: Optional[str] = None
    cumulative_buying_volume: Optional[float] = None
    cumulative_selling_volume: Optional[float] = None
    volume_strength_wave: Optional[float] = None
    ema_volume_strength_wave: Optional[float] = None
    latest_buying_volume: Optional[float] = None
    latest_selling_volume: Optional[float] = None
    buying_volume_pct: Optional[float] = None
    selling_volume_pct: Optional[float] = None
    volume_pressure: Optional[str] = None  # buying/neutral/selling
    
    # EOM Data (Layer 4)
    eom: Optional[float] = None
    eom_prev: Optional[float] = None
    eom_hl2_change: Optional[float] = None
    eom_distance: Optional[float] = None
    
    # Divergences
    price_rising_volume_falling: Optional[bool] = None
    price_falling_volume_rising: Optional[bool] = None


@dataclass
class DivergenceData:
    """All divergence data from Layer 3"""
    # MACD Divergence Counts
    macd_regular_bearish_count: int = 0
    macd_hidden_bearish_count: int = 0
    macd_regular_bullish_count: int = 0
    macd_hidden_bullish_count: int = 0
    macd_total_bearish: int = 0
    macd_total_bullish: int = 0
    
    # RSI Divergence Counts
    rsi_regular_bearish_count: int = 0
    rsi_hidden_bearish_count: int = 0
    rsi_regular_bullish_count: int = 0
    rsi_hidden_bullish_count: int = 0
    rsi_total_bearish: int = 0
    rsi_total_bullish: int = 0
    
    # Combined Totals
    total_bearish_divergences: int = 0
    total_bullish_divergences: int = 0
    total_regular_divergences: int = 0
    total_hidden_divergences: int = 0
    
    # Net Divergence
    net_divergence: int = 0  # bullish - bearish
    divergence_bias: Optional[str] = None  # bullish/neutral/bearish
    
    # Latest Values
    latest_rsi: Optional[float] = None
    latest_macd_1h: Optional[float] = None
    latest_macd_4h: Optional[float] = None
    latest_macd_1d: Optional[float] = None


@dataclass
class TrendData:
    """All trend data from Layer 5"""
    # SuperTrend Core
    supertrend_value: Optional[float] = None
    supertrend_direction: Optional[int] = None  # 1=bullish, -1=bearish
    supertrend_bullish: Optional[bool] = None
    supertrend_bearish: Optional[bool] = None
    trend_changed: Optional[bool] = None
    raw_buy_signal: Optional[bool] = None
    raw_sell_signal: Optional[bool] = None
    
    # Volatility from Layer 5
    atr: Optional[float] = None
    atr_percent: Optional[float] = None
    volatility_high: Optional[bool] = None
    volatility_low: Optional[bool] = None
    volatility_normal: Optional[bool] = None
    adaptive_multiplier: Optional[float] = None
    
    # Market Regime (ADX/DMI)
    adx: Optional[float] = None
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None
    trending: Optional[bool] = None
    weak_trend: Optional[bool] = None
    choppy: Optional[bool] = None
    
    # Whipsaw Detection
    whipsaw_mode: Optional[bool] = None
    flip_count: Optional[int] = None
    bars_since_flip: Optional[int] = None
    
    # Volume Confirmation
    volume_ratio: Optional[float] = None
    avg_volume: Optional[float] = None
    current_volume: Optional[float] = None
    volume_confirmed: Optional[bool] = None
    
    # HTF Alignment
    htf_timeframe: Optional[str] = None
    htf_bullish: Optional[bool] = None
    htf_bearish: Optional[bool] = None
    htf_aligned: Optional[bool] = None
    
    # RSI Alignment
    rsi: Optional[float] = None
    rsi_aligned: Optional[bool] = None
    
    # Time of Day
    is_first_30min: Optional[bool] = None
    is_lunch_hours: Optional[bool] = None
    is_close_risk: Optional[bool] = None
    is_optimal_hours: Optional[bool] = None
    
    # Persistence
    bars_in_trend: Optional[int] = None


@dataclass
class StructureData:
    """All market structure data from Layer 6"""
    # Pivot Data
    last_pivot_high: Optional[float] = None
    last_pivot_low: Optional[float] = None
    last_pivot_high_index: Optional[int] = None
    last_pivot_low_index: Optional[int] = None
    pivot_high_count: int = 0
    pivot_low_count: int = 0
    
    # Recent Pivot Arrays
    recent_pivot_highs: List[float] = field(default_factory=list)
    recent_pivot_lows: List[float] = field(default_factory=list)
    recent_pivot_high_indices: List[int] = field(default_factory=list)
    recent_pivot_low_indices: List[int] = field(default_factory=list)
    
    # Consecutive Counts (HH/HL/LH/LL)
    consecutive_higher_highs: int = 0
    consecutive_higher_lows: int = 0
    consecutive_lower_highs: int = 0
    consecutive_lower_lows: int = 0
    
    # Pattern Facts
    highs_ascending: Optional[bool] = None
    lows_ascending: Optional[bool] = None
    highs_descending: Optional[bool] = None
    lows_descending: Optional[bool] = None
    
    # Price Position Facts
    price_above_last_pivot_high: Optional[bool] = None
    price_below_last_pivot_high: Optional[bool] = None
    price_above_last_pivot_low: Optional[bool] = None
    price_below_last_pivot_low: Optional[bool] = None
    price_above_all_recent_lows: Optional[bool] = None
    price_below_all_recent_highs: Optional[bool] = None
    
    # Distance Facts
    distance_to_last_pivot_high: Optional[float] = None
    distance_to_last_pivot_high_pct: Optional[float] = None
    distance_to_last_pivot_low: Optional[float] = None
    distance_to_last_pivot_low_pct: Optional[float] = None
    
    # Swing Range
    current_swing_range: Optional[float] = None
    current_swing_range_pct: Optional[float] = None
    price_position_in_range_pct: Optional[float] = None
    
    # CHoCH Detection
    choch_bull_detected: bool = False
    choch_bear_detected: bool = False
    choch_bull_quality: float = 0
    choch_bear_quality: float = 0
    choch_bull_delta: float = 0
    choch_bear_delta: float = 0
    
    # BOS Detection
    bos_bull_detected: bool = False
    bos_bear_detected: bool = False
    bos_bull_quality: float = 0
    bos_bear_quality: float = 0
    bos_bull_delta: float = 0
    bos_bear_delta: float = 0
    
    # Structure Totals
    total_choch_bull: int = 0
    total_choch_bear: int = 0
    total_bos_bull: int = 0
    total_bos_bear: int = 0
    current_trend: int = 0  # 1=bullish, -1=bearish, 0=neutral
    
    # Order Blocks
    ob_bull_detected: bool = False
    ob_bear_detected: bool = False
    ob_bull_quality: float = 0
    ob_bear_quality: float = 0
    ob_bull_top: Optional[float] = None
    ob_bull_btm: Optional[float] = None
    ob_bear_top: Optional[float] = None
    ob_bear_btm: Optional[float] = None
    total_ob_bull: int = 0
    total_ob_bear: int = 0
    
    # Fair Value Gaps
    fvg_bull_detected: bool = False
    fvg_bear_detected: bool = False
    fvg_bull_quality: float = 0
    fvg_bear_quality: float = 0
    fvg_bull_top: Optional[float] = None
    fvg_bull_btm: Optional[float] = None
    fvg_bear_top: Optional[float] = None
    fvg_bear_btm: Optional[float] = None
    total_fvg_bull: int = 0
    total_fvg_bear: int = 0
    
    # Liquidity Detection
    liq_buy_detected: bool = False
    liq_sell_detected: bool = False
    liq_buy_level: Optional[float] = None
    liq_sell_level: Optional[float] = None
    
    # Trend Context
    trend_ema: Optional[float] = None
    price_vs_trend_ema: Optional[float] = None
    price_vs_trend_ema_pct: Optional[float] = None
    is_above_trend_ema: Optional[bool] = None


@dataclass
class LiquidityData:
    """All liquidity data from Layer 7"""
    # LuxAlgo Sweep Data
    bull_sweep_detected: bool = False
    bear_sweep_detected: bool = False
    bull_sweep_type: Optional[str] = None
    bear_sweep_type: Optional[str] = None
    bull_sweep_zone_top: Optional[float] = None
    bull_sweep_zone_bottom: Optional[float] = None
    bear_sweep_zone_top: Optional[float] = None
    bear_sweep_zone_bottom: Optional[float] = None
    total_bull_sweeps: int = 0
    total_bear_sweeps: int = 0
    
    # ICT Liquidity
    ict_buy_liq_detected: bool = False
    ict_sell_liq_detected: bool = False
    ict_buy_liq_level: Optional[float] = None
    ict_sell_liq_level: Optional[float] = None
    
    # Stop Hunt / Liquidity Grab
    bullish_grab_detected: bool = False
    bearish_grab_detected: bool = False
    bull_grab_level: Optional[float] = None
    bear_grab_level: Optional[float] = None
    bull_grab_strength: float = 0
    bear_grab_strength: float = 0
    bull_grab_retrace: float = 0
    bear_grab_retrace: float = 0
    bull_grab_type: Optional[str] = None
    bear_grab_type: Optional[str] = None
    bull_is_round_number: bool = False
    bear_is_round_number: bool = False
    bull_is_equal_level: bool = False
    bear_is_equal_level: bool = False
    total_bull_grabs: int = 0
    total_bear_grabs: int = 0
    
    # Statistics
    total_sweeps: int = 0
    total_grabs: int = 0
    successful_bull_grabs: int = 0
    successful_bear_grabs: int = 0
    
    # Trend Context
    trend_ema: Optional[float] = None
    price_vs_trend_ema: Optional[float] = None
    is_above_trend_ema: Optional[bool] = None
    adx: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    
    # Volume Context
    current_volume: Optional[float] = None
    avg_volume: Optional[float] = None
    volume_ratio: Optional[float] = None
    volume_spike: bool = False


@dataclass
class VolatilityData:
    """All volatility data from Layer 8"""
    # ATR Data
    atr: Optional[float] = None
    atr_prev: Optional[float] = None
    atr_change_pct: Optional[float] = None
    
    # ATRP Data
    atrp: Optional[float] = None
    atrp_prev: Optional[float] = None
    atrp_smoothed: Optional[float] = None
    atrp_trend_5bar: Optional[float] = None
    
    # Percentile Data
    percentile_rank: Optional[float] = None
    percentile_bucket: Optional[str] = None  # 0-20, 20-40, etc.
    p20_threshold: Optional[float] = None
    p40_threshold: Optional[float] = None
    p60_threshold: Optional[float] = None
    p80_threshold: Optional[float] = None
    
    # Volatility State
    is_below_p20: bool = False
    is_below_p40: bool = False
    is_above_p60: bool = False
    is_above_p80: bool = False
    volatility_expanding: bool = False
    volatility_contracting: bool = False
    
    # Volatility Regime Classification
    regime: Optional[str] = None  # very_low/low/normal/high/extreme


@dataclass
class MTFConfirmationData:
    """All MTF confirmation data from Layer 9"""
    # Current Timeframe
    current_timeframe: Optional[str] = None
    current_st_direction: Optional[int] = None
    current_st_bullish: Optional[bool] = None
    current_st_bearish: Optional[bool] = None
    current_adx: Optional[float] = None
    current_st_line: Optional[float] = None
    
    # Per-Timeframe Data
    tf_5min_direction: Optional[int] = None
    tf_5min_bullish: Optional[bool] = None
    tf_5min_adx: Optional[float] = None
    tf_5min_weight: int = 25
    
    tf_15min_direction: Optional[int] = None
    tf_15min_bullish: Optional[bool] = None
    tf_15min_adx: Optional[float] = None
    tf_15min_weight: int = 25
    
    tf_1h_direction: Optional[int] = None
    tf_1h_bullish: Optional[bool] = None
    tf_1h_adx: Optional[float] = None
    tf_1h_weight: int = 25
    
    tf_4h_direction: Optional[int] = None
    tf_4h_bullish: Optional[bool] = None
    tf_4h_adx: Optional[float] = None
    tf_4h_weight: int = 15
    
    tf_1d_direction: Optional[int] = None
    tf_1d_bullish: Optional[bool] = None
    tf_1d_adx: Optional[float] = None
    tf_1d_weight: int = 10
    
    # Alignment Metrics
    bull_count: int = 0
    bear_count: int = 0
    total_timeframes: int = 0
    aligned_with_current_count: int = 0
    not_aligned_count: int = 0
    
    # Weighted Alignment
    weighted_aligned: float = 0
    weighted_not_aligned: float = 0
    total_weight: float = 0
    alignment_pct: float = 0
    
    # HTF Context
    htf_bullish_count: int = 0
    htf_bearish_count: int = 0
    htf_aligned: bool = False
    
    # Derived
    mtf_dominant_trend: Optional[str] = None  # bullish/bearish/mixed


@dataclass
class CandlePatternData:
    """All candlestick pattern data from Layer 10"""
    # Three White Soldiers
    tws_detected: bool = False
    tws_quality: float = 0
    tws_high_quality: bool = False
    tws_pattern_base: bool = False
    tws_volume_strong: bool = False
    tws_after_downtrend: bool = False
    tws_near_support: bool = False
    tws_entry: Optional[float] = None
    tws_stop: Optional[float] = None
    tws_target1: Optional[float] = None
    tws_target2: Optional[float] = None
    
    # Inside Bar
    ib_detected: bool = False
    ib_bullish_breakout: bool = False
    ib_bearish_breakout: bool = False
    ib_quality: float = 0
    ib_mother_high: Optional[float] = None
    ib_mother_low: Optional[float] = None
    ib_inside_ratio_pct: float = 0
    ib_bull_entry: Optional[float] = None
    ib_bull_stop: Optional[float] = None
    ib_bull_target1: Optional[float] = None
    ib_bull_target2: Optional[float] = None
    ib_bear_entry: Optional[float] = None
    ib_bear_stop: Optional[float] = None
    ib_bear_target1: Optional[float] = None
    ib_bear_target2: Optional[float] = None
    
    # Pattern Lists
    patterns_detected: List[str] = field(default_factory=list)
    bullish_patterns: List[str] = field(default_factory=list)
    bearish_patterns: List[str] = field(default_factory=list)
    
    # Pattern Counts
    bullish_pattern_count: int = 0
    bearish_pattern_count: int = 0
    neutral_pattern_count: int = 0
    
    # Individual Pattern Detection
    doji_detected: bool = False
    hammer_detected: bool = False
    inverted_hammer_detected: bool = False
    hanging_man_detected: bool = False
    shooting_star_detected: bool = False
    morning_star_detected: bool = False
    evening_star_detected: bool = False
    bullish_engulfing_detected: bool = False
    bearish_engulfing_detected: bool = False
    bullish_harami_detected: bool = False
    bearish_harami_detected: bool = False
    piercing_line_detected: bool = False
    dark_cloud_detected: bool = False
    bullish_kicker_detected: bool = False
    bearish_kicker_detected: bool = False
    
    # Pattern Quality
    patterns_bull_quality: float = 0
    patterns_bear_quality: float = 0
    
    # Star Patterns (Pro)
    morning_star_pro_detected: bool = False
    evening_star_pro_detected: bool = False
    morning_star_quality: float = 0
    evening_star_quality: float = 0
    morning_star_high_quality: bool = False
    evening_star_high_quality: bool = False
    morning_star_entry: Optional[float] = None
    morning_star_stop: Optional[float] = None
    morning_star_target1: Optional[float] = None
    morning_star_target2: Optional[float] = None
    evening_star_entry: Optional[float] = None
    evening_star_stop: Optional[float] = None
    evening_star_target1: Optional[float] = None
    evening_star_target2: Optional[float] = None
    
    # Summary
    total_bullish_patterns: int = 0
    total_bearish_patterns: int = 0
    total_patterns_detected: int = 0
    
    # Current Candle Context
    current_candle_bullish: Optional[bool] = None
    current_candle_bearish: Optional[bool] = None
    current_body_size: Optional[float] = None
    current_range: Optional[float] = None
    current_upper_wick: Optional[float] = None
    current_lower_wick: Optional[float] = None
    avg_body: Optional[float] = None
    body_vs_avg_ratio: Optional[float] = None


@dataclass
class SupportResistanceData:
    """All S/R data from Layer 11"""
    # Fractal Data
    fractal_high_current: Optional[float] = None
    fractal_low_current: Optional[float] = None
    fractal_high_count: int = 0
    fractal_low_count: int = 0
    fractal_highs_recent: List[float] = field(default_factory=list)
    fractal_lows_recent: List[float] = field(default_factory=list)
    
    # S/R Channel Data
    sr_channel_count: int = 0
    sr_strongest_top: Optional[float] = None
    sr_strongest_bottom: Optional[float] = None
    sr_strongest_strength: Optional[float] = None
    sr_strongest_type: Optional[str] = None
    sr_resistance_broken: bool = False
    sr_support_broken: bool = False
    sr_channels: List[Dict] = field(default_factory=list)
    
    # Daily Pivot Points
    daily_pp: Optional[float] = None
    daily_r1: Optional[float] = None
    daily_r2: Optional[float] = None
    daily_r3: Optional[float] = None
    daily_s1: Optional[float] = None
    daily_s2: Optional[float] = None
    daily_s3: Optional[float] = None
    
    # Weekly Pivot Points
    weekly_pp: Optional[float] = None
    weekly_r1: Optional[float] = None
    weekly_r2: Optional[float] = None
    weekly_r3: Optional[float] = None
    weekly_s1: Optional[float] = None
    weekly_s2: Optional[float] = None
    weekly_s3: Optional[float] = None
    
    # Monthly Pivot Points
    monthly_pp: Optional[float] = None
    monthly_r1: Optional[float] = None
    monthly_r2: Optional[float] = None
    monthly_r3: Optional[float] = None
    monthly_s1: Optional[float] = None
    monthly_s2: Optional[float] = None
    monthly_s3: Optional[float] = None
    
    # MTF Levels
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    pwh: Optional[float] = None
    pwl: Optional[float] = None
    pmh: Optional[float] = None
    pml: Optional[float] = None
    ath: Optional[float] = None
    atl: Optional[float] = None
    
    # MTF Touches
    pdh_touch: bool = False
    pdl_touch: bool = False
    pwh_touch: bool = False
    pwl_touch: bool = False
    pmh_touch: bool = False
    pml_touch: bool = False
    
    # MTF Breaks
    pdh_break: bool = False
    pdl_break: bool = False
    pwh_break: bool = False
    pwl_break: bool = False
    
    # Confluence Zones
    confluence_zone_count: int = 0
    confluence_zones: List[Dict] = field(default_factory=list)
    
    # Nearest Levels
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    distance_to_support: Optional[float] = None
    distance_to_resistance: Optional[float] = None
    distance_to_support_pct: Optional[float] = None
    distance_to_resistance_pct: Optional[float] = None
    
    # Price vs Pivots
    price_above_daily_pp: Optional[bool] = None
    price_above_weekly_pp: Optional[bool] = None
    price_above_pdh: Optional[bool] = None
    price_below_pdl: Optional[bool] = None


@dataclass
class VWAPData:
    """All VWAP data from Layer 12"""
    # VWAP Core
    vwap: Optional[float] = None
    stdev: Optional[float] = None
    
    # Band Levels
    upper_1sd: Optional[float] = None
    lower_1sd: Optional[float] = None
    upper_2sd: Optional[float] = None
    lower_2sd: Optional[float] = None
    
    # Price vs VWAP
    price_vs_vwap: Optional[float] = None
    price_vs_vwap_pct: Optional[float] = None
    price_above_vwap: Optional[bool] = None
    price_below_vwap: Optional[bool] = None
    
    # Position Data
    stdev_distance: float = 0
    at_upper_1sd: bool = False
    at_lower_1sd: bool = False
    at_upper_2sd: bool = False
    at_lower_2sd: bool = False
    between_bands: bool = False
    
    # Volume Context
    current_volume: Optional[float] = None
    avg_volume_20: Optional[float] = None
    volume_ratio: float = 1.0
    is_high_volume: bool = False
    
    # Rejection Data
    rejection_count: int = 0
    is_strong_level: bool = False
    
    # Zone Data
    current_zone: str = "NEUTRAL"
    bars_in_zone: int = 0
    accepted_above: bool = False
    accepted_below: bool = False
    
    # Slope Data
    vwap_slope: float = 0
    vwap_slope_pct: float = 0
    slope_above_bull_threshold: bool = False
    slope_below_bear_threshold: bool = False
    slope_neutral: bool = True
    
    # Crossover Data
    crossed_above_vwap: bool = False
    crossed_below_vwap: bool = False
    crossed_above_lower_1sd: bool = False
    crossed_below_upper_1sd: bool = False
    crossed_above_upper_2sd: bool = False
    crossed_below_lower_2sd: bool = False
    
    # Distance to Bands
    distance_to_upper_1sd: Optional[float] = None
    distance_to_lower_1sd: Optional[float] = None
    distance_to_upper_2sd: Optional[float] = None
    distance_to_lower_2sd: Optional[float] = None


@dataclass
class VolumeProfileData:
    """All Volume Profile data from Layer 13"""
    # POC Data
    poc_price: Optional[float] = None
    poc_level: Optional[int] = None
    poc_volume: Optional[float] = None
    poc_volume_pct: float = 0
    
    # Value Area
    vah_price: Optional[float] = None
    val_price: Optional[float] = None
    va_volume_pct: float = 0
    va_width: Optional[float] = None
    
    # Profile Range
    profile_high: Optional[float] = None
    profile_low: Optional[float] = None
    profile_range: Optional[float] = None
    profile_levels: int = 0
    total_volume: Optional[float] = None
    
    # Price vs POC
    price_vs_poc: Optional[float] = None
    price_vs_poc_pct: Optional[float] = None
    price_above_poc: Optional[bool] = None
    price_below_poc: Optional[bool] = None
    
    # Price vs Value Area
    price_vs_vah: Optional[float] = None
    price_vs_val: Optional[float] = None
    in_value_area: bool = False
    above_value_area: bool = False
    below_value_area: bool = False
    position_location: str = "UNKNOWN"
    
    # POC Touch Data
    touching_poc: bool = False
    poc_touch_count: int = 0
    poc_is_strong: bool = False
    distance_to_poc_pct: float = 100
    
    # VAH/VAL Touch
    touching_vah: bool = False
    touching_val: bool = False
    
    # Rejection Data
    rejection_from_above: bool = False
    rejection_from_below: bool = False
    poc_rejection_bull: bool = False
    poc_rejection_bear: bool = False
    upper_wick_pct: float = 0
    lower_wick_pct: float = 0
    
    # Acceptance Data
    bars_at_poc: int = 0
    bars_above_vah: int = 0
    bars_below_val: int = 0
    accepted_at_poc: bool = False
    accepted_above_va: bool = False
    accepted_below_va: bool = False
    
    # Buy Pressure
    buy_pressure_pct: float = 50
    strong_buying: bool = False
    strong_selling: bool = False
    current_level_volume: Optional[float] = None
    current_level_buy_volume: Optional[float] = None
    
    # Volume Context
    current_volume: Optional[float] = None
    avg_volume_20: Optional[float] = None
    volume_ratio: float = 1.0
    is_high_volume: bool = False
    volume_spike: bool = False
    
    # Crossovers
    crossed_above_vah: bool = False
    crossed_below_vah: bool = False
    crossed_above_val: bool = False
    crossed_below_val: bool = False
    crossed_above_poc: bool = False
    crossed_below_poc: bool = False


@dataclass
class IVAnalysisData:
    """All IV Analysis data from Layer 14"""
    # HV Data
    hv_current: Optional[float] = None
    hv_smoothed: Optional[float] = None
    hv_high_52w: Optional[float] = None
    hv_low_52w: Optional[float] = None
    hv_range_52w: Optional[float] = None
    
    # IV Rank/Percentile
    iv_rank: Optional[float] = None
    iv_rank_valid: bool = False
    iv_percentile: Optional[float] = None
    iv_percentile_valid: bool = False
    
    # Threshold Comparisons
    iv_above_extreme_high: Optional[bool] = None
    iv_above_high: Optional[bool] = None
    iv_below_low: Optional[bool] = None
    iv_below_extreme_low: Optional[bool] = None
    iv_in_normal_range: Optional[bool] = None
    
    # State Classification
    iv_state: str = "UNKNOWN"
    
    # Expected Move - Current DTE
    em_dte: Optional[int] = None
    em_1sd: Optional[float] = None
    em_2sd: Optional[float] = None
    em_1sd_pct: Optional[float] = None
    em_2sd_pct: Optional[float] = None
    em_upper_1sd: Optional[float] = None
    em_lower_1sd: Optional[float] = None
    em_upper_2sd: Optional[float] = None
    em_lower_2sd: Optional[float] = None
    
    # Expected Move - Multiple DTEs
    em_7d_1sd_pct: float = 0
    em_14d_1sd_pct: float = 0
    em_30d_1sd_pct: float = 0
    em_45d_1sd_pct: float = 0
    em_60d_1sd_pct: float = 0
    
    # HV Trend
    hv_rising: Optional[bool] = None
    hv_falling: Optional[bool] = None
    hv_stable: Optional[bool] = None
    hv_5d_change: Optional[float] = None
    hv_10d_change: Optional[float] = None
    hv_vs_avg: Optional[float] = None
    
    # Thresholds
    threshold_extreme_high: float = 80
    threshold_high: float = 60
    threshold_low: float = 40
    threshold_extreme_low: float = 20
    
    # Distance to Thresholds
    distance_to_extreme_high: Optional[float] = None
    distance_to_high: Optional[float] = None
    distance_to_low: Optional[float] = None
    distance_to_extreme_low: Optional[float] = None


@dataclass
class GammaMaxPainData:
    """All Gamma & Max Pain data from Layer 15"""
    success: bool = False
    
    # Max Pain Data
    max_pain: Optional[float] = None
    max_pain_total_loss: Optional[float] = None
    max_pain_confidence: Optional[str] = None
    strike_min: Optional[float] = None
    strike_max: Optional[float] = None
    total_call_oi: int = 0
    total_put_oi: int = 0
    total_oi: int = 0
    put_call_oi_ratio: Optional[float] = None
    
    # Price vs Max Pain
    price_vs_max_pain: Optional[float] = None
    price_above_max_pain: Optional[bool] = None
    price_below_max_pain: Optional[bool] = None
    distance_to_max_pain: Optional[float] = None
    distance_to_max_pain_pct: Optional[float] = None
    
    # Distance Thresholds
    within_extreme_danger: Optional[bool] = None
    within_danger: Optional[bool] = None
    within_caution: Optional[bool] = None
    within_safe: Optional[bool] = None
    beyond_safe: Optional[bool] = None
    
    # GEX Data
    gex_total: float = 0
    gex_call: float = 0
    gex_put: float = 0
    gex_regime: str = "UNKNOWN"
    gamma_wall: Optional[float] = None
    
    # GEX Comparisons
    gex_is_positive: Optional[bool] = None
    gex_is_negative: Optional[bool] = None
    gex_above_high_threshold: Optional[bool] = None
    gex_below_neg_high_threshold: Optional[bool] = None
    gex_above_medium_threshold: Optional[bool] = None
    gex_below_neg_medium_threshold: Optional[bool] = None
    
    # Expiration Data
    expiration: Optional[str] = None
    days_to_expiry: Optional[int] = None
    is_expiry_day: Optional[bool] = None
    is_expiry_week: Optional[bool] = None
    is_monthly: Optional[bool] = None
    
    # Pin Probability
    pin_probability_pct: Optional[float] = None
    
    # Thresholds
    threshold_extreme_danger_pct: float = 0.5
    threshold_danger_pct: float = 1.0
    threshold_caution_pct: float = 2.0
    threshold_safe_pct: float = 5.0
    threshold_gex_high: float = 1000000
    threshold_gex_medium: float = 100000
    
    # Strikes
    strikes_analyzed: int = 0


@dataclass
class PutCallRatioData:
    """All Put/Call Ratio data from Layer 16"""
    success: bool = False
    
    # Current PCR
    pcr_current: Optional[float] = None
    pcr_ma_200: Optional[float] = None
    pcr_stdev: Optional[float] = None
    pcr_upper_band: Optional[float] = None
    pcr_lower_band: Optional[float] = None
    
    # PCR vs Bands
    pcr_above_upper_band: Optional[bool] = None
    pcr_below_lower_band: Optional[bool] = None
    pcr_within_bands: Optional[bool] = None
    
    # Distance from Bands
    distance_from_ma: Optional[float] = None
    distance_from_upper: Optional[float] = None
    distance_from_lower: Optional[float] = None
    z_score: Optional[float] = None
    
    # Sentiment Thresholds
    above_extreme_fear: Optional[bool] = None
    above_fear: Optional[bool] = None
    in_neutral_zone: Optional[bool] = None
    below_greed: Optional[bool] = None
    below_extreme_greed: Optional[bool] = None
    
    # Sentiment State
    sentiment_state: str = "UNKNOWN"
    is_extreme_sentiment: Optional[bool] = None
    
    # Put vs Call
    more_puts_than_calls: Optional[bool] = None
    more_calls_than_puts: Optional[bool] = None
    balanced: Optional[bool] = None
    
    # Volume Data
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_volume: int = 0
    call_volume_pct: float = 0
    put_volume_pct: float = 0
    
    # History
    pcr_history_length: int = 0
    has_sufficient_history: bool = False
    
    # Thresholds
    threshold_extreme_fear: float = 1.3
    threshold_fear: float = 1.1
    threshold_neutral_upper: float = 1.0
    threshold_neutral_lower: float = 0.9
    threshold_greed: float = 0.8
    threshold_extreme_greed: float = 0.7
    
    # Distance to Thresholds
    distance_to_extreme_fear: Optional[float] = None
    distance_to_fear: Optional[float] = None
    distance_to_greed: Optional[float] = None
    distance_to_extreme_greed: Optional[float] = None


@dataclass
class GreeksAnalysisData:
    """All Greeks Analysis data from Layer 17"""
    # Best Strike Data
    best_strike: Optional[float] = None
    best_strike_type: Optional[str] = None
    best_strike_score: Optional[float] = None
    best_delta: Optional[float] = None
    best_gamma: Optional[float] = None
    best_theta: Optional[float] = None
    best_vega: Optional[float] = None
    best_iv: Optional[float] = None
    best_dte: Optional[int] = None
    
    # Best Strike Scores
    best_delta_score: Optional[float] = None
    best_gamma_score: Optional[float] = None
    best_theta_score: Optional[float] = None
    best_vega_iv_score: Optional[float] = None
    
    # Best Strike Classifications
    best_delta_is_atm: Optional[bool] = None
    best_delta_is_itm: Optional[bool] = None
    best_delta_is_otm: Optional[bool] = None
    best_gamma_is_high: Optional[bool] = None
    best_gamma_is_low: Optional[bool] = None
    best_theta_is_low: Optional[bool] = None
    best_theta_is_high: Optional[bool] = None
    best_vega_is_high: Optional[bool] = None
    best_vega_is_low: Optional[bool] = None
    
    # Greeks Ranges
    delta_min: Optional[float] = None
    delta_max: Optional[float] = None
    delta_avg: Optional[float] = None
    gamma_peak_strike: Optional[float] = None
    theta_avg: Optional[float] = None
    vega_avg: Optional[float] = None
    
    # Strike Analysis
    total_strikes_analyzed: int = 0
    strikes_with_high_gamma: int = 0
    strikes_with_low_theta: int = 0
    strikes_atm: int = 0
    
    # IV Context
    iv_rank: Optional[float] = None
    iv_rank_is_low: Optional[bool] = None
    dte: Optional[int] = None
    dte_is_short: Optional[bool] = None


@dataclass
class DataQuality:
    """Data quality and completeness tracking"""
    total_layers: int = 17
    layers_with_data: int = 0
    layers_missing: int = 0
    missing_layer_list: List[str] = field(default_factory=list)
    data_completeness_pct: float = 0
    
    # Per-Layer Status
    layer_status: Dict[str, bool] = field(default_factory=dict)
    
    # Data Freshness
    timestamp: Optional[str] = None
    is_market_hours: Optional[bool] = None
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


@dataclass
class MasterAggregatorResult:
    """Complete aggregated data output"""
    # Meta
    ticker: str
    mode: str
    timeframe: str
    current_price: float
    timestamp: str
    
    # All Data Categories
    price_context: PriceContext
    momentum: MomentumData
    volume: VolumeData
    divergences: DivergenceData
    trend: TrendData
    structure: StructureData
    liquidity: LiquidityData
    volatility: VolatilityData
    mtf_confirmation: MTFConfirmationData
    candle_patterns: CandlePatternData
    support_resistance: SupportResistanceData
    vwap: VWAPData
    volume_profile: VolumeProfileData
    iv_analysis: IVAnalysisData
    gamma_max_pain: GammaMaxPainData
    put_call_ratio: PutCallRatioData
    greeks: GreeksAnalysisData
    
    # Quality
    data_quality: DataQuality
    
    # Raw Layers (for debugging)
    raw_layers: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MAIN AGGREGATOR CLASS
# =============================================================================

class Layer18MasterAggregator:
    """
    Master Data Aggregator - Extract 100% of data from all 17 layers.
    
    NO trading decisions, NO scoring - Pure Data Aggregation.
    AI receives this organized data and makes the trade decision.
    """
    
    def __init__(self):
        self.version = "5.0"
    
    def analyze(
        self,
        ticker: str,
        layer_results: Dict[str, Any],
        current_price: float,
        mode: TradeMode = TradeMode.SCALP
    ) -> MasterAggregatorResult:
        """
        Aggregate all layer data into organized structure.
        
        Args:
            ticker: Stock symbol
            layer_results: Dict with layer_1 through layer_17 results
            current_price: Current stock price
            mode: Trading mode (SCALP, INTRADAY, SWING, LEAPS)
            
        Returns:
            MasterAggregatorResult with 100% of data organized
        """
        # Extract all categories
        price_context = self._extract_price_context(layer_results, current_price)
        momentum = self._extract_momentum(layer_results)
        volume = self._extract_volume(layer_results)
        divergences = self._extract_divergences(layer_results)
        trend = self._extract_trend(layer_results)
        structure = self._extract_structure(layer_results)
        liquidity = self._extract_liquidity(layer_results)
        volatility = self._extract_volatility(layer_results)
        mtf_confirmation = self._extract_mtf_confirmation(layer_results)
        candle_patterns = self._extract_candle_patterns(layer_results)
        support_resistance = self._extract_support_resistance(layer_results)
        vwap = self._extract_vwap(layer_results)
        volume_profile = self._extract_volume_profile(layer_results)
        iv_analysis = self._extract_iv_analysis(layer_results)
        gamma_max_pain = self._extract_gamma_max_pain(layer_results)
        put_call_ratio = self._extract_put_call_ratio(layer_results)
        greeks = self._extract_greeks(layer_results)
        
        # Calculate data quality
        data_quality = self._calculate_data_quality(layer_results)
        
        # Determine timeframe from mode
        timeframe_map = {
            TradeMode.SCALP: "5min",
            TradeMode.INTRADAY: "15min",
            TradeMode.SWING: "daily",
            TradeMode.LEAPS: "daily"
        }
        
        return MasterAggregatorResult(
            ticker=ticker,
            mode=mode.value if isinstance(mode, TradeMode) else str(mode),
            timeframe=timeframe_map.get(mode, "daily") if isinstance(mode, TradeMode) else "daily",
            current_price=current_price,
            timestamp=datetime.now().isoformat(),
            price_context=price_context,
            momentum=momentum,
            volume=volume,
            divergences=divergences,
            trend=trend,
            structure=structure,
            liquidity=liquidity,
            volatility=volatility,
            mtf_confirmation=mtf_confirmation,
            candle_patterns=candle_patterns,
            support_resistance=support_resistance,
            vwap=vwap,
            volume_profile=volume_profile,
            iv_analysis=iv_analysis,
            gamma_max_pain=gamma_max_pain,
            put_call_ratio=put_call_ratio,
            greeks=greeks,
            data_quality=data_quality,
            raw_layers=layer_results
        )
    
    # =========================================================================
    # EXTRACTION METHODS - 100% DATA EXTRACTION
    # =========================================================================
    
    def _extract_price_context(self, layers: Dict, current_price: float) -> PriceContext:
        """Extract price context from layers 5, 6, 12"""
        l5 = layers.get("layer_5", {})
        l6 = layers.get("layer_6", {})
        l12 = layers.get("layer_12", {})
        
        return PriceContext(
            current_price=current_price,
            trend_ema_50=l6.get("trend_ema"),
            price_vs_trend_ema=l6.get("price_vs_trend_ema"),
            price_vs_trend_ema_pct=l6.get("price_vs_trend_ema_pct"),
            is_above_trend_ema=l6.get("is_above_trend_ema"),
            atr=l5.get("atr"),
            atr_percent=l5.get("atr_percent"),
            supertrend_value=l5.get("supertrend_value"),
            supertrend_bullish=l5.get("supertrend_bullish"),
            supertrend_bearish=l5.get("supertrend_bearish"),
            price_vs_supertrend=(current_price - l5.get("supertrend_value")) if l5.get("supertrend_value") else None
        )
    
    def _extract_momentum(self, layers: Dict) -> MomentumData:
        """Extract 100% of momentum data from Layer 1"""
        l1 = layers.get("layer_1", {})
        
        rsi_14 = l1.get("rsi_14")
        rsi_prev = l1.get("rsi_prev")
        stoch_k = l1.get("stoch_k")
        stoch_k_prev = l1.get("stoch_k_prev")
        macd_hist = l1.get("macd_histogram")
        macd_hist_prev = l1.get("macd_histogram_prev")
        
        return MomentumData(
            rsi_14=rsi_14,
            rsi_7=l1.get("rsi_7"),
            rsi_prev=rsi_prev,
            rsi_direction=self._get_direction(rsi_14, rsi_prev),
            rsi_zone=self._get_rsi_zone(rsi_14),
            macd_line=l1.get("macd_line"),
            macd_signal_line=l1.get("macd_signal_line"),
            macd_histogram=macd_hist,
            macd_histogram_prev=macd_hist_prev,
            macd_histogram_rising=l1.get("macd_histogram_rising"),
            macd_above_signal=(l1.get("macd_line", 0) or 0) > (l1.get("macd_signal_line", 0) or 0),
            macd_bullish_cross=(macd_hist or 0) > 0 and (macd_hist_prev or 0) <= 0,
            macd_bearish_cross=(macd_hist or 0) < 0 and (macd_hist_prev or 0) >= 0,
            stoch_k=stoch_k,
            stoch_d=l1.get("stoch_d"),
            stoch_k_prev=stoch_k_prev,
            stoch_zone=self._get_stoch_zone(stoch_k),
            stoch_bullish_cross=(stoch_k or 0) > (l1.get("stoch_d") or 0) and (stoch_k_prev or 0) <= (l1.get("stoch_d") or 0),
            stoch_bearish_cross=(stoch_k or 0) < (l1.get("stoch_d") or 0) and (stoch_k_prev or 0) >= (l1.get("stoch_d") or 0),
            cmf=l1.get("cmf"),
            cmf_prev=l1.get("cmf_prev"),
            cmf_positive=(l1.get("cmf") or 0) > 0,
            cmf_direction=self._get_direction(l1.get("cmf"), l1.get("cmf_prev")),
            adx=l1.get("adx"),
            plus_di=l1.get("plus_di"),
            minus_di=l1.get("minus_di"),
            di_diff=l1.get("di_diff"),
            adx_trend_strength=self._get_adx_strength(l1.get("adx")),
            di_bullish=(l1.get("plus_di") or 0) > (l1.get("minus_di") or 0),
            di_bearish=(l1.get("minus_di") or 0) > (l1.get("plus_di") or 0),
            ichimoku_conv=l1.get("ichimoku_conv"),
            ichimoku_base=l1.get("ichimoku_base"),
            ichimoku_lead1=l1.get("ichimoku_lead1"),
            ichimoku_lead2=l1.get("ichimoku_lead2"),
            price_vs_cloud_top=l1.get("price_vs_cloud_top"),
            price_vs_cloud_bottom=l1.get("price_vs_cloud_bottom"),
            price_above_cloud=(l1.get("price_vs_cloud_bottom") or 0) > 0,
            price_below_cloud=(l1.get("price_vs_cloud_top") or 0) < 0,
            price_in_cloud=(l1.get("price_vs_cloud_top") or 0) >= 0 and (l1.get("price_vs_cloud_bottom") or 0) <= 0
        )
    
    def _extract_volume(self, layers: Dict) -> VolumeData:
        """Extract 100% of volume data from Layers 2, 4"""
        l2 = layers.get("layer_2", {})
        l4 = layers.get("layer_4", {})
        
        cvd = l4.get("cvd")
        cvd_prev = l4.get("cvd_prev")
        buying_pct = l4.get("buying_volume_pct", 50)
        selling_pct = l4.get("selling_volume_pct", 50)
        
        return VolumeData(
            current_volume=l2.get("current_volume"),
            avg_volume_20=l2.get("avg_volume_20"),
            volume_ratio=l2.get("volume_ratio"),
            volume_change_5bar_pct=l2.get("volume_change_5bar_pct"),
            volume_interpretation=self._get_volume_interpretation(l2.get("volume_ratio")),
            obv=l2.get("obv"),
            obv_ma=l2.get("obv_ma"),
            obv_prev=l2.get("obv_prev"),
            obv_slope=l2.get("obv_slope"),
            obv_vs_ma=l2.get("obv_vs_ma"),
            obv_trend=self._get_slope_trend(l2.get("obv_slope")),
            ad_line=l2.get("ad_line"),
            ad_ma=l2.get("ad_ma"),
            ad_prev=l2.get("ad_prev"),
            ad_slope=l2.get("ad_slope"),
            ad_vs_ma=l2.get("ad_vs_ma"),
            ad_trend=self._get_slope_trend(l2.get("ad_slope")),
            cvd=cvd,
            cvd_prev=cvd_prev,
            cvd_trend="rising" if (cvd or 0) > (cvd_prev or 0) else "falling" if (cvd or 0) < (cvd_prev or 0) else "flat",
            cumulative_buying_volume=l4.get("cumulative_buying_volume"),
            cumulative_selling_volume=l4.get("cumulative_selling_volume"),
            volume_strength_wave=l4.get("volume_strength_wave"),
            ema_volume_strength_wave=l4.get("ema_volume_strength_wave"),
            latest_buying_volume=l4.get("latest_buying_volume"),
            latest_selling_volume=l4.get("latest_selling_volume"),
            buying_volume_pct=buying_pct,
            selling_volume_pct=selling_pct,
            volume_pressure="buying" if (buying_pct or 50) > 55 else "selling" if (selling_pct or 50) > 55 else "neutral",
            eom=l4.get("eom"),
            eom_prev=l4.get("eom_prev"),
            eom_hl2_change=l4.get("eom_hl2_change"),
            eom_distance=l4.get("eom_distance"),
            price_rising_volume_falling=l2.get("price_rising_volume_falling"),
            price_falling_volume_rising=l2.get("price_falling_volume_rising")
        )
    
    def _extract_divergences(self, layers: Dict) -> DivergenceData:
        """Extract 100% of divergence data from Layer 3"""
        l3 = layers.get("layer_3", {})
        counts = l3.get("divergence_counts", {})
        
        total_bull = counts.get("total_bullish_divergences", 0)
        total_bear = counts.get("total_bearish_divergences", 0)
        net = total_bull - total_bear
        
        return DivergenceData(
            macd_regular_bearish_count=counts.get("macd_regular_bearish_count", 0),
            macd_hidden_bearish_count=counts.get("macd_hidden_bearish_count", 0),
            macd_regular_bullish_count=counts.get("macd_regular_bullish_count", 0),
            macd_hidden_bullish_count=counts.get("macd_hidden_bullish_count", 0),
            macd_total_bearish=counts.get("macd_total_bearish", 0),
            macd_total_bullish=counts.get("macd_total_bullish", 0),
            rsi_regular_bearish_count=counts.get("rsi_regular_bearish_count", 0),
            rsi_hidden_bearish_count=counts.get("rsi_hidden_bearish_count", 0),
            rsi_regular_bullish_count=counts.get("rsi_regular_bullish_count", 0),
            rsi_hidden_bullish_count=counts.get("rsi_hidden_bullish_count", 0),
            rsi_total_bearish=counts.get("rsi_total_bearish", 0),
            rsi_total_bullish=counts.get("rsi_total_bullish", 0),
            total_bearish_divergences=total_bear,
            total_bullish_divergences=total_bull,
            total_regular_divergences=counts.get("total_regular_divergences", 0),
            total_hidden_divergences=counts.get("total_hidden_divergences", 0),
            net_divergence=net,
            divergence_bias="bullish" if net > 0 else "bearish" if net < 0 else "neutral",
            latest_rsi=l3.get("latest_rsi"),
            latest_macd_1h=l3.get("latest_macd_1h"),
            latest_macd_4h=l3.get("latest_macd_4h"),
            latest_macd_1d=l3.get("latest_macd_1d")
        )
    
    def _extract_trend(self, layers: Dict) -> TrendData:
        """Extract 100% of trend data from Layer 5"""
        l5 = layers.get("layer_5", {})
        
        return TrendData(
            supertrend_value=l5.get("supertrend_value"),
            supertrend_direction=l5.get("supertrend_direction"),
            supertrend_bullish=l5.get("supertrend_bullish"),
            supertrend_bearish=l5.get("supertrend_bearish"),
            trend_changed=l5.get("trend_changed"),
            raw_buy_signal=l5.get("raw_buy_signal"),
            raw_sell_signal=l5.get("raw_sell_signal"),
            atr=l5.get("atr"),
            atr_percent=l5.get("atr_percent"),
            volatility_high=l5.get("volatility_high"),
            volatility_low=l5.get("volatility_low"),
            volatility_normal=l5.get("volatility_normal"),
            adaptive_multiplier=l5.get("adaptive_multiplier"),
            adx=l5.get("adx"),
            di_plus=l5.get("di_plus"),
            di_minus=l5.get("di_minus"),
            trending=l5.get("trending"),
            weak_trend=l5.get("weak_trend"),
            choppy=l5.get("choppy"),
            whipsaw_mode=l5.get("whipsaw_mode"),
            flip_count=l5.get("flip_count"),
            bars_since_flip=l5.get("bars_since_flip"),
            volume_ratio=l5.get("volume_ratio"),
            avg_volume=l5.get("avg_volume"),
            current_volume=l5.get("current_volume"),
            volume_confirmed=l5.get("volume_confirmed"),
            htf_timeframe=l5.get("htf_timeframe"),
            htf_bullish=l5.get("htf_bullish"),
            htf_bearish=l5.get("htf_bearish"),
            htf_aligned=l5.get("htf_aligned"),
            rsi=l5.get("rsi"),
            rsi_aligned=l5.get("rsi_aligned"),
            is_first_30min=l5.get("is_first_30min"),
            is_lunch_hours=l5.get("is_lunch_hours"),
            is_close_risk=l5.get("is_close_risk"),
            is_optimal_hours=l5.get("is_optimal_hours"),
            bars_in_trend=l5.get("bars_in_trend")
        )
    
    def _extract_structure(self, layers: Dict) -> StructureData:
        """Extract 100% of structure data from Layer 6"""
        l6 = layers.get("layer_6", {})
        
        return StructureData(
            last_pivot_high=l6.get("last_pivot_high"),
            last_pivot_low=l6.get("last_pivot_low"),
            last_pivot_high_index=l6.get("last_pivot_high_index"),
            last_pivot_low_index=l6.get("last_pivot_low_index"),
            pivot_high_count=l6.get("pivot_high_count", 0),
            pivot_low_count=l6.get("pivot_low_count", 0),
            recent_pivot_highs=l6.get("recent_pivot_highs", []),
            recent_pivot_lows=l6.get("recent_pivot_lows", []),
            recent_pivot_high_indices=l6.get("recent_pivot_high_indices", []),
            recent_pivot_low_indices=l6.get("recent_pivot_low_indices", []),
            consecutive_higher_highs=l6.get("consecutive_higher_highs", 0),
            consecutive_higher_lows=l6.get("consecutive_higher_lows", 0),
            consecutive_lower_highs=l6.get("consecutive_lower_highs", 0),
            consecutive_lower_lows=l6.get("consecutive_lower_lows", 0),
            highs_ascending=l6.get("highs_ascending"),
            lows_ascending=l6.get("lows_ascending"),
            highs_descending=l6.get("highs_descending"),
            lows_descending=l6.get("lows_descending"),
            price_above_last_pivot_high=l6.get("price_above_last_pivot_high"),
            price_below_last_pivot_high=l6.get("price_below_last_pivot_high"),
            price_above_last_pivot_low=l6.get("price_above_last_pivot_low"),
            price_below_last_pivot_low=l6.get("price_below_last_pivot_low"),
            price_above_all_recent_lows=l6.get("price_above_all_recent_lows"),
            price_below_all_recent_highs=l6.get("price_below_all_recent_highs"),
            distance_to_last_pivot_high=l6.get("distance_to_last_pivot_high"),
            distance_to_last_pivot_high_pct=l6.get("distance_to_last_pivot_high_pct"),
            distance_to_last_pivot_low=l6.get("distance_to_last_pivot_low"),
            distance_to_last_pivot_low_pct=l6.get("distance_to_last_pivot_low_pct"),
            current_swing_range=l6.get("current_swing_range"),
            current_swing_range_pct=l6.get("current_swing_range_pct"),
            price_position_in_range_pct=l6.get("price_position_in_range_pct"),
            choch_bull_detected=l6.get("choch_bull_detected", False),
            choch_bear_detected=l6.get("choch_bear_detected", False),
            choch_bull_quality=l6.get("choch_bull_quality", 0),
            choch_bear_quality=l6.get("choch_bear_quality", 0),
            choch_bull_delta=l6.get("choch_bull_delta", 0),
            choch_bear_delta=l6.get("choch_bear_delta", 0),
            bos_bull_detected=l6.get("bos_bull_detected", False),
            bos_bear_detected=l6.get("bos_bear_detected", False),
            bos_bull_quality=l6.get("bos_bull_quality", 0),
            bos_bear_quality=l6.get("bos_bear_quality", 0),
            bos_bull_delta=l6.get("bos_bull_delta", 0),
            bos_bear_delta=l6.get("bos_bear_delta", 0),
            total_choch_bull=l6.get("total_choch_bull", 0),
            total_choch_bear=l6.get("total_choch_bear", 0),
            total_bos_bull=l6.get("total_bos_bull", 0),
            total_bos_bear=l6.get("total_bos_bear", 0),
            current_trend=l6.get("current_trend", 0),
            ob_bull_detected=l6.get("ob_bull_detected", False),
            ob_bear_detected=l6.get("ob_bear_detected", False),
            ob_bull_quality=l6.get("ob_bull_quality", 0),
            ob_bear_quality=l6.get("ob_bear_quality", 0),
            ob_bull_top=l6.get("ob_bull_top"),
            ob_bull_btm=l6.get("ob_bull_btm"),
            ob_bear_top=l6.get("ob_bear_top"),
            ob_bear_btm=l6.get("ob_bear_btm"),
            total_ob_bull=l6.get("total_ob_bull", 0),
            total_ob_bear=l6.get("total_ob_bear", 0),
            fvg_bull_detected=l6.get("fvg_bull_detected", False),
            fvg_bear_detected=l6.get("fvg_bear_detected", False),
            fvg_bull_quality=l6.get("fvg_bull_quality", 0),
            fvg_bear_quality=l6.get("fvg_bear_quality", 0),
            fvg_bull_top=l6.get("fvg_bull_top"),
            fvg_bull_btm=l6.get("fvg_bull_btm"),
            fvg_bear_top=l6.get("fvg_bear_top"),
            fvg_bear_btm=l6.get("fvg_bear_btm"),
            total_fvg_bull=l6.get("total_fvg_bull", 0),
            total_fvg_bear=l6.get("total_fvg_bear", 0),
            liq_buy_detected=l6.get("liq_buy_detected", False),
            liq_sell_detected=l6.get("liq_sell_detected", False),
            liq_buy_level=l6.get("liq_buy_level"),
            liq_sell_level=l6.get("liq_sell_level"),
            trend_ema=l6.get("trend_ema"),
            price_vs_trend_ema=l6.get("price_vs_trend_ema"),
            price_vs_trend_ema_pct=l6.get("price_vs_trend_ema_pct"),
            is_above_trend_ema=l6.get("is_above_trend_ema")
        )
    
    def _extract_liquidity(self, layers: Dict) -> LiquidityData:
        """Extract 100% of liquidity data from Layer 7"""
        l7 = layers.get("layer_7", {})
        
        return LiquidityData(
            bull_sweep_detected=l7.get("bull_sweep_detected", False),
            bear_sweep_detected=l7.get("bear_sweep_detected", False),
            bull_sweep_type=l7.get("bull_sweep_type"),
            bear_sweep_type=l7.get("bear_sweep_type"),
            bull_sweep_zone_top=l7.get("bull_sweep_zone_top"),
            bull_sweep_zone_bottom=l7.get("bull_sweep_zone_bottom"),
            bear_sweep_zone_top=l7.get("bear_sweep_zone_top"),
            bear_sweep_zone_bottom=l7.get("bear_sweep_zone_bottom"),
            total_bull_sweeps=l7.get("total_bull_sweeps", 0),
            total_bear_sweeps=l7.get("total_bear_sweeps", 0),
            ict_buy_liq_detected=l7.get("ict_buy_liq_detected", False),
            ict_sell_liq_detected=l7.get("ict_sell_liq_detected", False),
            ict_buy_liq_level=l7.get("ict_buy_liq_level"),
            ict_sell_liq_level=l7.get("ict_sell_liq_level"),
            bullish_grab_detected=l7.get("bullish_grab_detected", False),
            bearish_grab_detected=l7.get("bearish_grab_detected", False),
            bull_grab_level=l7.get("bull_grab_level"),
            bear_grab_level=l7.get("bear_grab_level"),
            bull_grab_strength=l7.get("bull_grab_strength", 0),
            bear_grab_strength=l7.get("bear_grab_strength", 0),
            bull_grab_retrace=l7.get("bull_grab_retrace", 0),
            bear_grab_retrace=l7.get("bear_grab_retrace", 0),
            bull_grab_type=l7.get("bull_grab_type"),
            bear_grab_type=l7.get("bear_grab_type"),
            bull_is_round_number=l7.get("bull_is_round_number", False),
            bear_is_round_number=l7.get("bear_is_round_number", False),
            bull_is_equal_level=l7.get("bull_is_equal_level", False),
            bear_is_equal_level=l7.get("bear_is_equal_level", False),
            total_bull_grabs=l7.get("total_bull_grabs", 0),
            total_bear_grabs=l7.get("total_bear_grabs", 0),
            total_sweeps=l7.get("total_sweeps", 0),
            total_grabs=l7.get("total_grabs", 0),
            successful_bull_grabs=l7.get("successful_bull_grabs", 0),
            successful_bear_grabs=l7.get("successful_bear_grabs", 0),
            trend_ema=l7.get("trend_ema"),
            price_vs_trend_ema=l7.get("price_vs_trend_ema"),
            is_above_trend_ema=l7.get("is_above_trend_ema"),
            adx=l7.get("adx"),
            plus_di=l7.get("plus_di"),
            minus_di=l7.get("minus_di"),
            current_volume=l7.get("current_volume"),
            avg_volume=l7.get("avg_volume"),
            volume_ratio=l7.get("volume_ratio"),
            volume_spike=l7.get("volume_spike", False)
        )
    
    def _extract_volatility(self, layers: Dict) -> VolatilityData:
        """Extract 100% of volatility data from Layer 8"""
        l8 = layers.get("layer_8", {})
        
        percentile = l8.get("percentile_rank")
        
        return VolatilityData(
            atr=l8.get("atr"),
            atr_prev=l8.get("atr_prev"),
            atr_change_pct=l8.get("atr_change_pct"),
            atrp=l8.get("atrp"),
            atrp_prev=l8.get("atrp_prev"),
            atrp_smoothed=l8.get("atrp_smoothed"),
            atrp_trend_5bar=l8.get("atrp_trend_5bar"),
            percentile_rank=percentile,
            percentile_bucket=l8.get("percentile_bucket"),
            p20_threshold=l8.get("p20_threshold"),
            p40_threshold=l8.get("p40_threshold"),
            p60_threshold=l8.get("p60_threshold"),
            p80_threshold=l8.get("p80_threshold"),
            is_below_p20=l8.get("is_below_p20", False),
            is_below_p40=l8.get("is_below_p40", False),
            is_above_p60=l8.get("is_above_p60", False),
            is_above_p80=l8.get("is_above_p80", False),
            volatility_expanding=l8.get("volatility_expanding", False),
            volatility_contracting=l8.get("volatility_contracting", False),
            regime=self._get_volatility_regime(percentile)
        )
    
    def _extract_mtf_confirmation(self, layers: Dict) -> MTFConfirmationData:
        """Extract 100% of MTF confirmation data from Layer 9"""
        l9 = layers.get("layer_9", {})
        
        bull = l9.get("bull_count", 0)
        bear = l9.get("bear_count", 0)
        
        return MTFConfirmationData(
            current_timeframe=l9.get("current_timeframe"),
            current_st_direction=l9.get("current_st_direction"),
            current_st_bullish=l9.get("current_st_bullish"),
            current_st_bearish=l9.get("current_st_bearish"),
            current_adx=l9.get("current_adx"),
            current_st_line=l9.get("current_st_line"),
            tf_5min_direction=l9.get("tf_5min_direction"),
            tf_5min_bullish=l9.get("tf_5min_bullish"),
            tf_5min_adx=l9.get("tf_5min_adx"),
            tf_5min_weight=l9.get("tf_5min_weight", 25),
            tf_15min_direction=l9.get("tf_15min_direction"),
            tf_15min_bullish=l9.get("tf_15min_bullish"),
            tf_15min_adx=l9.get("tf_15min_adx"),
            tf_15min_weight=l9.get("tf_15min_weight", 25),
            tf_1h_direction=l9.get("tf_1h_direction"),
            tf_1h_bullish=l9.get("tf_1h_bullish"),
            tf_1h_adx=l9.get("tf_1h_adx"),
            tf_1h_weight=l9.get("tf_1h_weight", 25),
            tf_4h_direction=l9.get("tf_4h_direction"),
            tf_4h_bullish=l9.get("tf_4h_bullish"),
            tf_4h_adx=l9.get("tf_4h_adx"),
            tf_4h_weight=l9.get("tf_4h_weight", 15),
            tf_1d_direction=l9.get("tf_1d_direction"),
            tf_1d_bullish=l9.get("tf_1d_bullish"),
            tf_1d_adx=l9.get("tf_1d_adx"),
            tf_1d_weight=l9.get("tf_1d_weight", 10),
            bull_count=bull,
            bear_count=bear,
            total_timeframes=l9.get("total_timeframes", 0),
            aligned_with_current_count=l9.get("aligned_with_current_count", 0),
            not_aligned_count=l9.get("not_aligned_count", 0),
            weighted_aligned=l9.get("weighted_aligned", 0),
            weighted_not_aligned=l9.get("weighted_not_aligned", 0),
            total_weight=l9.get("total_weight", 0),
            alignment_pct=l9.get("alignment_pct", 0),
            htf_bullish_count=l9.get("htf_bullish_count", 0),
            htf_bearish_count=l9.get("htf_bearish_count", 0),
            htf_aligned=l9.get("htf_aligned", False),
            mtf_dominant_trend="bullish" if bull > bear else "bearish" if bear > bull else "mixed"
        )
    
    def _extract_candle_patterns(self, layers: Dict) -> CandlePatternData:
        """Extract 100% of candle pattern data from Layer 10"""
        l10 = layers.get("layer_10", {})
        
        return CandlePatternData(
            tws_detected=l10.get("tws_detected", False),
            tws_quality=l10.get("tws_quality", 0),
            tws_high_quality=l10.get("tws_high_quality", False),
            tws_pattern_base=l10.get("tws_pattern_base", False),
            tws_volume_strong=l10.get("tws_volume_strong", False),
            tws_after_downtrend=l10.get("tws_after_downtrend", False),
            tws_near_support=l10.get("tws_near_support", False),
            tws_entry=l10.get("tws_entry"),
            tws_stop=l10.get("tws_stop"),
            tws_target1=l10.get("tws_target1"),
            tws_target2=l10.get("tws_target2"),
            ib_detected=l10.get("ib_detected", False),
            ib_bullish_breakout=l10.get("ib_bullish_breakout", False),
            ib_bearish_breakout=l10.get("ib_bearish_breakout", False),
            ib_quality=l10.get("ib_quality", 0),
            ib_mother_high=l10.get("ib_mother_high"),
            ib_mother_low=l10.get("ib_mother_low"),
            ib_inside_ratio_pct=l10.get("ib_inside_ratio_pct", 0),
            ib_bull_entry=l10.get("ib_bull_entry"),
            ib_bull_stop=l10.get("ib_bull_stop"),
            ib_bull_target1=l10.get("ib_bull_target1"),
            ib_bull_target2=l10.get("ib_bull_target2"),
            ib_bear_entry=l10.get("ib_bear_entry"),
            ib_bear_stop=l10.get("ib_bear_stop"),
            ib_bear_target1=l10.get("ib_bear_target1"),
            ib_bear_target2=l10.get("ib_bear_target2"),
            patterns_detected=l10.get("patterns_detected", []),
            bullish_patterns=l10.get("bullish_patterns", []),
            bearish_patterns=l10.get("bearish_patterns", []),
            bullish_pattern_count=l10.get("bullish_pattern_count", 0),
            bearish_pattern_count=l10.get("bearish_pattern_count", 0),
            neutral_pattern_count=l10.get("neutral_pattern_count", 0),
            doji_detected=l10.get("doji_detected", False),
            hammer_detected=l10.get("hammer_detected", False),
            inverted_hammer_detected=l10.get("inverted_hammer_detected", False),
            hanging_man_detected=l10.get("hanging_man_detected", False),
            shooting_star_detected=l10.get("shooting_star_detected", False),
            morning_star_detected=l10.get("morning_star_detected", False),
            evening_star_detected=l10.get("evening_star_detected", False),
            bullish_engulfing_detected=l10.get("bullish_engulfing_detected", False),
            bearish_engulfing_detected=l10.get("bearish_engulfing_detected", False),
            bullish_harami_detected=l10.get("bullish_harami_detected", False),
            bearish_harami_detected=l10.get("bearish_harami_detected", False),
            piercing_line_detected=l10.get("piercing_line_detected", False),
            dark_cloud_detected=l10.get("dark_cloud_detected", False),
            bullish_kicker_detected=l10.get("bullish_kicker_detected", False),
            bearish_kicker_detected=l10.get("bearish_kicker_detected", False),
            patterns_bull_quality=l10.get("patterns_bull_quality", 0),
            patterns_bear_quality=l10.get("patterns_bear_quality", 0),
            morning_star_pro_detected=l10.get("morning_star_pro_detected", False),
            evening_star_pro_detected=l10.get("evening_star_pro_detected", False),
            morning_star_quality=l10.get("morning_star_quality", 0),
            evening_star_quality=l10.get("evening_star_quality", 0),
            morning_star_high_quality=l10.get("morning_star_high_quality", False),
            evening_star_high_quality=l10.get("evening_star_high_quality", False),
            morning_star_entry=l10.get("morning_star_entry"),
            morning_star_stop=l10.get("morning_star_stop"),
            morning_star_target1=l10.get("morning_star_target1"),
            morning_star_target2=l10.get("morning_star_target2"),
            evening_star_entry=l10.get("evening_star_entry"),
            evening_star_stop=l10.get("evening_star_stop"),
            evening_star_target1=l10.get("evening_star_target1"),
            evening_star_target2=l10.get("evening_star_target2"),
            total_bullish_patterns=l10.get("total_bullish_patterns", 0),
            total_bearish_patterns=l10.get("total_bearish_patterns", 0),
            total_patterns_detected=l10.get("total_patterns_detected", 0),
            current_candle_bullish=l10.get("current_candle_bullish"),
            current_candle_bearish=l10.get("current_candle_bearish"),
            current_body_size=l10.get("current_body_size"),
            current_range=l10.get("current_range"),
            current_upper_wick=l10.get("current_upper_wick"),
            current_lower_wick=l10.get("current_lower_wick"),
            avg_body=l10.get("avg_body"),
            body_vs_avg_ratio=l10.get("body_vs_avg_ratio")
        )
    
    def _extract_support_resistance(self, layers: Dict) -> SupportResistanceData:
        """Extract 100% of S/R data from Layer 11"""
        l11 = layers.get("layer_11", {})
        
        return SupportResistanceData(
            fractal_high_current=l11.get("fractal_high_current"),
            fractal_low_current=l11.get("fractal_low_current"),
            fractal_high_count=l11.get("fractal_high_count", 0),
            fractal_low_count=l11.get("fractal_low_count", 0),
            fractal_highs_recent=l11.get("fractal_highs_recent", []),
            fractal_lows_recent=l11.get("fractal_lows_recent", []),
            sr_channel_count=l11.get("sr_channel_count", 0),
            sr_strongest_top=l11.get("sr_strongest_top"),
            sr_strongest_bottom=l11.get("sr_strongest_bottom"),
            sr_strongest_strength=l11.get("sr_strongest_strength"),
            sr_strongest_type=l11.get("sr_strongest_type"),
            sr_resistance_broken=l11.get("sr_resistance_broken", False),
            sr_support_broken=l11.get("sr_support_broken", False),
            sr_channels=l11.get("sr_channels", []),
            daily_pp=l11.get("daily_pp"),
            daily_r1=l11.get("daily_r1"),
            daily_r2=l11.get("daily_r2"),
            daily_r3=l11.get("daily_r3"),
            daily_s1=l11.get("daily_s1"),
            daily_s2=l11.get("daily_s2"),
            daily_s3=l11.get("daily_s3"),
            weekly_pp=l11.get("weekly_pp"),
            weekly_r1=l11.get("weekly_r1"),
            weekly_r2=l11.get("weekly_r2"),
            weekly_r3=l11.get("weekly_r3"),
            weekly_s1=l11.get("weekly_s1"),
            weekly_s2=l11.get("weekly_s2"),
            weekly_s3=l11.get("weekly_s3"),
            monthly_pp=l11.get("monthly_pp"),
            monthly_r1=l11.get("monthly_r1"),
            monthly_r2=l11.get("monthly_r2"),
            monthly_r3=l11.get("monthly_r3"),
            monthly_s1=l11.get("monthly_s1"),
            monthly_s2=l11.get("monthly_s2"),
            monthly_s3=l11.get("monthly_s3"),
            pdh=l11.get("pdh"),
            pdl=l11.get("pdl"),
            pwh=l11.get("pwh"),
            pwl=l11.get("pwl"),
            pmh=l11.get("pmh"),
            pml=l11.get("pml"),
            ath=l11.get("ath"),
            atl=l11.get("atl"),
            pdh_touch=l11.get("pdh_touch", False),
            pdl_touch=l11.get("pdl_touch", False),
            pwh_touch=l11.get("pwh_touch", False),
            pwl_touch=l11.get("pwl_touch", False),
            pmh_touch=l11.get("pmh_touch", False),
            pml_touch=l11.get("pml_touch", False),
            pdh_break=l11.get("pdh_break", False),
            pdl_break=l11.get("pdl_break", False),
            pwh_break=l11.get("pwh_break", False),
            pwl_break=l11.get("pwl_break", False),
            confluence_zone_count=l11.get("confluence_zone_count", 0),
            confluence_zones=l11.get("confluence_zones", []),
            nearest_support=l11.get("nearest_support"),
            nearest_resistance=l11.get("nearest_resistance"),
            distance_to_support=l11.get("distance_to_support"),
            distance_to_resistance=l11.get("distance_to_resistance"),
            distance_to_support_pct=l11.get("distance_to_support_pct"),
            distance_to_resistance_pct=l11.get("distance_to_resistance_pct"),
            price_above_daily_pp=l11.get("price_above_daily_pp"),
            price_above_weekly_pp=l11.get("price_above_weekly_pp"),
            price_above_pdh=l11.get("price_above_pdh"),
            price_below_pdl=l11.get("price_below_pdl")
        )
    
    def _extract_vwap(self, layers: Dict) -> VWAPData:
        """Extract 100% of VWAP data from Layer 12"""
        l12 = layers.get("layer_12", {})
        
        return VWAPData(
            vwap=l12.get("vwap"),
            stdev=l12.get("stdev"),
            upper_1sd=l12.get("upper_1sd"),
            lower_1sd=l12.get("lower_1sd"),
            upper_2sd=l12.get("upper_2sd"),
            lower_2sd=l12.get("lower_2sd"),
            price_vs_vwap=l12.get("price_vs_vwap"),
            price_vs_vwap_pct=l12.get("price_vs_vwap_pct"),
            price_above_vwap=l12.get("price_above_vwap"),
            price_below_vwap=l12.get("price_below_vwap"),
            stdev_distance=l12.get("stdev_distance", 0),
            at_upper_1sd=l12.get("at_upper_1sd", False),
            at_lower_1sd=l12.get("at_lower_1sd", False),
            at_upper_2sd=l12.get("at_upper_2sd", False),
            at_lower_2sd=l12.get("at_lower_2sd", False),
            between_bands=l12.get("between_bands", False),
            current_volume=l12.get("current_volume"),
            avg_volume_20=l12.get("avg_volume_20"),
            volume_ratio=l12.get("volume_ratio", 1.0),
            is_high_volume=l12.get("is_high_volume", False),
            rejection_count=l12.get("rejection_count", 0),
            is_strong_level=l12.get("is_strong_level", False),
            current_zone=l12.get("current_zone", "NEUTRAL"),
            bars_in_zone=l12.get("bars_in_zone", 0),
            accepted_above=l12.get("accepted_above", False),
            accepted_below=l12.get("accepted_below", False),
            vwap_slope=l12.get("vwap_slope", 0),
            vwap_slope_pct=l12.get("vwap_slope_pct", 0),
            slope_above_bull_threshold=l12.get("slope_above_bull_threshold", False),
            slope_below_bear_threshold=l12.get("slope_below_bear_threshold", False),
            slope_neutral=l12.get("slope_neutral", True),
            crossed_above_vwap=l12.get("crossed_above_vwap", False),
            crossed_below_vwap=l12.get("crossed_below_vwap", False),
            crossed_above_lower_1sd=l12.get("crossed_above_lower_1sd", False),
            crossed_below_upper_1sd=l12.get("crossed_below_upper_1sd", False),
            crossed_above_upper_2sd=l12.get("crossed_above_upper_2sd", False),
            crossed_below_lower_2sd=l12.get("crossed_below_lower_2sd", False),
            distance_to_upper_1sd=l12.get("distance_to_upper_1sd"),
            distance_to_lower_1sd=l12.get("distance_to_lower_1sd"),
            distance_to_upper_2sd=l12.get("distance_to_upper_2sd"),
            distance_to_lower_2sd=l12.get("distance_to_lower_2sd")
        )
    
    def _extract_volume_profile(self, layers: Dict) -> VolumeProfileData:
        """Extract 100% of volume profile data from Layer 13"""
        l13 = layers.get("layer_13", {})
        
        return VolumeProfileData(
            poc_price=l13.get("poc_price"),
            poc_level=l13.get("poc_level"),
            poc_volume=l13.get("poc_volume"),
            poc_volume_pct=l13.get("poc_volume_pct", 0),
            vah_price=l13.get("vah_price"),
            val_price=l13.get("val_price"),
            va_volume_pct=l13.get("va_volume_pct", 0),
            va_width=l13.get("va_width"),
            profile_high=l13.get("profile_high"),
            profile_low=l13.get("profile_low"),
            profile_range=l13.get("profile_range"),
            profile_levels=l13.get("profile_levels", 0),
            total_volume=l13.get("total_volume"),
            price_vs_poc=l13.get("price_vs_poc"),
            price_vs_poc_pct=l13.get("price_vs_poc_pct"),
            price_above_poc=l13.get("price_above_poc"),
            price_below_poc=l13.get("price_below_poc"),
            price_vs_vah=l13.get("price_vs_vah"),
            price_vs_val=l13.get("price_vs_val"),
            in_value_area=l13.get("in_value_area", False),
            above_value_area=l13.get("above_value_area", False),
            below_value_area=l13.get("below_value_area", False),
            position_location=l13.get("position_location", "UNKNOWN"),
            touching_poc=l13.get("touching_poc", False),
            poc_touch_count=l13.get("poc_touch_count", 0),
            poc_is_strong=l13.get("poc_is_strong", False),
            distance_to_poc_pct=l13.get("distance_to_poc_pct", 100),
            touching_vah=l13.get("touching_vah", False),
            touching_val=l13.get("touching_val", False),
            rejection_from_above=l13.get("rejection_from_above", False),
            rejection_from_below=l13.get("rejection_from_below", False),
            poc_rejection_bull=l13.get("poc_rejection_bull", False),
            poc_rejection_bear=l13.get("poc_rejection_bear", False),
            upper_wick_pct=l13.get("upper_wick_pct", 0),
            lower_wick_pct=l13.get("lower_wick_pct", 0),
            bars_at_poc=l13.get("bars_at_poc", 0),
            bars_above_vah=l13.get("bars_above_vah", 0),
            bars_below_val=l13.get("bars_below_val", 0),
            accepted_at_poc=l13.get("accepted_at_poc", False),
            accepted_above_va=l13.get("accepted_above_va", False),
            accepted_below_va=l13.get("accepted_below_va", False),
            buy_pressure_pct=l13.get("buy_pressure_pct", 50),
            strong_buying=l13.get("strong_buying", False),
            strong_selling=l13.get("strong_selling", False),
            current_level_volume=l13.get("current_level_volume"),
            current_level_buy_volume=l13.get("current_level_buy_volume"),
            current_volume=l13.get("current_volume"),
            avg_volume_20=l13.get("avg_volume_20"),
            volume_ratio=l13.get("volume_ratio", 1.0),
            is_high_volume=l13.get("is_high_volume", False),
            volume_spike=l13.get("volume_spike", False),
            crossed_above_vah=l13.get("crossed_above_vah", False),
            crossed_below_vah=l13.get("crossed_below_vah", False),
            crossed_above_val=l13.get("crossed_above_val", False),
            crossed_below_val=l13.get("crossed_below_val", False),
            crossed_above_poc=l13.get("crossed_above_poc", False),
            crossed_below_poc=l13.get("crossed_below_poc", False)
        )
    
    def _extract_iv_analysis(self, layers: Dict) -> IVAnalysisData:
        """Extract 100% of IV analysis data from Layer 14"""
        l14 = layers.get("layer_14", {})
        
        return IVAnalysisData(
            hv_current=l14.get("hv_current"),
            hv_smoothed=l14.get("hv_smoothed"),
            hv_high_52w=l14.get("hv_high_52w"),
            hv_low_52w=l14.get("hv_low_52w"),
            hv_range_52w=l14.get("hv_range_52w"),
            iv_rank=l14.get("iv_rank"),
            iv_rank_valid=l14.get("iv_rank_valid", False),
            iv_percentile=l14.get("iv_percentile"),
            iv_percentile_valid=l14.get("iv_percentile_valid", False),
            iv_above_extreme_high=l14.get("iv_above_extreme_high"),
            iv_above_high=l14.get("iv_above_high"),
            iv_below_low=l14.get("iv_below_low"),
            iv_below_extreme_low=l14.get("iv_below_extreme_low"),
            iv_in_normal_range=l14.get("iv_in_normal_range"),
            iv_state=l14.get("iv_state", "UNKNOWN"),
            em_dte=l14.get("em_dte"),
            em_1sd=l14.get("em_1sd"),
            em_2sd=l14.get("em_2sd"),
            em_1sd_pct=l14.get("em_1sd_pct"),
            em_2sd_pct=l14.get("em_2sd_pct"),
            em_upper_1sd=l14.get("em_upper_1sd"),
            em_lower_1sd=l14.get("em_lower_1sd"),
            em_upper_2sd=l14.get("em_upper_2sd"),
            em_lower_2sd=l14.get("em_lower_2sd"),
            em_7d_1sd_pct=l14.get("em_7d_1sd_pct", 0),
            em_14d_1sd_pct=l14.get("em_14d_1sd_pct", 0),
            em_30d_1sd_pct=l14.get("em_30d_1sd_pct", 0),
            em_45d_1sd_pct=l14.get("em_45d_1sd_pct", 0),
            em_60d_1sd_pct=l14.get("em_60d_1sd_pct", 0),
            hv_rising=l14.get("hv_rising"),
            hv_falling=l14.get("hv_falling"),
            hv_stable=l14.get("hv_stable"),
            hv_5d_change=l14.get("hv_5d_change"),
            hv_10d_change=l14.get("hv_10d_change"),
            hv_vs_avg=l14.get("hv_vs_avg"),
            threshold_extreme_high=l14.get("threshold_extreme_high", 80),
            threshold_high=l14.get("threshold_high", 60),
            threshold_low=l14.get("threshold_low", 40),
            threshold_extreme_low=l14.get("threshold_extreme_low", 20),
            distance_to_extreme_high=l14.get("distance_to_extreme_high"),
            distance_to_high=l14.get("distance_to_high"),
            distance_to_low=l14.get("distance_to_low"),
            distance_to_extreme_low=l14.get("distance_to_extreme_low")
        )
    
    def _extract_gamma_max_pain(self, layers: Dict) -> GammaMaxPainData:
        """Extract 100% of gamma/max pain data from Layer 15"""
        l15 = layers.get("layer_15", {})
        
        return GammaMaxPainData(
            success=l15.get("success", False),
            max_pain=l15.get("max_pain"),
            max_pain_total_loss=l15.get("max_pain_total_loss"),
            max_pain_confidence=l15.get("max_pain_confidence"),
            strike_min=l15.get("strike_min"),
            strike_max=l15.get("strike_max"),
            total_call_oi=l15.get("total_call_oi", 0),
            total_put_oi=l15.get("total_put_oi", 0),
            total_oi=l15.get("total_oi", 0),
            put_call_oi_ratio=l15.get("put_call_oi_ratio"),
            price_vs_max_pain=l15.get("price_vs_max_pain"),
            price_above_max_pain=l15.get("price_above_max_pain"),
            price_below_max_pain=l15.get("price_below_max_pain"),
            distance_to_max_pain=l15.get("distance_to_max_pain"),
            distance_to_max_pain_pct=l15.get("distance_to_max_pain_pct"),
            within_extreme_danger=l15.get("within_extreme_danger"),
            within_danger=l15.get("within_danger"),
            within_caution=l15.get("within_caution"),
            within_safe=l15.get("within_safe"),
            beyond_safe=l15.get("beyond_safe"),
            gex_total=l15.get("gex_total", 0),
            gex_call=l15.get("gex_call", 0),
            gex_put=l15.get("gex_put", 0),
            gex_regime=l15.get("gex_regime", "UNKNOWN"),
            gamma_wall=l15.get("gamma_wall"),
            gex_is_positive=l15.get("gex_is_positive"),
            gex_is_negative=l15.get("gex_is_negative"),
            gex_above_high_threshold=l15.get("gex_above_high_threshold"),
            gex_below_neg_high_threshold=l15.get("gex_below_neg_high_threshold"),
            gex_above_medium_threshold=l15.get("gex_above_medium_threshold"),
            gex_below_neg_medium_threshold=l15.get("gex_below_neg_medium_threshold"),
            expiration=l15.get("expiration"),
            days_to_expiry=l15.get("days_to_expiry"),
            is_expiry_day=l15.get("is_expiry_day"),
            is_expiry_week=l15.get("is_expiry_week"),
            is_monthly=l15.get("is_monthly"),
            pin_probability_pct=l15.get("pin_probability_pct"),
            threshold_extreme_danger_pct=l15.get("threshold_extreme_danger_pct", 0.5),
            threshold_danger_pct=l15.get("threshold_danger_pct", 1.0),
            threshold_caution_pct=l15.get("threshold_caution_pct", 2.0),
            threshold_safe_pct=l15.get("threshold_safe_pct", 5.0),
            threshold_gex_high=l15.get("threshold_gex_high", 1000000),
            threshold_gex_medium=l15.get("threshold_gex_medium", 100000),
            strikes_analyzed=l15.get("strikes_analyzed", 0)
        )
    
    def _extract_put_call_ratio(self, layers: Dict) -> PutCallRatioData:
        """Extract 100% of PCR data from Layer 16"""
        l16 = layers.get("layer_16", {})
        
        return PutCallRatioData(
            success=l16.get("success", False),
            pcr_current=l16.get("pcr_current"),
            pcr_ma_200=l16.get("pcr_ma_200"),
            pcr_stdev=l16.get("pcr_stdev"),
            pcr_upper_band=l16.get("pcr_upper_band"),
            pcr_lower_band=l16.get("pcr_lower_band"),
            pcr_above_upper_band=l16.get("pcr_above_upper_band"),
            pcr_below_lower_band=l16.get("pcr_below_lower_band"),
            pcr_within_bands=l16.get("pcr_within_bands"),
            distance_from_ma=l16.get("distance_from_ma"),
            distance_from_upper=l16.get("distance_from_upper"),
            distance_from_lower=l16.get("distance_from_lower"),
            z_score=l16.get("z_score"),
            above_extreme_fear=l16.get("above_extreme_fear"),
            above_fear=l16.get("above_fear"),
            in_neutral_zone=l16.get("in_neutral_zone"),
            below_greed=l16.get("below_greed"),
            below_extreme_greed=l16.get("below_extreme_greed"),
            sentiment_state=l16.get("sentiment_state", "UNKNOWN"),
            is_extreme_sentiment=l16.get("is_extreme_sentiment"),
            more_puts_than_calls=l16.get("more_puts_than_calls"),
            more_calls_than_puts=l16.get("more_calls_than_puts"),
            balanced=l16.get("balanced"),
            total_call_volume=l16.get("total_call_volume", 0),
            total_put_volume=l16.get("total_put_volume", 0),
            total_volume=l16.get("total_volume", 0),
            call_volume_pct=l16.get("call_volume_pct", 0),
            put_volume_pct=l16.get("put_volume_pct", 0),
            pcr_history_length=l16.get("pcr_history_length", 0),
            has_sufficient_history=l16.get("has_sufficient_history", False),
            threshold_extreme_fear=l16.get("threshold_extreme_fear", 1.3),
            threshold_fear=l16.get("threshold_fear", 1.1),
            threshold_neutral_upper=l16.get("threshold_neutral_upper", 1.0),
            threshold_neutral_lower=l16.get("threshold_neutral_lower", 0.9),
            threshold_greed=l16.get("threshold_greed", 0.8),
            threshold_extreme_greed=l16.get("threshold_extreme_greed", 0.7),
            distance_to_extreme_fear=l16.get("distance_to_extreme_fear"),
            distance_to_fear=l16.get("distance_to_fear"),
            distance_to_greed=l16.get("distance_to_greed"),
            distance_to_extreme_greed=l16.get("distance_to_extreme_greed")
        )
    
    def _extract_greeks(self, layers: Dict) -> GreeksAnalysisData:
        """Extract 100% of Greeks data from Layer 17"""
        l17 = layers.get("layer_17", {})
        
        return GreeksAnalysisData(
            best_strike=l17.get("best_strike"),
            best_strike_type=l17.get("best_strike_type"),
            best_strike_score=l17.get("best_strike_score"),
            best_delta=l17.get("best_delta"),
            best_gamma=l17.get("best_gamma"),
            best_theta=l17.get("best_theta"),
            best_vega=l17.get("best_vega"),
            best_iv=l17.get("best_iv"),
            best_dte=l17.get("best_dte"),
            best_delta_score=l17.get("best_delta_score"),
            best_gamma_score=l17.get("best_gamma_score"),
            best_theta_score=l17.get("best_theta_score"),
            best_vega_iv_score=l17.get("best_vega_iv_score"),
            best_delta_is_atm=l17.get("best_delta_is_atm"),
            best_delta_is_itm=l17.get("best_delta_is_itm"),
            best_delta_is_otm=l17.get("best_delta_is_otm"),
            best_gamma_is_high=l17.get("best_gamma_is_high"),
            best_gamma_is_low=l17.get("best_gamma_is_low"),
            best_theta_is_low=l17.get("best_theta_is_low"),
            best_theta_is_high=l17.get("best_theta_is_high"),
            best_vega_is_high=l17.get("best_vega_is_high"),
            best_vega_is_low=l17.get("best_vega_is_low"),
            delta_min=l17.get("delta_min"),
            delta_max=l17.get("delta_max"),
            delta_avg=l17.get("delta_avg"),
            gamma_peak_strike=l17.get("gamma_peak_strike"),
            theta_avg=l17.get("theta_avg"),
            vega_avg=l17.get("vega_avg"),
            total_strikes_analyzed=l17.get("total_strikes_analyzed", 0),
            strikes_with_high_gamma=l17.get("strikes_with_high_gamma", 0),
            strikes_with_low_theta=l17.get("strikes_with_low_theta", 0),
            strikes_atm=l17.get("strikes_atm", 0),
            iv_rank=l17.get("iv_rank"),
            iv_rank_is_low=l17.get("iv_rank_is_low"),
            dte=l17.get("dte"),
            dte_is_short=l17.get("dte_is_short")
        )
    
    def _calculate_data_quality(self, layers: Dict) -> DataQuality:
        """Calculate data quality metrics"""
        layers_with_data = []
        layers_missing = []
        layer_status = {}
        warnings = []
        
        for i in range(1, 18):
            layer_key = f"layer_{i}"
            layer_data = layers.get(layer_key, {})
            
            has_data = layer_data and any(
                v is not None and v != {} and v != []
                for v in layer_data.values()
            )
            
            layer_status[layer_key] = has_data
            
            if has_data:
                layers_with_data.append(layer_key)
            else:
                layers_missing.append(layer_key)
                warnings.append(f"{layer_key} has no data")
        
        completeness = len(layers_with_data) / 17 * 100
        
        return DataQuality(
            total_layers=17,
            layers_with_data=len(layers_with_data),
            layers_missing=len(layers_missing),
            missing_layer_list=layers_missing,
            data_completeness_pct=round(completeness, 1),
            layer_status=layer_status,
            timestamp=datetime.now().isoformat(),
            warnings=warnings
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_direction(self, current: Optional[float], prev: Optional[float]) -> Optional[str]:
        """Determine direction from current vs previous value"""
        if current is None or prev is None:
            return None
        if current > prev:
            return "rising"
        elif current < prev:
            return "falling"
        return "flat"
    
    def _get_rsi_zone(self, rsi: Optional[float]) -> Optional[str]:
        """Classify RSI into zones"""
        if rsi is None:
            return None
        if rsi >= 70:
            return "overbought"
        elif rsi >= 60:
            return "bullish"
        elif rsi >= 40:
            return "neutral"
        elif rsi >= 30:
            return "bearish"
        return "oversold"
    
    def _get_stoch_zone(self, stoch: Optional[float]) -> Optional[str]:
        """Classify Stochastic into zones"""
        if stoch is None:
            return None
        if stoch >= 80:
            return "overbought"
        elif stoch <= 20:
            return "oversold"
        return "neutral"
    
    def _get_adx_strength(self, adx: Optional[float]) -> Optional[str]:
        """Classify ADX trend strength"""
        if adx is None:
            return None
        if adx >= 50:
            return "very_strong"
        elif adx >= 35:
            return "strong"
        elif adx >= 25:
            return "trending"
        elif adx >= 15:
            return "weak"
        return "no_trend"
    
    def _get_volume_interpretation(self, ratio: Optional[float]) -> Optional[str]:
        """Interpret volume ratio"""
        if ratio is None:
            return None
        if ratio >= 2.0:
            return "very_high"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.0:
            return "normal"
        elif ratio >= 0.5:
            return "low"
        return "very_low"
    
    def _get_slope_trend(self, slope: Optional[float]) -> Optional[str]:
        """Convert slope to trend direction"""
        if slope is None:
            return None
        if slope > 0.01:
            return "rising"
        elif slope < -0.01:
            return "falling"
        return "flat"
    
    def _get_volatility_regime(self, percentile: Optional[float]) -> Optional[str]:
        """Classify volatility regime from percentile"""
        if percentile is None:
            return None
        if percentile >= 80:
            return "extreme"
        elif percentile >= 60:
            return "high"
        elif percentile >= 40:
            return "normal"
        elif percentile >= 20:
            return "low"
        return "very_low"
    
    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================
    
    def to_dict(self, result: MasterAggregatorResult, include_raw: bool = False) -> Dict:
        """Convert result to dictionary"""
        data = {
            "meta": {
                "ticker": result.ticker,
                "mode": result.mode,
                "timeframe": result.timeframe,
                "current_price": result.current_price,
                "timestamp": result.timestamp,
                "version": self.version
            },
            "price_context": asdict(result.price_context),
            "momentum": asdict(result.momentum),
            "volume": asdict(result.volume),
            "divergences": asdict(result.divergences),
            "trend": asdict(result.trend),
            "structure": asdict(result.structure),
            "liquidity": asdict(result.liquidity),
            "volatility": asdict(result.volatility),
            "mtf_confirmation": asdict(result.mtf_confirmation),
            "candle_patterns": asdict(result.candle_patterns),
            "support_resistance": asdict(result.support_resistance),
            "vwap": asdict(result.vwap),
            "volume_profile": asdict(result.volume_profile),
            "iv_analysis": asdict(result.iv_analysis),
            "gamma_max_pain": asdict(result.gamma_max_pain),
            "put_call_ratio": asdict(result.put_call_ratio),
            "greeks": asdict(result.greeks),
            "data_quality": asdict(result.data_quality)
        }
        
        if include_raw:
            data["raw_layers"] = result.raw_layers
        
        return data
    
    def to_json(self, result: MasterAggregatorResult, include_raw: bool = False, indent: int = 2) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(result, include_raw), indent=indent, default=str)
    
    def to_compact_summary(self, result: MasterAggregatorResult) -> Dict:
        """Generate compact summary for quick AI consumption"""
        return {
            "ticker": result.ticker,
            "price": result.current_price,
            "mode": result.mode,
            
            # Quick Bias Indicators
            "supertrend": "BULL" if result.trend.supertrend_bullish else "BEAR" if result.trend.supertrend_bearish else "NONE",
            "rsi": result.momentum.rsi_14,
            "rsi_zone": result.momentum.rsi_zone,
            "adx": result.trend.adx,
            "trend_strength": result.momentum.adx_trend_strength,
            
            # Structure
            "higher_highs": result.structure.consecutive_higher_highs,
            "higher_lows": result.structure.consecutive_higher_lows,
            "lower_highs": result.structure.consecutive_lower_highs,
            "lower_lows": result.structure.consecutive_lower_lows,
            
            # Key Levels
            "vwap": result.vwap.vwap,
            "above_vwap": result.vwap.price_above_vwap,
            "poc": result.volume_profile.poc_price,
            "nearest_support": result.support_resistance.nearest_support,
            "nearest_resistance": result.support_resistance.nearest_resistance,
            
            # Options Context
            "max_pain": result.gamma_max_pain.max_pain,
            "distance_to_max_pain_pct": result.gamma_max_pain.distance_to_max_pain_pct,
            "gex_regime": result.gamma_max_pain.gex_regime,
            "iv_rank": result.iv_analysis.iv_rank,
            "iv_state": result.iv_analysis.iv_state,
            "pcr": result.put_call_ratio.pcr_current,
            "sentiment": result.put_call_ratio.sentiment_state,
            
            # Confirmations
            "mtf_alignment_pct": result.mtf_confirmation.alignment_pct,
            "mtf_dominant": result.mtf_confirmation.mtf_dominant_trend,
            "bullish_patterns": result.candle_patterns.total_bullish_patterns,
            "bearish_patterns": result.candle_patterns.total_bearish_patterns,
            
            # Volume
            "volume_ratio": result.volume.volume_ratio,
            "volume_pressure": result.volume.volume_pressure,
            
            # Data Quality
            "data_completeness_pct": result.data_quality.data_completeness_pct,
            "layers_missing": result.data_quality.missing_layer_list
        }


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Alias for existing code
Layer18BrainV3 = Layer18MasterAggregator
Layer18DataAggregator = Layer18MasterAggregator


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    aggregator = Layer18MasterAggregator()
    
    # Mock layer results (in production, these come from layers 1-17)
    layer_results = {
        "layer_1": {"rsi_14": 55.5, "macd_line": 0.5, "macd_histogram": 0.1},
        "layer_5": {"supertrend_bullish": True, "adx": 28.5},
        # ... other layers
    }
    
    result = aggregator.analyze(
        ticker="AAPL",
        layer_results=layer_results,
        current_price=150.00,
        mode=TradeMode.SCALP
    )
    
    # Output as JSON for AI
    print(aggregator.to_json(result))
    
    # Or compact summary
    print(aggregator.to_compact_summary(result))
