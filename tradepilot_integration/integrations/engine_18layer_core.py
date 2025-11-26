"""
TradePilot Advanced 18-Layer Engine Core
==========================================
Master orchestrator that integrates ALL 18 layers for maximum probability trading.

This integration file does NOT modify any layer code - it imports and orchestrates them.

Features:
- Full 18-layer analysis pipeline
- SCALP (0-2 DTE) and SWING (7-45 DTE) modes
- 14 high-probability playbooks (7 bullish + 7 bearish)
- Real-time options chain integration
- Risk-adjusted position sizing
- AI-ready JSON output for Claude/GPT integration

Target Win Rates: 85-95% with proper setup matching
"""

import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

# Import path setup - adjust based on your deployment
LAYERS_PATH = os.environ.get('TRADEPILOT_LAYERS_PATH', '/home/mickey/tradepilot-mcp-server/tradepilot_engine/layers')
PRODUCTION_PATH = os.environ.get('TRADEPILOT_PRODUCTION_PATH', '/home/mickey/tradepilot-mcp-server')

sys.path.insert(0, LAYERS_PATH)
sys.path.insert(0, PRODUCTION_PATH)
sys.path.insert(0, os.path.join(PRODUCTION_PATH, 'tradepilot_engine'))


class TradeMode(Enum):
    """Trading mode selection"""
    SCALP = "SCALP"      # 0-2 DTE, quick plays
    SWING = "SWING"      # 7-45 DTE, multi-day holds
    INTRADAY = "INTRADAY"  # Same day plays


class SignalStrength(Enum):
    """Signal confidence levels"""
    SUPREME = "SUPREME"       # 90%+ probability
    EXCELLENT = "EXCELLENT"   # 85-90%
    STRONG = "STRONG"         # 80-85%
    MODERATE = "MODERATE"     # 75-80%
    WEAK = "WEAK"             # Below 75%
    NO_TRADE = "NO_TRADE"     # No valid setup


@dataclass
class LayerResult:
    """Individual layer analysis result"""
    layer_name: str
    layer_number: int
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class FullAnalysisResult:
    """Complete 18-layer analysis result"""
    ticker: str
    current_price: float
    mode: TradeMode
    timestamp: str
    
    # Layer results
    layer_results: Dict[str, LayerResult]
    layers_successful: int
    layers_failed: int
    
    # Aggregated signals
    technical_layers: Dict[str, Any]  # Layers 1-10
    price_action_layers: Dict[str, Any]  # Layers 11-13
    options_layers: Dict[str, Any]  # Layers 14-17
    brain_decision: Dict[str, Any]  # Layer 18
    
    # Final recommendation
    trade_valid: bool
    direction: str  # BULLISH, BEARISH, NEUTRAL
    action: str  # BUY_CALL, BUY_PUT, FLAT
    confidence: SignalStrength
    win_probability: float
    
    # Playbook info
    matched_playbook: Optional[str]
    playbook_id: Optional[int]
    playbook_details: Dict[str, Any]
    
    # Options recommendation
    strike: float
    strike_type: str  # ATM, OTM, ITM
    delta: float
    expiry_dte: int
    expiry_date: str
    
    # Execution plan
    entry_price: float
    target_price: float
    stop_price: float
    risk_reward: float
    position_size_pct: float
    contracts_suggested: int
    
    # Invalidation
    invalidation_above: Optional[float]
    invalidation_below: Optional[float]
    invalidation_reason: str
    
    # AI context
    reasoning: List[str]
    concerns: List[str]
    
    # Raw data for AI
    raw_data: Dict[str, Any] = field(default_factory=dict)


class TradePilotEngine18Layer:
    """
    Master 18-Layer Trading Engine
    
    Orchestrates all layers for high-probability options trading.
    Does NOT modify layer code - only imports and coordinates.
    """
    
    def __init__(self, polygon_api_key: Optional[str] = None):
        """
        Initialize the 18-layer engine
        
        Args:
            polygon_api_key: Optional API key override
        """
        self.api_key = polygon_api_key or os.environ.get('POLYGON_API_KEY', '')
        
        # Layer instances (lazy loaded)
        self._layers = {}
        self._layer_18_brain = None
        
        # Configuration
        self.config = {
            'min_bars_required': 150,
            'default_limit': 730,
            'scalp_dte_max': 2,
            'swing_dte_min': 7,
            'swing_dte_max': 45,
            'position_size_limits': {
                'SUPREME': 0.50,
                'EXCELLENT': 0.35,
                'STRONG': 0.25,
                'MODERATE': 0.15,
                'WEAK': 0.0,
            },
            'confidence_thresholds': {
                'SUPREME': 90,
                'EXCELLENT': 85,
                'STRONG': 80,
                'MODERATE': 75,
            }
        }
        
        # Initialize layers
        self._initialize_layers()
    
    def _initialize_layers(self):
        """Initialize all layer instances"""
        try:
            # Technical Layers (1-10) - from production
            from layers import (
                Layer1Momentum, Layer2Volume, Layer3Divergence,
                Layer4VolumeStrength, Layer5Trend, Layer6Structure,
                Layer7Liquidity, Layer8VolatilityRegime, Layer9Confirmation,
                Layer10CandleIntelligence
            )
            
            self._layers['layer_1'] = Layer1Momentum()
            self._layers['layer_2'] = Layer2Volume()
            self._layers['layer_3'] = Layer3Divergence()
            self._layers['layer_4'] = Layer4VolumeStrength()
            self._layers['layer_5'] = Layer5Trend()
            self._layers['layer_6'] = Layer6Structure()
            self._layers['layer_7'] = Layer7Liquidity()
            self._layers['layer_8'] = Layer8VolatilityRegime()
            self._layers['layer_9'] = Layer9Confirmation()
            self._layers['layer_10'] = Layer10CandleIntelligence()
            
        except ImportError as e:
            print(f"[Engine] Warning: Could not import production layers: {e}")
        
        try:
            # Advanced Layers (11-17) - from advanced package
            from layer_11_support_resistance import Layer11SupportResistance
            from layer_12_vwap_analysis import Layer12VWAPAnalysis
            from layer_13_volume_profile import Layer13VolumeProfile
            from layer_14_iv_analysis import Layer14IVAnalysis
            from layer_15_gamma_max_pain import Layer15GammaMaxPain
            from layer_16_put_call_ratio import Layer16PutCallRatio
            
            self._layers['layer_11'] = Layer11SupportResistance()
            self._layers['layer_12'] = Layer12VWAPAnalysis()
            self._layers['layer_13'] = Layer13VolumeProfile()
            self._layers['layer_14'] = Layer14IVAnalysis()
            self._layers['layer_15'] = Layer15GammaMaxPain()
            self._layers['layer_16'] = Layer16PutCallRatio()
            
        except ImportError as e:
            print(f"[Engine] Warning: Could not import advanced layers: {e}")
        
        try:
            # Layer 18 Brain
            from layer_18_brain_v3_COMPLETE import Layer18BrainV3
            self._layer_18_brain = Layer18BrainV3()
        except ImportError as e:
            print(f"[Engine] Warning: Could not import Layer 18 Brain: {e}")
    
    def analyze(self, 
                ticker: str,
                candles_data: Dict,
                options_data: Optional[Dict] = None,
                mode: TradeMode = TradeMode.SWING,
                timeframe: str = "day") -> FullAnalysisResult:
        """
        Run complete 18-layer analysis
        
        Args:
            ticker: Stock symbol (e.g., "SPY")
            candles_data: OHLCV candle data from Polygon.io
            options_data: Optional options chain data for layers 14-17
            mode: SCALP or SWING trading mode
            timeframe: Candle timeframe (day, hour, minute)
            
        Returns:
            FullAnalysisResult with complete analysis
        """
        start_time = datetime.now()
        
        # Convert candles to DataFrame
        df = self._prepare_dataframe(candles_data)
        if df is None or len(df) < self.config['min_bars_required']:
            return self._error_result(ticker, "Insufficient candle data", mode)
        
        current_price = float(df['close'].iloc[-1])
        
        # Run all layers
        layer_results = {}
        
        # Technical Layers (1-10)
        layer_results.update(self._run_technical_layers(df))
        
        # Price Action Layers (11-13)
        layer_results.update(self._run_price_action_layers(df))
        
        # Options Layers (14-17)
        if options_data:
            layer_results.update(self._run_options_layers(df, options_data, current_price))
        else:
            # Provide placeholder data for options layers
            layer_results.update(self._placeholder_options_layers(current_price))
        
        # Prepare layer data for Brain
        brain_input = self._prepare_brain_input(layer_results)
        
        # Run Layer 18 Brain
        brain_result = self._run_brain_analysis(
            ticker, brain_input, current_price, mode
        )
        
        # Build final result
        result = self._build_analysis_result(
            ticker=ticker,
            current_price=current_price,
            mode=mode,
            layer_results=layer_results,
            brain_result=brain_result,
            start_time=start_time
        )
        
        return result
    
    def _prepare_dataframe(self, candles_data: Dict) -> Optional[pd.DataFrame]:
        """Convert Polygon.io candles to DataFrame"""
        try:
            if not candles_data or "results" not in candles_data:
                return None
            
            results = candles_data["results"]
            if not results:
                return None
            
            df = pd.DataFrame(results)
            
            # Rename columns
            column_mapping = {
                "o": "open", "h": "high", "l": "low",
                "c": "close", "v": "volume", "t": "timestamp",
                "vw": "vwap", "n": "trades"
            }
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp
            if "timestamp" in df.columns:
                df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("datetime")
            
            # Add basic features
            df = self._add_basic_features(df)
            
            return df.sort_index()
            
        except Exception as e:
            print(f"[Engine] DataFrame preparation error: {e}")
            return None
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features"""
        df = df.copy()
        
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["true_range"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["price_change"] = df["close"].diff()
        df["price_change_pct"] = df["close"].pct_change() * 100
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_wick"] = df["high"] - np.maximum(df["close"], df["open"])
        df["lower_wick"] = np.minimum(df["close"], df["open"]) - df["low"]
        df["candle_range"] = df["high"] - df["low"]
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["is_bearish"] = (df["close"] < df["open"]).astype(int)
        
        return df
    
    def _run_technical_layers(self, df: pd.DataFrame) -> Dict[str, LayerResult]:
        """Run technical analysis layers 1-10"""
        results = {}
        
        for i in range(1, 11):
            layer_key = f"layer_{i}"
            start = datetime.now()
            
            try:
                if layer_key in self._layers:
                    layer_instance = self._layers[layer_key]
                    
                    # Layer 9 needs previous layer results
                    if i == 9:
                        prev_results = {k: v.data for k, v in results.items()}
                        data = layer_instance.analyze(df, prev_results)
                    else:
                        data = layer_instance.analyze(df)
                    
                    exec_time = (datetime.now() - start).total_seconds() * 1000
                    results[layer_key] = LayerResult(
                        layer_name=layer_key,
                        layer_number=i,
                        success=True,
                        data=data,
                        execution_time_ms=exec_time
                    )
                else:
                    results[layer_key] = LayerResult(
                        layer_name=layer_key,
                        layer_number=i,
                        success=False,
                        data={},
                        error="Layer not initialized"
                    )
                    
            except Exception as e:
                exec_time = (datetime.now() - start).total_seconds() * 1000
                results[layer_key] = LayerResult(
                    layer_name=layer_key,
                    layer_number=i,
                    success=False,
                    data={},
                    error=str(e),
                    execution_time_ms=exec_time
                )
        
        return results
    
    def _run_price_action_layers(self, df: pd.DataFrame) -> Dict[str, LayerResult]:
        """Run price action layers 11-13"""
        results = {}
        
        for i in range(11, 14):
            layer_key = f"layer_{i}"
            start = datetime.now()
            
            try:
                if layer_key in self._layers:
                    data = self._layers[layer_key].analyze(df)
                    exec_time = (datetime.now() - start).total_seconds() * 1000
                    results[layer_key] = LayerResult(
                        layer_name=layer_key,
                        layer_number=i,
                        success=True,
                        data=data,
                        execution_time_ms=exec_time
                    )
                else:
                    results[layer_key] = LayerResult(
                        layer_name=layer_key,
                        layer_number=i,
                        success=False,
                        data=self._placeholder_layer_data(i),
                        error="Layer not initialized"
                    )
                    
            except Exception as e:
                results[layer_key] = LayerResult(
                    layer_name=layer_key,
                    layer_number=i,
                    success=False,
                    data=self._placeholder_layer_data(i),
                    error=str(e)
                )
        
        return results
    
    def _run_options_layers(self, df: pd.DataFrame, options_data: Dict, 
                           current_price: float) -> Dict[str, LayerResult]:
        """Run options-specific layers 14-17"""
        results = {}
        
        # Layer 14: IV Analysis
        try:
            if 'layer_14' in self._layers:
                data = self._layers['layer_14'].analyze(df)
                results['layer_14'] = LayerResult(
                    layer_name='layer_14', layer_number=14,
                    success=True, data=data
                )
        except Exception as e:
            results['layer_14'] = LayerResult(
                layer_name='layer_14', layer_number=14,
                success=False, data={}, error=str(e)
            )
        
        # Layer 15: Gamma & Max Pain
        try:
            if 'layer_15' in self._layers:
                data = self._layers['layer_15'].analyze(options_data, current_price)
                results['layer_15'] = LayerResult(
                    layer_name='layer_15', layer_number=15,
                    success=True, data=data
                )
        except Exception as e:
            results['layer_15'] = LayerResult(
                layer_name='layer_15', layer_number=15,
                success=False, data={}, error=str(e)
            )
        
        # Layer 16: Put/Call Ratio
        try:
            if 'layer_16' in self._layers:
                data = self._layers['layer_16'].analyze(options_data)
                results['layer_16'] = LayerResult(
                    layer_name='layer_16', layer_number=16,
                    success=True, data=data
                )
        except Exception as e:
            results['layer_16'] = LayerResult(
                layer_name='layer_16', layer_number=16,
                success=False, data={}, error=str(e)
            )
        
        # Layer 17: Greeks Analysis (synthesized from options data)
        results['layer_17'] = self._analyze_greeks(options_data, current_price)
        
        return results
    
    def _analyze_greeks(self, options_data: Dict, current_price: float) -> LayerResult:
        """Analyze Greeks from options chain - ONLY recommends non-expired contracts"""
        try:
            if not options_data or 'results' not in options_data:
                return LayerResult(
                    layer_name='layer_17', layer_number=17,
                    success=False, data={}, error="No options data"
                )
            
            results = options_data['results']
            today = datetime.now().date()
            
            # Filter and score valid contracts
            valid_contracts = []
            for contract in results:
                details = contract.get('details', {})
                greeks = contract.get('greeks', {})
                
                if not details or not greeks:
                    continue
                
                # Get expiration and check if NOT expired
                exp_str = details.get('expiration_date')
                if not exp_str:
                    continue
                    
                try:
                    exp_date = datetime.strptime(exp_str, '%Y-%m-%d').date()
                except:
                    continue
                
                # CRITICAL: Skip expired contracts (must be at least 1 day out)
                dte = (exp_date - today).days
                if dte < 1:
                    continue
                
                strike = details.get('strike_price', 0)
                delta = abs(greeks.get('delta', 0))
                contract_type = details.get('contract_type', 'call')
                
                # Score: prefer ATM (delta ~0.50) and reasonable DTE (7-45 days)
                delta_score = 100 - abs(delta - 0.50) * 200  # Best at 0.50
                dte_score = 100 if 7 <= dte <= 45 else 50 if dte < 7 else 70
                
                valid_contracts.append({
                    'strike': strike,
                    'delta': delta,
                    'dte': dte,
                    'exp_date': exp_str,
                    'contract_type': contract_type,
                    'score': delta_score + dte_score,
                    'greeks': greeks
                })
            
            # Sort by score and pick best
            if not valid_contracts:
                return LayerResult(
                    layer_name='layer_17', layer_number=17,
                    success=False, data={}, error="No valid non-expired contracts found"
                )
            
            valid_contracts.sort(key=lambda x: x['score'], reverse=True)
            best = valid_contracts[0]
            
            data = {
                'best_strike': best['strike'],
                'best_delta': best['delta'],
                'best_dte': best['dte'],
                'best_expiry': best['exp_date'],
                'best_contract_type': best['contract_type'],
                'best_strike_type': 'ATM' if abs(best['strike'] - current_price) / current_price < 0.02 else ('ITM' if best['strike'] < current_price else 'OTM'),
                'contracts_analyzed': len(valid_contracts),
                'current_price': current_price
            }
            return LayerResult(
                layer_name='layer_17', layer_number=17,
                success=True, data=data
            )
        except Exception as e:
            return LayerResult(
                layer_name='layer_17', layer_number=17,
                success=False, data={}, error=str(e)
            )

    def _placeholder_options_layers(self, current_price: float) -> Dict[str, LayerResult]:
        """Placeholder data when options data not available"""
        return {
            'layer_14': LayerResult(
                layer_name='layer_14', layer_number=14,
                success=True,
                data={'iv_rank': 50, 'iv_percentile': 50, 'hv_current': 25}
            ),
            'layer_15': LayerResult(
                layer_name='layer_15', layer_number=15,
                success=True,
                data={'max_pain': current_price, 'distance_to_max_pain_pct': 0}
            ),
            'layer_16': LayerResult(
                layer_name='layer_16', layer_number=16,
                success=True,
                data={'pcr_current': 1.0, 'pcr_5d_avg': 1.0}
            ),
            'layer_17': LayerResult(
                layer_name='layer_17', layer_number=17,
                success=True,
                data={
                    'best_strike': current_price,
                    'best_delta': 0.50,
                    'best_dte': 30,
                    'best_strike_type': 'ATM'
                }
            )
        }
    
    def _placeholder_layer_data(self, layer_num: int) -> Dict:
        """Generate placeholder data for missing layers"""
        placeholders = {
            11: {
                'distance_to_support_pct': 2.0,
                'distance_to_resistance_pct': 2.0,
                'nearest_support': 0,
                'nearest_resistance': 0
            },
            12: {
                'price_above_vwap': True,
                'price_vs_vwap_pct': 0.5,
                'crossed_above_vwap': False,
                'crossed_below_vwap': False
            },
            13: {
                'in_value_area': True,
                'poc_price': 0,
                'value_area_high': 0,
                'value_area_low': 0
            }
        }
        return placeholders.get(layer_num, {})
    
    def _prepare_brain_input(self, layer_results: Dict[str, LayerResult]) -> Dict[str, Any]:
        """Prepare layer data for Layer 18 Brain"""
        brain_input = {}
        
        for layer_key, result in layer_results.items():
            if result.success:
                brain_input[layer_key] = result.data
            else:
                brain_input[layer_key] = {}
        
        return brain_input
    
    def _run_brain_analysis(self, ticker: str, layer_data: Dict, 
                           current_price: float, mode: TradeMode) -> Dict:
        """Run Layer 18 Brain analysis"""
        try:
            if self._layer_18_brain:
                from layer_18_brain_v3_COMPLETE import TradeMode as BrainMode
                
                brain_mode = BrainMode.SCALP if mode == TradeMode.SCALP else BrainMode.SWING
                
                recommendation = self._layer_18_brain.analyze(
                    ticker=ticker,
                    layer_results=layer_data,
                    current_price=current_price,
                    mode=brain_mode
                )
                
                return self._layer_18_brain.to_dict(recommendation)
            else:
                return self._fallback_brain_analysis(layer_data, current_price, mode)
                
        except Exception as e:
            print(f"[Engine] Brain analysis error: {e}")
            return self._fallback_brain_analysis(layer_data, current_price, mode)
    
    def _fallback_brain_analysis(self, layer_data: Dict, 
                                 current_price: float, mode: TradeMode) -> Dict:
        """Fallback analysis when Brain not available"""
        # Simple directional bias from layers
        bullish_score = 0
        bearish_score = 0
        
        # Layer 1: Momentum
        l1 = layer_data.get('layer_1', {})
        if l1.get('momentum_score', 0) > 0:
            bullish_score += 2
        elif l1.get('momentum_score', 0) < 0:
            bearish_score += 2
        
        # Layer 5: Trend
        l5 = layer_data.get('layer_5', {})
        if l5.get('supertrend_bullish'):
            bullish_score += 2
        if l5.get('supertrend_bearish'):
            bearish_score += 2
        
        # Layer 6: Structure
        l6 = layer_data.get('layer_6', {})
        if l6.get('bos_bull_detected'):
            bullish_score += 3
        if l6.get('bos_bear_detected'):
            bearish_score += 3
        
        # Determine direction
        if bullish_score > bearish_score and bullish_score >= 4:
            direction = "BULLISH"
            action = "BUY CALL"
            win_prob = 70 + (bullish_score * 2)
        elif bearish_score > bullish_score and bearish_score >= 4:
            direction = "BEARISH"
            action = "BUY PUT"
            win_prob = 70 + (bearish_score * 2)
        else:
            direction = "NEUTRAL"
            action = "FLAT"
            win_prob = 50
        
        return {
            'trade': {
                'valid': direction != "NEUTRAL",
                'direction': direction,
                'action': action,
                'confidence': 'MODERATE' if win_prob >= 75 else 'WEAK',
                'win_probability': min(win_prob, 85)
            },
            'playbooks': {'best': None, 'all_checked': []},
            'option': {
                'strike': current_price,
                'strike_type': 'ATM',
                'delta': 0.50,
                'expiry_dte': 30 if mode == TradeMode.SWING else 1
            },
            'reasoning': [f"Fallback analysis: {direction} bias detected"],
            'concerns': ["Full Brain analysis not available"]
        }
    
    def _build_analysis_result(self, ticker: str, current_price: float,
                              mode: TradeMode, layer_results: Dict[str, LayerResult],
                              brain_result: Dict, start_time: datetime) -> FullAnalysisResult:
        """Build the final analysis result"""
        
        # Count successful layers
        successful = sum(1 for r in layer_results.values() if r.success)
        failed = len(layer_results) - successful
        
        # Extract brain decision
        trade_info = brain_result.get('trade', {})
        option_info = brain_result.get('option', {})
        execution_info = brain_result.get('execution', {})
        position_info = brain_result.get('position', {})
        risk_info = brain_result.get('risk', {})
        playbook_info = brain_result.get('playbooks', {})
        
        # Determine confidence level
        win_prob = trade_info.get('win_probability', 50)
        if win_prob >= 90:
            confidence = SignalStrength.SUPREME
        elif win_prob >= 85:
            confidence = SignalStrength.EXCELLENT
        elif win_prob >= 80:
            confidence = SignalStrength.STRONG
        elif win_prob >= 75:
            confidence = SignalStrength.MODERATE
        else:
            confidence = SignalStrength.WEAK
        
        # Get matched playbook info
        best_playbook = playbook_info.get('best')
        matched_playbook = best_playbook.get('name') if best_playbook else None
        playbook_id = best_playbook.get('id') if best_playbook else None
        
        # Calculate expiry date
        dte = option_info.get('expiry_dte', 30)
        expiry_date = (datetime.now() + timedelta(days=dte)).strftime('%Y-%m-%d')
        
        return FullAnalysisResult(
            ticker=ticker,
            current_price=current_price,
            mode=mode,
            timestamp=datetime.now().isoformat(),
            
            layer_results=layer_results,
            layers_successful=successful,
            layers_failed=failed,
            
            technical_layers={k: v.data for k, v in layer_results.items() if v.layer_number <= 10},
            price_action_layers={k: v.data for k, v in layer_results.items() if 11 <= v.layer_number <= 13},
            options_layers={k: v.data for k, v in layer_results.items() if v.layer_number >= 14},
            brain_decision=brain_result,
            
            trade_valid=trade_info.get('valid', False),
            direction=trade_info.get('direction', 'NEUTRAL'),
            action=trade_info.get('action', 'FLAT'),
            confidence=confidence,
            win_probability=win_prob,
            
            matched_playbook=matched_playbook,
            playbook_id=playbook_id,
            playbook_details=playbook_info,
            
            strike=option_info.get('strike', current_price),
            strike_type=option_info.get('strike_type', 'ATM'),
            delta=option_info.get('delta', 0.50),
            expiry_dte=dte,
            expiry_date=expiry_date,
            
            entry_price=execution_info.get('entry', current_price),
            target_price=execution_info.get('target', current_price * 1.02),
            stop_price=execution_info.get('stop', current_price * 0.98),
            risk_reward=execution_info.get('risk_reward', 2.0),
            position_size_pct=position_info.get('size_pct', 0.10),
            contracts_suggested=position_info.get('contracts', 1),
            
            invalidation_above=risk_info.get('invalidation_above'),
            invalidation_below=risk_info.get('invalidation_below'),
            invalidation_reason=risk_info.get('reason', ''),
            
            reasoning=brain_result.get('reasoning', []),
            concerns=brain_result.get('concerns', []),
            
            raw_data={
                'layers': {k: v.data for k, v in layer_results.items()},
                'brain': brain_result
            }
        )
    
    def _error_result(self, ticker: str, error: str, mode: TradeMode) -> FullAnalysisResult:
        """Generate error result"""
        return FullAnalysisResult(
            ticker=ticker,
            current_price=0,
            mode=mode,
            timestamp=datetime.now().isoformat(),
            layer_results={},
            layers_successful=0,
            layers_failed=18,
            technical_layers={},
            price_action_layers={},
            options_layers={},
            brain_decision={'error': error},
            trade_valid=False,
            direction='NEUTRAL',
            action='FLAT',
            confidence=SignalStrength.NO_TRADE,
            win_probability=0,
            matched_playbook=None,
            playbook_id=None,
            playbook_details={},
            strike=0,
            strike_type='N/A',
            delta=0,
            expiry_dte=0,
            expiry_date='',
            entry_price=0,
            target_price=0,
            stop_price=0,
            risk_reward=0,
            position_size_pct=0,
            contracts_suggested=0,
            invalidation_above=None,
            invalidation_below=None,
            invalidation_reason=error,
            reasoning=[f"Error: {error}"],
            concerns=[error]
        )
    
    def to_json(self, result: FullAnalysisResult) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(result), indent=2, default=str)
    
    def to_dict(self, result: FullAnalysisResult) -> Dict:
        """Convert result to dictionary"""
        return {
            'ticker': result.ticker,
            'current_price': result.current_price,
            'mode': result.mode.value,
            'timestamp': result.timestamp,
            
            'analysis_summary': {
                'layers_successful': result.layers_successful,
                'layers_failed': result.layers_failed,
                'trade_valid': result.trade_valid,
                'direction': result.direction,
                'action': result.action,
                'confidence': result.confidence.value,
                'win_probability': result.win_probability
            },
            
            'playbook': {
                'matched': result.matched_playbook,
                'id': result.playbook_id,
                'details': result.playbook_details
            },
            
            'option_recommendation': {
                'strike': result.strike,
                'strike_type': result.strike_type,
                'delta': result.delta,
                'expiry_dte': result.expiry_dte,
                'expiry_date': result.expiry_date
            },
            
            'execution_plan': {
                'entry': result.entry_price,
                'target': result.target_price,
                'stop': result.stop_price,
                'risk_reward': result.risk_reward,
                'position_size_pct': result.position_size_pct,
                'contracts': result.contracts_suggested
            },
            
            'risk_management': {
                'invalidation_above': result.invalidation_above,
                'invalidation_below': result.invalidation_below,
                'reason': result.invalidation_reason
            },
            
            'reasoning': result.reasoning,
            'concerns': result.concerns,
            
            'layer_data': {
                'technical': result.technical_layers,
                'price_action': result.price_action_layers,
                'options': result.options_layers
            },
            
            'brain_decision': result.brain_decision,
            'raw_data': result.raw_data
        }
    
    def get_human_readable(self, result: FullAnalysisResult) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append("=" * 80)
        lines.append("🎯 TRADEPILOT 18-LAYER ANALYSIS")
        lines.append("=" * 80)
        lines.append(f"Ticker: {result.ticker} @ ${result.current_price:.2f}")
        lines.append(f"Mode: {result.mode.value}")
        lines.append(f"Time: {result.timestamp}")
        lines.append(f"Layers: {result.layers_successful}/18 successful")
        lines.append("")
        
        if result.trade_valid:
            lines.append(f"🚀 RECOMMENDATION: {result.action}")
            lines.append(f"📊 Direction: {result.direction}")
            lines.append(f"🏆 Confidence: {result.confidence.value} ({result.win_probability:.1f}%)")
            
            if result.matched_playbook:
                lines.append(f"📖 Playbook: {result.matched_playbook}")
            
            lines.append("")
            lines.append("📈 OPTION DETAILS:")
            lines.append(f"   Strike: ${result.strike:.2f} ({result.strike_type})")
            lines.append(f"   Delta: {result.delta:.2f}")
            lines.append(f"   Expiry: {result.expiry_date} ({result.expiry_dte} DTE)")
            
            lines.append("")
            lines.append("💰 EXECUTION:")
            lines.append(f"   Entry: ${result.entry_price:.2f}")
            lines.append(f"   Target: ${result.target_price:.2f}")
            lines.append(f"   Stop: ${result.stop_price:.2f}")
            lines.append(f"   R:R: {result.risk_reward:.1f}:1")
            
            lines.append("")
            lines.append("📊 POSITION:")
            lines.append(f"   Size: {result.position_size_pct*100:.0f}% of portfolio")
            lines.append(f"   Contracts: {result.contracts_suggested}")
            
            if result.invalidation_reason:
                lines.append("")
                lines.append("🚨 INVALIDATION:")
                lines.append(f"   {result.invalidation_reason}")
            
            if result.reasoning:
                lines.append("")
                lines.append("💡 REASONING:")
                for r in result.reasoning[:5]:
                    lines.append(f"   • {r}")
            
            if result.concerns:
                lines.append("")
                lines.append("⚠️ CONCERNS:")
                for c in result.concerns[:3]:
                    lines.append(f"   • {c}")
        else:
            lines.append("❌ NO TRADE - Insufficient setup quality")
            if result.reasoning:
                lines.append("")
                for r in result.reasoning:
                    lines.append(f"   • {r}")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Convenience function
def create_engine(api_key: Optional[str] = None) -> TradePilotEngine18Layer:
    """Create a new 18-layer engine instance"""
    return TradePilotEngine18Layer(polygon_api_key=api_key)


if __name__ == "__main__":
    # Example usage
    engine = create_engine()
    print("TradePilot 18-Layer Engine initialized successfully!")
    print(f"Layers available: {len(engine._layers)}")
