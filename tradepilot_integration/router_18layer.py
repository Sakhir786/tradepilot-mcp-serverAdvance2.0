"""
TradePilot Advanced 18-Layer API Router
========================================
FastAPI endpoints for the complete 18-layer trading engine.

Endpoints:
- /engine18/analyze - Full 18-layer analysis
- /engine18/quick - Quick signal check
- /engine18/scan - Scan multiple tickers
- /engine18/compare - Compare setups
- /engine18/health - System health check

Author: TradePilot Integration
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List
from datetime import datetime
import json
import asyncio
from enum import Enum

# Import engine (adjust path as needed)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine_18layer_core import (
    TradePilotEngine18Layer, 
    TradeMode, 
    SignalStrength,
    FullAnalysisResult
)


class TradeModeParam(str, Enum):
    """API parameter for trade mode"""
    scalp = "scalp"
    swing = "swing"
    intraday = "intraday"
    leaps = "leaps"


class OutputFormat(str, Enum):
    """Output format options"""
    json = "json"
    human = "human"
    ai = "ai"



def convert_numpy_types(obj):
    """Convert numpy types to Python native types"""
    import numpy as np
    import math
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


class NumpyJSONResponse(JSONResponse):
    """Custom JSON response that handles numpy types"""
    def render(self, content) -> bytes:
        return json.dumps(
            convert_numpy_types(content),
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


# Create router
router = APIRouter(prefix="/engine18", tags=["TradePilot 18-Layer Engine"])

# Global engine instance (lazy initialization)

_engine: Optional[TradePilotEngine18Layer] = None


def get_engine() -> TradePilotEngine18Layer:
    """Get or create engine instance"""
    global _engine
    if _engine is None:
        _engine = TradePilotEngine18Layer()
    return _engine


# Import data client based on DATA_SOURCE config
try:
    from config import DATA_SOURCE
except ImportError:
    DATA_SOURCE = "polygon"

if DATA_SOURCE == "ibkr":
    try:
        from ibkr_client import (
            get_candles_for_mode,
            get_candles,
            get_option_chain_snapshot,
            get_full_option_chain_snapshot,
            get_ticker_details,
            get_market_context,
        )
        print("[Router] Data source: IBKR")
    except ImportError as e:
        raise ImportError(f"IBKR client not available (DATA_SOURCE=ibkr): {e}")
else:
    try:
        from polygon_client import (
            get_candles_for_mode,
            get_candles,
            get_option_chain_snapshot,
            get_full_option_chain_snapshot,
            get_ticker_details,
            get_market_context,
        )
        print("[Router] Data source: Polygon")
    except ImportError:
        # Fallback stubs
        def get_candles(symbol, tf="day", limit=730):
            return {"error": "Polygon client not available"}

        def get_candles_for_mode(symbol, mode="swing"):
            return {"error": "Polygon client not available"}

        def get_option_chain_snapshot(symbol, cursor=None, limit=50):
            return {"error": "Polygon client not available"}

        def get_full_option_chain_snapshot(symbol, limit=100):
            return {"error": "Polygon client not available"}

        def get_ticker_details(symbol):
            return {"error": "Polygon client not available"}

        def get_market_context(mode="swing"):
            return {"market_bias": "neutral", "warnings": ["Polygon client not available"]}


@router.get("/analyze")
async def full_analysis(
    symbol: str = Query(..., description="Stock symbol (e.g., SPY, AAPL)"),
    mode: TradeModeParam = Query(TradeModeParam.swing, description="Trading mode"),
    tf: str = Query("day", description="Timeframe (day, hour, minute)"),
    limit: int = Query(730, description="Number of candles to fetch"),
    include_options: bool = Query(True, description="Include options chain analysis"),
    output: OutputFormat = Query(OutputFormat.json, description="Output format")
):
    """
    🎯 Full 18-Layer Analysis
    
    Runs complete analysis through all 18 layers including:
    - Technical indicators (Layers 1-10)
    - Price action analysis (Layers 11-13)
    - Options analysis (Layers 14-17)
    - Master Brain decision (Layer 18)
    
    Returns comprehensive trade recommendation with:
    - Option strike/expiry recommendations
    - Entry/Target/Stop levels
    - Risk management parameters
    """
    try:
        engine = get_engine()
        symbol = symbol.upper()

        # Convert mode - now includes LEAPS
        trade_mode = {
            TradeModeParam.scalp: TradeMode.SCALP,
            TradeModeParam.swing: TradeMode.SWING,
            TradeModeParam.intraday: TradeMode.INTRADAY,
            TradeModeParam.leaps: TradeMode.LEAPS
        }.get(mode, TradeMode.SWING)

        # Use mode-specific data fetching (5m/15m/daily/weekly based on mode)
        candles_data = get_candles_for_mode(symbol, mode=mode.value)

        # Extract timeframe from mode config for engine
        mode_config = candles_data.get("_mode_config", {})
        tf = f"{mode_config.get('multiplier', 1)}{mode_config.get('timespan', 'day')[0]}"
        
        if not candles_data or "results" not in candles_data:
            raise HTTPException(
                status_code=400, 
                detail=f"Unable to fetch candle data for {symbol}"
            )
        
        if len(candles_data.get("results", [])) < 150:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {symbol}: {len(candles_data.get('results', []))} bars"
            )
        
        # Fetch options data if requested
        options_data = None
        if include_options:
            try:
                options_data = get_full_option_chain_snapshot(symbol, limit=100)
            except Exception as e:
                print(f"[Router] Options fetch warning: {e}")

        # Fetch market-wide context (SPY + VIX)
        market_ctx = {}
        try:
            market_ctx = get_market_context(mode=mode.value)
        except Exception as e:
            print(f"[Router] Market context warning: {e}")

        # Run analysis
        result = engine.analyze(
            ticker=symbol,
            candles_data=candles_data,
            options_data=options_data,
            mode=trade_mode,
            timeframe=tf,
            market_context=market_ctx
        )
        
        # Format output
        if output == OutputFormat.human:
            return {"text": engine.get_human_readable(result)}
        elif output == OutputFormat.ai:
            return convert_numpy_types({
                "summary": engine.get_human_readable(result),
                "data": engine.to_dict(result)
            })
        else:
            return convert_numpy_types(engine.to_dict(result))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/quick")
async def quick_signal(
    symbol: str = Query(..., description="Stock symbol"),
    mode: TradeModeParam = Query(TradeModeParam.swing, description="Trading mode")
):
    """
    ⚡ Quick Signal Check
    
    Fast analysis returning just the essential signals:
    - Direction (BULLISH/BEARISH/NEUTRAL)
    - Action (BUY CALL/BUY PUT/FLAT)
    - Confidence level
    - Win probability
    - Recommended strike/expiry
    """
    try:
        engine = get_engine()
        symbol = symbol.upper()
        
        trade_mode = TradeMode.SCALP if mode == TradeModeParam.scalp else TradeMode.SWING
        
        # Fetch minimal candle data
        candles_data = get_candles(symbol, tf="day", limit=200)

        if not candles_data or "results" not in candles_data:
            raise HTTPException(status_code=400, detail=f"Unable to fetch data for {symbol}")

        # Fetch market context
        market_ctx = {}
        try:
            market_ctx = get_market_context(mode=mode.value)
        except Exception as e:
            print(f"[Router] Quick market context warning: {e}")

        # Run analysis (without options for speed)
        result = engine.analyze(
            ticker=symbol,
            candles_data=candles_data,
            options_data=None,
            mode=trade_mode,
            timeframe="day",
            market_context=market_ctx
        )

        return convert_numpy_types({
            "ticker": symbol,
            "mode": mode.value,
            "timestamp": datetime.now().isoformat(),
            "signal": {
                "direction": result.direction,
                "action": result.action,
                "confidence": result.confidence.value,
                "win_probability": result.win_probability,
                "trade_valid": result.trade_valid
            },
            "option": {
                "strike": result.strike,
                "delta": result.delta,
                "expiry_dte": result.expiry_dte
            },
            "market_context": {
                "market_bias": market_ctx.get("market_bias", "unknown"),
                "vix_level": market_ctx.get("vix", {}).get("level"),
                "spy_trend": market_ctx.get("spy", {}).get("trend"),
                "risk_level": market_ctx.get("risk_level", "unknown"),
                "favor_calls": market_ctx.get("favor_calls", False),
                "favor_puts": market_ctx.get("favor_puts", False),
                "warnings": market_ctx.get("warnings", [])
            },
            "quick_summary": f"{'🟢' if result.direction == 'BULLISH' else '🔴' if result.direction == 'BEARISH' else '⚪'} {result.action} | {result.confidence.value} ({result.win_probability:.0f}%)"
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick signal failed: {str(e)}")


@router.get("/scan")
async def scan_tickers(
    symbols: str = Query(..., description="Comma-separated symbols (e.g., SPY,QQQ,AAPL)"),
    mode: TradeModeParam = Query(TradeModeParam.swing, description="Trading mode")
):
    """
    Multi-Ticker Scanner - PURE DATA OUTPUT
    Returns key signals from all layers for each ticker.
    NO filtering, NO blocking - AI decides what to trade.
    """
    try:
        engine = get_engine()
        ticker_list = [s.strip().upper() for s in symbols.split(",")][:20]
        trade_mode = TradeMode.SCALP if mode == TradeModeParam.scalp else TradeMode.SWING
        results = []

        # Fetch market context ONCE for all tickers
        market_ctx = {}
        try:
            market_ctx = get_market_context(mode=mode.value)
        except Exception as e:
            print(f"[Router] Scan market context warning: {e}")

        for symbol in ticker_list:
            try:
                candles_data = get_candles_for_mode(symbol, mode=mode.value)
                if candles_data and "results" in candles_data and len(candles_data["results"]) >= 50:
                    result = engine.analyze(
                        ticker=symbol,
                        candles_data=candles_data,
                        options_data=None,
                        mode=trade_mode,
                        market_context=market_ctx
                    )
                    
                    tech = result.technical_layers or {}
                    layer1 = tech.get('layer_1', {})
                    layer2 = tech.get('layer_2', {})
                    layer5 = tech.get('layer_5', {})
                    layer6 = tech.get('layer_6', {})
                    opts = result.options_layers or {}
                    layer14 = opts.get('layer_14', {})
                    layer15 = opts.get('layer_15', {})
                    layer16 = opts.get('layer_16', {})
                    
                    results.append({
                        "ticker": symbol,
                        "price": result.current_price,
                        "layers_ok": result.layers_successful,
                        "momentum": {
                            "rsi_14": layer1.get('rsi_14'),
                            "rsi_7": layer1.get('rsi_7'),
                            "macd_hist": layer1.get('macd_histogram'),
                            "macd_rising": layer1.get('macd_histogram_rising'),
                            "stoch_k": layer1.get('stoch_k'),
                            "cmf": layer1.get('cmf')
                        },
                        "volume": {
                            "obv_slope": layer2.get('obv_slope'),
                            "volume_ratio": layer2.get('volume_ratio')
                        },
                        "trend": {
                            "supertrend": layer5.get('supertrend_direction'),
                            "supertrend_bullish": layer5.get('supertrend_bullish'),
                            "adx": layer5.get('adx'),
                            "trending": layer5.get('trending'),
                            "weak_trend": layer5.get('weak_trend'),
                            "choppy": layer5.get('choppy'),
                            "htf_aligned": layer5.get('htf_aligned')
                        },
                        "structure": {
                            "current_trend": layer6.get('current_trend'),
                            "choch_bull": layer6.get('choch_bull_detected'),
                            "choch_bear": layer6.get('choch_bear_detected'),
                            "bos_bull": layer6.get('bos_bull_detected'),
                            "bos_bear": layer6.get('bos_bear_detected'),
                            "ob_bull": layer6.get('ob_bull_detected'),
                            "ob_bear": layer6.get('ob_bear_detected'),
                            "fvg_bull": layer6.get('fvg_bull_detected'),
                            "fvg_bear": layer6.get('fvg_bear_detected'),
                            "bull_bear_ratio": layer6.get('bull_bear_ratio')
                        },
                        "options": {
                            "iv_rank": layer14.get('iv_rank'),
                            "max_pain": layer15.get('max_pain'),
                            "pcr": layer16.get('pcr_current')
                        }
                    })
                else:
                    results.append({"ticker": symbol, "error": "Need 50+ candles"})
            except Exception as e:
                results.append({"ticker": symbol, "error": str(e)})
        
        return convert_numpy_types({
            "scan_time": datetime.now().isoformat(),
            "mode": mode.value,
            "count": len(ticker_list),
            "market_context": {
                "market_bias": market_ctx.get("market_bias", "unknown"),
                "spy_trend": market_ctx.get("spy", {}).get("trend"),
                "spy_price": market_ctx.get("spy", {}).get("price"),
                "vix_level": market_ctx.get("vix", {}).get("level"),
                "vix_regime": market_ctx.get("vix", {}).get("regime"),
                "risk_level": market_ctx.get("risk_level", "unknown"),
                "favor_calls": market_ctx.get("favor_calls", False),
                "favor_puts": market_ctx.get("favor_puts", False),
                "warnings": market_ctx.get("warnings", [])
            },
            "results": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")



@router.get("/compare")
async def compare_setups(
    symbols: str = Query(..., description="Comma-separated symbols to compare"),
    mode: TradeModeParam = Query(TradeModeParam.swing, description="Trading mode")
):
    """
    📊 Compare Multiple Setups
    
    Side-by-side comparison of setups for decision making.
    Shows which setup has the highest probability edge.
    """
    try:
        engine = get_engine()
        ticker_list = [s.strip().upper() for s in symbols.split(",")][:5]  # Max 5 for comparison

        trade_mode = TradeMode.SCALP if mode == TradeModeParam.scalp else TradeMode.SWING

        # Fetch market context ONCE for all comparisons
        market_ctx = {}
        try:
            market_ctx = get_market_context(mode=mode.value)
        except Exception as e:
            print(f"[Router] Compare market context warning: {e}")

        comparisons = []

        for symbol in ticker_list:
            try:
                candles_data = get_candles(symbol, tf="day", limit=730)
                options_data = None
                try:
                    options_data = get_option_chain_snapshot(symbol, limit=50)
                except:
                    pass

                if candles_data and "results" in candles_data:
                    result = engine.analyze(
                        ticker=symbol,
                        candles_data=candles_data,
                        options_data=options_data,
                        mode=trade_mode,
                        market_context=market_ctx
                    )
                    
                    comparisons.append({
                        "rank": 0,  # Will be set after sorting
                        "ticker": symbol,
                        "recommendation": result.action,
                        "direction": result.direction,
                        "confidence": result.confidence.value,
                        "win_probability": result.win_probability,
                        "option": {
                            "strike": result.strike,
                            "delta": result.delta,
                            "expiry_dte": result.expiry_dte
                        },
                        "execution": {
                            "entry": result.entry_price,
                            "target": result.target_price,
                            "stop": result.stop_price,
                            "risk_reward": result.risk_reward
                        },
                        "reasoning": result.reasoning[:3],
                        "concerns": result.concerns[:2]
                    })
                    
            except Exception as e:
                comparisons.append({
                    "ticker": symbol,
                    "error": str(e)
                })
        
        # Sort and rank
        valid_comparisons = [c for c in comparisons if "error" not in c]
        valid_comparisons.sort(key=lambda x: x.get("win_probability", 0), reverse=True)
        
        for i, comp in enumerate(valid_comparisons):
            comp["rank"] = i + 1
        
        # Find best setup
        best_setup = valid_comparisons[0] if valid_comparisons else None
        
        return convert_numpy_types({
            "comparison_time": datetime.now().isoformat(),
            "mode": mode.value,
            "market_context": {
                "market_bias": market_ctx.get("market_bias", "unknown"),
                "spy_trend": market_ctx.get("spy", {}).get("trend"),
                "vix_level": market_ctx.get("vix", {}).get("level"),
                "risk_level": market_ctx.get("risk_level", "unknown"),
                "favor_calls": market_ctx.get("favor_calls", False),
                "favor_puts": market_ctx.get("favor_puts", False),
                "warnings": market_ctx.get("warnings", [])
            },
            "best_setup": best_setup["ticker"] if best_setup else None,
            "best_probability": best_setup["win_probability"] if best_setup else 0,
            "ranked_setups": valid_comparisons,
            "errors": [c for c in comparisons if "error" in c]
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/layers")
async def list_layers():
    """
    📚 List All 18 Layers
    
    Complete breakdown of all analysis layers and their outputs.
    """
    return convert_numpy_types({
        "total_layers": 18,
        "categories": {
            "technical": {
                "layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "description": "Core technical indicators and price analysis"
            },
            "price_action": {
                "layers": [11, 12, 13],
                "description": "Support/Resistance, VWAP, Volume Profile"
            },
            "options": {
                "layers": [14, 15, 16, 17],
                "description": "Options-specific analysis (IV, Greeks, GEX)"
            },
            "decision": {
                "layers": [18],
                "description": "Master Brain - pure data aggregation for AI"
            }
        },
        "layer_details": [
            {"id": 1, "name": "Momentum", "outputs": ["RSI", "MACD", "Stochastic", "CMF", "ADX", "Ichimoku"]},
            {"id": 2, "name": "Volume", "outputs": ["OBV", "A/D Line", "CMF", "Volume divergence"]},
            {"id": 3, "name": "Divergence", "outputs": ["RSI divergence", "MACD divergence", "CDV analysis"]},
            {"id": 4, "name": "Volume Strength", "outputs": ["RVOL", "Volume spike detection", "CVD"]},
            {"id": 5, "name": "Trend", "outputs": ["SuperTrend", "ADX/DMI", "Moving averages"]},
            {"id": 6, "name": "Structure", "outputs": ["CHoCH", "BOS", "Order blocks", "FVG"]},
            {"id": 7, "name": "Liquidity", "outputs": ["Liquidity sweeps", "Hunt detection"]},
            {"id": 8, "name": "Volatility Regime", "outputs": ["ATR percentile", "Volatility classification"]},
            {"id": 9, "name": "Confirmation", "outputs": ["Multi-timeframe confirmation"]},
            {"id": 10, "name": "Candle Intelligence", "outputs": ["Advanced candlestick patterns"]},
            {"id": 11, "name": "Support/Resistance", "outputs": ["Fractals", "S/R channels", "Pivot points", "MTF levels"]},
            {"id": 12, "name": "VWAP Analysis", "outputs": ["VWAP levels", "Standard deviation bands"]},
            {"id": 13, "name": "Volume Profile", "outputs": ["POC", "Value Area", "HVN/LVN"]},
            {"id": 14, "name": "IV Analysis", "outputs": ["HV", "IV Rank", "IV Percentile", "Expected Move"]},
            {"id": 15, "name": "Gamma & Max Pain", "outputs": ["Max Pain", "GEX", "Pin Risk"]},
            {"id": 16, "name": "Put/Call Ratio", "outputs": ["PCR current", "PCR trend", "Volume imbalance"]},
            {"id": 17, "name": "Greeks Analysis", "outputs": ["Best strike selection", "Delta/Gamma/Theta/Vega analysis"]},
            {"id": 18, "name": "Master Brain", "outputs": ["Data aggregation", "Trade recommendation", "Risk management"]}
        ]
    })


@router.get("/layer/{layer_number}")
async def get_layer_analysis(
    layer_number: int,
    symbol: str = Query(..., description="Stock symbol"),
    tf: str = Query("day", description="Timeframe")
):
    """
    🔬 Single Layer Analysis
    
    Get output from a specific layer for debugging or detailed analysis.
    """
    if layer_number < 1 or layer_number > 18:
        raise HTTPException(status_code=400, detail="Layer number must be between 1 and 18")
    
    try:
        engine = get_engine()
        symbol = symbol.upper()
        
        candles_data = get_candles(symbol, tf=tf, limit=730)
        
        if not candles_data or "results" not in candles_data:
            raise HTTPException(status_code=400, detail=f"Unable to fetch data for {symbol}")
        
        # Fetch options data for layers 14-17
        options_data = None
        if layer_number >= 14 and layer_number <= 17:
            try:
                options_data = get_full_option_chain_snapshot(symbol, limit=100)
            except:
                pass
        
        # Run full analysis to get layer data
        result = engine.analyze(
            ticker=symbol,
            candles_data=candles_data,
            options_data=options_data,
            mode=TradeMode.SWING
        )
        
        layer_key = f"layer_{layer_number}"
        layer_data = result.layer_results.get(layer_key)
        
        if layer_data:
            return convert_numpy_types({
                "symbol": symbol,
                "timeframe": tf,
                "layer": layer_number,
                "success": layer_data.success,
                "data": layer_data.data,
                "error": layer_data.error,
                "execution_time_ms": layer_data.execution_time_ms
            })
        else:
            raise HTTPException(status_code=404, detail=f"Layer {layer_number} data not available")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer analysis failed: {str(e)}")


@router.get("/health")
async def engine_health():
    """
    💚 Engine Health Check
    
    Returns system status and layer availability.
    """
    try:
        engine = get_engine()
        
        available_layers = list(engine._layers.keys())
        brain_available = engine._layer_18_brain is not None
        
        return convert_numpy_types({
            "status": "healthy",
            "engine": "TradePilot 18-Layer Engine",
            "version": "3.0",
            "timestamp": datetime.now().isoformat(),
            "layers_loaded": len(available_layers) + (1 if brain_available else 0),
            "technical_layers": [k for k in available_layers if int(k.split("_")[1]) <= 10],
            "price_action_layers": [k for k in available_layers if 11 <= int(k.split("_")[1]) <= 13],
            "options_layers": [k for k in available_layers if int(k.split("_")[1]) >= 14],
            "brain_available": brain_available,
            "configuration": engine.config
        })
    
    except Exception as e:
        return convert_numpy_types({
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })


@router.get("/ai-prompt")
async def get_ai_prompt(
    symbol: str = Query(..., description="Stock symbol"),
    mode: TradeModeParam = Query(TradeModeParam.swing, description="Trading mode")
):
    """
    🤖 AI Integration Prompt
    
    Returns analysis formatted for Claude/GPT integration with full context.
    Perfect for automated AI trading analysis workflows.
    """
    try:
        engine = get_engine()
        symbol = symbol.upper()
        
        trade_mode = TradeMode.SCALP if mode == TradeModeParam.scalp else TradeMode.SWING
        
        candles_data = get_candles(symbol, tf="day", limit=730)
        options_data = None
        try:
            options_data = get_option_chain_snapshot(symbol, limit=50)
        except:
            pass

        # Fetch market context
        market_ctx = {}
        try:
            market_ctx = get_market_context(mode=mode.value)
        except Exception as e:
            print(f"[Router] AI-prompt market context warning: {e}")

        if candles_data and "results" in candles_data:
            result = engine.analyze(
                ticker=symbol,
                candles_data=candles_data,
                options_data=options_data,
                mode=trade_mode,
                market_context=market_ctx
            )

            # Market context summary for prompt
            spy_info = market_ctx.get("spy", {})
            vix_info = market_ctx.get("vix", {})
            mkt_bias = market_ctx.get("market_bias", "unknown")
            mkt_warnings = market_ctx.get("warnings", [])

            # Build AI-friendly prompt
            prompt = f"""## TradePilot 18-Layer Analysis for {symbol}

### Market Context (SPY + VIX)
- **SPY**: ${spy_info.get('price', '?')} | Trend: {spy_info.get('trend', '?')} | vs 200MA: {spy_info.get('distance_from_200ma_pct', '?')}%
- **VIX**: {vix_info.get('level', '?')} | Regime: {vix_info.get('regime', '?')} | Percentile: {vix_info.get('percentile', '?')}
- **Market Bias**: {mkt_bias} | Risk: {market_ctx.get('risk_level', '?')}
- **Favor**: {'CALLS' if market_ctx.get('favor_calls') else 'PUTS' if market_ctx.get('favor_puts') else 'NEUTRAL'}
{chr(10).join('- ⚠️ ' + w for w in mkt_warnings) if mkt_warnings else ''}

### Quick Summary
- **Ticker**: {result.ticker} @ ${result.current_price:.2f}
- **Mode**: {result.mode.value}
- **Trade Valid**: {'YES' if result.trade_valid else 'NO'}
- **Direction**: {result.direction}
- **Action**: {result.action}
- **Confidence**: {result.confidence.value} ({result.win_probability:.1f}%)
### Option Recommendation
- Strike: ${result.strike:.2f} ({result.strike_type})
- Delta: {result.delta:.2f}
- Expiry: {result.expiry_date} ({result.expiry_dte} DTE)

### Execution Plan
- Entry: ${result.entry_price:.2f}
- Target: ${result.target_price:.2f}
- Stop: ${result.stop_price:.2f}
- Risk/Reward: {result.risk_reward:.1f}:1
- Position Size: {result.position_size_pct*100:.0f}%

### Key Reasoning
{chr(10).join('- ' + r for r in result.reasoning[:5])}

### Concerns
{chr(10).join('- ' + c for c in result.concerns) if result.concerns else '- None identified'}

### Raw Layer Data
Available for detailed analysis in the `layer_data` and `market_context` fields.
"""

            return convert_numpy_types({
                "prompt": prompt,
                "structured_data": engine.to_dict(result),
                "usage_instructions": "Pass this prompt to Claude or GPT for detailed trading analysis. The structured_data contains all layer outputs AND market_context (SPY+VIX) for deep analysis."
            })
        else:
            raise HTTPException(status_code=400, detail=f"Unable to analyze {symbol}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI prompt generation failed: {str(e)}")


@router.get("/backtest")
async def backtest_signal(
    symbol: str = Query(..., description="Stock symbol (e.g., SPY, AAPL, TSLA)"),
    mode: TradeModeParam = Query(TradeModeParam.swing, description="Trading mode"),
    checkpoints: int = Query(10, description="Number of historical checkpoints to test (5-30)", ge=5, le=30),
    max_hold: int = Query(30, description="Max bars to hold a trade before timeout", ge=5, le=60),
    min_confidence: str = Query("MODERATE", description="Minimum confidence to take a trade")
):
    """
    Paper Trading Backtest

    Tests the 18-layer engine against historical data:
    1. Slides a window through 2 years of price history
    2. At each checkpoint runs the full analysis
    3. Simulates trades using entry/target/stop levels
    4. Tracks wins, losses, P&L, and real win rate

    Returns actual performance metrics - not theoretical.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from paper_trade import run_backtest
        result = run_backtest(
            symbol=symbol.upper(),
            mode=mode.value,
            checkpoints=checkpoints,
            max_hold_bars=max_hold,
            min_confidence=min_confidence
        )

        # Save to database
        try:
            _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            import database as db
            db.save_backtest(
                symbol=symbol.upper(),
                mode=mode.value,
                total_signals=result.total_signals,
                trades_taken=result.trades_taken,
                wins=result.wins,
                losses=result.losses,
                timeouts=result.timeouts,
                win_rate=result.win_rate,
                total_pnl=result.total_pnl_pct,
                full_result=json.dumps({
                    "by_confidence": result.by_confidence,
                    "avg_pnl_pct": result.avg_pnl_pct,
                    "best_trade_pnl": result.best_trade_pnl,
                    "worst_trade_pnl": result.worst_trade_pnl,
                    "avg_bars_held": result.avg_bars_held,
                    "avg_risk_reward": result.avg_risk_reward,
                    "trades": result.trades
                }, default=str)
            )
        except Exception as e:
            print(f"[Router] Backtest save warning: {e}")

        from dataclasses import asdict
        return convert_numpy_types(asdict(result))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@router.get("/strategies")
async def get_strategies(
    symbol: str = Query(..., description="Stock symbol (e.g., SPY, AAPL)"),
    mode: str = Query("swing", description="Mode: scalp (0-2 DTE), swing (7-45 DTE), leaps (180-400 DTE)")
):
    """
    🎯 Get All 6 Option Strategies
    
    Returns analysis + 6 best strategies for the given mode:
    - Long Call
    - Long Put  
    - Bull Call Spread (debit)
    - Bear Put Spread (debit)
    - Bull Put Spread (credit)
    - Bear Call Spread (credit)
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from polygon_client import analyze_with_strategies

    try:
        result = analyze_with_strategies(symbol.upper(), mode.lower())

        # Add market context to strategies output
        try:
            market_ctx = get_market_context(mode=mode.lower())
            result["market_context"] = {
                "market_bias": market_ctx.get("market_bias", "unknown"),
                "spy_trend": market_ctx.get("spy", {}).get("trend"),
                "spy_price": market_ctx.get("spy", {}).get("price"),
                "vix_level": market_ctx.get("vix", {}).get("level"),
                "vix_regime": market_ctx.get("vix", {}).get("regime"),
                "risk_level": market_ctx.get("risk_level", "unknown"),
                "favor_calls": market_ctx.get("favor_calls", False),
                "favor_puts": market_ctx.get("favor_puts", False),
                "warnings": market_ctx.get("warnings", [])
            }
        except Exception as e:
            print(f"[Router] Strategies market context warning: {e}")

        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
