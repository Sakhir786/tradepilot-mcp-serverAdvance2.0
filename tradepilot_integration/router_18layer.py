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
    if DATA_SOURCE == "ibkr":
        from ibkr_client import analyze_with_strategies
    else:
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


# ---------------------------------------------------------------------------
# IBKR Execution Endpoints (only available when DATA_SOURCE=ibkr)
# ---------------------------------------------------------------------------

if DATA_SOURCE == "ibkr":
    from pydantic import BaseModel, Field
    from typing import List as TypingList

    class OptionOrderRequest(BaseModel):
        symbol: str = Field(..., description="Underlying symbol (e.g. SPY)")
        expiry: str = Field(..., description="Expiration YYYYMMDD or YYYY-MM-DD")
        strike: float = Field(..., description="Strike price")
        right: str = Field(..., description="C for call, P for put")
        action: str = Field("BUY", description="BUY or SELL")
        quantity: int = Field(1, description="Number of contracts", ge=1)
        order_type: str = Field("LMT", description="LMT, MKT, or STP")
        limit_price: Optional[float] = Field(None, description="Limit price (auto mid-price if None for LMT)")

    class SpreadLeg(BaseModel):
        expiry: str
        strike: float
        right: str
        action: str
        ratio: int = 1

    class SpreadOrderRequest(BaseModel):
        symbol: str
        legs: TypingList[SpreadLeg]
        action: str = "BUY"
        quantity: int = Field(1, ge=1)
        order_type: str = "LMT"
        limit_price: Optional[float] = None

    class CloseRequest(BaseModel):
        symbol: str
        con_id: int = 0
        quantity: Optional[int] = None
        order_type: str = "MKT"

    @router.post("/execute")
    async def execute_option_order(req: OptionOrderRequest):
        """
        Place a single-leg option order via IBKR.
        Requires DATA_SOURCE=ibkr and active IB Gateway connection.
        """
        try:
            from ibkr_client import place_option_order
            result = place_option_order(
                symbol=req.symbol,
                expiry=req.expiry,
                strike=req.strike,
                right=req.right,
                action=req.action,
                quantity=req.quantity,
                order_type=req.order_type,
                limit_price=req.limit_price,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Order failed: {str(e)}")

    @router.post("/execute/spread")
    async def execute_spread_order(req: SpreadOrderRequest):
        """
        Place a multi-leg spread order via IBKR.
        Supports verticals, iron condors, etc.
        """
        try:
            from ibkr_client import place_spread_order
            legs = [leg.model_dump() for leg in req.legs]
            result = place_spread_order(
                symbol=req.symbol,
                legs=legs,
                action=req.action,
                quantity=req.quantity,
                order_type=req.order_type,
                limit_price=req.limit_price,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Spread order failed: {str(e)}")

    @router.post("/close")
    async def close_position_endpoint(req: CloseRequest):
        """Close an existing position and update trade record in DB."""
        try:
            from ibkr_client import close_position
            result = close_position(
                symbol=req.symbol,
                con_id=req.con_id,
                quantity=req.quantity,
                order_type=req.order_type,
            )
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])

            # Update DB: find matching trade by symbol and close it
            try:
                import database as db
                open_trades = db.get_live_trades(status="OPEN", symbol=req.symbol)
                if open_trades:
                    # Close the most recent matching trade
                    trade = open_trades[0]
                    fill_price = result.get("avg_fill_price", 0)
                    close_reason = "MANUAL_CLOSE"
                    db.close_live_trade(trade["order_id"], fill_price, close_reason)
            except Exception as e:
                print(f"[Router] Close DB update warning: {e}")

            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Close failed: {str(e)}")

    @router.get("/positions")
    async def list_positions():
        """Get all current IBKR positions."""
        try:
            from ibkr_client import get_positions
            return {"positions": get_positions()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Positions failed: {str(e)}")

    @router.get("/orders")
    async def list_open_orders():
        """Get all open/pending orders."""
        try:
            from ibkr_client import get_open_orders
            return {"orders": get_open_orders()}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Orders failed: {str(e)}")

    @router.delete("/orders/{order_id}")
    async def cancel_order_endpoint(order_id: int):
        """Cancel an open order."""
        try:
            from ibkr_client import cancel_order
            result = cancel_order(order_id)
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cancel failed: {str(e)}")

    @router.get("/account")
    async def account_summary():
        """Get account balance, buying power, P&L."""
        try:
            from ibkr_client import get_account_summary
            return get_account_summary()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Account summary failed: {str(e)}")

    # -------------------------------------------------------------------
    # AI-Critical Execution Endpoints
    # -------------------------------------------------------------------

    class ExecuteSignalRequest(BaseModel):
        symbol: str = Field(..., description="Stock symbol to analyze and execute")
        mode: str = Field("swing", description="Trading mode: scalp, intraday, swing, leaps")
        quantity: int = Field(1, ge=1, description="Number of contracts (0 = use engine suggestion)")
        order_type: str = Field("LMT", description="LMT or MKT")
        limit_price: Optional[float] = Field(None, description="Override limit price")
        dry_run: bool = Field(False, description="Preview without executing")
        bracket: bool = Field(False, description="Place bracket order with TP/SL")

    class ModifyOrderRequest(BaseModel):
        order_id: int
        new_limit_price: Optional[float] = None
        new_quantity: Optional[int] = None

    class RollOptionRequest(BaseModel):
        symbol: str
        old_con_id: int = Field(..., description="conId of position to close")
        new_expiry: str = Field(..., description="New expiration YYYYMMDD")
        new_strike: Optional[float] = None
        new_right: Optional[str] = None
        quantity: Optional[int] = None
        order_type: str = "MKT"

    class BracketOrderRequest(BaseModel):
        symbol: str
        expiry: str
        strike: float
        right: str
        action: str = "BUY"
        quantity: int = Field(1, ge=1)
        entry_price: float
        take_profit_price: float
        stop_loss_price: float

    @router.post("/execute/signal")
    async def execute_from_signal(req: ExecuteSignalRequest):
        """
        AI's primary action: analyze a symbol, then auto-execute the trade.
        Runs full 18-layer analysis → extracts recommendation → places order.
        Set dry_run=true to preview without executing.
        Set bracket=true to place entry + TP + SL in one shot.
        """
        try:
            engine = get_engine()
            symbol = req.symbol.upper()

            trade_mode = {
                "scalp": TradeMode.SCALP,
                "swing": TradeMode.SWING,
                "intraday": TradeMode.INTRADAY,
                "leaps": TradeMode.LEAPS,
            }.get(req.mode, TradeMode.SWING)

            # Step 1: Run analysis
            candles_data = get_candles_for_mode(symbol, mode=req.mode)
            if not candles_data or "results" not in candles_data:
                raise HTTPException(status_code=400, detail=f"No candle data for {symbol}")

            mode_config = candles_data.get("_mode_config", {})
            tf = f"{mode_config.get('multiplier', 1)}{mode_config.get('timespan', 'day')[0]}"

            options_data = None
            try:
                options_data = get_full_option_chain_snapshot(symbol, limit=100)
            except Exception:
                pass

            market_ctx = {}
            try:
                market_ctx = get_market_context(mode=req.mode)
            except Exception:
                pass

            result = engine.analyze(
                ticker=symbol,
                candles_data=candles_data,
                options_data=options_data,
                mode=trade_mode,
                timeframe=tf,
                market_context=market_ctx,
            )

            analysis_dict = convert_numpy_types(engine.to_dict(result))

            # Step 2: Execute
            from ibkr_client import execute_signal, execute_signal_bracket

            if req.bracket:
                exec_result = execute_signal_bracket(
                    analysis_result=analysis_dict,
                    quantity=req.quantity,
                )
            else:
                exec_result = execute_signal(
                    analysis_result=analysis_dict,
                    quantity=req.quantity,
                    order_type=req.order_type,
                    limit_price=req.limit_price,
                    dry_run=req.dry_run,
                )

            # Step 3: Record in database
            if "order_id" in exec_result or "parent" in exec_result:
                try:
                    import database as db
                    order_id = exec_result.get("order_id", exec_result.get("parent", {}).get("order_id", 0))
                    signal = exec_result.get("engine_signal", {})
                    opt_rec = analysis_dict.get("option_recommendation", {})
                    db.save_live_trade(
                        order_id=order_id,
                        symbol=symbol,
                        mode=req.mode,
                        action=signal.get("action", exec_result.get("action", "")),
                        right=exec_result.get("contract", {}).get("right", ""),
                        strike=opt_rec.get("strike", 0),
                        expiry=opt_rec.get("expiry_date", ""),
                        quantity=req.quantity,
                        entry_price=signal.get("entry", 0),
                        stop_price=signal.get("stop", 0),
                        target_price=signal.get("target", 0),
                        confidence=signal.get("confidence", ""),
                        win_probability=signal.get("win_probability", 0),
                        signal_data=analysis_dict.get("analysis_summary", {}),
                    )
                except Exception as e:
                    print(f"[Router] Trade record warning: {e}")

            return convert_numpy_types({
                "analysis": {
                    "ticker": symbol,
                    "direction": analysis_dict.get("analysis_summary", {}).get("direction"),
                    "action": analysis_dict.get("analysis_summary", {}).get("action"),
                    "confidence": analysis_dict.get("analysis_summary", {}).get("confidence"),
                    "win_probability": analysis_dict.get("analysis_summary", {}).get("win_probability"),
                    "trade_valid": analysis_dict.get("analysis_summary", {}).get("trade_valid"),
                },
                "execution": exec_result,
            })

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execute signal failed: {str(e)}")

    @router.post("/execute/bracket")
    async def execute_bracket_order(req: BracketOrderRequest):
        """
        Place a bracket order: entry + take-profit + stop-loss.
        IBKR handles OCO (one-cancels-other) server-side.
        """
        try:
            from ibkr_client import place_bracket_order
            result = place_bracket_order(
                symbol=req.symbol,
                expiry=req.expiry,
                strike=req.strike,
                right=req.right,
                action=req.action,
                quantity=req.quantity,
                entry_price=req.entry_price,
                take_profit_price=req.take_profit_price,
                stop_loss_price=req.stop_loss_price,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Bracket order failed: {str(e)}")

    @router.put("/orders/{order_id}")
    async def modify_order_endpoint(order_id: int, req: ModifyOrderRequest):
        """Modify an open order's price or quantity."""
        try:
            from ibkr_client import modify_order
            result = modify_order(
                order_id=order_id,
                new_limit_price=req.new_limit_price,
                new_quantity=req.new_quantity,
            )
            if "error" in result:
                raise HTTPException(status_code=404, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Modify failed: {str(e)}")

    @router.post("/roll")
    async def roll_option_endpoint(req: RollOptionRequest):
        """
        Roll an option: close expiring position → open new DTE.
        AI uses this to manage expiring positions automatically.
        """
        try:
            from ibkr_client import roll_option
            result = roll_option(
                symbol=req.symbol,
                old_con_id=req.old_con_id,
                new_expiry=req.new_expiry,
                new_strike=req.new_strike,
                new_right=req.new_right,
                quantity=req.quantity,
                order_type=req.order_type,
            )
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Roll failed: {str(e)}")

    @router.post("/close/all")
    async def close_all_endpoint():
        """Emergency flatten: close ALL positions with market orders."""
        try:
            from ibkr_client import close_all_positions
            return close_all_positions(order_type="MKT")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Close all failed: {str(e)}")

    @router.delete("/orders/all")
    async def cancel_all_orders_endpoint():
        """Cancel ALL open orders immediately."""
        try:
            from ibkr_client import cancel_all_orders
            return cancel_all_orders()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cancel all failed: {str(e)}")

    @router.get("/positions/pnl")
    async def position_pnl(symbol: Optional[str] = Query(None)):
        """Get real-time P&L for each position."""
        try:
            from ibkr_client import get_position_pnl
            return {"positions": get_position_pnl(symbol=symbol)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PnL failed: {str(e)}")

    @router.get("/portfolio/risk")
    async def portfolio_risk():
        """
        Portfolio-level risk: total Greeks, exposure %, concentration, warnings.
        AI should check this BEFORE placing any new trade.
        """
        try:
            from ibkr_client import get_portfolio_risk
            return get_portfolio_risk()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Portfolio risk failed: {str(e)}")

    @router.get("/orders/{order_id}/wait")
    async def wait_for_fill_endpoint(
        order_id: int,
        timeout: int = Query(60, ge=5, le=300, description="Max seconds to wait"),
    ):
        """Wait for an order to fill (polls until filled or timeout)."""
        try:
            from ibkr_client import wait_for_fill
            return wait_for_fill(order_id=order_id, timeout_seconds=timeout)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Wait failed: {str(e)}")

    @router.get("/trades")
    async def list_trades(
        status: Optional[str] = Query(None, description="Filter: OPEN, CLOSED, PENDING"),
        symbol: Optional[str] = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ):
        """Get recorded live trades from database."""
        try:
            import database as db
            return {"trades": db.get_live_trades(status=status, symbol=symbol, limit=limit)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Trades query failed: {str(e)}")

    @router.get("/trades/stats")
    async def trade_stats():
        """Aggregate trade statistics: win rate, total P&L, best/worst trade."""
        try:
            import database as db
            return db.get_trade_stats()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")

    class ExecuteStrategyRequest(BaseModel):
        symbol: str = Field(..., description="Stock symbol")
        mode: str = Field("swing", description="Trading mode")
        strategy_name: str = Field(..., description="Strategy: long_call, long_put, bull_call_spread, bear_put_spread, bull_put_spread, bear_call_spread")
        quantity: int = Field(1, ge=1)
        order_type: str = Field("LMT", description="LMT or MKT")

    class PreTradeCheckRequest(BaseModel):
        cost_estimate: float = Field(..., description="Estimated trade cost in dollars")
        max_portfolio_pct: float = Field(5.0, description="Max % of NLV for single trade")
        max_positions: int = Field(10, description="Max open positions")
        min_buying_power_pct: float = Field(20.0, description="Min buying power % to keep")

    @router.post("/execute/strategy")
    async def execute_strategy_endpoint(req: ExecuteStrategyRequest):
        """
        Fetch strategies for a symbol, then auto-execute the chosen one.
        Combines /strategies + order placement in one call.
        """
        try:
            from ibkr_client import analyze_with_strategies, execute_strategy

            strategies_result = analyze_with_strategies(req.symbol.upper(), req.mode)
            if "error" in strategies_result:
                raise HTTPException(status_code=400, detail=strategies_result["error"])

            exec_result = execute_strategy(
                strategy_result=strategies_result,
                strategy_name=req.strategy_name,
                quantity=req.quantity,
                order_type=req.order_type,
            )

            if "error" in exec_result:
                raise HTTPException(status_code=400, detail=exec_result["error"])

            # Record in DB
            try:
                import database as db
                strat = strategies_result["strategies"].get(req.strategy_name, {})
                order_id = exec_result.get("order_id", 0)
                details = strat.get("details", {})
                is_spread = "long" in details
                right = "C" if "call" in req.strategy_name else "P"

                if is_spread:
                    strike = details.get("long", {}).get("strike", 0)
                    expiry = details.get("long", {}).get("expiry", "")
                    entry = strat.get("cost", strat.get("credit", 0)) / 100
                else:
                    strike = details.get("strike", 0)
                    expiry = details.get("expiry", "")
                    entry = details.get("price", 0)

                db.save_live_trade(
                    order_id=order_id,
                    symbol=req.symbol.upper(),
                    mode=req.mode,
                    action="BUY" if strat.get("type") == "DEBIT" else "SELL",
                    right=right,
                    strike=strike,
                    expiry=expiry,
                    quantity=req.quantity,
                    entry_price=entry,
                    confidence=req.strategy_name,
                    signal_data=strat,
                )
            except Exception as e:
                print(f"[Router] Strategy trade record warning: {e}")

            return convert_numpy_types({
                "strategy": req.strategy_name,
                "strategy_details": strategies_result["strategies"].get(req.strategy_name),
                "execution": exec_result,
            })

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Execute strategy failed: {str(e)}")

    @router.post("/pre-trade-check")
    async def pre_trade_check_endpoint(req: PreTradeCheckRequest):
        """
        Pre-trade risk gate. AI MUST call this before every trade.
        Returns approved/denied with reasons.
        """
        try:
            from ibkr_client import pre_trade_check
            return pre_trade_check(
                cost_estimate=req.cost_estimate,
                max_portfolio_pct=req.max_portfolio_pct,
                max_positions=req.max_positions,
                min_buying_power_pct=req.min_buying_power_pct,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pre-trade check failed: {str(e)}")

    @router.get("/dashboard")
    async def dashboard():
        """
        Single-call dashboard: account + positions + P&L + risk + orders.
        AI calls this once for full situational awareness.
        """
        try:
            from ibkr_client import get_dashboard
            return get_dashboard()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Dashboard failed: {str(e)}")

    @router.post("/sync")
    async def sync_orders():
        """
        Reconcile IBKR order state with local DB.
        Call periodically to catch fills/cancels that happened while disconnected.
        """
        try:
            from ibkr_client import sync_order_status
            return sync_order_status()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
