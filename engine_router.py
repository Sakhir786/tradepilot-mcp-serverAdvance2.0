"""
Engine Router - FastAPI endpoints for TradePilot Engine
"""
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import sys
sys.path.append('.')

from tradepilot_engine import TradePilotEngine
from polygon_client import get_candles

router = APIRouter(prefix="/engine", tags=["TradePilot Engine"])

# Initialize engine
engine = TradePilotEngine()


@router.get("/analyze")
async def analyze_symbol(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL)"),
    tf: str = Query("day", description="Timeframe (day, hour, minute)"),
    limit: int = Query(730, description="Number of candles to fetch")
):
    """
    Run complete 10-layer analysis on a symbol
    
    Returns comprehensive analysis from all layers
    """
    try:
        # Fetch candles (Massive API compatible)
        candles_data = get_candles(symbol.upper(), tf=tf, limit=limit)

        # --- Safe data check ---
        if not candles_data or "results" not in candles_data or len(candles_data["results"]) == 0:
            raise HTTPException(status_code=400, detail=f"No candle data returned for {symbol}")

        # --- Clean structure for engine ---
        clean_candles = [
            {
                "t": c.get("t"),
                "o": c.get("o"),
                "h": c.get("h"),
                "l": c.get("l"),
                "c": c.get("c"),
                "v": c.get("v"),
            }
            for c in candles_data["results"]
        ]

        if len(clean_candles) < 10:
            raise HTTPException(status_code=400, detail=f"Not enough candles for analysis ({len(clean_candles)})")

        # --- Auto Mode Detection ---
        if tf == "minute":
            mode_label = "⚡ SCALP PLAY (1–2 hr window)"
        elif tf == "hour":
            mode_label = "📈 INTRADAY PLAY (1–6 hr window)"
        else:
            mode_label = "📆 SWING PLAY (Multi-day window)"

        print(f"[Engine] {mode_label} → {symbol.upper()} | {len(clean_candles)} candles")

        # --- Run analysis ---
        results = engine.analyze({"results": clean_candles}, symbol.upper(), tf)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        # Attach mode label to response
        results["mode"] = mode_label

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/signal-summary")
async def get_signal_summary(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL)"),
    tf: str = Query("day", description="Timeframe"),
    limit: int = Query(730, description="Number of candles")
):
    """
    Get condensed signal summary for quick decision making
    
    Returns key metrics and overall recommendation
    """
    try:
        # Fetch candles
        candles_data = get_candles(symbol.upper(), tf=tf, limit=limit)

        if not candles_data or "results" not in candles_data:
            raise HTTPException(status_code=400, detail="Unable to fetch candle data")

        # Get summary
        summary = engine.get_signal_summary(candles_data, symbol.upper())

        if "error" in summary:
            raise HTTPException(status_code=400, detail=summary["error"])

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(e)}")


@router.get("/layer/{layer_name}")
async def get_layer_analysis(
    layer_name: str,
    symbol: str = Query(..., description="Stock symbol"),
    tf: str = Query("day", description="Timeframe"),
    limit: int = Query(730, description="Number of candles")
):
    """
    Get analysis from a specific layer only
    
    Available layers:
    - layer_1_momentum
    - layer_2_volume
    - layer_3_divergence
    - layer_4_volume_strength
    - layer_5_trend
    - layer_6_structure
    - layer_7_liquidity
    - layer_8_volatility_regime
    - layer_9_confirmation
    - layer_10_candle_intelligence
    """
    try:
        # Validate layer name
        if layer_name not in engine.layers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid layer name. Available: {list(engine.layers.keys())}"
            )

        # Fetch candles
        candles_data = get_candles(symbol.upper(), tf=tf, limit=limit)

        if not candles_data or "results" not in candles_data:
            raise HTTPException(status_code=400, detail="Unable to fetch candle data")

        # Run full analysis to get the layer result
        full_results = engine.analyze(candles_data, symbol.upper(), tf)

        if "error" in full_results:
            raise HTTPException(status_code=400, detail=full_results["error"])

        # Return specific layer
        layer_result = full_results["layers"].get(layer_name, {})

        return {
            "symbol": symbol.upper(),
            "timeframe": tf,
            "layer": layer_name,
            "result": layer_result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Layer analysis failed: {str(e)}")


@router.get("/health")
async def engine_health():
    """
    Check engine health status
    """
    return {
        "status": "healthy",
        "engine": "TradePilot v2.0",
        "layers": len(engine.layers),
        "available_layers": list(engine.layers.keys())
    }


@router.get("/layers")
async def list_layers():
    """
    List all available analysis layers
    """
    return {
        "total_layers": len(engine.layers),
        "layers": [
            {
                "name": "layer_1_momentum",
                "description": "RSI, MACD, Stochastic, CMF, ADX, Ichimoku"
            },
            {
                "name": "layer_2_volume",
                "description": "OBV, A/D Line, CMF, Volume divergence"
            },
            {
                "name": "layer_3_divergence",
                "description": "Delta divergence detection, CDV analysis"
            },
            {
                "name": "layer_4_volume_strength",
                "description": "RVOL, Volume spike detection"
            },
            {
                "name": "layer_5_trend",
                "description": "SuperTrend, ADX/DMI, Moving averages"
            },
            {
                "name": "layer_6_structure",
                "description": "CHoCH, BOS, Order blocks, FVG"
            },
            {
                "name": "layer_7_liquidity",
                "description": "Liquidity sweeps and hunt detection"
            },
            {
                "name": "layer_8_volatility_regime",
                "description": "ATR percentile, Volatility classification"
            },
            {
                "name": "layer_9_confirmation",
                "description": "Multi-timeframe confirmation system"
            },
            {
                "name": "layer_10_candle_intelligence",
                "description": "Advanced candlestick patterns"
            }
        ]
    }
