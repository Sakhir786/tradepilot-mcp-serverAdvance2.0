"""
TradePilot Integration - Main Application
==========================================
Unified entry point that integrates all 18 layers with:
- Advanced API router
- Multi-ticker scanner
- Alert notifications
- Risk management
- AI-ready output

This is the main file to run for the complete TradePilot system.

Usage:
    python main_18layer.py
    
    or with uvicorn:
    uvicorn main_18layer:app --host 0.0.0.0 --port 10000 --reload

Author: TradePilot Integration
"""

import os
import sys
from typing import Optional
from datetime import datetime

# Add paths
INTEGRATION_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, INTEGRATION_PATH)
sys.path.insert(0, os.path.join(INTEGRATION_PATH, 'integrations'))
sys.path.insert(0, os.path.join(INTEGRATION_PATH, 'scanners'))
sys.path.insert(0, os.path.join(INTEGRATION_PATH, 'alerts'))
sys.path.insert(0, os.path.join(INTEGRATION_PATH, 'risk'))

# Production path (for polygon_client and base layers)
PRODUCTION_PATH = os.environ.get('TRADEPILOT_PRODUCTION_PATH', '/home/claude/production/tradepilot-mcp-server-main')
sys.path.insert(0, PRODUCTION_PATH)
sys.path.insert(0, os.path.join(PRODUCTION_PATH, 'tradepilot_engine'))

# Layers path
LAYERS_PATH = os.environ.get('TRADEPILOT_LAYERS_PATH', '/home/claude/layers/tradepilot-mcp-serverAdvance2.0-main')
sys.path.insert(0, LAYERS_PATH)

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import production components
try:
    from polygon_client import (
        get_symbol_lookup, get_candles, get_options_chain,
        get_news, get_last_trade, get_ticker_details,
        get_fundamentals, get_previous_day_bar, get_single_stock_snapshot,
        get_all_option_contracts, get_option_aggregates,
        get_option_previous_day_bar, get_option_contract_snapshot,
        get_option_chain_snapshot
    )
    POLYGON_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Polygon client not available: {e}")
    POLYGON_AVAILABLE = False

# Import integration components
try:
    from integrations.router_18layer import router as engine18_router
    ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: 18-layer router not available: {e}")
    ROUTER_AVAILABLE = False

try:
    from integrations.engine_18layer_core import TradePilotEngine18Layer, TradeMode
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: 18-layer engine not available: {e}")
    ENGINE_AVAILABLE = False

try:
    from scanners.multi_ticker_scanner import TradePilotScanner, ScannerMode
    SCANNER_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Scanner not available: {e}")
    SCANNER_AVAILABLE = False

try:
    from alerts.notification_system import TradePilotAlerts, NotificationChannel
    ALERTS_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Alerts not available: {e}")
    ALERTS_AVAILABLE = False

try:
    from risk.risk_manager import TradePilotRiskManager, RiskProfile
    RISK_AVAILABLE = True
except ImportError as e:
    print(f"[Main] Warning: Risk manager not available: {e}")
    RISK_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="TradePilot 18-Layer MCP Server",
    description="""
    🎯 **Advanced Options Trading Intelligence Engine**
    
    Complete 18-layer technical analysis system with:
    - 14 high-probability playbooks (7 bullish + 7 bearish)
    - Target 85-95% win rates
    - SCALP (0-2 DTE) and SWING (7-45 DTE) modes
    - Real-time options chain analysis
    - Risk-adjusted position sizing
    - AI-ready JSON output
    
    **Layers:**
    - 1-10: Technical indicators (Momentum, Volume, Trend, Structure)
    - 11-13: Price action (S/R, VWAP, Volume Profile)
    - 14-17: Options analysis (IV, Greeks, Gamma, Put/Call)
    - 18: Master Brain Decision Engine
    """,
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include 18-layer router
if ROUTER_AVAILABLE:
    app.include_router(engine18_router)

# Global instances
_engine: Optional[TradePilotEngine18Layer] = None
_scanner: Optional[TradePilotScanner] = None
_alerts: Optional[TradePilotAlerts] = None
_risk_mgr: Optional[TradePilotRiskManager] = None


def get_engine():
    """Get or create engine instance"""
    global _engine
    if _engine is None and ENGINE_AVAILABLE:
        _engine = TradePilotEngine18Layer()
    return _engine


def get_scanner():
    """Get or create scanner instance"""
    global _scanner
    if _scanner is None and SCANNER_AVAILABLE:
        _scanner = TradePilotScanner()
    return _scanner


def get_alerts():
    """Get or create alerts instance"""
    global _alerts
    if _alerts is None and ALERTS_AVAILABLE:
        _alerts = TradePilotAlerts()
        _alerts.add_channel(NotificationChannel.CONSOLE)
        
        # Add Discord if configured
        discord_url = os.environ.get('TRADEPILOT_DISCORD_WEBHOOK')
        if discord_url:
            _alerts.add_channel(NotificationChannel.DISCORD, webhook_url=discord_url)
    
    return _alerts


def get_risk_manager():
    """Get or create risk manager instance"""
    global _risk_mgr
    if _risk_mgr is None and RISK_AVAILABLE:
        portfolio_value = float(os.environ.get('TRADEPILOT_PORTFOLIO_VALUE', '100000'))
        _risk_mgr = TradePilotRiskManager(
            portfolio_value=portfolio_value,
            risk_profile=RiskProfile.MODERATE
        )
    return _risk_mgr


# ==================== ROOT ENDPOINTS ====================

@app.get("/")
def root():
    """Root endpoint with system status"""
    return {
        "message": "🎯 TradePilot 18-Layer MCP Server v3.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "engine": "18-layer technical analysis system",
        "documentation": "/docs",
        "endpoints": {
            "18_layer_analysis": "/engine18/analyze",
            "quick_signal": "/engine18/quick",
            "scanner": "/engine18/scan",
            "playbooks": "/engine18/playbooks",
            "health": "/engine18/health",
            "ai_integration": "/engine18/ai-prompt"
        },
        "components": {
            "polygon_client": POLYGON_AVAILABLE,
            "engine_18layer": ENGINE_AVAILABLE,
            "router_18layer": ROUTER_AVAILABLE,
            "scanner": SCANNER_AVAILABLE,
            "alerts": ALERTS_AVAILABLE,
            "risk_manager": RISK_AVAILABLE
        }
    }


@app.get("/health")
def system_health():
    """Complete system health check"""
    engine = get_engine()
    scanner = get_scanner()
    alerts = get_alerts()
    risk_mgr = get_risk_manager()
    
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0",
        "components": {
            "polygon_api": {
                "available": POLYGON_AVAILABLE,
                "api_key_set": bool(os.environ.get('POLYGON_API_KEY'))
            },
            "engine_18layer": {
                "available": ENGINE_AVAILABLE,
                "layers_loaded": len(engine._layers) if engine else 0,
                "brain_available": engine._layer_18_brain is not None if engine else False
            },
            "scanner": {
                "available": SCANNER_AVAILABLE,
                "watchlists": list(scanner._watchlists.keys()) if scanner else []
            },
            "alerts": {
                "available": ALERTS_AVAILABLE,
                "channels_configured": len(alerts._channels) if alerts else 0
            },
            "risk_manager": {
                "available": RISK_AVAILABLE,
                "profile": risk_mgr.risk_profile.value if risk_mgr else None,
                "portfolio_value": risk_mgr.portfolio_value if risk_mgr else 0
            }
        },
        "playbooks": {
            "total": 14,
            "bullish": 7,
            "bearish": 7
        }
    }
    
    # Check overall health
    critical_components = [POLYGON_AVAILABLE, ENGINE_AVAILABLE]
    if not all(critical_components):
        health["status"] = "degraded"
    
    return health


# ==================== POLYGON ENDPOINTS (from production) ====================

if POLYGON_AVAILABLE:
    @app.get("/symbol-lookup")
    def symbol_lookup(query: str):
        """Search for ticker symbols"""
        return get_symbol_lookup(query)
    
    @app.get("/candles")
    def candles(symbol: str, tf: str = "day", limit: int = 730):
        """Fetch OHLCV candles"""
        return get_candles(symbol.upper(), tf=tf, limit=limit)
    
    @app.get("/news")
    def news(symbol: str):
        """Get latest news for symbol"""
        return get_news(symbol.upper())
    
    @app.get("/last-trade")
    def last_trade(symbol: str):
        """Get last trade for symbol"""
        return get_last_trade(symbol.upper())
    
    @app.get("/ticker-details")
    def ticker_details(symbol: str):
        """Get ticker details"""
        return get_ticker_details(symbol.upper())
    
    @app.get("/fundamentals")
    def fundamentals(symbol: str):
        """Get company fundamentals"""
        return get_fundamentals(symbol.upper())
    
    @app.get("/stock-snapshot/{ticker}")
    def stock_snapshot(ticker: str):
        """Get stock snapshot"""
        return get_single_stock_snapshot(ticker.upper())
    
    @app.get("/option-chain-snapshot/{underlying}")
    def option_chain_snapshot(underlying: str, limit: int = 50):
        """Get options chain snapshot"""
        return get_option_chain_snapshot(underlying.upper(), limit=limit)


# ==================== QUICK ANALYSIS ENDPOINTS ====================

@app.get("/quick-analyze")
async def quick_analyze(symbol: str = Query(..., description="Stock symbol")):
    """
    ⚡ Quick one-click analysis endpoint
    
    Returns essential trading signals without full analysis.
    Perfect for quick checks and alerts.
    """
    try:
        engine = get_engine()
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        if not POLYGON_AVAILABLE:
            raise HTTPException(status_code=503, detail="Polygon API not available")
        
        symbol = symbol.upper()
        candles_data = get_candles(symbol, tf="day", limit=200)
        
        if not candles_data or "results" not in candles_data:
            raise HTTPException(status_code=400, detail=f"No data for {symbol}")
        
        result = engine.analyze(
            ticker=symbol,
            candles_data=candles_data,
            mode=TradeMode.SWING
        )
        
        # Simplified response
        return {
            "ticker": symbol,
            "timestamp": datetime.now().isoformat(),
            "signal": {
                "valid": result.trade_valid,
                "direction": result.direction,
                "action": result.action,
                "confidence": result.confidence.value,
                "probability": result.win_probability
            },
            "playbook": result.matched_playbook,
            "option": {
                "strike": result.strike,
                "delta": result.delta,
                "dte": result.expiry_dte
            },
            "summary": f"{'🟢' if result.direction == 'BULLISH' else '🔴' if result.direction == 'BEARISH' else '⚪'} {result.action} | {result.confidence.value} ({result.win_probability:.0f}%)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scan-market")
async def scan_market(
    watchlist: str = Query("etfs", description="Watchlist to scan (etfs, tech, mega_caps, etc.)"),
    min_probability: float = Query(75, description="Minimum win probability")
):
    """
    🔍 Quick market scan endpoint
    
    Scans a predefined watchlist and returns top setups.
    """
    try:
        scanner = get_scanner()
        if not scanner:
            raise HTTPException(status_code=503, detail="Scanner not available")
        
        scanner.clear_filters()
        scanner.add_filter("win_probability", ">=", min_probability)
        
        try:
            summary = scanner.scan_watchlist(watchlist)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        return {
            "scan_id": summary.scan_id,
            "watchlist": watchlist,
            "duration_seconds": summary.duration_seconds,
            "setups_found": summary.setups_found,
            "top_setups": [
                {
                    "ticker": r.ticker,
                    "action": r.action,
                    "confidence": r.confidence,
                    "probability": r.win_probability,
                    "playbook": r.matched_playbook
                }
                for r in summary.results[:5]
            ],
            "available_watchlists": list(scanner._watchlists.keys())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RISK MANAGEMENT ENDPOINTS ====================

@app.get("/risk/metrics")
async def get_risk_metrics():
    """Get current portfolio risk metrics"""
    risk_mgr = get_risk_manager()
    if not risk_mgr:
        raise HTTPException(status_code=503, detail="Risk manager not available")
    
    metrics = risk_mgr.get_risk_metrics()
    
    return {
        "portfolio": {
            "total_value": metrics.total_portfolio_value,
            "cash_available": metrics.cash_available,
            "positions_value": metrics.positions_value
        },
        "exposure": {
            "total_pct": metrics.total_exposure_pct * 100,
            "bullish_pct": metrics.bullish_exposure_pct * 100,
            "bearish_pct": metrics.bearish_exposure_pct * 100,
            "net_pct": metrics.net_exposure_pct * 100,
            "remaining_capacity_pct": metrics.remaining_capacity_pct * 100
        },
        "positions": {
            "total": metrics.total_positions,
            "bullish": metrics.bullish_positions,
            "bearish": metrics.bearish_positions
        },
        "pnl": {
            "unrealized": metrics.unrealized_pnl,
            "unrealized_pct": metrics.unrealized_pnl_pct * 100,
            "daily": metrics.daily_pnl
        },
        "risk": {
            "current_drawdown_pct": metrics.current_drawdown_pct * 100,
            "max_drawdown_pct": metrics.max_drawdown_pct * 100,
            "risk_score": metrics.overall_risk_score,
            "concentration_risk": metrics.concentration_risk
        }
    }


@app.get("/risk/position-size")
async def calculate_position_size(
    symbol: str = Query(..., description="Stock symbol"),
    option_price: float = Query(..., description="Option contract price"),
    mode: str = Query("SWING", description="Trading mode (SCALP/SWING)")
):
    """Calculate recommended position size for a trade"""
    try:
        engine = get_engine()
        risk_mgr = get_risk_manager()
        
        if not engine or not risk_mgr:
            raise HTTPException(status_code=503, detail="Engine or risk manager not available")
        
        if not POLYGON_AVAILABLE:
            raise HTTPException(status_code=503, detail="Polygon API not available")
        
        symbol = symbol.upper()
        candles_data = get_candles(symbol, tf="day", limit=200)
        
        if not candles_data or "results" not in candles_data:
            raise HTTPException(status_code=400, detail=f"No data for {symbol}")
        
        trade_mode = TradeMode.SCALP if mode.upper() == "SCALP" else TradeMode.SWING
        
        result = engine.analyze(
            ticker=symbol,
            candles_data=candles_data,
            mode=trade_mode
        )
        
        sizing = risk_mgr.calculate_position_size(result, option_price)
        
        return {
            "ticker": symbol,
            "option_price": option_price,
            "analysis": {
                "direction": result.direction,
                "confidence": result.confidence.value,
                "probability": result.win_probability
            },
            "recommended": {
                "contracts": sizing.recommended_contracts,
                "cost": sizing.recommended_cost,
                "pct_of_portfolio": sizing.recommended_pct_of_portfolio * 100
            },
            "limits": {
                "max_contracts": sizing.max_contracts_allowed,
                "max_cost": sizing.max_cost_allowed
            },
            "kelly": {
                "fraction": sizing.kelly_fraction * 100,
                "contracts": sizing.kelly_contracts
            },
            "risk": {
                "per_contract": sizing.risk_per_contract,
                "total": sizing.total_risk
            },
            "reasoning": sizing.reasoning,
            "warnings": sizing.warnings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== SSE ENDPOINT ====================

@app.get("/sse")
async def sse():
    """Server-Sent Events endpoint"""
    async def event_generator():
        yield "data: TradePilot 18-Layer MCP Server connected\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("\n" + "=" * 60)
    print("🎯 TradePilot 18-Layer MCP Server Starting...")
    print("=" * 60)
    
    # Initialize components
    engine = get_engine()
    scanner = get_scanner()
    alerts = get_alerts()
    risk_mgr = get_risk_manager()
    
    print(f"\n📊 Components Status:")
    print(f"   • Polygon API: {'✅' if POLYGON_AVAILABLE else '❌'}")
    print(f"   • 18-Layer Engine: {'✅' if engine else '❌'}")
    print(f"   • Scanner: {'✅' if scanner else '❌'}")
    print(f"   • Alerts: {'✅' if alerts else '❌'}")
    print(f"   • Risk Manager: {'✅' if risk_mgr else '❌'}")
    
    if engine:
        print(f"\n🔧 Engine Details:")
        print(f"   • Layers loaded: {len(engine._layers)}")
        print(f"   • Brain available: {engine._layer_18_brain is not None}")
    
    print("\n" + "=" * 60)
    print("✅ Server Ready!")
    print("📖 Documentation: http://localhost:10000/docs")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 TradePilot 18-Layer MCP Server Shutting Down...")
    
    # Stop alert worker if running
    alerts = get_alerts()
    if alerts:
        alerts.stop_async_worker()
    
    print("👋 Goodbye!\n")


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"\n🚀 Starting TradePilot 18-Layer MCP Server on {host}:{port}")
    
    uvicorn.run(
        "main_18layer:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
