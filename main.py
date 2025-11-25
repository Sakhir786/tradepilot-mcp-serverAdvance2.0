from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from polygon_client import (
    get_symbol_lookup,
    get_candles,
    get_options_chain,
    get_news,
    get_last_trade,
    get_ticker_details,
    get_fundamentals,
    get_previous_day_bar,
    get_single_stock_snapshot,
    get_all_option_contracts,
    get_option_aggregates,
    get_option_previous_day_bar,
    get_option_contract_snapshot,
    get_option_chain_snapshot,
)
from fastapi.openapi.utils import get_openapi
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add 18-layer integration
import sys
sys.path.insert(0, './tradepilot_integration')

try:
    from integrations.router_18layer import router as engine18_router
    ROUTER_18_AVAILABLE = True
except ImportError as e:
    print(f"[Main] 18-layer router not available: {e}")
    ROUTER_18_AVAILABLE = False

app = FastAPI(
    title="TradePilot MCP Server",
    description="18-layer trading intelligence engine powered by Polygon.io",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include 18-layer engine routes
if ROUTER_18_AVAILABLE:
    app.include_router(engine18_router)
    print("✅ 18-Layer Engine integrated")

# ---------------- Root ----------------
@app.get("/")
def root():
    return {
        "message": "TradePilot MCP Server v3.0",
        "status": "running",
        "engine": "18-layer technical analysis system",
        "documentation": "/docs",
        "health": "/engine18/health"
    }

# ---------------- Core endpoints ----------------
@app.get("/symbol-lookup")
def symbol_lookup(query: str):
    return get_symbol_lookup(query)

@app.get("/candles")
def candles(symbol: str, tf: str = "day", limit: int = 730):
    """Fetch up to 2 years of OHLCV candles (default 730 daily bars)."""
    return get_candles(symbol.upper(), tf=tf, limit=limit)

@app.get("/news")
def news(symbol: str):
    return get_news(symbol.upper())

@app.get("/last-trade")
def last_trade(symbol: str):
    return get_last_trade(symbol.upper())

@app.get("/ticker-details")
def ticker_details(symbol: str):
    return get_ticker_details(symbol.upper())

@app.get("/fundamentals")
def fundamentals(symbol: str):
    return get_fundamentals(symbol.upper())

# ---------------- Stock endpoints ----------------
@app.get("/previous-day-bar/{ticker}")
def previous_day_bar(ticker: str):
    return get_previous_day_bar(ticker.upper())

@app.get("/stock-snapshot/{ticker}")
def stock_snapshot(ticker: str):
    return get_single_stock_snapshot(ticker.upper())

# ---------------- Options endpoints with expiry filtering ----------------
def filter_by_expiry(results: list, expiry_bucket: str | None = None):
    """Filter options contracts between today and +2 years. Optionally narrow by bucket."""
    today = datetime.utcnow().date()
    max_date = today + timedelta(days=730)

    filtered = []
    for c in results:
        expiry = c.get("expiration_date")
        if not expiry:
            continue
        try:
            expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        except Exception:
            continue
        if today <= expiry_date <= max_date:
            filtered.append(c)

    if expiry_bucket:
        bucket_days = {
            "otd": 0,
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "365d": 365,
            "730d": 730,
        }
        if expiry_bucket in bucket_days:
            cutoff = today + timedelta(days=bucket_days[expiry_bucket])
            filtered = [
                c for c in filtered
                if datetime.strptime(c["expiration_date"], "%Y-%m-%d").date() <= cutoff
            ]

    return filtered

@app.get("/options")
def options(symbol: str,
            type: str = "call",
            days_out: int = 30,
            expiry_bucket: str | None = Query(None, enum=["otd","7d","30d","90d","365d","730d"])):
    chain = get_options_chain(symbol.upper(), option_type=type.lower(), days_out=days_out)
    if "results" in chain:
        chain["results"] = filter_by_expiry(chain["results"], expiry_bucket)
    return chain

@app.get("/all-option-contracts")
def all_option_contracts(underlying_ticker: str,
                         expiration_date: str | None = None,
                         limit: int = 50,
                         expiry_bucket: str | None = Query(None, enum=["otd","7d","30d","90d","365d","730d"])):
    """Fetch all option contracts for a given stock, with expiry filtering."""
    contracts = get_all_option_contracts(underlying_ticker.upper(), expiration_date, limit)
    if "results" in contracts:
        contracts["results"] = filter_by_expiry(contracts["results"], expiry_bucket)
    return contracts

@app.get("/option-aggregates/{options_ticker}")
def option_aggregates(options_ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str):
    return get_option_aggregates(options_ticker.upper(), multiplier, timespan, from_date, to_date)

@app.get("/option-previous-day-bar/{options_ticker}")
def option_previous_day_bar(options_ticker: str):
    return get_option_previous_day_bar(options_ticker.upper())

@app.get("/option-contract-snapshot/{underlying}/{contract}")
def option_contract_snapshot_route(underlying: str, contract: str):
    """Snapshot for a single option contract (requires both underlying + contract)."""
    result = get_option_contract_snapshot(underlying.upper(), contract.upper())
    if "error" in result:
        return JSONResponse(status_code=400, content=result)

    expiry = result.get("results", {}).get("expiration_date")
    if expiry:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        today = datetime.utcnow().date()
        if expiry_date < today or expiry_date > today + timedelta(days=730):
            return JSONResponse(status_code=400, content={"error": "Expired or too far contract"})
    return result

@app.get("/option-chain-snapshot/{underlying_asset}")
def option_chain_snapshot_route(underlying_asset: str,
                                expiry_bucket: str | None = Query(None, enum=["otd","7d","30d","90d","365d","730d"]),
                                cursor: str | None = None,
                                limit: int = 50):
    """Snapshot of full option chain for a stock (supports pagination)."""
    chain = get_option_chain_snapshot(underlying_asset.upper(), cursor=cursor, limit=limit)
    if "results" in chain:
        chain["results"] = filter_by_expiry(chain["results"], expiry_bucket)
    return chain

# ---------------- SSE ----------------
@app.get("/sse")
async def sse():
    async def event_generator():
        yield "data: TradePilot MCP Server connected\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ---------------- OpenAPI ----------------
@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    return JSONResponse(get_openapi(
        title=app.title,
        version="3.0.0",
        routes=app.routes
    ))
