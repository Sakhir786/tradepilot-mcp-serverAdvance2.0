import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.massive.com"
# MODE CONFIGURATION - Add near top of file after imports
MODE_DATA_CONFIG = {
    "scalp": {
        "multiplier": 5,
        "timespan": "minute",
        "limit": 1500,
        "lookback_days": 14,      # Calendar days to fetch (~5 trading days)
        "delay_minutes": 20,     # Developer tier delay
        "dte_range": (0, 2),
        "description": "5-minute bars for scalping"
    },
    "intraday": {
        "multiplier": 15,
        "timespan": "minute",
        "limit": 50000,
        "lookback_days": 90,     # ~60 trading days
        "delay_minutes": 20,
        "dte_range": (0, 5),
        "description": "15-minute bars for intraday"
    },
    "swing": {
        "multiplier": 1,
        "timespan": "day",
        "limit": 730,
        "lookback_days": 730,
        "delay_minutes": 0,
        "dte_range": (7, 45),
        "description": "Daily bars for swing trading"
    },
    "leaps": {
        "multiplier": 1,
        "timespan": "day",
        "limit": 730,
        "lookback_days": 730,    # ~2 years daily data
        "delay_minutes": 0,
        "dte_range": (180, 720),
        "description": "Weekly bars for LEAPS"
    }
}


def get_candles_for_mode(symbol: str, mode: str = "swing"):
    """
    Get OHLCV candles configured for specific trading mode.
    
    Args:
        symbol: Ticker symbol (e.g., 'SPY')
        mode: Trading mode - scalp, intraday, swing, or leaps
        
    Returns:
        Dict with candle data and mode config metadata
    """
    mode = mode.lower()
    if mode not in MODE_DATA_CONFIG:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(MODE_DATA_CONFIG.keys())}")
    
    config = MODE_DATA_CONFIG[mode]
    now_utc = datetime.utcnow()
    
    # Apply delay for intraday (Developer tier)
    if config["delay_minutes"] > 0:
        end_time = now_utc - timedelta(minutes=config["delay_minutes"])
    else:
        end_time = now_utc
    
    # Calculate start time
    start_time = end_time - timedelta(days=config["lookback_days"])
    
    # Format dates
    start = start_time.strftime("%Y-%m-%d")
    end = end_time.strftime("%Y-%m-%d")
    
    # Build URL with multiplier/timespan
    # Format: /v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/{config['multiplier']}/{config['timespan']}/{start}/{end}"
        f"?adjusted=true&sort=asc&limit={config['limit']}&apiKey={API_KEY}"
    )
    
    response = requests.get(url)
    data = response.json()
    
    # Add mode metadata
    data["_mode_config"] = {
        "mode": mode,
        "multiplier": config["multiplier"],
        "timespan": config["timespan"],
        "requested_limit": config["limit"],
        "actual_bars": data.get("resultsCount", 0),
        "date_range": {"start": start, "end": end},
        "dte_range": config["dte_range"]
    }
    
    # Handle delayed/empty responses
    if not data.get("results") or data.get("resultsCount", 0) == 0:
        data["status"] = "DELAYED"
        data["message"] = f"No candles available for {mode} mode - possible delay window"
    
    print(
        f"[Polygon] {symbol} | Mode={mode.upper()} | {config['multiplier']}{config['timespan'][0]} | "
        f"Bars={data.get('resultsCount', 0)}/{config['limit']} | {start} to {end}"
    )
    
    return data

# ---------------- Core endpoints ----------------

def get_symbol_lookup(query: str):
    url = f"{BASE_URL}/v3/reference/tickers?search={query}&active=true&apiKey={API_KEY}"
    return requests.get(url).json()


def get_candles(symbol: str, tf: str = "day", limit: int = 730):
    """
    Get OHLCV candles dynamically (default = 730 days ≈ 2 years).
    Applies a 20-minute offset for Developer-tier intraday requests
    to account for delayed Polygon data availability.
    """
    now_utc = datetime.utcnow()

    # Apply 20-min offset for intraday timeframes
    if tf in ["minute", "hour"]:
        end_time = now_utc - timedelta(minutes=20)
    else:
        end_time = now_utc

    # Compute start time based on timeframe and limit
    if tf == "minute":
        start_time = end_time - timedelta(minutes=limit)
    elif tf == "hour":
        start_time = end_time - timedelta(hours=limit)
    elif tf == "day":
        start_time = end_time - timedelta(days=limit)
    else:
        raise ValueError("Invalid timeframe provided")

    # Format to YYYY-MM-DD for Polygon API
    start = start_time.strftime("%Y-%m-%d")
    end = end_time.strftime("%Y-%m-%d")

    # Build Polygon Aggregates URL
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/{tf}/{start}/{end}"
        f"?adjusted=true&sort=asc&limit={limit}&apiKey={API_KEY}"
    )

    response = requests.get(url)
    data = response.json()

    # Gracefully handle delayed or empty responses
    if not data.get("results") or data.get("resultsCount", 0) == 0:
        data["status"] = "DELAYED"
        data["message"] = (
            "No recent candles yet — Developer tier delay window active (≈20 min)"
        )

    # Optional: Debug log (safe to remove in production)
    print(
        f"[Polygon Fetch] {symbol} | TF={tf} | Start={start} | End={end} | "
        f"Status={data.get('status', 'OK')}"
    )

    return data


def get_news(symbol: str):
    url = f"{BASE_URL}/v2/reference/news?ticker={symbol}&limit=5&apiKey={API_KEY}"
    return requests.get(url).json()


def get_last_trade(symbol: str):
    url = f"{BASE_URL}/v2/last/trade/{symbol}?apiKey={API_KEY}"
    return requests.get(url).json()


def get_ticker_details(symbol: str):
    url = f"{BASE_URL}/v3/reference/tickers/{symbol}?apiKey={API_KEY}"
    return requests.get(url).json()


def get_fundamentals(symbol: str):
    """
    Get latest company financials (quarterly).
    """
    url = f"{BASE_URL}/v2/reference/financials?ticker={symbol.upper()}&limit=1&apiKey={API_KEY}"
    return requests.get(url).json()


def get_previous_day_bar(ticker: str):
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/prev?apiKey={API_KEY}"
    return requests.get(url).json()


def get_single_stock_snapshot(ticker: str):
    url = f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}?apiKey={API_KEY}"
    return requests.get(url).json()

# ---------------- Options endpoints ----------------

def get_all_option_contracts(underlying_ticker: str, expiration_date: str | None = None, limit: int = 50):
    """
    List all option contracts for a given underlying.
    """
    url = f"{BASE_URL}/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&limit={limit}&apiKey={API_KEY}"
    if expiration_date:
        url += f"&expiration_date.gte={expiration_date}"
    return requests.get(url).json()


def get_options_chain(symbol: str, option_type: str = "call", days_out: int = 30):
    """
    Fetch filtered option contracts by type (call/put) and expiry window.
    """
    today = datetime.utcnow().date()
    target_date = today + timedelta(days=days_out)

    url = (
        f"{BASE_URL}/v3/reference/options/contracts?"
        f"underlying_ticker={symbol}&contract_type={option_type}&"
        f"expiration_date.gte={today}&expiration_date.lte={target_date}&limit=100&apiKey={API_KEY}"
    )
    return requests.get(url).json()


def get_option_aggregates(options_ticker: str, multiplier: int, timespan: str, from_date: str, to_date: str):
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{options_ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        f"?apiKey={API_KEY}"
    )
    return requests.get(url).json()


def get_option_previous_day_bar(options_ticker: str):
    url = f"{BASE_URL}/v2/aggs/ticker/{options_ticker}/prev?apiKey={API_KEY}"
    return requests.get(url).json()


def get_option_chain_snapshot(underlying_asset: str, cursor: str | None = None, limit: int = 50):
    """
    Get paginated option chain snapshot for an underlying.
    """
    url = f"{BASE_URL}/v3/snapshot/options/{underlying_asset}?limit={limit}&apiKey={API_KEY}"
    if cursor:
        url += f"&cursor={cursor}"
    return requests.get(url).json()


def get_option_contract_snapshot(underlying: str, contract: str):
    """
    Snapshot for a single option contract.
    """
    url = f"{BASE_URL}/v3/snapshot/options/{underlying}/{contract}?apiKey={API_KEY}"
    return requests.get(url).json()


def get_full_option_chain_snapshot(underlying_asset: str, limit: int = 100, min_dte: int = 1):
    """
    Get full option chain snapshot with BOTH calls and puts.
    Filters strikes around current price for relevance.
    """
    from datetime import datetime, timedelta
    
    # Get tomorrow's date to filter out 0DTE
    tomorrow = (datetime.now() + timedelta(days=min_dte)).strftime('%Y-%m-%d')
    
    # Get current price first
    current_price = 0
    try:
        price_resp = requests.get(f"{BASE_URL}/v2/aggs/ticker/{underlying_asset}/prev?apiKey={API_KEY}").json()
        current_price = price_resp.get("results", [{}])[0].get("c", 0)
    except:
        pass
    
    # Fallback prices
    if not current_price or current_price == 0:
        defaults = {"SPY": 675, "QQQ": 500, "AAPL": 230, "TSLA": 350, "NVDA": 140, "AMD": 140}
        current_price = defaults.get(underlying_asset.upper(), 100)
    
    # Filter strikes within 10% of current price
    strike_low = int(current_price * 0.90)
    strike_high = int(current_price * 1.10)
    
    base_params = f"expiration_date.gte={tomorrow}&strike_price.gte={strike_low}&strike_price.lte={strike_high}&limit={limit}&apiKey={API_KEY}"
    
    calls_url = f"{BASE_URL}/v3/snapshot/options/{underlying_asset}?contract_type=call&{base_params}"
    puts_url = f"{BASE_URL}/v3/snapshot/options/{underlying_asset}?contract_type=put&{base_params}"
    
    calls_resp = requests.get(calls_url).json()
    puts_resp = requests.get(puts_url).json()
    
    results = []
    if calls_resp.get("results"):
        results.extend(calls_resp["results"])
    if puts_resp.get("results"):
        results.extend(puts_resp["results"])
    
    return {
        "status": "OK" if results else "ERROR",
        "results": results,
        "call_count": len(calls_resp.get("results", [])),
        "put_count": len(puts_resp.get("results", [])),
        "current_price": current_price,
        "strike_range": [strike_low, strike_high]
    }

def get_spread_options(underlying_asset: str, strategy: str = "bull_call_spread", 
                       dte_target: int = 30, width: int = 5, limit: int = 250):
    """
    Fetch options for spread strategies.
    
    Strategies:
    - bull_call_spread (debit): Buy ATM call, Sell OTM call
    - bear_put_spread (debit): Buy ATM put, Sell OTM put  
    - bull_put_spread (credit): Sell ATM put, Buy OTM put
    - bear_call_spread (credit): Sell ATM call, Buy OTM call
    - iron_condor: Sell OTM put + call, Buy further OTM put + call
    
    Args:
        width: Strike width between legs (e.g., 5 = $5 apart)
        dte_target: Target days to expiration
    """
    from datetime import datetime, timedelta
    
    # Get current price with fallbacks
    current_price = 0
    try:
        price_resp = requests.get(f"{BASE_URL}/v2/aggs/ticker/{underlying_asset}/prev?apiKey={API_KEY}").json()
        current_price = price_resp.get("results", [{}])[0].get("c", 0)
    except:
        pass
    
    if not current_price:
        defaults = {"SPY": 675, "QQQ": 500, "AAPL": 230, "TSLA": 350, "NVDA": 140, "AMD": 140}
        current_price = defaults.get(underlying_asset.upper(), 100)
    
    # Calculate date range around target DTE
    today = datetime.now()
    min_exp = (today + timedelta(days=max(1, dte_target - 7))).strftime('%Y-%m-%d')
    max_exp = (today + timedelta(days=dte_target + 14)).strftime('%Y-%m-%d')
    
    # Strike range for spreads (ATM ± 10%)
    strike_low = int(current_price * 0.92)
    strike_high = int(current_price * 1.08)
    
    params = f"expiration_date.gte={min_exp}&expiration_date.lte={max_exp}"
    params += f"&strike_price.gte={strike_low}&strike_price.lte={strike_high}"
    params += f"&limit={limit}&apiKey={API_KEY}"
    
    # Fetch both calls and puts for flexibility
    calls_resp = requests.get(f"{BASE_URL}/v3/snapshot/options/{underlying_asset}?contract_type=call&{params}").json()
    puts_resp = requests.get(f"{BASE_URL}/v3/snapshot/options/{underlying_asset}?contract_type=put&{params}").json()
    
    calls = calls_resp.get("results", [])
    puts = puts_resp.get("results", [])
    
    # Find optimal spread based on strategy
    spread_result = _find_optimal_spread(calls, puts, current_price, strategy, width, dte_target)
    
    return {
        "status": "OK",
        "strategy": strategy,
        "current_price": current_price,
        "dte_target": dte_target,
        "width": width,
        "spread": spread_result,
        "available_calls": len(calls),
        "available_puts": len(puts)
    }


def _find_optimal_spread(calls, puts, current_price, strategy, width, dte_target):
    """Find the optimal spread contracts for a given strategy."""
    from datetime import datetime
    
    def get_contract_info(c):
        d = c.get("details", {})
        g = c.get("greeks", {})
        q = c.get("last_quote", {})
        day = c.get("day", {})
        exp = d.get("expiration_date", "")
        try:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days
        except:
            dte = 0
        # Use last_quote if available, otherwise use day data
        bid = q.get("bid") or day.get("close", 0) * 0.98  # Estimate bid as 98% of close
        ask = q.get("ask") or day.get("close", 0) * 1.02  # Estimate ask as 102% of close
        mid = (bid + ask) / 2 if bid and ask else day.get("vwap", day.get("close", 0))
        return {
            "ticker": d.get("ticker"),
            "strike": d.get("strike_price", 0),
            "expiry": exp,
            "dte": dte,
            "type": d.get("contract_type"),
            "delta": g.get("delta", 0),
            "theta": g.get("theta", 0),
            "iv": c.get("implied_volatility", 0),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "mid": round(mid, 2),
            "oi": c.get("open_interest", 0)
        }
    
    # Process contracts
    call_list = [get_contract_info(c) for c in calls if c.get("details", {}).get("strike_price")]
    put_list = [get_contract_info(p) for p in puts if p.get("details", {}).get("strike_price")]
    
    # Sort by strike
    call_list.sort(key=lambda x: x["strike"])
    put_list.sort(key=lambda x: x["strike"])
    
    # Find ATM strike
    atm_strike = round(current_price)
    
    result = {"legs": [], "metrics": {}}
    
    if strategy == "bull_call_spread":
        # Buy lower strike call (ATM), Sell higher strike call (OTM)
        long_calls = [c for c in call_list if c["strike"] <= atm_strike and c["dte"] >= dte_target - 7]
        short_calls = [c for c in call_list if c["strike"] >= atm_strike + width and c["dte"] >= dte_target - 7]
        
        if long_calls and short_calls:
            long_leg = max(long_calls, key=lambda x: x["strike"])  # Closest to ATM
            # Find matching expiry
            short_candidates = [c for c in short_calls if c["expiry"] == long_leg["expiry"]]
            if short_candidates:
                short_leg = min(short_candidates, key=lambda x: x["strike"])  # Closest OTM
                
                max_profit = (short_leg["strike"] - long_leg["strike"]) * 100
                net_debit = (long_leg["ask"] - short_leg["bid"]) * 100
                max_loss = net_debit
                
                result = {
                    "type": "DEBIT",
                    "direction": "BULLISH",
                    "legs": [
                        {"action": "BUY", "contract": long_leg},
                        {"action": "SELL", "contract": short_leg}
                    ],
                    "metrics": {
                        "net_debit": round(net_debit, 2),
                        "max_profit": round(max_profit - net_debit, 2),
                        "max_loss": round(max_loss, 2),
                        "breakeven": round(long_leg["strike"] + net_debit/100, 2),
                        "risk_reward": round((max_profit - net_debit) / net_debit, 2) if net_debit > 0 else 0
                    }
                }
    
    elif strategy == "bear_put_spread":
        # Buy higher strike put (ATM), Sell lower strike put (OTM)
        long_puts = [p for p in put_list if p["strike"] >= atm_strike and p["dte"] >= dte_target - 7]
        short_puts = [p for p in put_list if p["strike"] <= atm_strike - width and p["dte"] >= dte_target - 7]
        
        if long_puts and short_puts:
            long_leg = min(long_puts, key=lambda x: x["strike"])  # Closest to ATM
            short_candidates = [p for p in short_puts if p["expiry"] == long_leg["expiry"]]
            if short_candidates:
                short_leg = max(short_candidates, key=lambda x: x["strike"])  # Closest OTM
                
                max_profit = (long_leg["strike"] - short_leg["strike"]) * 100
                net_debit = (long_leg["ask"] - short_leg["bid"]) * 100
                
                result = {
                    "type": "DEBIT",
                    "direction": "BEARISH",
                    "legs": [
                        {"action": "BUY", "contract": long_leg},
                        {"action": "SELL", "contract": short_leg}
                    ],
                    "metrics": {
                        "net_debit": round(net_debit, 2),
                        "max_profit": round(max_profit - net_debit, 2),
                        "max_loss": round(net_debit, 2),
                        "breakeven": round(long_leg["strike"] - net_debit/100, 2),
                        "risk_reward": round((max_profit - net_debit) / net_debit, 2) if net_debit > 0 else 0
                    }
                }
    
    elif strategy == "bull_put_spread":
        # Sell higher strike put (ATM), Buy lower strike put (OTM) - CREDIT
        short_puts = [p for p in put_list if p["strike"] <= atm_strike and p["dte"] >= dte_target - 7]
        long_puts = [p for p in put_list if p["strike"] <= atm_strike - width and p["dte"] >= dte_target - 7]
        
        if short_puts and long_puts:
            short_leg = max(short_puts, key=lambda x: x["strike"])  # Highest OTM put to sell
            long_candidates = [p for p in long_puts if p["expiry"] == short_leg["expiry"] and p["strike"] < short_leg["strike"]]
            if long_candidates:
                long_leg = max(long_candidates, key=lambda x: x["strike"])  # Closest protection
                
                net_credit = (short_leg["bid"] - long_leg["ask"]) * 100
                max_loss = (short_leg["strike"] - long_leg["strike"]) * 100 - net_credit
                
                result = {
                    "type": "CREDIT",
                    "direction": "BULLISH",
                    "legs": [
                        {"action": "SELL", "contract": short_leg},
                        {"action": "BUY", "contract": long_leg}
                    ],
                    "metrics": {
                        "net_credit": round(net_credit, 2),
                        "max_profit": round(net_credit, 2),
                        "max_loss": round(max_loss, 2),
                        "breakeven": round(short_leg["strike"] - net_credit/100, 2),
                        "risk_reward": round(net_credit / max_loss, 2) if max_loss > 0 else 0
                    }
                }
    
    elif strategy == "bear_call_spread":
        # Sell lower strike call (ATM), Buy higher strike call (OTM) - CREDIT
        short_calls = [c for c in call_list if c["strike"] >= atm_strike and c["dte"] >= dte_target - 7]
        long_calls = [c for c in call_list if c["strike"] >= atm_strike + width and c["dte"] >= dte_target - 7]
        
        if short_calls and long_calls:
            short_leg = min(short_calls, key=lambda x: x["strike"])  # Lowest OTM call to sell
            long_candidates = [c for c in long_calls if c["expiry"] == short_leg["expiry"] and c["strike"] > short_leg["strike"]]
            if long_candidates:
                long_leg = min(long_candidates, key=lambda x: x["strike"])  # Closest protection
                
                net_credit = (short_leg["bid"] - long_leg["ask"]) * 100
                max_loss = (long_leg["strike"] - short_leg["strike"]) * 100 - net_credit
                
                result = {
                    "type": "CREDIT",
                    "direction": "BEARISH",
                    "legs": [
                        {"action": "SELL", "contract": short_leg},
                        {"action": "BUY", "contract": long_leg}
                    ],
                    "metrics": {
                        "net_credit": round(net_credit, 2),
                        "max_profit": round(net_credit, 2),
                        "max_loss": round(max_loss, 2),
                        "breakeven": round(short_leg["strike"] + net_credit/100, 2),
                        "risk_reward": round(net_credit / max_loss, 2) if max_loss > 0 else 0
                    }
                }
    
    return result


def get_single_option(underlying_asset: str, option_type: str = "call", 
                      dte_target: int = 30, delta_target: float = 0.50, limit: int = 100):
    """
    Find optimal single option (long call or long put).
    
    Args:
        option_type: "call" or "put"
        dte_target: Target days to expiration
        delta_target: Target delta (0.50 = ATM, 0.30 = OTM, 0.70 = ITM)
    """
    from datetime import datetime, timedelta
    
    # Get current price
    current_price = 0
    try:
        price_resp = requests.get(f"{BASE_URL}/v2/aggs/ticker/{underlying_asset}/prev?apiKey={API_KEY}").json()
        current_price = price_resp.get("results", [{}])[0].get("c", 0)
    except:
        pass
    
    if not current_price:
        defaults = {"SPY": 675, "QQQ": 500, "AAPL": 230, "TSLA": 350, "NVDA": 140}
        current_price = defaults.get(underlying_asset.upper(), 100)
    
    # Date range
    today = datetime.now()
    min_exp = (today + timedelta(days=max(1, dte_target - 10))).strftime('%Y-%m-%d')
    max_exp = (today + timedelta(days=dte_target + 14)).strftime('%Y-%m-%d')
    
    # Strike range (wider for finding target delta)
    strike_low = int(current_price * 0.85)
    strike_high = int(current_price * 1.15)
    
    params = f"contract_type={option_type}&expiration_date.gte={min_exp}&expiration_date.lte={max_exp}"
    params += f"&strike_price.gte={strike_low}&strike_price.lte={strike_high}"
    params += f"&limit={limit}&apiKey={API_KEY}"
    
    resp = requests.get(f"{BASE_URL}/v3/snapshot/options/{underlying_asset}?{params}").json()
    contracts = resp.get("results", [])
    
    if not contracts:
        return {"status": "ERROR", "error": "No contracts found"}
    
    # Process and find best match for target delta
    best_contract = None
    best_delta_diff = float('inf')
    
    for c in contracts:
        d = c.get("details", {})
        g = c.get("greeks", {})
        day = c.get("day", {})
        
        if not g.get("delta"):
            continue
        
        exp = d.get("expiration_date", "")
        try:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days
        except:
            continue
        
        if dte < 1:
            continue
        
        delta = abs(g.get("delta", 0))
        delta_diff = abs(delta - delta_target)
        
        # Also prefer closer to target DTE
        dte_diff = abs(dte - dte_target)
        score = delta_diff + (dte_diff * 0.01)  # Delta weighted more
        
        if score < best_delta_diff:
            best_delta_diff = score
            mid = day.get("vwap") or day.get("close", 0)
            best_contract = {
                "ticker": d.get("ticker"),
                "type": option_type,
                "strike": d.get("strike_price"),
                "expiry": exp,
                "dte": dte,
                "delta": g.get("delta"),
                "gamma": g.get("gamma"),
                "theta": g.get("theta"),
                "vega": g.get("vega"),
                "iv": c.get("implied_volatility"),
                "mid": round(mid, 2),
                "bid": round(day.get("close", 0) * 0.98, 2),
                "ask": round(day.get("close", 0) * 1.02, 2),
                "oi": c.get("open_interest", 0),
                "volume": day.get("volume", 0)
            }
    
    if not best_contract:
        return {"status": "ERROR", "error": "No suitable contract found"}
    
    # Calculate metrics
    premium = best_contract["mid"] * 100
    
    if option_type == "call":
        breakeven = best_contract["strike"] + best_contract["mid"]
        direction = "BULLISH"
    else:
        breakeven = best_contract["strike"] - best_contract["mid"]
        direction = "BEARISH"
    
    return {
        "status": "OK",
        "strategy": f"long_{option_type}",
        "direction": direction,
        "current_price": current_price,
        "contract": best_contract,
        "metrics": {
            "premium": round(premium, 2),
            "max_profit": "Unlimited" if option_type == "call" else round((best_contract["strike"] - best_contract["mid"]) * 100, 2),
            "max_loss": round(premium, 2),
            "breakeven": round(breakeven, 2),
            "delta_exposure": round(best_contract["delta"] * 100, 1)
        }
    }


def get_all_strategies(underlying_asset: str, dte_target: int = 30):
    """
    Get all available option strategies for a ticker.
    Returns long call, long put, and all 4 spreads.
    """
    results = {}
    
    # Single options
    print(f"Fetching long call...")
    results["long_call"] = get_single_option(underlying_asset, "call", dte_target, delta_target=0.50)
    
    print(f"Fetching long put...")
    results["long_put"] = get_single_option(underlying_asset, "put", dte_target, delta_target=0.50)
    
    # Spreads
    spreads = ["bull_call_spread", "bear_put_spread", "bull_put_spread", "bear_call_spread"]
    for strat in spreads:
        print(f"Fetching {strat}...")
        results[strat] = get_spread_options(underlying_asset, strat, dte_target, width=5)
    
    return results


def get_trade_recommendation(underlying_asset: str, dte_target: int = 30):
    """
    Complete trade recommendation with analysis + best strategy.
    This is what AI will use to make final recommendations.
    """
    from datetime import datetime
    
    # Get all strategies
    strategies = {
        "long_call": get_single_option(underlying_asset, "call", dte_target, 0.50),
        "long_put": get_single_option(underlying_asset, "put", dte_target, 0.50),
        "bull_call_spread": get_spread_options(underlying_asset, "bull_call_spread", dte_target, 5),
        "bear_put_spread": get_spread_options(underlying_asset, "bear_put_spread", dte_target, 5),
        "bull_put_spread": get_spread_options(underlying_asset, "bull_put_spread", dte_target, 5),
        "bear_call_spread": get_spread_options(underlying_asset, "bear_call_spread", dte_target, 5),
    }
    
    # Format for easy reading
    result = {
        "ticker": underlying_asset,
        "timestamp": datetime.now().isoformat(),
        "dte_target": dte_target,
        "strategies": {}
    }
    
    for name, data in strategies.items():
        if name.startswith("long_"):
            if data.get("contract"):
                c = data["contract"]
                m = data["metrics"]
                result["strategies"][name] = {
                    "direction": data["direction"],
                    "contract": f"{c['type'].upper()} ${c['strike']} exp:{c['expiry']}",
                    "cost": m["premium"],
                    "max_loss": m["max_loss"],
                    "breakeven": m["breakeven"],
                    "delta": c["delta"]
                }
        else:
            if data.get("spread") and data["spread"].get("legs"):
                s = data["spread"]
                m = s["metrics"]
                legs = " / ".join([f"{l['action']} ${l['contract']['strike']}" for l in s["legs"]])
                cost_key = "net_debit" if s["type"] == "DEBIT" else "net_credit"
                result["strategies"][name] = {
                    "type": s["type"],
                    "direction": s["direction"],
                    "legs": legs,
                    "cost": m.get(cost_key, 0),
                    "max_profit": m["max_profit"],
                    "max_loss": m["max_loss"],
                    "breakeven": m["breakeven"],
                    "risk_reward": m["risk_reward"]
                }
    
    return result


def analyze_with_strategies(symbol: str, mode: str = "swing"):
    """
    Complete analysis + all 6 strategies in one call.
    
    Mode: scalp (0-2 DTE), swing (7-45 DTE), leaps (180-400 DTE)
    
    Returns:
    - 18-layer analysis summary
    - 6 strategies with best contracts for the mode
    - AI recommendation based on direction
    """
    from datetime import datetime, timedelta
    import requests
    
    # Mode config
    mode_config = {
        "scalp": {"min_dte": 0, "max_dte": 2, "strike_pct": 0.05},
        "swing": {"min_dte": 7, "max_dte": 45, "strike_pct": 0.10},
        "leaps": {"min_dte": 180, "max_dte": 400, "strike_pct": 0.15},
    }
    
    config = mode_config.get(mode.lower(), mode_config["swing"])
    
    # Get current price
    current_price = 0
    try:
        price_resp = requests.get(f"{BASE_URL}/v2/aggs/ticker/{symbol}/prev?apiKey={API_KEY}").json()
        current_price = price_resp.get("results", [{}])[0].get("c", 0)
    except:
        pass
    
    if not current_price:
        defaults = {"SPY": 675, "QQQ": 500, "AAPL": 230, "TSLA": 350, "NVDA": 140, "AMD": 140}
        current_price = defaults.get(symbol.upper(), 100)
    
    # Date range for mode
    today = datetime.now()
    min_exp = (today + timedelta(days=config["min_dte"])).strftime('%Y-%m-%d')
    max_exp = (today + timedelta(days=config["max_dte"])).strftime('%Y-%m-%d')
    
    # Strike range
    strike_low = int(current_price * (1 - config["strike_pct"]))
    strike_high = int(current_price * (1 + config["strike_pct"]))
    
    # Fetch contracts for this mode
    params = f"expiration_date.gte={min_exp}&expiration_date.lte={max_exp}"
    params += f"&strike_price.gte={strike_low}&strike_price.lte={strike_high}"
    params += f"&limit=250&apiKey={API_KEY}"
    
    calls_resp = requests.get(f"{BASE_URL}/v3/snapshot/options/{symbol}?contract_type=call&{params}").json()
    puts_resp = requests.get(f"{BASE_URL}/v3/snapshot/options/{symbol}?contract_type=put&{params}").json()
    
    calls = calls_resp.get("results", [])
    puts = puts_resp.get("results", [])
    
    # Process contracts
    def process_contract(c):
        d = c.get("details", {})
        g = c.get("greeks", {})
        day = c.get("day", {})
        exp = d.get("expiration_date", "")
        try:
            dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days
        except:
            dte = 0
        
        price = day.get("vwap") or day.get("close", 0)
        return {
            "ticker": d.get("ticker"),
            "type": d.get("contract_type"),
            "strike": d.get("strike_price", 0),
            "expiry": exp,
            "dte": dte,
            "delta": round(g.get("delta", 0), 3),
            "gamma": round(g.get("gamma", 0), 5),
            "theta": round(g.get("theta", 0), 3),
            "vega": round(g.get("vega", 0), 3),
            "iv": round(c.get("implied_volatility", 0), 3),
            "price": round(price, 2),
            "oi": c.get("open_interest", 0),
            "volume": day.get("volume", 0)
        }
    
    call_list = [process_contract(c) for c in calls if c.get("details", {}).get("strike_price")]
    put_list = [process_contract(c) for c in puts if c.get("details", {}).get("strike_price")]
    
    # Filter valid contracts - require minimum price and liquidity
    call_list = [c for c in call_list if c["dte"] >= config["min_dte"] and c["price"] >= 0.50 and c["oi"] >= 100]
    put_list = [p for p in put_list if p["dte"] >= config["min_dte"] and p["price"] >= 0.50 and p["oi"] >= 100]
    
    # Sort by strike
    call_list.sort(key=lambda x: (x["expiry"], x["strike"]))
    put_list.sort(key=lambda x: (x["expiry"], x["strike"]))
    
    # Build 6 strategies
    strategies = {}
    
    # 1. LONG CALL - Find best ATM call (delta ~0.50)
    best_call = None
    best_call_diff = float('inf')
    for c in call_list:
        diff = abs(abs(c["delta"]) - 0.50)
        if diff < best_call_diff:
            best_call_diff = diff
            best_call = c
    
    if best_call:
        strategies["long_call"] = {
            "direction": "BULLISH",
            "type": "DEBIT",
            "contract": f"{best_call['type'].upper()} ${best_call['strike']} {best_call['expiry']}",
            "cost": round(best_call["price"] * 100, 0),
            "max_profit": "UNLIMITED",
            "max_loss": round(best_call["price"] * 100, 0),
            "breakeven": round(best_call["strike"] + best_call["price"], 2),
            "delta": best_call["delta"],
            "dte": best_call["dte"],
            "details": best_call
        }
    
    # 2. LONG PUT - Find best ATM put (delta ~-0.50)
    best_put = None
    best_put_diff = float('inf')
    for p in put_list:
        diff = abs(abs(p["delta"]) - 0.50)
        if diff < best_put_diff:
            best_put_diff = diff
            best_put = p
    
    if best_put:
        strategies["long_put"] = {
            "direction": "BEARISH",
            "type": "DEBIT",
            "contract": f"{best_put['type'].upper()} ${best_put['strike']} {best_put['expiry']}",
            "cost": round(best_put["price"] * 100, 0),
            "max_profit": round((best_put["strike"] - best_put["price"]) * 100, 0),
            "max_loss": round(best_put["price"] * 100, 0),
            "breakeven": round(best_put["strike"] - best_put["price"], 2),
            "delta": best_put["delta"],
            "dte": best_put["dte"],
            "details": best_put
        }
    
    # Helper to find matching expiry contracts
    def find_spread_legs(contracts, is_call, is_bull, width=5):
        # Group by expiry
        by_expiry = {}
        for c in contracts:
            if c["expiry"] not in by_expiry:
                by_expiry[c["expiry"]] = []
            by_expiry[c["expiry"]].append(c)
        
        best_spread = None
        best_score = -999
        
        for expiry, exp_contracts in by_expiry.items():
            exp_contracts.sort(key=lambda x: x["strike"])
            
            for i, long_leg in enumerate(exp_contracts):
                for short_leg in exp_contracts:
                    if is_bull:
                        # Bull: buy lower, sell higher
                        if short_leg["strike"] <= long_leg["strike"]:
                            continue
                        spread_width = short_leg["strike"] - long_leg["strike"]
                    else:
                        # Bear: buy higher, sell lower
                        if short_leg["strike"] >= long_leg["strike"]:
                            continue
                        spread_width = long_leg["strike"] - short_leg["strike"]
                    
                    # Prefer $5 spreads, allow $1-$10
                    if spread_width < 1 or spread_width > 10:
                        continue
                    
                    # Calculate debit/credit
                    if is_bull == is_call:
                        # Bull Call or Bear Put = DEBIT
                        net = long_leg["price"] - short_leg["price"]
                    else:
                        # Bear Call or Bull Put = CREDIT
                        net = short_leg["price"] - long_leg["price"]
                    
                    if net <= 0 or net < 0.50:
                        continue
                    
                    # Score: PRIORITIZE ATM (delta), then R:R, then liquidity
                    # Skip if too far OTM (delta too low)
                    if abs(long_leg["delta"]) < 0.15:
                        continue
                    # Skip low liquidity
                    if long_leg["volume"] < 10 or short_leg["volume"] < 10:
                        continue
                    
                    atm_diff = abs(long_leg["strike"] - current_price)
                    delta_score = 100 - abs(abs(long_leg["delta"]) - 0.50) * 200  # Best at 0.50
                    rr = (spread_width - net) / net if net > 0 else 0
                    score = delta_score + (rr * 10) - (atm_diff / 10)
                    
                    if score > best_score:
                        best_score = score
                        best_spread = {
                            "long": long_leg,
                            "short": short_leg,
                            "width": spread_width,
                            "net": net,
                            "rr": rr
                        }
        
        return best_spread
    
    # 3. BULL CALL SPREAD - Buy call, sell higher call
    bull_call = find_spread_legs(call_list, is_call=True, is_bull=True)
    if bull_call:
        net_debit = bull_call["net"] * 100
        max_profit = (bull_call["width"] * 100) - net_debit
        strategies["bull_call_spread"] = {
            "direction": "BULLISH",
            "type": "DEBIT",
            "legs": f"BUY ${bull_call['long']['strike']} / SELL ${bull_call['short']['strike']}",
            "expiry": bull_call["long"]["expiry"],
            "cost": round(net_debit, 0),
            "max_profit": round(max_profit, 0),
            "max_loss": round(net_debit, 0),
            "breakeven": round(bull_call["long"]["strike"] + bull_call["net"], 2),
            "risk_reward": round(max_profit / net_debit, 2) if net_debit > 0 else 0,
            "dte": bull_call["long"]["dte"],
            "details": {"long": bull_call["long"], "short": bull_call["short"]}
        }
    
    # 4. BEAR PUT SPREAD - Buy put, sell lower put
    bear_put = find_spread_legs(put_list, is_call=False, is_bull=False)
    if bear_put:
        net_debit = bear_put["net"] * 100
        max_profit = (bear_put["width"] * 100) - net_debit
        strategies["bear_put_spread"] = {
            "direction": "BEARISH",
            "type": "DEBIT",
            "legs": f"BUY ${bear_put['long']['strike']} / SELL ${bear_put['short']['strike']}",
            "expiry": bear_put["long"]["expiry"],
            "cost": round(net_debit, 0),
            "max_profit": round(max_profit, 0),
            "max_loss": round(net_debit, 0),
            "breakeven": round(bear_put["long"]["strike"] - bear_put["net"], 2),
            "risk_reward": round(max_profit / net_debit, 2) if net_debit > 0 else 0,
            "dte": bear_put["long"]["dte"],
            "details": {"long": bear_put["long"], "short": bear_put["short"]}
        }
    
    # 5. BULL PUT SPREAD (Credit) - Sell OTM put, buy lower put for protection
    # Short leg should be BELOW current price (OTM)
    bull_put = None
    best_score = -999
    by_expiry = {}
    for p in put_list:
        if p["expiry"] not in by_expiry:
            by_expiry[p["expiry"]] = []
        by_expiry[p["expiry"]].append(p)
    
    for expiry, exp_puts in by_expiry.items():
        exp_puts.sort(key=lambda x: x["strike"])
        for short_leg in exp_puts:
            # SHORT LEG MUST BE OTM (below current price)
            if short_leg["strike"] >= current_price:
                continue
            # Skip if delta too high (too close to ATM)
            if abs(short_leg["delta"]) > 0.45:
                continue
            for long_leg in exp_puts:
                if long_leg["strike"] >= short_leg["strike"]:
                    continue
                width = short_leg["strike"] - long_leg["strike"]
                if width < 3 or width > 10:
                    continue
                credit = short_leg["price"] - long_leg["price"]
                if credit < 0.30:
                    continue
                max_loss = (width - credit) * 100
                if max_loss <= 0:
                    continue
                rr = (credit * 100) / max_loss
                # Score: prefer slightly OTM (delta ~0.30), good credit
                delta_score = 100 - abs(abs(short_leg["delta"]) - 0.30) * 200
                score = delta_score + (credit * 50)
                if score > best_score:
                    best_score = score
                    bull_put = {"short": short_leg, "long": long_leg, "width": width, "credit": credit, "rr": rr}
    
    if bull_put:
        net_credit = bull_put["credit"] * 100
        max_loss = (bull_put["width"] * 100) - net_credit
        strategies["bull_put_spread"] = {
            "direction": "BULLISH",
            "type": "CREDIT",
            "legs": f"SELL ${bull_put['short']['strike']} / BUY ${bull_put['long']['strike']}",
            "expiry": bull_put["short"]["expiry"],
            "credit": round(net_credit, 0),
            "max_profit": round(net_credit, 0),
            "max_loss": round(max_loss, 0),
            "breakeven": round(bull_put["short"]["strike"] - bull_put["credit"], 2),
            "risk_reward": round(net_credit / max_loss, 2) if max_loss > 0 else 0,
            "dte": bull_put["short"]["dte"],
            "details": {"short": bull_put["short"], "long": bull_put["long"]}
        }
    
    # 6. BEAR CALL SPREAD (Credit) - Sell OTM call, buy higher call for protection
    # Short leg should be ABOVE current price (OTM)
    bear_call = None
    best_score = -999
    by_expiry = {}
    for c in call_list:
        if c["expiry"] not in by_expiry:
            by_expiry[c["expiry"]] = []
        by_expiry[c["expiry"]].append(c)
    
    for expiry, exp_calls in by_expiry.items():
        exp_calls.sort(key=lambda x: x["strike"])
        for short_leg in exp_calls:
            # SHORT LEG MUST BE OTM (above current price)
            if short_leg["strike"] <= current_price:
                continue
            # Skip if delta too high (too close to ATM)
            if abs(short_leg["delta"]) > 0.45:
                continue
            for long_leg in exp_calls:
                if long_leg["strike"] <= short_leg["strike"]:
                    continue
                width = long_leg["strike"] - short_leg["strike"]
                if width < 3 or width > 10:
                    continue
                credit = short_leg["price"] - long_leg["price"]
                if credit < 0.30:
                    continue
                max_loss = (width - credit) * 100
                if max_loss <= 0:
                    continue
                rr = (credit * 100) / max_loss
                # Score: prefer slightly OTM (delta ~0.30), good credit
                delta_score = 100 - abs(abs(short_leg["delta"]) - 0.30) * 200
                score = delta_score + (credit * 50)
                if score > best_score:
                    best_score = score
                    bear_call = {"short": short_leg, "long": long_leg, "width": width, "credit": credit, "rr": rr}
    
    if bear_call:
        net_credit = bear_call["credit"] * 100
        max_loss = (bear_call["width"] * 100) - net_credit
        strategies["bear_call_spread"] = {
            "direction": "BEARISH",
            "type": "CREDIT",
            "legs": f"SELL ${bear_call['short']['strike']} / BUY ${bear_call['long']['strike']}",
            "expiry": bear_call["short"]["expiry"],
            "credit": round(net_credit, 0),
            "max_profit": round(net_credit, 0),
            "max_loss": round(max_loss, 0),
            "breakeven": round(bear_call["short"]["strike"] + bear_call["credit"], 2),
            "risk_reward": round(net_credit / max_loss, 2) if max_loss > 0 else 0,
            "dte": bear_call["short"]["dte"],
            "details": {"short": bear_call["short"], "long": bear_call["long"]}
        }
    
    return {
        "symbol": symbol.upper(),
        "mode": mode.upper(),
        "current_price": current_price,
        "timestamp": datetime.now().isoformat(),
        "dte_range": [config["min_dte"], config["max_dte"]],
        "contracts_fetched": {"calls": len(call_list), "puts": len(put_list)},
        "strategies": strategies
    }
