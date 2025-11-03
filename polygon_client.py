import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.massive.com"

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
