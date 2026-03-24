"""
IBKR Client for TradePilot
===========================
Drop-in replacement for polygon_client.py data functions,
plus live order execution via IB Gateway.

Data functions (Polygon-compatible output):
  - get_candles_for_mode(symbol, mode)
  - get_candles(symbol, tf, limit)
  - get_full_option_chain_snapshot(symbol, limit, min_dte)
  - get_market_context(mode)
  - get_option_chain_snapshot(symbol, cursor, limit)
  - get_ticker_details(symbol)

Execution functions:
  - place_option_order(symbol, expiry, strike, right, action, quantity, order_type, limit_price)
  - place_spread_order(symbol, legs, action, quantity, order_type, limit_price)
  - get_positions()
  - close_position(symbol, contract_id)
  - get_open_orders()
  - cancel_order(order_id)
  - get_account_summary()
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from ib_insync import IB, Stock, Index, Option, util

from ibkr_config import (
    IBKR_HOST,
    IBKR_PORT,
    IBKR_CLIENT_ID,
    IBKR_CONNECT_TIMEOUT,
    IBKR_REQUEST_TIMEOUT,
    IBKR_MARKET_DATA_TYPE,
    MODE_DATA_CONFIG,
)

# ---------------------------------------------------------------------------
# Connection singleton
# ---------------------------------------------------------------------------

_ib: Optional[IB] = None
_lock = threading.Lock()


def _get_ib() -> IB:
    """Get or create a connected IB instance (thread-safe)."""
    global _ib
    with _lock:
        if _ib is None or not _ib.isConnected():
            _ib = IB()
            _ib.connect(
                IBKR_HOST,
                IBKR_PORT,
                clientId=IBKR_CLIENT_ID,
                timeout=IBKR_CONNECT_TIMEOUT,
            )
            _ib.reqMarketDataType(IBKR_MARKET_DATA_TYPE)
            print(f"[IBKR] Connected to {IBKR_HOST}:{IBKR_PORT} (client {IBKR_CLIENT_ID})")
        return _ib


def disconnect():
    """Disconnect from IB Gateway."""
    global _ib
    with _lock:
        if _ib and _ib.isConnected():
            _ib.disconnect()
            print("[IBKR] Disconnected")
        _ib = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_contract(symbol: str):
    """Create an IB contract for a symbol. Handles index symbols."""
    sym = symbol.upper()
    # Index symbols
    if sym in ("SPX", "VIX", "NDX", "DJX", "RUT") or sym.startswith("I:"):
        clean = sym.replace("I:", "")
        return Index(clean, "CBOE")
    return Stock(sym, "SMART", "USD")


def _bars_to_polygon_format(bars) -> list:
    """Convert ib_insync BarData list to Polygon OHLCV format."""
    results = []
    for bar in bars:
        ts = int(bar.date.timestamp() * 1000) if hasattr(bar.date, "timestamp") else 0
        results.append({
            "o": float(bar.open),
            "h": float(bar.high),
            "l": float(bar.low),
            "c": float(bar.close),
            "v": int(bar.volume) if bar.volume > 0 else 0,
            "vw": round(float((bar.open + bar.high + bar.low + bar.close) / 4), 4),
            "t": ts,
            "n": 0,
        })
    return results


def _current_price(ib: IB, contract) -> float:
    """Get current/last price for a contract."""
    ib.qualifyContracts(contract)
    ticker = ib.reqMktData(contract, "", False, False)
    ib.sleep(2)
    price = ticker.marketPrice()
    if price != price:  # NaN check
        price = ticker.close
    ib.cancelMktData(contract)
    return float(price) if price == price else 0.0


# ---------------------------------------------------------------------------
# Public API — mirrors polygon_client.py signatures
# ---------------------------------------------------------------------------

def get_candles_for_mode(symbol: str, mode: str = "swing") -> dict:
    """
    Get OHLCV candles configured for specific trading mode.
    Returns data in Polygon-compatible format.
    """
    mode = mode.lower()
    if mode not in MODE_DATA_CONFIG:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {list(MODE_DATA_CONFIG.keys())}")

    cfg = MODE_DATA_CONFIG[mode]
    ib = _get_ib()
    contract = _make_contract(symbol)
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=cfg["duration"],
        barSizeSetting=cfg["bar_size"],
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
        timeout=IBKR_REQUEST_TIMEOUT,
    )

    results = _bars_to_polygon_format(bars)

    now = datetime.utcnow()
    start_str = (now - timedelta(days=730)).strftime("%Y-%m-%d")
    end_str = now.strftime("%Y-%m-%d")

    data = {
        "status": "OK" if results else "DELAYED",
        "resultsCount": len(results),
        "results": results,
        "_mode_config": {
            "mode": mode,
            "multiplier": cfg["multiplier"],
            "timespan": cfg["timespan"],
            "requested_limit": len(results),
            "actual_bars": len(results),
            "date_range": {"start": start_str, "end": end_str},
            "dte_range": cfg["dte_range"],
        },
    }

    if not results:
        data["message"] = f"No bars returned for {symbol} in {mode} mode"

    print(
        f"[IBKR] {symbol} | Mode={mode.upper()} | {cfg['bar_size']} | "
        f"Bars={len(results)} | {start_str} to {end_str}"
    )
    return data


def get_candles(symbol: str, tf: str = "day", limit: int = 730) -> dict:
    """
    Get OHLCV candles (simple version). Matches polygon_client.get_candles.
    """
    bar_size_map = {
        "minute": "1 min",
        "hour": "1 hour",
        "day": "1 day",
    }
    duration_map = {
        "minute": f"{min(limit, 1800)} S",
        "hour": f"{min(limit * 3600, 86400 * 30)} S",
        "day": f"{min(limit, 730)} D" if limit <= 365 else "2 Y",
    }

    bar_size = bar_size_map.get(tf, "1 day")
    duration = duration_map.get(tf, "2 Y")

    ib = _get_ib()
    contract = _make_contract(symbol)
    ib.qualifyContracts(contract)

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
        timeout=IBKR_REQUEST_TIMEOUT,
    )

    results = _bars_to_polygon_format(bars)

    data = {
        "status": "OK" if results else "DELAYED",
        "resultsCount": len(results),
        "results": results,
    }
    if not results:
        data["message"] = f"No candles for {symbol} tf={tf}"

    print(f"[IBKR] {symbol} | TF={tf} | Bars={len(results)}")
    return data


def get_full_option_chain_snapshot(
    underlying_asset: str, limit: int = 100, min_dte: int = 1
) -> dict:
    """
    Get full option chain snapshot with calls and puts.
    Returns data in the exact Polygon snapshot format the engine expects.
    """
    ib = _get_ib()
    sym = underlying_asset.upper()

    # Get current price
    stock = _make_contract(sym)
    ib.qualifyContracts(stock)
    price = _current_price(ib, stock)
    if price <= 0:
        return {"status": "ERROR", "results": [], "current_price": 0}

    # Strike range: 90%-110% of current price
    strike_low = price * 0.90
    strike_high = price * 1.10

    # Get available option chains
    chains = ib.reqSecDefOptParams(sym, "", stock.secType, stock.conId)
    if not chains:
        return {
            "status": "ERROR",
            "results": [],
            "current_price": price,
            "call_count": 0,
            "put_count": 0,
            "strike_range": [int(strike_low), int(strike_high)],
        }

    # Pick the SMART exchange chain (or first available)
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

    # Filter expirations: min_dte to ~90 days out
    tomorrow = datetime.now() + timedelta(days=min_dte)
    max_exp = datetime.now() + timedelta(days=90)
    valid_exps = sorted(
        exp
        for exp in chain.expirations
        if tomorrow.strftime("%Y%m%d") <= exp <= max_exp.strftime("%Y%m%d")
    )[:6]  # Keep 6 nearest expirations

    # Filter strikes near current price
    valid_strikes = sorted(
        s for s in chain.strikes if strike_low <= s <= strike_high
    )

    # Build option contracts for both calls and puts
    contracts = []
    for exp in valid_exps:
        for strike in valid_strikes:
            for right in ("C", "P"):
                contracts.append(
                    Option(sym, exp, strike, right, chain.exchange)
                )
                if len(contracts) >= limit * 2:
                    break
            if len(contracts) >= limit * 2:
                break
        if len(contracts) >= limit * 2:
            break

    # Qualify and fetch snapshots in batches
    qualified = []
    for batch_start in range(0, len(contracts), 50):
        batch = contracts[batch_start : batch_start + 50]
        try:
            ib.qualifyContracts(*batch)
            qualified.extend(batch)
        except Exception as e:
            print(f"[IBKR] Qualify batch warning: {e}")

    # Request market data for all qualified contracts
    tickers = []
    for con in qualified:
        tickers.append(ib.reqMktData(con, "100,101,104,106", False, False))

    ib.sleep(3)  # Let data stream in

    results = []
    call_count = 0
    put_count = 0

    for ticker in tickers:
        con = ticker.contract
        exp_date = f"{con.lastTradeDateOrContractMonth[:4]}-{con.lastTradeDateOrContractMonth[4:6]}-{con.lastTradeDateOrContractMonth[6:]}"
        contract_type = "call" if con.right == "C" else "put"

        # Extract greeks from model
        greeks = ticker.modelGreeks
        delta = float(greeks.delta) if greeks and greeks.delta == greeks.delta else 0.0
        gamma = float(greeks.gamma) if greeks and greeks.gamma == greeks.gamma else 0.0
        theta = float(greeks.theta) if greeks and greeks.theta == greeks.theta else 0.0
        vega = float(greeks.vega) if greeks and greeks.vega == greeks.vega else 0.0
        iv = float(greeks.impliedVol) if greeks and greeks.impliedVol == greeks.impliedVol else 0.0

        bid = float(ticker.bid) if ticker.bid == ticker.bid and ticker.bid > 0 else 0.0
        ask = float(ticker.ask) if ticker.ask == ticker.ask and ticker.ask > 0 else 0.0
        last = float(ticker.last) if ticker.last == ticker.last and ticker.last > 0 else (bid + ask) / 2 if (bid + ask) > 0 else 0.0
        volume = int(ticker.volume) if ticker.volume == ticker.volume and ticker.volume > 0 else 0

        # Build Polygon-compatible result
        entry = {
            "details": {
                "ticker": f"{sym}_{contract_type[0].upper()}_{int(con.strike)}_{con.lastTradeDateOrContractMonth}",
                "strike_price": float(con.strike),
                "contract_type": contract_type,
                "expiration_date": exp_date,
            },
            "greeks": {
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
            },
            "last_quote": {
                "bid": bid,
                "ask": ask,
            },
            "day": {
                "close": last,
                "vwap": round((bid + ask) / 2, 4) if (bid + ask) > 0 else last,
                "volume": volume,
            },
            "implied_volatility": iv,
            "open_interest": int(ticker.callOpenInterest if contract_type == "call" else ticker.putOpenInterest) if hasattr(ticker, "callOpenInterest") else 0,
        }

        results.append(entry)
        if contract_type == "call":
            call_count += 1
        else:
            put_count += 1

        ib.cancelMktData(con)

    print(
        f"[IBKR] Options {sym} | Calls={call_count} Puts={put_count} | "
        f"Price=${price:.2f} | Strikes={int(strike_low)}-{int(strike_high)}"
    )

    return {
        "status": "OK" if results else "ERROR",
        "results": results,
        "call_count": call_count,
        "put_count": put_count,
        "current_price": price,
        "strike_range": [int(strike_low), int(strike_high)],
    }


def get_option_chain_snapshot(
    underlying_asset: str, cursor: str | None = None, limit: int = 50
) -> dict:
    """Paginated option chain snapshot (wraps get_full_option_chain_snapshot)."""
    return get_full_option_chain_snapshot(underlying_asset, limit=limit)


def get_ticker_details(symbol: str) -> dict:
    """Get basic ticker details from IBKR."""
    ib = _get_ib()
    contract = _make_contract(symbol)
    ib.qualifyContracts(contract)
    details = ib.reqContractDetails(contract)

    if not details:
        return {"status": "ERROR", "results": []}

    d = details[0]
    return {
        "status": "OK",
        "results": {
            "ticker": symbol.upper(),
            "name": d.longName,
            "market": d.contract.exchange,
            "locale": "us",
            "primary_exchange": d.contract.primaryExchange,
            "type": d.contract.secType,
            "currency_name": d.contract.currency,
        },
    }


def get_market_context(mode: str = "swing") -> dict:
    """
    Fetch SPY + VIX data and compute market-wide context.
    Returns the same dict format as polygon_client.get_market_context.
    """
    context = {
        "spy": {},
        "vix": {},
        "market_bias": "neutral",
        "market_regime": "unknown",
        "risk_level": "normal",
        "favor_puts": False,
        "favor_calls": False,
        "warnings": [],
    }

    # --- SPY ---
    try:
        spy_data = get_candles_for_mode("SPY", mode="swing")
        spy_bars = spy_data.get("results", [])

        if spy_bars and len(spy_bars) >= 200:
            closes = [bar["c"] for bar in spy_bars]
            highs = [bar["h"] for bar in spy_bars]
            lows = [bar["l"] for bar in spy_bars]
            volumes = [bar.get("v", 0) for bar in spy_bars]

            current = closes[-1]
            prev_close = closes[-2] if len(closes) > 1 else current

            ma_20 = np.mean(closes[-20:])
            ma_50 = np.mean(closes[-50:])
            ma_200 = np.mean(closes[-200:])

            daily_change_pct = ((current - prev_close) / prev_close) * 100
            weekly_change_pct = ((current - closes[-6]) / closes[-6]) * 100 if len(closes) >= 6 else 0
            monthly_change_pct = ((current - closes[-22]) / closes[-22]) * 100 if len(closes) >= 22 else 0
            distance_from_200ma_pct = ((current - ma_200) / ma_200) * 100

            # RSI-14
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses_arr = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses_arr) if len(losses_arr) > 0 else 0
            spy_rsi = 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss > 0 else 100

            above_20ma = current > ma_20
            above_50ma = current > ma_50
            above_200ma = current > ma_200
            ma_20_above_50 = ma_20 > ma_50
            ma_50_above_200 = ma_50 > ma_200

            bull_count = sum([above_20ma, above_50ma, above_200ma, ma_20_above_50, ma_50_above_200])

            if bull_count >= 4:
                spy_trend = "BULLISH"
            elif bull_count >= 3:
                spy_trend = "SLIGHTLY_BULLISH"
            elif bull_count <= 1:
                spy_trend = "BEARISH"
            elif bull_count <= 2:
                spy_trend = "SLIGHTLY_BEARISH"
            else:
                spy_trend = "NEUTRAL"

            high_52w = max(highs[-252:]) if len(highs) >= 252 else max(highs)
            low_52w = min(lows[-252:]) if len(lows) >= 252 else min(lows)
            pct_from_52w_high = ((current - high_52w) / high_52w) * 100

            avg_vol_20 = np.mean(volumes[-20:]) if volumes else 0
            vol_ratio = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

            context["spy"] = {
                "price": round(current, 2),
                "daily_change_pct": round(daily_change_pct, 2),
                "weekly_change_pct": round(weekly_change_pct, 2),
                "monthly_change_pct": round(monthly_change_pct, 2),
                "trend": spy_trend,
                "rsi_14": round(float(spy_rsi), 2),
                "ma_20": round(ma_20, 2),
                "ma_50": round(ma_50, 2),
                "ma_200": round(ma_200, 2),
                "above_20ma": above_20ma,
                "above_50ma": above_50ma,
                "above_200ma": above_200ma,
                "distance_from_200ma_pct": round(distance_from_200ma_pct, 2),
                "pct_from_52w_high": round(pct_from_52w_high, 2),
                "volume_ratio": round(float(vol_ratio), 2),
            }
        else:
            context["warnings"].append("SPY: insufficient data (need 200+ bars)")

    except Exception as e:
        context["warnings"].append(f"SPY fetch failed: {str(e)}")

    # --- VIX ---
    try:
        vix_data = get_candles("VIX", tf="day", limit=252)
        vix_bars = vix_data.get("results", [])

        if vix_bars and len(vix_bars) >= 20:
            vix_closes = [bar["c"] for bar in vix_bars]
            vix_current = vix_closes[-1]
            vix_prev = vix_closes[-2] if len(vix_closes) > 1 else vix_current

            vix_ma_20 = np.mean(vix_closes[-20:])
            vix_daily_change = ((vix_current - vix_prev) / vix_prev) * 100
            vix_percentile = (sum(1 for v in vix_closes if v <= vix_current) / len(vix_closes)) * 100

            if vix_current >= 30:
                vix_regime = "EXTREME_FEAR"
            elif vix_current >= 25:
                vix_regime = "HIGH_FEAR"
            elif vix_current >= 20:
                vix_regime = "ELEVATED"
            elif vix_current >= 15:
                vix_regime = "NORMAL"
            elif vix_current >= 12:
                vix_regime = "LOW"
            else:
                vix_regime = "EXTREME_COMPLACENCY"

            context["vix"] = {
                "level": round(vix_current, 2),
                "daily_change_pct": round(vix_daily_change, 2),
                "ma_20": round(float(vix_ma_20), 2),
                "above_ma_20": vix_current > vix_ma_20,
                "percentile": round(float(vix_percentile), 1),
                "regime": vix_regime,
                "is_spiking": vix_daily_change > 10,
                "is_elevated": vix_current >= 20,
                "is_extreme": vix_current >= 30,
            }
        else:
            context["warnings"].append("VIX: insufficient data")

    except Exception as e:
        context["warnings"].append(f"VIX fetch failed: {str(e)}")

    # --- Compute overall market bias (same logic as polygon_client) ---
    spy_info = context.get("spy", {})
    vix_info = context.get("vix", {})

    spy_trend = spy_info.get("trend", "NEUTRAL")
    vix_regime = vix_info.get("regime", "NORMAL")
    vix_level = vix_info.get("level", 18)
    spy_above_200 = spy_info.get("above_200ma", True)
    spy_rsi_val = spy_info.get("rsi_14", 50)

    bull_points = 0
    bear_points = 0

    if spy_trend == "BULLISH":
        bull_points += 3
    elif spy_trend == "SLIGHTLY_BULLISH":
        bull_points += 1
    elif spy_trend == "BEARISH":
        bear_points += 3
    elif spy_trend == "SLIGHTLY_BEARISH":
        bear_points += 1

    if spy_above_200:
        bull_points += 2
    else:
        bear_points += 2

    if vix_regime in ("EXTREME_FEAR", "HIGH_FEAR"):
        bear_points += 3
    elif vix_regime == "ELEVATED":
        bear_points += 1
    elif vix_regime in ("LOW", "EXTREME_COMPLACENCY"):
        bull_points += 2
    elif vix_regime == "NORMAL":
        bull_points += 1

    if spy_rsi_val > 70:
        bear_points += 1
        context["warnings"].append("SPY RSI overbought (>70) — pullback risk")
    elif spy_rsi_val < 30:
        bull_points += 1
        context["warnings"].append("SPY RSI oversold (<30) — bounce potential")

    if bull_points > bear_points + 2:
        context["market_bias"] = "STRONG_BULLISH"
        context["favor_calls"] = True
    elif bull_points > bear_points:
        context["market_bias"] = "BULLISH"
        context["favor_calls"] = True
    elif bear_points > bull_points + 2:
        context["market_bias"] = "STRONG_BEARISH"
        context["favor_puts"] = True
    elif bear_points > bull_points:
        context["market_bias"] = "BEARISH"
        context["favor_puts"] = True
    else:
        context["market_bias"] = "NEUTRAL"

    if vix_regime == "EXTREME_FEAR" and not spy_above_200:
        context["market_regime"] = "CRISIS"
        context["warnings"].append("CRISIS MODE: VIX extreme + SPY below 200MA. Favor puts or cash.")
    elif vix_regime in ("HIGH_FEAR", "EXTREME_FEAR"):
        context["market_regime"] = "HIGH_VOLATILITY"
        context["warnings"].append("High volatility: use tighter stops, smaller positions")
    elif vix_regime in ("LOW", "EXTREME_COMPLACENCY") and spy_trend == "BULLISH":
        context["market_regime"] = "CALM_BULL"
    elif spy_trend == "BEARISH":
        context["market_regime"] = "DOWNTREND"
        context["warnings"].append("Market in downtrend: favor puts or wait for reversal")
    else:
        context["market_regime"] = "NORMAL"

    if vix_level >= 30:
        context["risk_level"] = "EXTREME"
    elif vix_level >= 25:
        context["risk_level"] = "HIGH"
    elif vix_level >= 20:
        context["risk_level"] = "ELEVATED"
    elif vix_level >= 15:
        context["risk_level"] = "NORMAL"
    else:
        context["risk_level"] = "LOW"

    print(
        f"[IBKR-MarketCtx] SPY={spy_info.get('price','?')} trend={spy_trend} | "
        f"VIX={vix_level} regime={vix_regime} | Bias={context['market_bias']}"
    )
    return context


# ---------------------------------------------------------------------------
# Order Execution
# ---------------------------------------------------------------------------

def place_option_order(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    action: str = "BUY",
    quantity: int = 1,
    order_type: str = "LMT",
    limit_price: Optional[float] = None,
) -> dict:
    """
    Place a single-leg option order.

    Args:
        symbol: Underlying (e.g. "SPY")
        expiry: Expiration in YYYYMMDD or YYYY-MM-DD format
        strike: Strike price
        right: "C" or "P"
        action: "BUY" or "SELL"
        quantity: Number of contracts
        order_type: "LMT", "MKT", or "STP"
        limit_price: Required for LMT orders

    Returns:
        Dict with order_id, status, and fill details.
    """
    from ib_insync import LimitOrder, MarketOrder, StopOrder

    ib = _get_ib()
    expiry_clean = expiry.replace("-", "")
    contract = Option(symbol.upper(), expiry_clean, strike, right.upper(), "SMART")
    ib.qualifyContracts(contract)

    if order_type == "MKT":
        order = MarketOrder(action.upper(), quantity)
    elif order_type == "STP":
        if limit_price is None:
            raise ValueError("limit_price required for STP orders")
        order = StopOrder(action.upper(), quantity, limit_price)
    else:
        if limit_price is None:
            # Auto-price: use midpoint
            ticker = ib.reqMktData(contract, "", False, False)
            ib.sleep(2)
            bid = float(ticker.bid) if ticker.bid == ticker.bid and ticker.bid > 0 else 0
            ask = float(ticker.ask) if ticker.ask == ticker.ask and ticker.ask > 0 else 0
            ib.cancelMktData(contract)
            if bid > 0 and ask > 0:
                limit_price = round((bid + ask) / 2, 2)
            else:
                raise ValueError("Cannot determine price — provide limit_price or use MKT")
        order = LimitOrder(action.upper(), quantity, limit_price)

    trade = ib.placeOrder(contract, order)
    ib.sleep(1)

    print(
        f"[IBKR-ORDER] {action} {quantity}x {symbol} {expiry_clean} "
        f"${strike} {right} @ {order_type} {limit_price or 'MKT'} | "
        f"OrderId={trade.order.orderId} Status={trade.orderStatus.status}"
    )

    return {
        "order_id": trade.order.orderId,
        "status": trade.orderStatus.status,
        "symbol": symbol.upper(),
        "contract": {
            "expiry": expiry_clean,
            "strike": strike,
            "right": right.upper(),
            "con_id": contract.conId,
        },
        "action": action.upper(),
        "quantity": quantity,
        "order_type": order_type,
        "limit_price": limit_price,
        "filled": trade.orderStatus.filled,
        "avg_fill_price": trade.orderStatus.avgFillPrice,
        "remaining": trade.orderStatus.remaining,
    }


def place_spread_order(
    symbol: str,
    legs: list,
    action: str = "BUY",
    quantity: int = 1,
    order_type: str = "LMT",
    limit_price: Optional[float] = None,
) -> dict:
    """
    Place a multi-leg spread order (vertical, iron condor, etc).

    Args:
        symbol: Underlying
        legs: List of dicts, each with:
            - expiry: YYYYMMDD
            - strike: float
            - right: "C" or "P"
            - action: "BUY" or "SELL"
            - ratio: int (default 1)
        quantity: Number of spreads
        order_type: "LMT" or "MKT"
        limit_price: Net debit (positive) or credit (negative) for LMT

    Returns:
        Dict with order_id and status.
    """
    from ib_insync import LimitOrder, MarketOrder, ComboLeg, Contract

    ib = _get_ib()

    # Build combo legs
    combo_legs = []
    for leg in legs:
        opt = Option(
            symbol.upper(),
            leg["expiry"].replace("-", ""),
            leg["strike"],
            leg["right"].upper(),
            "SMART",
        )
        ib.qualifyContracts(opt)
        combo_legs.append(
            ComboLeg(
                conId=opt.conId,
                ratio=leg.get("ratio", 1),
                action=leg["action"].upper(),
                exchange="SMART",
            )
        )

    # Build combo contract
    combo = Contract()
    combo.symbol = symbol.upper()
    combo.secType = "BAG"
    combo.currency = "USD"
    combo.exchange = "SMART"
    combo.comboLegs = combo_legs

    if order_type == "MKT":
        order = MarketOrder(action.upper(), quantity)
    else:
        if limit_price is None:
            raise ValueError("limit_price required for LMT spread orders")
        order = LimitOrder(action.upper(), quantity, limit_price)

    trade = ib.placeOrder(combo, order)
    ib.sleep(1)

    print(
        f"[IBKR-SPREAD] {action} {quantity}x {symbol} {len(legs)}-leg spread | "
        f"OrderId={trade.order.orderId} Status={trade.orderStatus.status}"
    )

    return {
        "order_id": trade.order.orderId,
        "status": trade.orderStatus.status,
        "symbol": symbol.upper(),
        "legs": len(legs),
        "action": action.upper(),
        "quantity": quantity,
        "order_type": order_type,
        "limit_price": limit_price,
        "filled": trade.orderStatus.filled,
        "avg_fill_price": trade.orderStatus.avgFillPrice,
        "remaining": trade.orderStatus.remaining,
    }


def get_positions() -> list:
    """Get all current positions."""
    ib = _get_ib()
    positions = ib.positions()

    result = []
    for pos in positions:
        con = pos.contract
        result.append({
            "account": pos.account,
            "symbol": con.symbol,
            "sec_type": con.secType,
            "con_id": con.conId,
            "quantity": pos.position,
            "avg_cost": pos.avgCost,
            "expiry": getattr(con, "lastTradeDateOrContractMonth", ""),
            "strike": getattr(con, "strike", 0),
            "right": getattr(con, "right", ""),
        })

    print(f"[IBKR] Positions: {len(result)}")
    return result


def close_position(
    symbol: str,
    con_id: int = 0,
    quantity: Optional[int] = None,
    order_type: str = "MKT",
) -> dict:
    """
    Close an existing position by symbol/conId.

    Args:
        symbol: Underlying symbol
        con_id: Contract ID (from get_positions). If 0, closes first matching position.
        quantity: Contracts to close. None = close entire position.
        order_type: "MKT" (default) or "LMT"

    Returns:
        Dict with order details.
    """
    from ib_insync import MarketOrder, Contract

    ib = _get_ib()
    positions = ib.positions()

    target = None
    for pos in positions:
        if pos.contract.symbol.upper() == symbol.upper():
            if con_id == 0 or pos.contract.conId == con_id:
                target = pos
                break

    if target is None:
        return {"error": f"No position found for {symbol} (conId={con_id})"}

    close_qty = abs(quantity or int(target.position))
    close_action = "SELL" if target.position > 0 else "BUY"

    contract = target.contract
    ib.qualifyContracts(contract)

    order = MarketOrder(close_action, close_qty)
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)

    print(
        f"[IBKR-CLOSE] {close_action} {close_qty}x {symbol} conId={contract.conId} | "
        f"OrderId={trade.order.orderId} Status={trade.orderStatus.status}"
    )

    return {
        "order_id": trade.order.orderId,
        "status": trade.orderStatus.status,
        "action": close_action,
        "quantity": close_qty,
        "symbol": symbol.upper(),
        "con_id": contract.conId,
    }


def get_open_orders() -> list:
    """Get all open/pending orders."""
    ib = _get_ib()
    trades = ib.openTrades()

    result = []
    for trade in trades:
        con = trade.contract
        result.append({
            "order_id": trade.order.orderId,
            "status": trade.orderStatus.status,
            "symbol": con.symbol,
            "sec_type": con.secType,
            "action": trade.order.action,
            "quantity": trade.order.totalQuantity,
            "order_type": trade.order.orderType,
            "limit_price": trade.order.lmtPrice,
            "filled": trade.orderStatus.filled,
            "remaining": trade.orderStatus.remaining,
            "expiry": getattr(con, "lastTradeDateOrContractMonth", ""),
            "strike": getattr(con, "strike", 0),
            "right": getattr(con, "right", ""),
        })

    print(f"[IBKR] Open orders: {len(result)}")
    return result


def cancel_order(order_id: int) -> dict:
    """Cancel an open order by order ID."""
    ib = _get_ib()
    trades = ib.openTrades()

    for trade in trades:
        if trade.order.orderId == order_id:
            ib.cancelOrder(trade.order)
            ib.sleep(1)
            print(f"[IBKR-CANCEL] OrderId={order_id} cancelled")
            return {
                "order_id": order_id,
                "status": "CANCELLED",
                "symbol": trade.contract.symbol,
            }

    return {"error": f"Order {order_id} not found in open orders"}


def get_account_summary() -> dict:
    """Get account balance, buying power, P&L."""
    ib = _get_ib()
    account_values = ib.accountSummary()

    summary = {}
    keys_we_want = {
        "NetLiquidation", "TotalCashValue", "BuyingPower",
        "GrossPositionValue", "MaintMarginReq", "AvailableFunds",
        "UnrealizedPnL", "RealizedPnL",
    }
    for av in account_values:
        if av.tag in keys_we_want and av.currency == "USD":
            summary[av.tag] = float(av.value)

    print(f"[IBKR] Account: NLV=${summary.get('NetLiquidation', '?')}")
    return summary
