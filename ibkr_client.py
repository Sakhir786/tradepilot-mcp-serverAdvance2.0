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


# ---------------------------------------------------------------------------
# AI-Critical Execution Functions
# ---------------------------------------------------------------------------

def execute_signal(
    analysis_result: dict,
    quantity: int = 1,
    order_type: str = "LMT",
    limit_price: Optional[float] = None,
    dry_run: bool = False,
) -> dict:
    """
    Execute a trade directly from engine analysis output.
    This is the AI's primary action — takes the 18-layer result and places the order.

    Args:
        analysis_result: Dict from /engine18/analyze (the engine's full output)
        quantity: Number of contracts (overrides engine suggestion if set)
        order_type: "LMT" (default, uses mid-price) or "MKT"
        limit_price: Override price. None = auto mid-price for LMT.
        dry_run: If True, returns what would be placed without executing.

    Returns:
        Dict with order details, or dry_run preview.
    """
    # Extract recommendation from analysis
    ticker = analysis_result.get("ticker", "")
    trade_valid = analysis_result.get("analysis_summary", {}).get("trade_valid",
                  analysis_result.get("trade_valid", False))
    action = analysis_result.get("analysis_summary", {}).get("action",
             analysis_result.get("action", "FLAT"))
    direction = analysis_result.get("analysis_summary", {}).get("direction",
                analysis_result.get("direction", "NEUTRAL"))
    confidence = analysis_result.get("analysis_summary", {}).get("confidence",
                 analysis_result.get("confidence", "NO_TRADE"))
    win_prob = analysis_result.get("analysis_summary", {}).get("win_probability",
               analysis_result.get("win_probability", 0))

    opt = analysis_result.get("option_recommendation", {})
    strike = opt.get("strike", analysis_result.get("strike", 0))
    expiry_dte = opt.get("expiry_dte", analysis_result.get("expiry_dte", 0))
    expiry_date = opt.get("expiry_date", analysis_result.get("expiry_date", ""))
    delta = opt.get("delta", analysis_result.get("delta", 0))

    exec_plan = analysis_result.get("execution_plan", {})
    entry = exec_plan.get("entry", analysis_result.get("entry_price", 0))
    target = exec_plan.get("target", analysis_result.get("target_price", 0))
    stop = exec_plan.get("stop", analysis_result.get("stop_price", 0))

    contracts = analysis_result.get("execution_plan", {}).get("contracts",
                analysis_result.get("contracts_suggested", quantity))

    # Determine right (C/P) from action
    if action in ("BUY_CALL", "SELL_PUT"):
        right = "C"
        order_action = "BUY" if action == "BUY_CALL" else "SELL"
    elif action in ("BUY_PUT", "SELL_CALL"):
        right = "P"
        order_action = "BUY" if action == "BUY_PUT" else "SELL"
    else:
        return {
            "error": "NO_TRADE",
            "reason": f"Engine action is {action} — no order to place",
            "trade_valid": trade_valid,
            "confidence": confidence,
        }

    if not trade_valid:
        return {
            "error": "TRADE_INVALID",
            "reason": "Engine says trade_valid=False",
            "action": action,
            "confidence": confidence,
            "win_probability": win_prob,
        }

    # Format expiry
    expiry_fmt = expiry_date.replace("-", "") if expiry_date else ""
    if not expiry_fmt:
        return {"error": "NO_EXPIRY", "reason": "Engine did not provide expiry_date"}

    use_qty = quantity if quantity > 1 else max(1, int(contracts or 1))

    preview = {
        "symbol": ticker,
        "expiry": expiry_fmt,
        "strike": strike,
        "right": right,
        "action": order_action,
        "quantity": use_qty,
        "order_type": order_type,
        "limit_price": limit_price or entry,
        "engine_signal": {
            "direction": direction,
            "action": action,
            "confidence": confidence,
            "win_probability": win_prob,
            "delta": delta,
            "entry": entry,
            "target": target,
            "stop": stop,
        },
    }

    if dry_run:
        preview["dry_run"] = True
        print(f"[IBKR-DRY] Would place: {order_action} {use_qty}x {ticker} {expiry_fmt} ${strike} {right}")
        return preview

    # Execute
    result = place_option_order(
        symbol=ticker,
        expiry=expiry_fmt,
        strike=strike,
        right=right,
        action=order_action,
        quantity=use_qty,
        order_type=order_type,
        limit_price=limit_price,
    )
    result["engine_signal"] = preview["engine_signal"]
    return result


def place_bracket_order(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    action: str = "BUY",
    quantity: int = 1,
    entry_price: float = 0,
    take_profit_price: float = 0,
    stop_loss_price: float = 0,
) -> dict:
    """
    Place a bracket order: entry + profit target + stop loss in one shot.
    IBKR handles the OCO (one-cancels-other) logic server-side.

    Args:
        symbol: Underlying
        expiry: YYYYMMDD
        strike: Strike price
        right: "C" or "P"
        action: "BUY" or "SELL"
        quantity: Contracts
        entry_price: Limit entry price
        take_profit_price: Limit exit price for profit
        stop_loss_price: Stop price for loss

    Returns:
        Dict with parent + child order IDs.
    """
    from ib_insync import LimitOrder, StopOrder, Order

    ib = _get_ib()
    expiry_clean = expiry.replace("-", "")
    contract = Option(symbol.upper(), expiry_clean, strike, right.upper(), "SMART")
    ib.qualifyContracts(contract)

    exit_action = "SELL" if action.upper() == "BUY" else "BUY"

    # Parent: limit entry
    parent = LimitOrder(action.upper(), quantity, entry_price)
    parent.orderId = ib.client.getReqId()
    parent.transmit = False

    # Child 1: take profit
    tp = LimitOrder(exit_action, quantity, take_profit_price)
    tp.orderId = ib.client.getReqId()
    tp.parentId = parent.orderId
    tp.transmit = False

    # Child 2: stop loss (OCO with take profit)
    sl = StopOrder(exit_action, quantity, stop_loss_price)
    sl.orderId = ib.client.getReqId()
    sl.parentId = parent.orderId
    sl.transmit = True  # Last child transmits the whole bracket

    trade_parent = ib.placeOrder(contract, parent)
    trade_tp = ib.placeOrder(contract, tp)
    trade_sl = ib.placeOrder(contract, sl)
    ib.sleep(1)

    print(
        f"[IBKR-BRACKET] {action} {quantity}x {symbol} ${strike}{right} | "
        f"Entry=${entry_price} TP=${take_profit_price} SL=${stop_loss_price}"
    )

    return {
        "parent": {
            "order_id": parent.orderId,
            "status": trade_parent.orderStatus.status,
            "type": "ENTRY",
            "price": entry_price,
        },
        "take_profit": {
            "order_id": tp.orderId,
            "status": trade_tp.orderStatus.status,
            "type": "TAKE_PROFIT",
            "price": take_profit_price,
        },
        "stop_loss": {
            "order_id": sl.orderId,
            "status": trade_sl.orderStatus.status,
            "type": "STOP_LOSS",
            "price": stop_loss_price,
        },
        "symbol": symbol.upper(),
        "contract": {"expiry": expiry_clean, "strike": strike, "right": right.upper()},
        "quantity": quantity,
    }


def execute_signal_bracket(
    analysis_result: dict,
    quantity: int = 1,
) -> dict:
    """
    Execute a bracket order directly from engine analysis output.
    Uses engine's entry/target/stop prices for the bracket.

    Args:
        analysis_result: Dict from /engine18/analyze
        quantity: Override contract count

    Returns:
        Bracket order result or error.
    """
    ticker = analysis_result.get("ticker", "")
    trade_valid = analysis_result.get("analysis_summary", {}).get("trade_valid",
                  analysis_result.get("trade_valid", False))
    action = analysis_result.get("analysis_summary", {}).get("action",
             analysis_result.get("action", "FLAT"))

    if not trade_valid or action in ("FLAT", "NO_TRADE"):
        return {"error": "NO_TRADE", "reason": f"action={action}, trade_valid={trade_valid}"}

    opt = analysis_result.get("option_recommendation", {})
    strike = opt.get("strike", analysis_result.get("strike", 0))
    expiry_date = opt.get("expiry_date", analysis_result.get("expiry_date", ""))
    expiry_fmt = expiry_date.replace("-", "")

    exec_plan = analysis_result.get("execution_plan", {})
    entry = exec_plan.get("entry", analysis_result.get("entry_price", 0))
    target = exec_plan.get("target", analysis_result.get("target_price", 0))
    stop = exec_plan.get("stop", analysis_result.get("stop_price", 0))

    contracts = exec_plan.get("contracts", analysis_result.get("contracts_suggested", quantity))
    use_qty = quantity if quantity > 1 else max(1, int(contracts or 1))

    if action in ("BUY_CALL", "SELL_PUT"):
        right = "C"
        order_action = "BUY" if action == "BUY_CALL" else "SELL"
    elif action in ("BUY_PUT", "SELL_CALL"):
        right = "P"
        order_action = "BUY" if action == "BUY_PUT" else "SELL"
    else:
        return {"error": "UNKNOWN_ACTION", "action": action}

    if not all([expiry_fmt, strike, entry, target, stop]):
        return {"error": "MISSING_DATA", "expiry": expiry_fmt, "strike": strike,
                "entry": entry, "target": target, "stop": stop}

    return place_bracket_order(
        symbol=ticker,
        expiry=expiry_fmt,
        strike=strike,
        right=right,
        action=order_action,
        quantity=use_qty,
        entry_price=entry,
        take_profit_price=target,
        stop_loss_price=stop,
    )


def modify_order(
    order_id: int,
    new_limit_price: Optional[float] = None,
    new_quantity: Optional[int] = None,
) -> dict:
    """
    Modify an existing open order (price and/or quantity).

    Args:
        order_id: The order ID to modify
        new_limit_price: New limit price (None = keep current)
        new_quantity: New quantity (None = keep current)

    Returns:
        Dict with updated order details.
    """
    ib = _get_ib()
    trades = ib.openTrades()

    for trade in trades:
        if trade.order.orderId == order_id:
            if new_limit_price is not None:
                trade.order.lmtPrice = new_limit_price
            if new_quantity is not None:
                trade.order.totalQuantity = new_quantity
            ib.placeOrder(trade.contract, trade.order)
            ib.sleep(1)
            print(
                f"[IBKR-MODIFY] OrderId={order_id} | "
                f"Price={new_limit_price or 'unchanged'} Qty={new_quantity or 'unchanged'}"
            )
            return {
                "order_id": order_id,
                "status": trade.orderStatus.status,
                "new_limit_price": new_limit_price,
                "new_quantity": new_quantity,
                "symbol": trade.contract.symbol,
            }

    return {"error": f"Order {order_id} not found in open orders"}


def roll_option(
    symbol: str,
    old_con_id: int,
    new_expiry: str,
    new_strike: Optional[float] = None,
    new_right: Optional[str] = None,
    quantity: Optional[int] = None,
    order_type: str = "MKT",
) -> dict:
    """
    Roll an option position: close current, open new DTE.

    Args:
        symbol: Underlying
        old_con_id: conId of position to close (from get_positions)
        new_expiry: New expiration YYYYMMDD
        new_strike: New strike (None = same strike)
        new_right: New right C/P (None = same right)
        quantity: Contracts to roll (None = full position)
        order_type: "MKT" or "LMT"

    Returns:
        Dict with close_order and open_order details.
    """
    ib = _get_ib()
    positions = ib.positions()

    old_pos = None
    for pos in positions:
        if pos.contract.conId == old_con_id:
            old_pos = pos
            break

    if old_pos is None:
        return {"error": f"Position conId={old_con_id} not found"}

    old_contract = old_pos.contract
    roll_qty = abs(quantity or int(old_pos.position))
    use_strike = new_strike or old_contract.strike
    use_right = (new_right or old_contract.right).upper()

    # Step 1: Close old position
    close_result = close_position(
        symbol=symbol,
        con_id=old_con_id,
        quantity=roll_qty,
        order_type=order_type,
    )

    if "error" in close_result:
        return {"error": f"Close failed: {close_result['error']}"}

    # Step 2: Open new position
    open_action = "BUY" if old_pos.position > 0 else "SELL"
    open_result = place_option_order(
        symbol=symbol,
        expiry=new_expiry.replace("-", ""),
        strike=use_strike,
        right=use_right,
        action=open_action,
        quantity=roll_qty,
        order_type=order_type,
    )

    print(
        f"[IBKR-ROLL] {symbol} | Close conId={old_con_id} → "
        f"Open {new_expiry} ${use_strike}{use_right} x{roll_qty}"
    )

    return {
        "close_order": close_result,
        "open_order": open_result,
        "rolled": {
            "symbol": symbol.upper(),
            "from_expiry": old_contract.lastTradeDateOrContractMonth,
            "from_strike": old_contract.strike,
            "to_expiry": new_expiry.replace("-", ""),
            "to_strike": use_strike,
            "right": use_right,
            "quantity": roll_qty,
        },
    }


def close_all_positions(order_type: str = "MKT") -> dict:
    """
    Emergency flatten: close ALL option positions immediately.

    Args:
        order_type: "MKT" (recommended for emergency) or "LMT"

    Returns:
        Dict with list of close orders placed.
    """
    ib = _get_ib()
    positions = ib.positions()

    if not positions:
        return {"closed": [], "message": "No positions to close"}

    results = []
    for pos in positions:
        if pos.position == 0:
            continue
        try:
            result = close_position(
                symbol=pos.contract.symbol,
                con_id=pos.contract.conId,
                quantity=abs(int(pos.position)),
                order_type=order_type,
            )
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "symbol": pos.contract.symbol,
                "con_id": pos.contract.conId,
            })

    print(f"[IBKR-FLATTEN] Closed {len(results)} positions")
    return {"closed": results, "count": len(results)}


def get_position_pnl(symbol: Optional[str] = None) -> list:
    """
    Get real-time P&L for each position (unrealized + realized).

    Args:
        symbol: Filter by symbol (None = all positions)

    Returns:
        List of position P&L dicts with market value, unrealized P&L, etc.
    """
    ib = _get_ib()
    ib.reqPnL(ib.managedAccounts()[0])
    ib.sleep(1)

    positions = ib.positions()
    results = []

    for pos in positions:
        if pos.position == 0:
            continue
        if symbol and pos.contract.symbol.upper() != symbol.upper():
            continue

        con = pos.contract
        ib.qualifyContracts(con)

        # Get current market price
        ticker = ib.reqMktData(con, "", False, False)
        ib.sleep(1)
        market_price = float(ticker.marketPrice()) if ticker.marketPrice() == ticker.marketPrice() else 0
        ib.cancelMktData(con)

        qty = float(pos.position)
        avg_cost = float(pos.avgCost)
        # Options: avgCost is per share, multiply by 100 for per-contract
        multiplier = 100 if con.secType == "OPT" else 1
        cost_basis = avg_cost * abs(qty)
        market_value = market_price * abs(qty) * multiplier
        unrealized_pnl = (market_price * multiplier - avg_cost) * qty

        results.append({
            "symbol": con.symbol,
            "sec_type": con.secType,
            "con_id": con.conId,
            "quantity": qty,
            "avg_cost": avg_cost,
            "market_price": market_price,
            "cost_basis": round(cost_basis, 2),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round((unrealized_pnl / cost_basis) * 100, 2) if cost_basis > 0 else 0,
            "expiry": getattr(con, "lastTradeDateOrContractMonth", ""),
            "strike": getattr(con, "strike", 0),
            "right": getattr(con, "right", ""),
            "dte": _calc_dte(getattr(con, "lastTradeDateOrContractMonth", "")),
        })

    return results


def _calc_dte(expiry_str: str) -> int:
    """Calculate days to expiration from YYYYMMDD string."""
    if not expiry_str or len(expiry_str) < 8:
        return -1
    try:
        exp_date = datetime.strptime(expiry_str[:8], "%Y%m%d").date()
        return (exp_date - datetime.now().date()).days
    except ValueError:
        return -1


def get_portfolio_risk() -> dict:
    """
    Get portfolio-level risk metrics: total Greeks, exposure, concentration.
    Critical for AI to check before placing new trades.

    Returns:
        Dict with aggregated Greeks, exposure by symbol, and risk flags.
    """
    ib = _get_ib()
    positions = ib.positions()

    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0
    total_exposure = 0.0
    total_cost = 0.0
    by_symbol = {}
    expiring_soon = []

    for pos in positions:
        if pos.position == 0:
            continue

        con = pos.contract
        qty = float(pos.position)
        ib.qualifyContracts(con)

        # Request greeks
        ticker = ib.reqMktData(con, "100,101,104,106", False, False)
        ib.sleep(1)

        greeks = ticker.modelGreeks
        market_price = float(ticker.marketPrice()) if ticker.marketPrice() == ticker.marketPrice() else 0
        ib.cancelMktData(con)

        multiplier = 100 if con.secType == "OPT" else 1
        position_value = abs(market_price * qty * multiplier)

        if greeks:
            d = float(greeks.delta) if greeks.delta == greeks.delta else 0
            g = float(greeks.gamma) if greeks.gamma == greeks.gamma else 0
            t = float(greeks.theta) if greeks.theta == greeks.theta else 0
            v = float(greeks.vega) if greeks.vega == greeks.vega else 0
            total_delta += d * qty * multiplier
            total_gamma += g * qty * multiplier
            total_theta += t * qty * multiplier
            total_vega += v * qty * multiplier

        total_exposure += position_value
        total_cost += abs(float(pos.avgCost) * qty)

        sym = con.symbol
        if sym not in by_symbol:
            by_symbol[sym] = {"contracts": 0, "exposure": 0, "delta": 0}
        by_symbol[sym]["contracts"] += abs(int(qty))
        by_symbol[sym]["exposure"] += position_value
        if greeks:
            by_symbol[sym]["delta"] += d * qty * multiplier

        # Flag positions expiring within 2 days
        dte = _calc_dte(getattr(con, "lastTradeDateOrContractMonth", ""))
        if 0 <= dte <= 2:
            expiring_soon.append({
                "symbol": sym,
                "con_id": con.conId,
                "strike": getattr(con, "strike", 0),
                "right": getattr(con, "right", ""),
                "dte": dte,
                "quantity": qty,
            })

    # Risk flags
    account = get_account_summary()
    nlv = account.get("NetLiquidation", 1)
    buying_power = account.get("BuyingPower", 0)

    exposure_pct = (total_exposure / nlv * 100) if nlv > 0 else 0
    warnings = []
    if exposure_pct > 50:
        warnings.append(f"HIGH EXPOSURE: {exposure_pct:.0f}% of NLV in options")
    if abs(total_delta) > 500:
        warnings.append(f"HIGH DELTA: net {total_delta:.0f} — large directional risk")
    if total_theta < -50:
        warnings.append(f"HIGH THETA DECAY: losing ${abs(total_theta):.0f}/day")
    if expiring_soon:
        warnings.append(f"{len(expiring_soon)} position(s) expiring within 2 days — roll or close")
    if buying_power < nlv * 0.2:
        warnings.append(f"LOW BUYING POWER: ${buying_power:,.0f} ({buying_power/nlv*100:.0f}% of NLV)")

    return {
        "total_greeks": {
            "delta": round(total_delta, 2),
            "gamma": round(total_gamma, 4),
            "theta": round(total_theta, 2),
            "vega": round(total_vega, 2),
        },
        "exposure": {
            "total_market_value": round(total_exposure, 2),
            "total_cost_basis": round(total_cost, 2),
            "pct_of_nlv": round(exposure_pct, 1),
        },
        "by_symbol": by_symbol,
        "positions_count": sum(1 for p in positions if p.position != 0),
        "expiring_soon": expiring_soon,
        "account": {
            "net_liquidation": nlv,
            "buying_power": buying_power,
        },
        "warnings": warnings,
    }


def wait_for_fill(order_id: int, timeout_seconds: int = 60) -> dict:
    """
    Wait for an order to fill (poll until filled or timeout).

    Args:
        order_id: Order ID to wait for
        timeout_seconds: Max seconds to wait (default 60)

    Returns:
        Dict with final order status, fill price, and fill time.
    """
    ib = _get_ib()
    start = time.time()

    while time.time() - start < timeout_seconds:
        trades = ib.trades()
        for trade in trades:
            if trade.order.orderId == order_id:
                status = trade.orderStatus.status
                if status in ("Filled",):
                    print(f"[IBKR-FILL] OrderId={order_id} FILLED @ ${trade.orderStatus.avgFillPrice}")
                    return {
                        "order_id": order_id,
                        "status": "Filled",
                        "avg_fill_price": trade.orderStatus.avgFillPrice,
                        "filled": trade.orderStatus.filled,
                        "fill_time": datetime.now().isoformat(),
                        "symbol": trade.contract.symbol,
                    }
                elif status in ("Cancelled", "ApiCancelled", "Inactive"):
                    return {
                        "order_id": order_id,
                        "status": status,
                        "reason": "Order was cancelled or rejected",
                    }
        ib.sleep(1)

    return {
        "order_id": order_id,
        "status": "TIMEOUT",
        "waited_seconds": timeout_seconds,
        "message": "Order not filled within timeout",
    }


def cancel_all_orders() -> dict:
    """Cancel ALL open orders immediately."""
    ib = _get_ib()
    ib.reqGlobalCancel()
    ib.sleep(1)
    print("[IBKR-CANCEL-ALL] Global cancel requested")
    return {"status": "GLOBAL_CANCEL_SENT", "timestamp": datetime.now().isoformat()}
