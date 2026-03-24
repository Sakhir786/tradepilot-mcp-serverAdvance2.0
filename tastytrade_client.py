"""
TastyTrade Sandbox Client for TradePilot
==========================================
Execute trades on TastyTrade's paper-trading sandbox using signals
from the 18-layer engine. Data still comes from IBKR/Polygon.

Sandbox details:
  - Base URL: https://api.cert.tastyworks.com
  - Resets every 24 hours (trades/positions cleared, accounts persist)
  - Quotes are 15-min delayed (we don't use them — IBKR provides data)

Execution functions (mirrors ibkr_client.py interface):
  - tt_login() -> session token
  - tt_get_accounts() -> list of sandbox accounts
  - tt_place_option_order(symbol, expiry, strike, right, action, quantity, order_type, limit_price)
  - tt_get_positions(account) -> current positions
  - tt_get_orders(account) -> open orders
  - tt_cancel_order(account, order_id)
  - tt_close_position(account, symbol, quantity)
  - tt_get_account_balance(account) -> buying power, cash, P&L
  - tt_execute_signal(analysis_result, ...) -> place trade from engine output
  - tt_dry_run(analysis_result) -> preview without executing
"""

import os
import time
import httpx
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TT_SANDBOX_URL = "https://api.cert.tastyworks.com"
TT_LIVE_URL = "https://api.tastyworks.com"

TT_USERNAME = os.getenv("TASTYTRADE_USERNAME", "")
TT_PASSWORD = os.getenv("TASTYTRADE_PASSWORD", "")
TT_SANDBOX = os.getenv("TASTYTRADE_SANDBOX", "true").lower() == "true"

# Session state
_session_token: Optional[str] = None
_session_expiry: Optional[datetime] = None
_account_number: Optional[str] = None


def _base_url() -> str:
    return TT_SANDBOX_URL if TT_SANDBOX else TT_LIVE_URL


def _headers() -> dict:
    h = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if _session_token:
        h["Authorization"] = _session_token
    return h


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def tt_login(username: str = "", password: str = "") -> dict:
    """
    Authenticate with TastyTrade and get a session token.
    Uses env vars if username/password not provided.

    Returns:
        Dict with session_token, user info, and accounts.
    """
    global _session_token, _session_expiry

    user = username or TT_USERNAME
    pwd = password or TT_PASSWORD

    if not user or not pwd:
        return {"error": "MISSING_CREDENTIALS", "reason": "Set TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD in .env"}

    url = f"{_base_url()}/sessions"
    payload = {
        "login": user,
        "password": pwd,
    }

    resp = httpx.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)

    if resp.status_code != 201:
        return {
            "error": "AUTH_FAILED",
            "status_code": resp.status_code,
            "detail": resp.text[:500],
        }

    data = resp.json().get("data", {})
    _session_token = data.get("session-token", "")
    _session_expiry = datetime.now() + timedelta(minutes=14)  # tokens last ~15min

    print(f"[TastyTrade] Logged in as {user} ({'SANDBOX' if TT_SANDBOX else 'LIVE'})")

    # Auto-fetch accounts
    accounts = tt_get_accounts()

    return {
        "status": "authenticated",
        "environment": "sandbox" if TT_SANDBOX else "live",
        "user": data.get("user", {}).get("email", user),
        "session_token": _session_token[:20] + "...",
        "accounts": accounts,
    }


def _ensure_session():
    """Auto-login if session expired or missing."""
    global _session_token, _session_expiry
    if not _session_token or (_session_expiry and datetime.now() > _session_expiry):
        result = tt_login()
        if "error" in result:
            raise ConnectionError(f"TastyTrade auth failed: {result.get('reason', result.get('detail', 'Unknown'))}")


def _get_account() -> str:
    """Get the active account number, auto-selecting the first one."""
    global _account_number
    if _account_number:
        return _account_number
    accounts = tt_get_accounts()
    if not accounts:
        raise ValueError("No TastyTrade accounts found. Create one in sandbox first.")
    _account_number = accounts[0]["account_number"]
    return _account_number


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------

def tt_get_accounts() -> list:
    """Get all accounts for the authenticated user."""
    global _account_number
    _ensure_session()

    url = f"{_base_url()}/customers/me/accounts"
    resp = httpx.get(url, headers=_headers(), timeout=15)

    if resp.status_code != 200:
        print(f"[TastyTrade] Failed to get accounts: {resp.status_code}")
        return []

    items = resp.json().get("data", {}).get("items", [])
    accounts = []
    for item in items:
        acct = item.get("account", {})
        accounts.append({
            "account_number": acct.get("account-number", ""),
            "nickname": acct.get("nickname", ""),
            "account_type": acct.get("account-type-name", ""),
            "is_margin": acct.get("margin-or-cash") == "Margin",
        })

    if accounts and not _account_number:
        _account_number = accounts[0]["account_number"]
        print(f"[TastyTrade] Using account: {_account_number}")

    return accounts


def tt_get_account_balance(account: str = "") -> dict:
    """Get account balances: cash, buying power, P&L."""
    _ensure_session()
    acct = account or _get_account()

    url = f"{_base_url()}/accounts/{acct}/balances"
    resp = httpx.get(url, headers=_headers(), timeout=15)

    if resp.status_code != 200:
        return {"error": f"Failed to get balance: {resp.status_code}"}

    data = resp.json().get("data", {})
    return {
        "account": acct,
        "environment": "sandbox" if TT_SANDBOX else "live",
        "cash_balance": float(data.get("cash-balance", 0)),
        "net_liquidating_value": float(data.get("net-liquidating-value", 0)),
        "option_buying_power": float(data.get("derivative-buying-power", 0)),
        "equity_buying_power": float(data.get("equity-buying-power", 0)),
        "maintenance_excess": float(data.get("maintenance-excess", 0)),
        "pending_cash": float(data.get("pending-cash", 0)),
        "day_trading_buying_power": float(data.get("day-trading-buying-power", 0)),
        "open_pl": float(data.get("pending-cash", 0)),
        "close_pl": float(data.get("closed-pl", 0)),
    }


# ---------------------------------------------------------------------------
# Option Symbol Builder
# ---------------------------------------------------------------------------

def _build_occ_symbol(symbol: str, expiry: str, right: str, strike: float) -> str:
    """
    Build OCC option symbol for TastyTrade.
    Format: SPY   260425C00575000
    - 6 char symbol (left-padded with spaces)
    - YYMMDD expiry
    - C or P
    - 8 digit strike (price * 1000, zero-padded)
    """
    sym = symbol.upper().ljust(6)
    exp = expiry.replace("-", "")
    if len(exp) == 8:  # YYYYMMDD -> YYMMDD
        exp = exp[2:]
    r = right.upper()[0]
    strike_int = int(strike * 1000)
    strike_str = str(strike_int).zfill(8)
    return f"{sym}{exp}{r}{strike_str}"


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def tt_place_option_order(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    action: str = "BUY",
    quantity: int = 1,
    order_type: str = "Limit",
    limit_price: Optional[float] = None,
    dry_run: bool = False,
) -> dict:
    """
    Place a single-leg option order on TastyTrade sandbox.

    Args:
        symbol: Underlying (e.g. "SPY")
        expiry: Expiration YYYYMMDD or YYYY-MM-DD
        strike: Strike price
        right: "C" or "P"
        action: "BUY" or "SELL"
        quantity: Number of contracts
        order_type: "Limit" or "Market"
        limit_price: Required for Limit orders
        dry_run: If True, validates without placing

    Returns:
        Dict with order details and status.
    """
    _ensure_session()
    acct = _get_account()

    occ_symbol = _build_occ_symbol(symbol, expiry, right, strike)

    # Map action
    if action.upper() == "BUY":
        tt_action = "Buy to Open"
    elif action.upper() == "SELL":
        tt_action = "Sell to Close"
    else:
        tt_action = action

    leg = {
        "instrument-type": "Equity Option",
        "symbol": occ_symbol,
        "action": tt_action,
        "quantity": quantity,
    }

    order_payload = {
        "time-in-force": "Day",
        "order-type": order_type,
        "legs": [leg],
    }

    if order_type == "Limit" and limit_price is not None:
        # TastyTrade: negative = debit (buying), positive = credit (selling)
        price = -abs(limit_price) if "Buy" in tt_action else abs(limit_price)
        order_payload["price"] = str(round(price, 2))

    # Dry run first
    if dry_run:
        url = f"{_base_url()}/accounts/{acct}/orders/dry-run"
    else:
        url = f"{_base_url()}/accounts/{acct}/orders"

    resp = httpx.post(url, json=order_payload, headers=_headers(), timeout=15)

    if resp.status_code not in (200, 201):
        return {
            "error": "ORDER_FAILED",
            "status_code": resp.status_code,
            "detail": resp.text[:500],
            "order_payload": order_payload,
        }

    data = resp.json().get("data", {})

    if dry_run:
        bp_effect = data.get("buying-power-effect", {})
        return {
            "dry_run": True,
            "symbol": symbol,
            "occ_symbol": occ_symbol.strip(),
            "action": tt_action,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "buying_power_effect": float(bp_effect.get("change-in-buying-power", 0)),
            "margin_requirement": float(bp_effect.get("initial-requirement", 0)),
            "warnings": data.get("warnings", []),
        }

    order_data = data.get("order", data)
    order_id = order_data.get("id", "")

    print(
        f"[TastyTrade-ORDER] {tt_action} {quantity}x {occ_symbol.strip()} "
        f"@ {order_type} {limit_price or 'MKT'} | OrderId={order_id} "
        f"Status={order_data.get('status', 'unknown')}"
    )

    return {
        "order_id": order_id,
        "status": order_data.get("status", "unknown"),
        "symbol": symbol.upper(),
        "occ_symbol": occ_symbol.strip(),
        "contract": {
            "expiry": expiry.replace("-", ""),
            "strike": strike,
            "right": right.upper(),
        },
        "action": tt_action,
        "quantity": quantity,
        "order_type": order_type,
        "limit_price": limit_price,
        "environment": "sandbox" if TT_SANDBOX else "live",
    }


def tt_place_bracket_order(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    quantity: int = 1,
    entry_price: float = 0,
    target_price: float = 0,
    stop_price: float = 0,
) -> dict:
    """
    Place OTOCO bracket: entry + take-profit + stop-loss.
    TastyTrade supports complex orders (OTOCO).
    """
    _ensure_session()
    acct = _get_account()

    occ_symbol = _build_occ_symbol(symbol, expiry, right, strike)

    # Entry order (trigger)
    entry_leg = {
        "instrument-type": "Equity Option",
        "symbol": occ_symbol,
        "action": "Buy to Open",
        "quantity": quantity,
    }
    trigger_order = {
        "time-in-force": "Day",
        "order-type": "Limit",
        "price": str(round(-abs(entry_price), 2)),
        "legs": [entry_leg],
    }

    # Take profit (sell at target)
    tp_leg = {
        "instrument-type": "Equity Option",
        "symbol": occ_symbol,
        "action": "Sell to Close",
        "quantity": quantity,
    }
    tp_order = {
        "time-in-force": "GTC",
        "order-type": "Limit",
        "price": str(round(abs(target_price), 2)),
        "legs": [tp_leg],
    }

    # Stop loss
    sl_leg = {
        "instrument-type": "Equity Option",
        "symbol": occ_symbol,
        "action": "Sell to Close",
        "quantity": quantity,
    }
    sl_order = {
        "time-in-force": "GTC",
        "order-type": "Stop",
        "stop-trigger": str(round(abs(stop_price), 2)),
        "legs": [sl_leg],
    }

    complex_payload = {
        "type": "OTOCO",
        "trigger-order": trigger_order,
        "orders": [tp_order, sl_order],
    }

    url = f"{_base_url()}/accounts/{acct}/complex-orders"
    resp = httpx.post(url, json=complex_payload, headers=_headers(), timeout=15)

    if resp.status_code not in (200, 201):
        return {
            "error": "BRACKET_FAILED",
            "status_code": resp.status_code,
            "detail": resp.text[:500],
        }

    data = resp.json().get("data", {})

    return {
        "order_type": "OTOCO_BRACKET",
        "symbol": symbol.upper(),
        "occ_symbol": occ_symbol.strip(),
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "quantity": quantity,
        "complex_order_id": data.get("id", ""),
        "status": data.get("status", "unknown"),
        "environment": "sandbox" if TT_SANDBOX else "live",
    }


# ---------------------------------------------------------------------------
# Positions & Orders
# ---------------------------------------------------------------------------

def tt_get_positions(account: str = "") -> list:
    """Get all current positions."""
    _ensure_session()
    acct = account or _get_account()

    url = f"{_base_url()}/accounts/{acct}/positions"
    resp = httpx.get(url, headers=_headers(), timeout=15)

    if resp.status_code != 200:
        return []

    items = resp.json().get("data", {}).get("items", [])
    positions = []
    for item in items:
        positions.append({
            "symbol": item.get("symbol", ""),
            "underlying": item.get("underlying-symbol", ""),
            "instrument_type": item.get("instrument-type", ""),
            "quantity": int(item.get("quantity", 0)),
            "direction": item.get("quantity-direction", ""),
            "avg_open_price": float(item.get("average-open-price", 0)),
            "close_price": float(item.get("close-price", 0)),
            "mark_price": float(item.get("mark-price", 0)),
            "realized_pnl": float(item.get("realized-day-gain", 0)),
            "unrealized_pnl": float(item.get("mark", 0)) - float(item.get("average-open-price", 0)),
            "multiplier": int(item.get("multiplier", 100)),
        })

    return positions


def tt_get_orders(account: str = "") -> list:
    """Get all live/open orders."""
    _ensure_session()
    acct = account or _get_account()

    url = f"{_base_url()}/accounts/{acct}/orders/live"
    resp = httpx.get(url, headers=_headers(), timeout=15)

    if resp.status_code != 200:
        return []

    items = resp.json().get("data", {}).get("items", [])
    orders = []
    for item in items:
        legs = item.get("legs", [])
        orders.append({
            "order_id": item.get("id", ""),
            "status": item.get("status", ""),
            "order_type": item.get("order-type", ""),
            "time_in_force": item.get("time-in-force", ""),
            "price": item.get("price", ""),
            "legs": [{
                "symbol": leg.get("symbol", ""),
                "action": leg.get("action", ""),
                "quantity": leg.get("quantity", 0),
                "fill_quantity": leg.get("remaining-quantity", 0),
            } for leg in legs],
            "created_at": item.get("received-at", ""),
        })

    return orders


def tt_cancel_order(order_id: str, account: str = "") -> dict:
    """Cancel an open order."""
    _ensure_session()
    acct = account or _get_account()

    url = f"{_base_url()}/accounts/{acct}/orders/{order_id}"
    resp = httpx.delete(url, headers=_headers(), timeout=15)

    if resp.status_code not in (200, 204):
        return {"error": "CANCEL_FAILED", "status_code": resp.status_code, "detail": resp.text[:300]}

    return {"status": "cancelled", "order_id": order_id}


def tt_close_position(
    symbol: str,
    expiry: str,
    strike: float,
    right: str,
    quantity: int = 1,
    order_type: str = "Market",
    limit_price: Optional[float] = None,
    account: str = "",
) -> dict:
    """Close an existing option position."""
    return tt_place_option_order(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        right=right,
        action="SELL",
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
    )


# ---------------------------------------------------------------------------
# AI Signal Execution (mirrors ibkr_client.execute_signal)
# ---------------------------------------------------------------------------

def tt_execute_signal(
    analysis_result: dict,
    quantity: int = 1,
    order_type: str = "Limit",
    limit_price: Optional[float] = None,
    dry_run: bool = False,
    bracket: bool = False,
) -> dict:
    """
    Execute a trade from 18-layer engine output on TastyTrade sandbox.
    This is the AI bridge: IBKR data -> engine signal -> TastyTrade execution.

    Args:
        analysis_result: Dict from /engine18/analyze or /engine18/signal
        quantity: Number of contracts (overrides engine suggestion if > 1)
        order_type: "Limit" or "Market"
        limit_price: Override price. None = use engine entry price.
        dry_run: If True, validates without placing.
        bracket: If True, places OTOCO bracket (entry + target + stop).

    Returns:
        Dict with order details and status.
    """
    # Extract signal from analysis result (same logic as ibkr_client)
    ticker = analysis_result.get("ticker", "")
    trade_valid = analysis_result.get("analysis_summary", {}).get("trade_valid",
                  analysis_result.get("trade_valid", False))
    action = analysis_result.get("analysis_summary", {}).get("action",
             analysis_result.get("action",
             analysis_result.get("signal", "FLAT")))
    direction = analysis_result.get("analysis_summary", {}).get("direction",
                analysis_result.get("direction", "NEUTRAL"))
    confidence = analysis_result.get("analysis_summary", {}).get("confidence",
                 analysis_result.get("confidence", "NO_TRADE"))
    win_prob = analysis_result.get("analysis_summary", {}).get("win_probability",
               analysis_result.get("win_probability", 0))

    # Option details
    opt = analysis_result.get("option_recommendation", analysis_result.get("trade", {}))
    strike = opt.get("strike", analysis_result.get("strike", 0))
    expiry_date = opt.get("expiry_date", opt.get("expiry", analysis_result.get("expiry_date", "")))
    expiry_dte = opt.get("expiry_dte", analysis_result.get("expiry_dte", 0))
    delta = opt.get("delta", analysis_result.get("delta", 0))

    # Execution plan
    exec_plan = analysis_result.get("execution_plan", analysis_result.get("plan", {}))
    entry = exec_plan.get("entry", analysis_result.get("entry_price", 0))
    target = exec_plan.get("target", analysis_result.get("target_price", 0))
    stop = exec_plan.get("stop", analysis_result.get("stop_price", 0))
    contracts = exec_plan.get("contracts", analysis_result.get("contracts_suggested", quantity))

    # Determine right (C/P)
    if action in ("BUY_CALL", "SELL_PUT"):
        right = "C"
    elif action in ("BUY_PUT", "SELL_CALL"):
        right = "P"
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

    if not expiry_date:
        return {"error": "NO_EXPIRY", "reason": "Engine did not provide expiry_date"}

    use_qty = quantity if quantity > 1 else max(1, int(contracts or 1))
    use_price = limit_price or entry

    # Build preview
    preview = {
        "broker": "tastytrade",
        "environment": "sandbox" if TT_SANDBOX else "live",
        "symbol": ticker,
        "expiry": expiry_date,
        "strike": strike,
        "right": right,
        "quantity": use_qty,
        "order_type": order_type,
        "limit_price": use_price,
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

    if bracket and target > 0 and stop > 0:
        if dry_run:
            preview["dry_run"] = True
            preview["bracket"] = {"entry": use_price, "target": target, "stop": stop}
            return preview

        result = tt_place_bracket_order(
            symbol=ticker,
            expiry=expiry_date,
            strike=strike,
            right=right,
            quantity=use_qty,
            entry_price=use_price,
            target_price=target,
            stop_price=stop,
        )
        result["engine_signal"] = preview["engine_signal"]
        return result

    if dry_run:
        # Use TastyTrade's dry-run endpoint for real validation
        result = tt_place_option_order(
            symbol=ticker,
            expiry=expiry_date,
            strike=strike,
            right=right,
            action="BUY",
            quantity=use_qty,
            order_type=order_type,
            limit_price=use_price,
            dry_run=True,
        )
        result["engine_signal"] = preview["engine_signal"]
        return result

    # Place the order
    result = tt_place_option_order(
        symbol=ticker,
        expiry=expiry_date,
        strike=strike,
        right=right,
        action="BUY",
        quantity=use_qty,
        order_type=order_type,
        limit_price=use_price,
    )
    result["engine_signal"] = preview["engine_signal"]
    return result


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

def tt_status() -> dict:
    """Check TastyTrade connection status."""
    global _session_token, _session_expiry

    connected = bool(_session_token and _session_expiry and datetime.now() < _session_expiry)

    result = {
        "broker": "tastytrade",
        "environment": "sandbox" if TT_SANDBOX else "live",
        "base_url": _base_url(),
        "connected": connected,
        "account": _account_number,
        "credentials_set": bool(TT_USERNAME and TT_PASSWORD),
    }

    if connected:
        try:
            balance = tt_get_account_balance()
            result["balance"] = balance
        except Exception:
            pass

    return result
