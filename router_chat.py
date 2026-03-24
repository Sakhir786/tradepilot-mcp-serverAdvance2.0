"""
TradePilot Chat API Router
============================
REST endpoints for the conversational AI chat interface.

Endpoints:
  POST /chat/send         - Send a message, get AI response
  GET  /chat/sessions     - List chat sessions
  GET  /chat/sessions/{id}/messages - Get messages for a session
  DELETE /chat/sessions/{id} - Delete a session
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

import database as db
from chat_engine import (
    process_message,
    get_or_create_session,
    format_signal_response,
    format_balance_response,
    format_positions_response,
    format_orders_response,
    format_execution_response,
    format_scan_response,
    format_watchlist_response,
    format_history_response,
)

# Lazy imports for heavy modules
_engine_available = False
_tt_available = False


def _get_engine_funcs():
    """Lazy-load engine functions to avoid import errors on startup."""
    global _engine_available
    try:
        from tradepilot_integration.router_18layer import (
            full_analysis,
            trade_signal,
        )
        from tradepilot_integration.engine_18layer_core import (
            TradePilotEngine18Layer,
            TradeMode,
        )
        from config import DATA_SOURCE
        if DATA_SOURCE == "ibkr":
            from ibkr_client import get_candles_for_mode, get_full_option_chain_snapshot, get_market_context
        else:
            from polygon_client import get_candles_for_mode, get_full_option_chain_snapshot, get_market_context

        _engine_available = True
        return {
            "engine_cls": TradePilotEngine18Layer,
            "TradeMode": TradeMode,
            "get_candles_for_mode": get_candles_for_mode,
            "get_full_option_chain_snapshot": get_full_option_chain_snapshot,
            "get_market_context": get_market_context,
        }
    except ImportError as e:
        print(f"[Chat] Engine not available: {e}")
        return None


def _get_tt_funcs():
    """Lazy-load TastyTrade functions."""
    global _tt_available
    try:
        from tastytrade_client import (
            tt_get_account_balance,
            tt_get_positions,
            tt_get_orders,
            tt_execute_signal,
            tt_status,
        )
        _tt_available = True
        return {
            "tt_get_account_balance": tt_get_account_balance,
            "tt_get_positions": tt_get_positions,
            "tt_get_orders": tt_get_orders,
            "tt_execute_signal": tt_execute_signal,
            "tt_status": tt_status,
        }
    except ImportError as e:
        print(f"[Chat] TastyTrade not available: {e}")
        return None


# Router
router = APIRouter(prefix="/chat", tags=["AI Chat"])

# Engine singleton
_engine_instance = None


def _get_engine():
    global _engine_instance
    funcs = _get_engine_funcs()
    if not funcs:
        return None, None
    if _engine_instance is None:
        _engine_instance = funcs["engine_cls"]()
    return _engine_instance, funcs


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str
    tool_used: str
    has_data: bool = False
    data: Optional[dict] = None


def _run_analysis(symbol: str, mode: str) -> dict:
    """Run 18-layer analysis and return result dict."""
    engine, funcs = _get_engine()
    if not engine:
        return {"error": "Engine not available. Check server logs."}

    TradeMode = funcs["TradeMode"]
    mode_map = {
        "scalp": TradeMode.SCALP,
        "swing": TradeMode.SWING,
        "intraday": TradeMode.INTRADAY,
        "leaps": TradeMode.LEAPS,
    }
    trade_mode = mode_map.get(mode, TradeMode.SWING)

    candles_data = funcs["get_candles_for_mode"](symbol, mode=mode)
    if not candles_data or "results" not in candles_data:
        return {"error": f"Could not fetch data for {symbol}"}

    if len(candles_data.get("results", [])) < 50:
        return {"error": f"Insufficient data for {symbol} ({len(candles_data.get('results', []))} bars)"}

    mode_config = candles_data.get("_mode_config", {})
    tf = f"{mode_config.get('multiplier', 1)}{mode_config.get('timespan', 'day')[0]}"

    options_data = None
    try:
        options_data = funcs["get_full_option_chain_snapshot"](symbol, limit=100)
    except Exception:
        pass

    market_ctx = {}
    try:
        market_ctx = funcs["get_market_context"](mode=mode)
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

    return engine.to_dict(result)


def _convert_numpy(obj):
    """Convert numpy types for JSON serialization."""
    import math
    try:
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if (math.isnan(val) or math.isinf(val)) else val
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


@router.post("/send")
async def send_message(msg: ChatMessage):
    """
    Send a natural language message to TradePilot AI.

    The AI interprets your message, runs the appropriate analysis/action,
    and responds conversationally.

    Examples:
      - "What's the play on SPY?"
      - "Analyze TSLA for scalp"
      - "My balance"
      - "Execute that trade"
      - "Compare AAPL MSFT GOOGL"
    """
    try:
        # Process through chat engine (intent detection + routing)
        result = process_message(msg.session_id, msg.message)
        session = get_or_create_session(result["session_id"])

        response = ChatResponse(
            session_id=result["session_id"],
            response=result["response"],
            intent=result["intent"],
            tool_used=result["tool_used"],
        )

        # Execute the action if needed
        action = result.get("needs_action", "")

        if action == "run_analysis":
            symbol = result["tool_data"].get("symbol", "")
            mode = result["tool_data"].get("mode", session.mode)
            try:
                data = _run_analysis(symbol, mode)
                data = _convert_numpy(data)
                session.last_analysis = data
                session.last_symbol = symbol
                response.response = format_signal_response(data)
                response.has_data = True
                response.data = data

                # Update session title with first analyzed ticker
                db.update_chat_session_title(session.session_id, f"{symbol} Analysis")
                # Save the real response
                db.save_chat_message(session.session_id, "assistant", response.response,
                                     tool_used="engine18_analyze")
            except Exception as e:
                response.response = f"Analysis failed for {symbol}: {str(e)}"
                print(f"[Chat] Analysis error: {traceback.format_exc()}")

        elif action == "run_execute_preview":
            tt = _get_tt_funcs()
            if not tt:
                response.response = "TastyTrade not configured. Set TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD in .env"
            else:
                try:
                    data = tt["tt_execute_signal"](
                        session.last_analysis,
                        dry_run=True
                    )
                    data = _convert_numpy(data)
                    response.response = format_execution_response(data)
                    response.has_data = True
                    response.data = data
                    db.save_chat_message(session.session_id, "assistant", response.response,
                                         tool_used="tt_dry_run")
                except Exception as e:
                    response.response = f"Execution preview failed: {str(e)}"

        elif action == "run_positions":
            tt = _get_tt_funcs()
            if not tt:
                response.response = "TastyTrade not configured."
            else:
                try:
                    positions = tt["tt_get_positions"]()
                    response.response = format_positions_response(positions)
                    response.has_data = True
                    response.data = {"positions": positions}
                    db.save_chat_message(session.session_id, "assistant", response.response,
                                         tool_used="tt_positions")
                except Exception as e:
                    response.response = f"Could not fetch positions: {str(e)}"

        elif action == "run_balance":
            tt = _get_tt_funcs()
            if not tt:
                response.response = "TastyTrade not configured."
            else:
                try:
                    balance = tt["tt_get_account_balance"]()
                    balance = _convert_numpy(balance)
                    response.response = format_balance_response(balance)
                    response.has_data = True
                    response.data = balance
                    db.save_chat_message(session.session_id, "assistant", response.response,
                                         tool_used="tt_balance")
                except Exception as e:
                    response.response = f"Could not fetch balance: {str(e)}"

        elif action == "run_orders":
            tt = _get_tt_funcs()
            if not tt:
                response.response = "TastyTrade not configured."
            else:
                try:
                    orders = tt["tt_get_orders"]()
                    response.response = format_orders_response(orders)
                    response.has_data = True
                    response.data = {"orders": orders}
                    db.save_chat_message(session.session_id, "assistant", response.response,
                                         tool_used="tt_orders")
                except Exception as e:
                    response.response = f"Could not fetch orders: {str(e)}"

        elif action == "run_scan":
            symbols = result["tool_data"].get("symbols", [])
            mode = result["tool_data"].get("mode", session.mode)
            try:
                scan_results = []
                for sym in symbols[:8]:  # Cap at 8
                    try:
                        data = _run_analysis(sym, mode)
                        data = _convert_numpy(data)
                        summary = data.get("analysis_summary", {})
                        scan_results.append({
                            "ticker": sym,
                            "direction": summary.get("direction", "NEUTRAL"),
                            "action": summary.get("action", "NO_TRADE"),
                            "confidence": summary.get("confidence", "WEAK"),
                            "win_probability": summary.get("win_probability", 0),
                        })
                    except Exception:
                        scan_results.append({"ticker": sym, "direction": "ERROR", "confidence": "N/A", "win_probability": 0, "action": "ERROR"})

                response.response = format_scan_response(scan_results)
                response.has_data = True
                response.data = {"scan": scan_results}
                db.save_chat_message(session.session_id, "assistant", response.response,
                                     tool_used="engine18_scan")
            except Exception as e:
                response.response = f"Scan failed: {str(e)}"

        elif action == "run_watchlist_show":
            try:
                watchlist = db.get_watchlist()
                response.response = format_watchlist_response(watchlist)
                response.has_data = True
                response.data = {"watchlist": watchlist}
            except Exception as e:
                response.response = f"Could not load watchlist: {str(e)}"

        elif action == "run_watchlist_add":
            sym = result["tool_data"].get("symbol", "")
            try:
                res = db.add_to_watchlist(sym)
                if res.get("status") == "added":
                    response.response = f"**{sym}** added to watchlist!"
                else:
                    response.response = f"**{sym}** is already in your watchlist."
            except Exception as e:
                response.response = f"Could not add to watchlist: {str(e)}"

        elif action == "run_watchlist_remove":
            sym = result["tool_data"].get("symbol", "")
            try:
                res = db.remove_from_watchlist(sym)
                if res.get("status") == "removed":
                    response.response = f"**{sym}** removed from watchlist."
                else:
                    response.response = f"**{sym}** was not in your watchlist."
            except Exception as e:
                response.response = f"Could not remove from watchlist: {str(e)}"

        elif action == "run_history":
            try:
                stats = db.get_trade_stats()
                response.response = format_history_response(stats)
                response.has_data = True
                response.data = stats
            except Exception as e:
                response.response = f"Could not load history: {str(e)}"

        elif action == "run_status":
            status_info = {"engine_available": _engine_available}
            tt = _get_tt_funcs()
            if tt:
                try:
                    tt_stat = tt["tt_status"]()
                    status_info["tastytrade"] = tt_stat
                except Exception:
                    status_info["tastytrade"] = {"connected": False}
            else:
                status_info["tastytrade"] = {"connected": False, "reason": "Not configured"}

            lines = [
                "**System Status**",
                f"Engine: {'Online' if _engine_available else 'Offline'}",
                f"TastyTrade: {'Connected' if status_info.get('tastytrade', {}).get('connected') else 'Not connected'}",
            ]
            tt_info = status_info.get("tastytrade", {})
            if tt_info.get("connected"):
                lines.append(f"Environment: {tt_info.get('environment', '?').upper()}")
                lines.append(f"Account: {tt_info.get('account', '?')}")
            response.response = "\n".join(lines)
            response.has_data = True
            response.data = status_info

        return response.model_dump()

    except Exception as e:
        print(f"[Chat] Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@router.get("/sessions")
async def list_sessions(limit: int = 50):
    """List all chat sessions."""
    return db.get_chat_sessions(limit)


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 100):
    """Get all messages for a chat session."""
    messages = db.get_chat_messages(session_id, limit)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")
    return messages


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and its messages."""
    deleted = db.delete_chat_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    # Clean up in-memory session
    from chat_engine import _sessions
    _sessions.pop(session_id, None)

    return {"status": "deleted", "session_id": session_id}
