"""
TradePilot Conversational AI Chat Engine (Claude-Powered)
==========================================================
Uses Claude API with tool_use to power natural conversations.
Claude decides which TradePilot tools to call, interprets results,
and responds intelligently.

Flow:
  User message → Claude API (with tools) → Tool calls → Results back to Claude → Response
"""

import os
import json
import uuid
import math
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv

import database as db

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_CHAT_MODEL", "claude-sonnet-4-20250514")

# ---------------------------------------------------------------------------
# Claude API Client (lightweight, no SDK dependency needed)
# ---------------------------------------------------------------------------

import httpx

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"


def _call_claude(messages: list, tools: list = None, system: str = "",
                 max_tokens: int = 2048) -> dict:
    """Call Claude API directly via HTTP (no SDK needed)."""
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not set in .env"}

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools

    resp = httpx.post(CLAUDE_API_URL, json=payload, headers=headers, timeout=120)

    if resp.status_code != 200:
        return {"error": f"Claude API error {resp.status_code}: {resp.text[:500]}"}

    return resp.json()


# ---------------------------------------------------------------------------
# Tool Definitions (what Claude can call)
# ---------------------------------------------------------------------------

TRADEPILOT_TOOLS = [
    {
        "name": "analyze_stock",
        "description": "Run full 18-layer technical + options analysis on a stock. Returns direction (BULLISH/BEARISH/NEUTRAL), confidence, option strike/expiry recommendations, entry/target/stop prices, and reasoning. Use this when the user asks about any stock, wants a signal, or asks what to trade.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. SPY, AAPL, TSLA)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["scalp", "swing", "intraday", "leaps"],
                    "description": "Trading mode. scalp=0-2 DTE, swing=7-45 DTE (default), intraday=0-1 DTE, leaps=180+ DTE"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "execute_trade",
        "description": "Execute a trade on TastyTrade sandbox (paper trading) based on the last analysis. Always does a dry-run preview first. Use when user says 'execute', 'send to sandbox', 'paper trade it', 'place the trade', etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "If true, actually places the order. If false (default), shows a dry-run preview."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_account_balance",
        "description": "Get TastyTrade account balance including cash, buying power, net liquidation value, and P&L.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_positions",
        "description": "Get all current open positions from TastyTrade sandbox account.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_orders",
        "description": "Get all open/pending orders from TastyTrade sandbox account.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "scan_stocks",
        "description": "Run 18-layer analysis on multiple stocks and compare them. Returns ranked results by win probability. Use when user wants to compare tickers or find the best setup.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock symbols to scan (2-8 tickers)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["scalp", "swing", "intraday", "leaps"],
                    "description": "Trading mode (default: swing)"
                }
            },
            "required": ["symbols"]
        }
    },
    {
        "name": "manage_watchlist",
        "description": "View, add to, or remove from the watchlist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["show", "add", "remove"],
                    "description": "What to do with the watchlist"
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol to add or remove (required for add/remove)"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "get_trade_history",
        "description": "Get trade performance statistics: win rate, total P&L, best/worst trades, etc.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_news",
        "description": "Get latest news for a stock symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["symbol"]
        }
    },
]

SYSTEM_PROMPT = """You are TradePilot AI — a conversational trading assistant built into the TradePilot 18-layer analysis engine.

You help users analyze stocks, execute paper trades on TastyTrade sandbox, manage their watchlist, and check their account.

Key behaviors:
- When a user mentions a stock ticker (SPY, AAPL, TSLA, etc.), analyze it using the analyze_stock tool
- When they say "execute", "send to sandbox", "paper trade it" — use execute_trade
- Keep responses concise and trading-focused
- Present analysis data clearly: direction, confidence, strike, entry/target/stop
- After showing analysis, remind them they can execute it on sandbox
- Default trading mode is "swing" unless the user specifies otherwise
- You have real data — the 18-layer engine runs actual technical analysis with real market data
- For execution: always do a dry-run preview first, then the user confirms

Personality: Direct, knowledgeable, trader-to-trader. Skip fluff. Lead with the signal.
"""


# ---------------------------------------------------------------------------
# Tool Execution (server-side)
# ---------------------------------------------------------------------------

def _convert_numpy(obj):
    """Convert numpy types for JSON serialization."""
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


# Lazy-loaded engine and TastyTrade
_engine_instance = None


def _get_engine():
    global _engine_instance
    try:
        from tradepilot_integration.engine_18layer_core import TradePilotEngine18Layer, TradeMode
        from config import DATA_SOURCE
        if DATA_SOURCE == "ibkr":
            from ibkr_client import get_candles_for_mode, get_full_option_chain_snapshot, get_market_context
        else:
            from polygon_client import get_candles_for_mode, get_full_option_chain_snapshot, get_market_context

        if _engine_instance is None:
            _engine_instance = TradePilotEngine18Layer()

        return {
            "engine": _engine_instance,
            "TradeMode": TradeMode,
            "get_candles_for_mode": get_candles_for_mode,
            "get_full_option_chain_snapshot": get_full_option_chain_snapshot,
            "get_market_context": get_market_context,
        }
    except ImportError as e:
        return {"error": str(e)}


def _get_tt():
    try:
        from tastytrade_client import (
            tt_get_account_balance,
            tt_get_positions,
            tt_get_orders,
            tt_execute_signal,
            tt_status,
        )
        return {
            "tt_get_account_balance": tt_get_account_balance,
            "tt_get_positions": tt_get_positions,
            "tt_get_orders": tt_get_orders,
            "tt_execute_signal": tt_execute_signal,
            "tt_status": tt_status,
        }
    except ImportError:
        return None


def _run_analysis(symbol: str, mode: str = "swing") -> dict:
    """Run 18-layer analysis."""
    e = _get_engine()
    if "error" in e:
        return {"error": f"Engine not available: {e['error']}"}

    engine = e["engine"]
    TradeMode = e["TradeMode"]
    mode_map = {
        "scalp": TradeMode.SCALP,
        "swing": TradeMode.SWING,
        "intraday": TradeMode.INTRADAY,
        "leaps": TradeMode.LEAPS,
    }
    trade_mode = mode_map.get(mode, TradeMode.SWING)

    candles_data = e["get_candles_for_mode"](symbol, mode=mode)
    if not candles_data or "results" not in candles_data:
        return {"error": f"Could not fetch data for {symbol}"}
    if len(candles_data.get("results", [])) < 50:
        return {"error": f"Insufficient data for {symbol}"}

    mode_config = candles_data.get("_mode_config", {})
    tf = f"{mode_config.get('multiplier', 1)}{mode_config.get('timespan', 'day')[0]}"

    options_data = None
    try:
        options_data = e["get_full_option_chain_snapshot"](symbol, limit=100)
    except Exception:
        pass

    market_ctx = {}
    try:
        market_ctx = e["get_market_context"](mode=mode)
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
    return _convert_numpy(engine.to_dict(result))


def execute_tool(tool_name: str, tool_input: dict, session: "ChatSession") -> dict:
    """Execute a tool call and return the result."""

    if tool_name == "analyze_stock":
        symbol = tool_input.get("symbol", "").upper()
        mode = tool_input.get("mode", session.mode or "swing")
        data = _run_analysis(symbol, mode)
        if "error" not in data:
            session.last_analysis = data
            session.last_symbol = symbol
            # Return a condensed version for Claude to interpret
            summary = data.get("analysis_summary", {})
            opt = data.get("option_recommendation", {})
            plan = data.get("execution_plan", {})
            return {
                "ticker": data.get("ticker"),
                "current_price": data.get("current_price"),
                "direction": summary.get("direction"),
                "action": summary.get("action"),
                "confidence": summary.get("confidence"),
                "win_probability": summary.get("win_probability"),
                "trade_valid": summary.get("trade_valid"),
                "strike": opt.get("strike"),
                "delta": opt.get("delta"),
                "expiry_date": opt.get("expiry_date"),
                "expiry_dte": opt.get("expiry_dte"),
                "entry": plan.get("entry"),
                "target": plan.get("target"),
                "stop": plan.get("stop"),
                "risk_reward": plan.get("risk_reward"),
                "contracts": plan.get("contracts"),
                "reasoning": data.get("reasoning", [])[:5],
                "concerns": data.get("concerns", [])[:3],
                "market_context": data.get("market_context", {}),
            }
        return data

    elif tool_name == "execute_trade":
        if not session.last_analysis:
            return {"error": "No analysis to execute. Analyze a stock first."}
        tt = _get_tt()
        if not tt:
            return {"error": "TastyTrade not configured. Set TASTYTRADE_USERNAME and TASTYTRADE_PASSWORD in .env"}
        confirm = tool_input.get("confirm", False)
        try:
            result = tt["tt_execute_signal"](
                session.last_analysis,
                dry_run=not confirm,
            )
            return _convert_numpy(result)
        except Exception as e:
            return {"error": f"Execution failed: {str(e)}"}

    elif tool_name == "get_account_balance":
        tt = _get_tt()
        if not tt:
            return {"error": "TastyTrade not configured."}
        try:
            return _convert_numpy(tt["tt_get_account_balance"]())
        except Exception as e:
            return {"error": str(e)}

    elif tool_name == "get_positions":
        tt = _get_tt()
        if not tt:
            return {"error": "TastyTrade not configured."}
        try:
            return {"positions": tt["tt_get_positions"]()}
        except Exception as e:
            return {"error": str(e)}

    elif tool_name == "get_orders":
        tt = _get_tt()
        if not tt:
            return {"error": "TastyTrade not configured."}
        try:
            return {"orders": tt["tt_get_orders"]()}
        except Exception as e:
            return {"error": str(e)}

    elif tool_name == "scan_stocks":
        symbols = tool_input.get("symbols", [])
        mode = tool_input.get("mode", session.mode or "swing")
        results = []
        for sym in symbols[:8]:
            try:
                data = _run_analysis(sym.upper(), mode)
                data = _convert_numpy(data)
                summary = data.get("analysis_summary", {})
                results.append({
                    "ticker": sym.upper(),
                    "direction": summary.get("direction", "ERROR"),
                    "action": summary.get("action", "ERROR"),
                    "confidence": summary.get("confidence", "N/A"),
                    "win_probability": summary.get("win_probability", 0),
                })
            except Exception:
                results.append({"ticker": sym.upper(), "direction": "ERROR", "error": "Analysis failed"})
        return {"scan_results": results}

    elif tool_name == "manage_watchlist":
        action = tool_input.get("action", "show")
        symbol = tool_input.get("symbol", "").upper()
        if action == "show":
            return {"watchlist": db.get_watchlist()}
        elif action == "add" and symbol:
            return db.add_to_watchlist(symbol)
        elif action == "remove" and symbol:
            return db.remove_from_watchlist(symbol)
        return {"error": "Invalid watchlist action"}

    elif tool_name == "get_trade_history":
        return db.get_trade_stats()

    elif tool_name == "get_news":
        symbol = tool_input.get("symbol", "").upper()
        try:
            from polygon_client import get_news
            data = get_news(symbol)
            results = data.get("results", [])[:5]
            return {"news": [{"title": n.get("title"), "published": n.get("published_utc"), "author": n.get("author")} for n in results]}
        except Exception as e:
            return {"error": str(e)}

    return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Chat Session Manager
# ---------------------------------------------------------------------------

class ChatSession:
    """Manages state for a single chat conversation."""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.mode = "swing"
        self.last_analysis = None
        self.last_symbol = None
        self.conversation_history: List[dict] = []  # Claude message format

    def add_user_message(self, content: str):
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content):
        """content can be str or list of content blocks."""
        if isinstance(content, str):
            self.conversation_history.append({"role": "assistant", "content": content})
        else:
            self.conversation_history.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_use_id: str, result: dict):
        self.conversation_history.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(result, default=str)[:8000],  # Cap size
            }]
        })

    def get_messages(self, max_turns: int = 20) -> list:
        """Get recent conversation history for Claude API."""
        # Keep last N messages to stay within context
        msgs = self.conversation_history[-max_turns * 2:]
        # Ensure first message is from user
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs


# Global session store
_sessions: Dict[str, ChatSession] = {}


def get_or_create_session(session_id: str = None) -> ChatSession:
    """Get existing session or create new one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]

    session = ChatSession(session_id)
    _sessions[session.session_id] = session

    try:
        db.create_chat_session(session.session_id)
    except Exception:
        pass

    return session


# ---------------------------------------------------------------------------
# Main Chat Processing
# ---------------------------------------------------------------------------

def process_message(session_id: str, message: str) -> Dict:
    """
    Process a user message through Claude API with tool use.

    Returns:
        {
            "session_id": str,
            "response": str,         # Claude's text response
            "tool_used": str,        # which tool was called (if any)
            "tool_data": dict,       # raw data from the tool (for UI)
            "has_analysis": bool,    # if analysis data is available for chart
            "analysis_data": dict,   # full analysis data for rendering
        }
    """
    session = get_or_create_session(session_id)

    result = {
        "session_id": session.session_id,
        "response": "",
        "tool_used": "",
        "tool_data": {},
        "has_analysis": False,
        "analysis_data": None,
    }

    # Check if Claude API is configured
    if not ANTHROPIC_API_KEY:
        result["response"] = (
            "**Claude API not configured.**\n\n"
            "Add your API key to `.env`:\n"
            "```\nANTHROPIC_API_KEY=sk-ant-...\n```\n"
            "Get one at: https://console.anthropic.com/settings/keys"
        )
        return result

    # Add user message to conversation
    session.add_user_message(message)

    # Save user message to DB
    db.save_chat_message(session.session_id, "user", message)

    # Call Claude with tools
    try:
        response = _call_claude(
            messages=session.get_messages(),
            tools=TRADEPILOT_TOOLS,
            system=SYSTEM_PROMPT,
        )

        if "error" in response:
            result["response"] = f"AI Error: {response['error']}"
            return result

        # Process response — handle tool use loop
        max_tool_rounds = 5
        rounds = 0

        while rounds < max_tool_rounds:
            rounds += 1
            content_blocks = response.get("content", [])
            stop_reason = response.get("stop_reason", "")

            # Check if Claude wants to use tools
            if stop_reason == "tool_use":
                # Save assistant's response (with tool_use blocks)
                session.add_assistant_message(content_blocks)

                # Execute each tool call
                for block in content_blocks:
                    if block.get("type") == "tool_use":
                        tool_name = block["name"]
                        tool_input = block.get("input", {})
                        tool_id = block["id"]

                        # Execute the tool
                        tool_result = execute_tool(tool_name, tool_input, session)

                        # Track what was used
                        result["tool_used"] = tool_name
                        result["tool_data"] = tool_result

                        # If it was an analysis, store for chart rendering
                        if tool_name == "analyze_stock" and session.last_analysis:
                            result["has_analysis"] = True
                            result["analysis_data"] = session.last_analysis

                        # Feed result back to Claude
                        session.add_tool_result(tool_id, tool_result)

                # Call Claude again with tool results
                response = _call_claude(
                    messages=session.get_messages(),
                    tools=TRADEPILOT_TOOLS,
                    system=SYSTEM_PROMPT,
                )

                if "error" in response:
                    result["response"] = f"AI Error: {response['error']}"
                    return result

            else:
                # Claude is done — extract text response
                text_parts = []
                for block in content_blocks:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])

                final_response = "\n".join(text_parts)
                session.add_assistant_message(final_response)
                result["response"] = final_response

                # Save to DB
                db.save_chat_message(
                    session.session_id, "assistant", final_response,
                    tool_used=result["tool_used"],
                    tool_data=result["tool_data"]
                )

                # Update session title on first analysis
                if session.last_symbol:
                    try:
                        db.update_chat_session_title(session.session_id, f"Chat: {session.last_symbol}")
                    except Exception:
                        pass

                break

        if not result["response"]:
            result["response"] = "I processed your request but didn't generate a response. Try again?"

    except Exception as e:
        import traceback
        print(f"[Chat] Error: {traceback.format_exc()}")
        result["response"] = f"Something went wrong: {str(e)}"

    return result
