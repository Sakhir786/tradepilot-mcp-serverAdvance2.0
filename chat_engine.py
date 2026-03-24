"""
TradePilot Conversational AI Chat Engine
==========================================
Interprets natural language messages and routes them to the appropriate
TradePilot endpoints/functions. No external AI API needed — this uses
pattern matching + the 18-layer engine to generate intelligent responses.

Flow:
  User message → Intent detection → Route to function → Format response
"""

import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import database as db


# ---------------------------------------------------------------------------
# Intent Detection
# ---------------------------------------------------------------------------

# Patterns mapped to intents
INTENT_PATTERNS = {
    "analyze": [
        r"(?:analyze|analysis|analyse|what.?s (?:the )?(?:play|setup|look)|how.?s|check|scan|look at|what about|thoughts on)\s+(\$?[A-Z]{1,5})",
        r"(\$?[A-Z]{1,5})\s+(?:analysis|setup|play|signal|outlook|look)",
        r"(?:run|do|get)\s+(?:an?\s+)?(?:analysis|signal|scan)\s+(?:on|for)\s+(\$?[A-Z]{1,5})",
        r"^(\$[A-Z]{1,5})$",
        r"^([A-Z]{2,5})\?*$",
    ],
    "signal": [
        r"(?:quick\s+)?signal\s+(?:on|for)\s+(\$?[A-Z]{1,5})",
        r"(?:what.?s the )?(?:trade|signal|call|play)\s+(?:on|for)\s+(\$?[A-Z]{1,5})",
        r"(?:should i|can i|do i)\s+(?:buy|sell|trade|short|long)\s+(\$?[A-Z]{1,5})",
        r"(?:best )?(?:play|trade|move)\s+(?:on|for)\s+(\$?[A-Z]{1,5})",
    ],
    "execute": [
        r"(?:execute|place|send|submit|paper\s*trade|sandbox)\s+(?:it|that|the (?:trade|order|signal))",
        r"(?:send|place|execute)\s+(?:it|that)\s+(?:on|to)\s+(?:sandbox|tastytrade|tt|paper)",
        r"(?:yeah|yes|yep|sure|do it|go ahead|send it|place it|execute it)",
        r"(?:buy|sell)\s+(?:the\s+)?(?:\d+\s*x?\s*)?(\$?[A-Z]{1,5})\s+(?:\$?\d+)\s*(?:call|put|c|p)",
    ],
    "positions": [
        r"(?:my\s+)?positions",
        r"(?:what.?s|show|get|check)\s+(?:my\s+)?(?:positions|holdings|trades|open\s+trades)",
        r"(?:am i|what am i)\s+(?:holding|in)",
    ],
    "balance": [
        r"(?:my\s+)?(?:balance|account|buying\s*power|portfolio|money|cash)",
        r"(?:how\s+(?:much|is))\s+(?:my|in my)\s+(?:account|balance|portfolio)",
        r"(?:what.?s|show|get|check)\s+(?:my\s+)?(?:balance|account|buying\s*power)",
    ],
    "orders": [
        r"(?:my\s+)?(?:open\s+)?orders",
        r"(?:show|get|check|list)\s+(?:my\s+)?(?:open\s+)?orders",
        r"(?:pending|active)\s+orders",
    ],
    "scan": [
        r"(?:scan|screen|compare)\s+((?:\$?[A-Z]{1,5}[\s,]+){2,}(?:\$?[A-Z]{1,5}))",
        r"(?:what.?s (?:the )?best|compare|which (?:is|one))\s+(?:between|among|of)\s+((?:\$?[A-Z]{1,5}[\s,]+){1,}(?:\$?[A-Z]{1,5}))",
        r"(?:scan|screen)\s+(?:the\s+)?(?:market|stocks|tickers)",
    ],
    "watchlist": [
        r"(?:show|get|my)\s+watchlist",
        r"(?:add)\s+(\$?[A-Z]{1,5})\s+(?:to\s+)?(?:my\s+)?watchlist",
        r"(?:remove|delete)\s+(\$?[A-Z]{1,5})\s+(?:from\s+)?(?:my\s+)?watchlist",
        r"watchlist",
    ],
    "history": [
        r"(?:my\s+)?(?:trade\s+)?history",
        r"(?:show|get|check)\s+(?:my\s+)?(?:past|previous|trade\s+)?(?:trades|history|stats|statistics)",
        r"(?:how\s+(?:have|am)\s+i\s+(?:been\s+)?(?:doing|performed|performing))",
        r"(?:win\s*rate|performance|pnl|p&l|profit|loss)",
    ],
    "mode": [
        r"(?:switch|change|set|use)\s+(?:to\s+)?(?:mode\s+)?(scalp|swing|intraday|leaps)",
        r"(scalp|swing|intraday|leaps)\s+mode",
    ],
    "help": [
        r"(?:help|what can you do|commands|how (?:does|do) (?:this|you) work)",
        r"(?:what|which)\s+(?:commands|things|stuff)\s+(?:can|do)\s+(?:you|i)",
    ],
    "greeting": [
        r"^(?:hi|hello|hey|yo|sup|what.?s up|gm|good morning|good evening)(?:\s|!|\.|$)",
    ],
    "status": [
        r"(?:server|system|engine)\s+(?:status|health|check)",
        r"(?:is the|are you)\s+(?:server|system|engine)\s+(?:running|online|up)",
        r"(?:status|health)\s+check",
    ],
}

# Mode mapping
MODE_KEYWORDS = {
    "scalp": "scalp", "scalping": "scalp",
    "swing": "swing", "swings": "swing",
    "intraday": "intraday", "day": "intraday", "daytrade": "intraday",
    "leaps": "leaps", "leap": "leaps", "long term": "leaps", "longterm": "leaps",
}


def detect_intent(message: str) -> Tuple[str, Dict]:
    """
    Detect user intent from natural language message.
    Returns (intent_name, extracted_params).
    """
    msg = message.strip()
    msg_lower = msg.lower()

    # Check for mode in the message
    mode = None
    for keyword, mode_val in MODE_KEYWORDS.items():
        if keyword in msg_lower:
            mode = mode_val
            break

    # Try each intent pattern
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, msg, re.IGNORECASE)
            if match:
                params = {"mode": mode}
                # Extract symbol(s) from match groups
                if match.groups():
                    raw = match.group(1)
                    # Clean up symbols
                    symbols = [s.strip().strip("$").upper() for s in re.split(r"[,\s]+", raw) if s.strip()]
                    symbols = [s for s in symbols if 1 <= len(s) <= 5 and s.isalpha()]
                    if symbols:
                        params["symbol"] = symbols[0]
                        if len(symbols) > 1:
                            params["symbols"] = symbols
                return intent, params

    # Fallback: check if the message is just a ticker
    ticker_match = re.match(r"^\$?([A-Z]{1,5})\s*\??$", msg)
    if ticker_match:
        return "analyze", {"symbol": ticker_match.group(1), "mode": mode}

    return "unknown", {"mode": mode}


def extract_symbol(message: str) -> Optional[str]:
    """Extract a stock symbol from a message."""
    match = re.search(r"\$?([A-Z]{1,5})", message.upper())
    if match:
        sym = match.group(1)
        # Filter out common words
        common_words = {"I", "A", "THE", "AND", "OR", "TO", "IS", "IT", "IN", "ON",
                        "MY", "ME", "DO", "IF", "AT", "FOR", "OF", "UP", "SO", "NO",
                        "YES", "OK", "GET", "SET", "RUN", "HOW", "CAN", "AM", "BE"}
        if sym not in common_words and len(sym) >= 2:
            return sym
    return None


# ---------------------------------------------------------------------------
# Response Formatters
# ---------------------------------------------------------------------------

def format_signal_response(data: dict) -> str:
    """Format engine analysis result into a chat-friendly message."""
    if "error" in data or "detail" in data:
        return f"Could not analyze: {data.get('error', data.get('detail', 'Unknown error'))}"

    ticker = data.get("ticker", "?")
    price = data.get("current_price", 0)

    # Get analysis summary
    summary = data.get("analysis_summary", {})
    direction = summary.get("direction", data.get("direction", "NEUTRAL"))
    action = summary.get("action", data.get("action", "NO_TRADE"))
    confidence = summary.get("confidence", data.get("confidence", "WEAK"))
    win_prob = summary.get("win_probability", data.get("win_probability", 0))
    trade_valid = summary.get("trade_valid", data.get("trade_valid", False))

    # Options recommendation
    opt = data.get("option_recommendation", {})
    strike = opt.get("strike", data.get("strike", 0))
    delta = opt.get("delta", data.get("delta", 0))
    expiry_date = opt.get("expiry_date", data.get("expiry_date", ""))
    expiry_dte = opt.get("expiry_dte", data.get("expiry_dte", 0))

    # Execution plan
    plan = data.get("execution_plan", {})
    entry = plan.get("entry", data.get("entry_price", 0))
    target = plan.get("target", data.get("target_price", 0))
    stop = plan.get("stop", data.get("stop_price", 0))
    rr = plan.get("risk_reward", data.get("risk_reward", 0))
    contracts = plan.get("contracts", data.get("contracts_suggested", 1))

    # Reasoning
    reasoning = data.get("reasoning", [])
    concerns = data.get("concerns", [])

    # Direction emoji
    dir_icon = "BULLISH" if "BULL" in direction.upper() else "BEARISH" if "BEAR" in direction.upper() else "NEUTRAL"

    lines = []
    lines.append(f"**{ticker}** @ ${price:.2f} — **{dir_icon}**")
    lines.append(f"Confidence: **{confidence}** ({win_prob:.0f}%)")
    lines.append("")

    if trade_valid and action not in ("FLAT", "NO_TRADE"):
        lines.append(f"**Action: {action}**")
        if strike:
            right = "CALL" if "CALL" in action else "PUT" if "PUT" in action else "?"
            lines.append(f"Strike: ${strike} {right} | Delta: {delta:.2f}")
        if expiry_date:
            lines.append(f"Expiry: {expiry_date} ({expiry_dte} DTE)")
        lines.append("")
        if entry:
            lines.append(f"Entry: ${entry:.2f}")
        if target:
            lines.append(f"Target: ${target:.2f}")
        if stop:
            lines.append(f"Stop: ${stop:.2f}")
        if rr:
            lines.append(f"Risk/Reward: {rr:.1f}:1")
        if contracts:
            lines.append(f"Contracts: {contracts}")
    else:
        lines.append("**No valid trade setup at this time.**")

    if reasoning:
        lines.append("")
        lines.append("**Why:**")
        for r in reasoning[:4]:
            lines.append(f"  - {r}")

    if concerns:
        lines.append("")
        lines.append("**Watch out:**")
        for c in concerns[:3]:
            lines.append(f"  - {c}")

    if trade_valid and action not in ("FLAT", "NO_TRADE"):
        lines.append("")
        lines.append('_Say "execute" or "send to sandbox" to paper trade this._')

    return "\n".join(lines)


def format_balance_response(data: dict) -> str:
    """Format TastyTrade balance into chat message."""
    if "error" in data:
        return f"Could not get balance: {data['error']}"

    env = data.get("environment", "sandbox").upper()
    lines = [
        f"**TastyTrade Account** ({env})",
        f"Account: {data.get('account', '?')}",
        "",
        f"Cash: ${data.get('cash_balance', 0):,.2f}",
        f"Net Liq: ${data.get('net_liquidating_value', 0):,.2f}",
        f"Option BP: ${data.get('option_buying_power', 0):,.2f}",
        f"Equity BP: ${data.get('equity_buying_power', 0):,.2f}",
    ]
    return "\n".join(lines)


def format_positions_response(positions: list) -> str:
    """Format TastyTrade positions into chat message."""
    if not positions:
        return "No open positions."

    lines = [f"**Open Positions** ({len(positions)} total)", ""]
    for p in positions:
        sym = p.get("symbol", "?")
        qty = p.get("quantity", 0)
        direction = p.get("direction", "")
        avg = p.get("avg_open_price", 0)
        mark = p.get("mark_price", 0)
        pnl = (mark - avg) * qty * p.get("multiplier", 100) if avg else 0
        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        lines.append(f"  {direction} {qty}x **{sym}** @ ${avg:.2f} → ${mark:.2f} ({pnl_str})")

    return "\n".join(lines)


def format_orders_response(orders: list) -> str:
    """Format TastyTrade orders into chat message."""
    if not orders:
        return "No open orders."

    lines = [f"**Open Orders** ({len(orders)} total)", ""]
    for o in orders:
        oid = o.get("order_id", "?")
        status = o.get("status", "?")
        price = o.get("price", "MKT")
        legs = o.get("legs", [])
        leg_str = ", ".join(f"{l.get('action')} {l.get('quantity')}x {l.get('symbol', '?')}" for l in legs)
        lines.append(f"  #{oid} [{status}] {leg_str} @ {price}")

    return "\n".join(lines)


def format_execution_response(data: dict) -> str:
    """Format trade execution result."""
    if "error" in data:
        reason = data.get("reason", data.get("detail", "Unknown"))
        return f"Could not execute: {data['error']} — {reason}"

    if data.get("dry_run"):
        lines = [
            "**Dry Run Preview (not executed)**",
            f"Symbol: {data.get('symbol', '?')}",
            f"Contract: {data.get('occ_symbol', '?')}",
            f"Action: {data.get('action', '?')}",
            f"Qty: {data.get('quantity', '?')}",
            f"Price: ${data.get('limit_price', 'MKT')}",
            f"BP Impact: ${data.get('buying_power_effect', 0):,.2f}",
        ]
        if data.get("warnings"):
            lines.append(f"Warnings: {', '.join(data['warnings'])}")
        lines.append("")
        lines.append('_Say "confirm" or "go ahead" to place the trade._')
        return "\n".join(lines)

    lines = [
        "**Trade Placed!**",
        f"Order ID: {data.get('order_id', '?')}",
        f"Status: {data.get('status', '?')}",
        f"Symbol: {data.get('symbol', '?')} ({data.get('occ_symbol', '')})",
        f"Action: {data.get('action', '?')}",
        f"Qty: {data.get('quantity', '?')}",
        f"Price: ${data.get('limit_price', 'MKT')}",
        f"Env: {data.get('environment', 'sandbox').upper()}",
    ]

    signal = data.get("engine_signal", {})
    if signal:
        lines.append("")
        lines.append(f"Engine: {signal.get('direction', '')} {signal.get('confidence', '')} ({signal.get('win_probability', 0):.0f}%)")

    return "\n".join(lines)


def format_scan_response(results: list) -> str:
    """Format multi-ticker scan results."""
    if not results:
        return "No scan results."

    lines = ["**Multi-Ticker Scan Results**", ""]
    for r in results:
        ticker = r.get("ticker", "?")
        direction = r.get("direction", "?")
        confidence = r.get("confidence", "?")
        prob = r.get("win_probability", 0)
        action = r.get("action", "?")
        icon = "UP" if "BULL" in str(direction).upper() else "DOWN" if "BEAR" in str(direction).upper() else "--"
        lines.append(f"  {icon} **{ticker}** — {direction} {confidence} ({prob:.0f}%) → {action}")

    return "\n".join(lines)


def format_watchlist_response(watchlist: list) -> str:
    """Format watchlist."""
    if not watchlist:
        return "Your watchlist is empty. Say \"add AAPL to watchlist\" to start."

    lines = [f"**Watchlist** ({len(watchlist)} symbols)", ""]
    for w in watchlist:
        lines.append(f"  - {w['symbol']}")
    return "\n".join(lines)


def format_history_response(stats: dict) -> str:
    """Format trade stats/history."""
    if not stats or not stats.get("total_trades"):
        return "No trade history yet."

    lines = [
        "**Trade Performance**",
        "",
        f"Total Trades: {stats.get('total_trades', 0)}",
        f"Open: {stats.get('open_trades', 0)} | Closed: {stats.get('closed_trades', 0)}",
        f"Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}",
        f"Win Rate: {stats.get('win_rate', 0):.1f}%",
        f"Total P&L: ${stats.get('total_pnl', 0):,.2f}",
        f"Avg P&L: {stats.get('avg_pnl_pct', 0):.1f}%",
        f"Best Trade: ${stats.get('best_trade', 0):,.2f}",
        f"Worst Trade: ${stats.get('worst_trade', 0):,.2f}",
    ]
    return "\n".join(lines)


HELP_TEXT = """**TradePilot AI Chat**

Just talk to me naturally! Here's what I can do:

**Analyze stocks:**
  "What's the play on SPY?"
  "Analyze TSLA for swing"
  "AAPL signal"
  "$NVDA"

**Execute trades (sandbox):**
  "Send it to sandbox"
  "Execute that trade"
  "Paper trade it"

**Account info:**
  "My balance"
  "Show positions"
  "Open orders"

**Scan multiple tickers:**
  "Compare SPY QQQ IWM"
  "Scan AAPL MSFT GOOGL AMZN"

**Watchlist:**
  "Show watchlist"
  "Add TSLA to watchlist"

**History:**
  "My trade history"
  "How am I doing?"
  "Win rate"

**Modes:**
  "Switch to scalp mode"
  "Use leaps mode"

Just type a ticker symbol to get started!
"""

GREETING_RESPONSES = [
    "Hey! Ready to find some trades. What ticker are you looking at?",
    "What's up! Drop a ticker and I'll run the 18-layer analysis.",
    "Hey! TradePilot AI at your service. What symbol should I analyze?",
]


# ---------------------------------------------------------------------------
# Chat Session Manager
# ---------------------------------------------------------------------------

class ChatSession:
    """Manages state for a single chat conversation."""

    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.mode = "swing"
        self.last_analysis = None  # Store last analysis result for "execute" commands
        self.last_symbol = None
        self._greeting_idx = 0

    def get_greeting(self) -> str:
        resp = GREETING_RESPONSES[self._greeting_idx % len(GREETING_RESPONSES)]
        self._greeting_idx += 1
        return resp


# Global session store (in-memory, keyed by session_id)
_sessions: Dict[str, ChatSession] = {}


def get_or_create_session(session_id: str = None) -> ChatSession:
    """Get existing session or create new one."""
    if session_id and session_id in _sessions:
        return _sessions[session_id]

    session = ChatSession(session_id)
    _sessions[session.session_id] = session

    # Persist to DB
    try:
        db.create_chat_session(session.session_id)
    except Exception:
        pass  # Already exists

    return session


def process_message(session_id: str, message: str) -> Dict:
    """
    Main entry point: process a user message and return a response.

    Returns:
        {
            "session_id": str,
            "response": str,
            "intent": str,
            "tool_used": str,     # which internal tool was called
            "tool_data": dict,    # raw data from the tool
            "needs_action": str,  # "execute", "confirm", etc. or ""
        }
    """
    session = get_or_create_session(session_id)
    intent, params = detect_intent(message)

    # Apply mode if found in message
    if params.get("mode"):
        session.mode = params["mode"]

    # Use last symbol if none found
    if not params.get("symbol") and session.last_symbol:
        if intent in ("analyze", "signal"):
            # Only use last symbol if the message doesn't look like it has one
            pass  # Don't auto-fill for these — require explicit symbol
        elif intent == "execute":
            params["symbol"] = session.last_symbol

    result = {
        "session_id": session.session_id,
        "response": "",
        "intent": intent,
        "tool_used": "",
        "tool_data": {},
        "needs_action": "",
    }

    # --- Route to handler ---

    if intent == "greeting":
        result["response"] = session.get_greeting()

    elif intent == "help":
        result["response"] = HELP_TEXT

    elif intent == "mode":
        mode = params.get("mode", "swing")
        session.mode = mode
        result["response"] = f"Mode switched to **{mode.upper()}**. All future analyses will use {mode} settings."
        result["tool_used"] = "mode_switch"

    elif intent in ("analyze", "signal"):
        symbol = params.get("symbol")
        if not symbol:
            result["response"] = "Which ticker? Just type a symbol like SPY, AAPL, or TSLA."
            return result

        result["tool_used"] = "engine18_analyze"
        result["tool_data"] = {
            "action": "analyze",
            "symbol": symbol,
            "mode": session.mode,
        }
        result["response"] = f"Analyzing **{symbol}** in {session.mode} mode..."
        result["needs_action"] = "run_analysis"
        session.last_symbol = symbol

    elif intent == "execute":
        if session.last_analysis and session.last_analysis.get("analysis_summary", {}).get("trade_valid"):
            result["tool_used"] = "tt_execute"
            result["tool_data"] = {
                "action": "execute",
                "analysis": session.last_analysis,
                "dry_run": True,  # Always preview first
            }
            result["response"] = "Previewing trade on TastyTrade sandbox..."
            result["needs_action"] = "run_execute_preview"
        elif session.last_analysis:
            result["response"] = "The last analysis didn't produce a valid trade signal. Try another ticker or timeframe."
        else:
            result["response"] = "No analysis to execute. Analyze a ticker first! e.g., \"What's the play on SPY?\""

    elif intent == "positions":
        result["tool_used"] = "tt_positions"
        result["tool_data"] = {"action": "positions"}
        result["response"] = "Fetching your positions..."
        result["needs_action"] = "run_positions"

    elif intent == "balance":
        result["tool_used"] = "tt_balance"
        result["tool_data"] = {"action": "balance"}
        result["response"] = "Checking your account..."
        result["needs_action"] = "run_balance"

    elif intent == "orders":
        result["tool_used"] = "tt_orders"
        result["tool_data"] = {"action": "orders"}
        result["response"] = "Fetching open orders..."
        result["needs_action"] = "run_orders"

    elif intent == "scan":
        symbols = params.get("symbols", [])
        if not symbols and params.get("symbol"):
            symbols = [params["symbol"]]
        if len(symbols) < 2:
            result["response"] = "Give me 2+ symbols to compare. e.g., \"Compare SPY QQQ IWM AAPL\""
            return result

        result["tool_used"] = "engine18_scan"
        result["tool_data"] = {
            "action": "scan",
            "symbols": symbols,
            "mode": session.mode,
        }
        result["response"] = f"Scanning {', '.join(symbols)}..."
        result["needs_action"] = "run_scan"

    elif intent == "watchlist":
        # Check for add/remove
        add_match = re.search(r"add\s+(\$?[A-Z]{1,5})", message, re.IGNORECASE)
        remove_match = re.search(r"(?:remove|delete)\s+(\$?[A-Z]{1,5})", message, re.IGNORECASE)

        if add_match:
            sym = add_match.group(1).strip("$").upper()
            result["tool_used"] = "watchlist_add"
            result["tool_data"] = {"action": "add", "symbol": sym}
            result["needs_action"] = "run_watchlist_add"
            result["response"] = f"Adding {sym} to watchlist..."
        elif remove_match:
            sym = remove_match.group(1).strip("$").upper()
            result["tool_used"] = "watchlist_remove"
            result["tool_data"] = {"action": "remove", "symbol": sym}
            result["needs_action"] = "run_watchlist_remove"
            result["response"] = f"Removing {sym} from watchlist..."
        else:
            result["tool_used"] = "watchlist_show"
            result["tool_data"] = {"action": "show"}
            result["needs_action"] = "run_watchlist_show"
            result["response"] = "Loading watchlist..."

    elif intent == "history":
        result["tool_used"] = "trade_stats"
        result["tool_data"] = {"action": "history"}
        result["needs_action"] = "run_history"
        result["response"] = "Pulling up your stats..."

    elif intent == "status":
        result["tool_used"] = "system_status"
        result["tool_data"] = {"action": "status"}
        result["needs_action"] = "run_status"
        result["response"] = "Checking system status..."

    else:
        # Unknown intent — try to extract a symbol
        symbol = extract_symbol(message)
        if symbol:
            result["tool_used"] = "engine18_analyze"
            result["tool_data"] = {"action": "analyze", "symbol": symbol, "mode": session.mode}
            result["response"] = f"Analyzing **{symbol}** in {session.mode} mode..."
            result["needs_action"] = "run_analysis"
            session.last_symbol = symbol
        else:
            result["response"] = (
                "I'm not sure what you mean. Try:\n"
                "  - A ticker symbol like **SPY** or **$TSLA**\n"
                "  - \"What's the play on AAPL?\"\n"
                "  - \"My balance\" or \"My positions\"\n"
                "  - \"help\" for all commands"
            )

    # Save messages to DB
    db.save_chat_message(session.session_id, "user", message)
    db.save_chat_message(
        session.session_id, "assistant", result["response"],
        tool_used=result["tool_used"],
        tool_data=result["tool_data"]
    )

    return result
