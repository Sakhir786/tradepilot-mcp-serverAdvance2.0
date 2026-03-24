"""
TradePilot Chat API Router
============================
REST endpoints for the Claude-powered conversational AI chat.

Claude is the brain — it decides which TradePilot tools to call,
interprets the results, and responds naturally.

Endpoints:
  POST /chat/send                    - Send a message, get AI response
  GET  /chat/sessions                - List chat sessions
  GET  /chat/sessions/{id}/messages  - Get messages for a session
  DELETE /chat/sessions/{id}         - Delete a session
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

import database as db
from chat_engine import (
    process_message,
    get_or_create_session,
    _sessions,
)

router = APIRouter(prefix="/chat", tags=["AI Chat"])


class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None


@router.post("/send")
async def send_message(msg: ChatMessage):
    """
    Send a natural language message to TradePilot AI (powered by Claude).

    Claude interprets your message, decides which TradePilot tools to call
    (18-layer analysis, TastyTrade execution, etc.), and responds conversationally.

    Examples:
      - "What's the play on SPY?"
      - "Should I buy TSLA?"
      - "Execute that trade on sandbox"
      - "How's my account?"
      - "Compare AAPL MSFT GOOGL"
      - "What's the news on NVDA?"
    """
    try:
        result = process_message(msg.session_id, msg.message)

        return {
            "session_id": result["session_id"],
            "response": result["response"],
            "tool_used": result.get("tool_used", ""),
            "has_data": result.get("has_analysis", False),
            "data": result.get("analysis_data"),
        }

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

    _sessions.pop(session_id, None)
    return {"status": "deleted", "session_id": session_id}
