"""
TradePilot Local Database
=========================
SQLite database for watchlists, settings, and analysis history.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tradepilot.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            notes TEXT DEFAULT '',
            added_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            signal TEXT,
            confidence REAL,
            summary TEXT,
            full_result TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        -- Default settings
        INSERT OR IGNORE INTO settings (key, value) VALUES ('default_mode', 'swing');
        INSERT OR IGNORE INTO settings (key, value) VALUES ('theme', 'dark');
    """)
    conn.commit()
    conn.close()


# --- Watchlist ---

def get_watchlist() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM watchlist ORDER BY added_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_to_watchlist(symbol: str, notes: str = "") -> Dict:
    conn = get_connection()
    try:
        conn.execute("INSERT INTO watchlist (symbol, notes) VALUES (?, ?)", (symbol.upper(), notes))
        conn.commit()
        return {"status": "added", "symbol": symbol.upper()}
    except sqlite3.IntegrityError:
        return {"status": "exists", "symbol": symbol.upper()}
    finally:
        conn.close()


def remove_from_watchlist(symbol: str) -> Dict:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
    conn.commit()
    conn.close()
    return {"status": "removed" if cursor.rowcount > 0 else "not_found", "symbol": symbol.upper()}


# --- Analysis History ---

def save_analysis(symbol: str, mode: str, signal: str, confidence: float, summary: str, full_result: dict) -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO analysis_history (symbol, mode, signal, confidence, summary, full_result) VALUES (?, ?, ?, ?, ?, ?)",
        (symbol.upper(), mode, signal, confidence, summary, json.dumps(full_result, default=str))
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_analysis_history(symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
    conn = get_connection()
    if symbol:
        rows = conn.execute(
            "SELECT id, symbol, mode, signal, confidence, summary, created_at FROM analysis_history WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol.upper(), limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, symbol, mode, signal, confidence, summary, created_at FROM analysis_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis_detail(analysis_id: int) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    if row:
        result = dict(row)
        result["full_result"] = json.loads(result["full_result"]) if result["full_result"] else {}
        return result
    return None


# --- Settings ---

def get_setting(key: str, default: str = "") -> str:
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    conn = get_connection()
    conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def get_all_settings() -> Dict[str, str]:
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


# Initialize on import
init_db()
