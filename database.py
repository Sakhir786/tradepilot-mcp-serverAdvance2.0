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

        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            total_signals INTEGER DEFAULT 0,
            trades_taken INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            timeouts INTEGER DEFAULT 0,
            win_rate REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            full_result TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS live_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER,
            symbol TEXT NOT NULL,
            mode TEXT DEFAULT 'swing',
            action TEXT NOT NULL,
            right TEXT NOT NULL,
            strike REAL NOT NULL,
            expiry TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            entry_price REAL DEFAULT 0,
            exit_price REAL DEFAULT 0,
            stop_price REAL DEFAULT 0,
            target_price REAL DEFAULT 0,
            status TEXT DEFAULT 'PENDING',
            pnl REAL DEFAULT 0,
            pnl_pct REAL DEFAULT 0,
            close_reason TEXT DEFAULT '',
            confidence TEXT DEFAULT '',
            win_probability REAL DEFAULT 0,
            signal_data TEXT DEFAULT '',
            entry_time TEXT DEFAULT (datetime('now')),
            exit_time TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_live_trades_symbol ON live_trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_live_trades_status ON live_trades(status);

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


# --- Paper Trades / Backtests ---

def save_backtest(symbol: str, mode: str, total_signals: int, trades_taken: int,
                  wins: int, losses: int, timeouts: int, win_rate: float,
                  total_pnl: float, full_result: str) -> int:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO paper_trades (symbol, mode, total_signals, trades_taken, wins, losses, timeouts, win_rate, total_pnl, full_result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (symbol.upper(), mode, total_signals, trades_taken, wins, losses, timeouts, win_rate, total_pnl, full_result)
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_backtests(symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
    conn = get_connection()
    if symbol:
        rows = conn.execute(
            "SELECT id, symbol, mode, total_signals, trades_taken, wins, losses, timeouts, win_rate, total_pnl, created_at FROM paper_trades WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol.upper(), limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, symbol, mode, total_signals, trades_taken, wins, losses, timeouts, win_rate, total_pnl, created_at FROM paper_trades ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_backtest_detail(backtest_id: int) -> Optional[Dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM paper_trades WHERE id = ?", (backtest_id,)).fetchone()
    conn.close()
    if row:
        result = dict(row)
        result["full_result"] = json.loads(result["full_result"]) if result["full_result"] else {}
        return result
    return None


# --- Live Trades ---

def save_live_trade(order_id: int, symbol: str, mode: str, action: str,
                    right: str, strike: float, expiry: str, quantity: int,
                    entry_price: float = 0, stop_price: float = 0,
                    target_price: float = 0, confidence: str = "",
                    win_probability: float = 0, signal_data: dict = None) -> int:
    """Record a new live trade execution."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO live_trades
           (order_id, symbol, mode, action, right, strike, expiry, quantity,
            entry_price, stop_price, target_price, status, confidence,
            win_probability, signal_data)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?)""",
        (order_id, symbol.upper(), mode, action, right, strike, expiry,
         quantity, entry_price, stop_price, target_price, confidence,
         win_probability, json.dumps(signal_data or {}, default=str))
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def update_live_trade_fill(order_id: int, fill_price: float, status: str = "FILLED") -> bool:
    """Update trade with fill price after execution."""
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE live_trades SET entry_price = ?, status = ?, updated_at = datetime('now') WHERE order_id = ? AND status = 'PENDING'",
        (fill_price, status, order_id)
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def close_live_trade(order_id: int, exit_price: float, close_reason: str = "MANUAL") -> bool:
    """Close a live trade and calculate P&L."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM live_trades WHERE order_id = ?", (order_id,)).fetchone()
    if not row:
        conn.close()
        return False

    entry = row["entry_price"]
    qty = row["quantity"]
    action = row["action"]

    # P&L calc: long = (exit - entry), short = (entry - exit), per contract * 100
    if action in ("BUY",):
        pnl = (exit_price - entry) * qty * 100
    else:
        pnl = (entry - exit_price) * qty * 100

    pnl_pct = ((exit_price - entry) / entry * 100) if entry > 0 else 0
    if action == "SELL":
        pnl_pct = -pnl_pct

    conn.execute(
        """UPDATE live_trades
           SET exit_price = ?, pnl = ?, pnl_pct = ?, status = 'CLOSED',
               close_reason = ?, exit_time = datetime('now'), updated_at = datetime('now')
           WHERE order_id = ?""",
        (exit_price, round(pnl, 2), round(pnl_pct, 2), close_reason, order_id)
    )
    conn.commit()
    conn.close()
    return True


def get_live_trades(status: Optional[str] = None, symbol: Optional[str] = None,
                    limit: int = 100) -> List[Dict]:
    """Get live trades, optionally filtered by status and symbol."""
    conn = get_connection()
    query = "SELECT * FROM live_trades WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol.upper())

    query += " ORDER BY entry_time DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trade_stats() -> Dict:
    """Get aggregate trade statistics."""
    conn = get_connection()
    stats = {}

    row = conn.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_trades,
            SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN status = 'CLOSED' AND pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN status = 'CLOSED' AND pnl <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN status = 'CLOSED' THEN pnl ELSE 0 END) as total_pnl,
            AVG(CASE WHEN status = 'CLOSED' THEN pnl_pct ELSE NULL END) as avg_pnl_pct,
            MAX(CASE WHEN status = 'CLOSED' THEN pnl ELSE NULL END) as best_trade,
            MIN(CASE WHEN status = 'CLOSED' THEN pnl ELSE NULL END) as worst_trade
        FROM live_trades
    """).fetchone()

    if row:
        stats = dict(row)
        closed = stats.get("closed_trades", 0) or 0
        wins = stats.get("wins", 0) or 0
        stats["win_rate"] = round((wins / closed * 100), 1) if closed > 0 else 0

    conn.close()
    return stats


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
