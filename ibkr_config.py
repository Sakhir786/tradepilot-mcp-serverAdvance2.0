"""
IBKR Gateway Configuration
===========================
Connection settings for Interactive Brokers TWS/Gateway.

Setup:
1. Install IB Gateway (headless) or TWS
2. Enable API connections in Gateway config
3. Set socket port (default 4001 for live, 4002 for paper)
4. Set env vars or edit defaults below
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Connection
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4002"))  # 4001=live, 4002=paper
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))

# Timeouts
IBKR_CONNECT_TIMEOUT = int(os.getenv("IBKR_CONNECT_TIMEOUT", "10"))
IBKR_REQUEST_TIMEOUT = int(os.getenv("IBKR_REQUEST_TIMEOUT", "30"))

# Market data type: 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen
IBKR_MARKET_DATA_TYPE = int(os.getenv("IBKR_MARKET_DATA_TYPE", "3"))

# Mode config (mirrors polygon_client.py MODE_DATA_CONFIG)
MODE_DATA_CONFIG = {
    "scalp": {
        "bar_size": "5 mins",
        "duration": "2 W",
        "timespan": "minute",
        "multiplier": 5,
        "dte_range": (0, 2),
    },
    "intraday": {
        "bar_size": "15 mins",
        "duration": "3 M",
        "timespan": "minute",
        "multiplier": 15,
        "dte_range": (0, 5),
    },
    "swing": {
        "bar_size": "1 day",
        "duration": "2 Y",
        "timespan": "day",
        "multiplier": 1,
        "dte_range": (7, 45),
    },
    "leaps": {
        "bar_size": "1 day",
        "duration": "2 Y",
        "timespan": "day",
        "multiplier": 1,
        "dte_range": (180, 720),
    },
}
