#!/usr/bin/env python3
"""
TradePilot - Personal Trading Intelligence Software
====================================================
Launch script - starts the server and opens the dashboard in your browser.

Usage:
    python run.py
    python run.py --port 8000
    python run.py --no-browser
"""

import argparse
import os
import sys
import threading
import time
import webbrowser

import uvicorn


def open_browser(port: int, delay: float = 1.5):
    """Open dashboard in browser after server starts."""
    time.sleep(delay)
    url = f"http://localhost:{port}"
    print(f"\n{'='*50}")
    print(f"  TradePilot Dashboard: {url}")
    print(f"  API Documentation:    {url}/docs")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="TradePilot Trading Intelligence")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "10000")), help="Port to run on")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check for .env
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            print("[!] No .env file found. Copy .env.example to .env and add your Polygon API key.")
            print("    cp .env.example .env")
            sys.exit(1)

    print(r"""
  _____ ____    _    ____  _____ ____ ___ _     ___ _____
 |_   _|  _ \  / \  |  _ \| ____|  _ \_ _| |   / _ \_   _|
   | | | |_) |/ _ \ | | | |  _| | |_) | || |  | | | || |
   | | |  _ </ ___ \| |_| | |___|  __/| || |__| |_| || |
   |_| |_| \_\_/  \_\____/|_____|_|  |___|_____\___/ |_|

     Personal Trading Intelligence Software v3.0
    """)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    uvicorn.run("main:app", host="0.0.0.0", port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
