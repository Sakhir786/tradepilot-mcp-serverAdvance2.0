#!/bin/bash
# TradePilot - One-click launcher (Linux/Mac)
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Setting up TradePilot for first time..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

python run.py "$@"
