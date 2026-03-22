@echo off
REM TradePilot - One-click launcher (Windows)
cd /d "%~dp0"

if not exist "venv" (
    echo Setting up TradePilot for first time...
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

python run.py %*
