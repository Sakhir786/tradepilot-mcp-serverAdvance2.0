# 🚀 TradePilot MCP Server v3.0
**Advanced 18-Layer Trading Intelligence Engine**

A production-grade MCP (Model Context Protocol) server that combines **Polygon.io real-time market data** with an **18-layer technical analysis engine** for autonomous options trading intelligence.

---

## 📊 What is This?

TradePilot is a **FastAPI-based MCP server** that:
- Fetches real-time and historical market data from Polygon.io
- Runs analysis through **18 specialized layers** (Technical + Price Action + Options + Master Brain)
- Provides **14 high-probability playbooks** targeting 85-95% win rates
- Delivers actionable options trading signals with precise strike/expiry recommendations
- Includes multi-ticker scanning, risk management, and alert notifications

Think of it as your **AI options trading copilot** powered by institutional-grade analysis.

---

## 🎯 Key Features

- **18-Layer Analysis System**
  - Layers 1-10: Technical indicators (Momentum, Volume, Trend, Structure)
  - Layers 11-13: Price action (Support/Resistance, VWAP, Volume Profile)
  - Layers 14-17: Options analysis (IV, Greeks, Gamma, Put/Call ratios)
  - Layer 18: Master Brain with 14 playbooks

- **Trading Modes**
  - SCALP: 0-2 DTE options
  - SWING: 7-45 DTE options
  - INTRADAY: Same-day trades

- **Advanced Features**
  - Multi-ticker scanner with filtering
  - Kelly Criterion position sizing
  - Risk management with drawdown protection
  - Discord/Slack/Telegram alerts
  - Backtesting engine
  - AI-ready JSON output

---

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/Sakhir786/tradepilot-mcp-server.git
cd tradepilot-mcp-server

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your POLYGON_API_KEY

# Run the server
python main.py
# Or with uvicorn:
uvicorn main:app --host 0.0.0.0 --port 10000
```

Access documentation at: http://localhost:10000/docs

---

## 📡 API Endpoints

### 18-Layer Analysis
- `GET /engine18/analyze` - Full 18-layer analysis
- `GET /engine18/quick` - Quick signal check
- `GET /engine18/scan` - Multi-ticker scanner
- `GET /engine18/playbooks` - List all 14 playbooks
- `GET /engine18/health` - System health check

### Market Data (Polygon.io)
- `GET /candles` - OHLCV data
- `GET /options` - Options chain
- `GET /news` - Latest news
- `GET /ticker-details` - Company info

---

## 🔧 Environment Variables
```bash
POLYGON_API_KEY=your_api_key_here
TRADEPILOT_PRODUCTION_PATH=/path/to/tradepilot-mcp-server
TRADEPILOT_LAYERS_PATH=/path/to/layers
TRADEPILOT_PORTFOLIO_VALUE=100000
TRADEPILOT_DISCORD_WEBHOOK=your_webhook_url
```

---

## 📊 Example Usage
```python
import requests

# Full analysis
response = requests.get(
    "http://localhost:10000/engine18/analyze",
    params={"symbol": "SPY", "mode": "swing"}
)
analysis = response.json()

print(f"Direction: {analysis['analysis_summary']['direction']}")
print(f"Win Probability: {analysis['analysis_summary']['win_probability']}%")
```

---

## 🎯 14 High-Probability Playbooks

**Bullish (CALLS):**
1. Liquidity Sweep + BOS (85-95%)
2. CHoCH Reversal (82-92%)
3. Trend Continuation (80-90%)
4. FVG Fill + Rejection (81-88%)
5. Order Block Bounce (79-87%)
6. Divergence + Structure (80-88%)
7. VWAP Reclaim (77-85%)

**Bearish (PUTS):**
8-14. Mirror patterns for bearish setups

---

## 🔐 Security & Best Practices

- Never commit `.env` files
- Keep API keys secure
- Use environment variables for sensitive data
- Enable rate limiting for production deployments

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## 📧 Support

For issues or questions, please open a GitHub issue.

**TradePilot v3.0 - Professional Options Trading Intelligence** 🚀📈
