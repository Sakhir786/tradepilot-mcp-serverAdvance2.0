# TradePilot MCP Server v3.0

Advanced 18-Layer Trading Intelligence Engine for AI-powered options trading.

TradePilot is a FastAPI-based MCP server that fetches real-time market data from Polygon.io, runs it through 18 specialized analysis layers, and outputs structured raw data for AI (Claude/GPT) to make trading decisions across all timeframes.

---

## How It Works

```
Polygon.io Market Data
        |
        v
18-Layer Analysis Engine
  Layers 1-10:  Technical (Momentum, Volume, Divergence, Trend, Structure, Liquidity, Volatility, MTF, Candles)
  Layers 11-13: Price Action (Support/Resistance, VWAP, Volume Profile)
  Layers 14-17: Options (IV, Gamma/Max Pain, Put/Call Ratio, Greeks)
  Layer 18:     Brain Aggregator (organizes all data for AI)
        |
        v
Structured Raw Data Output (JSON)
        |
        v
AI reads ALL the data and makes the trading decision
```

The server outputs pure data. No scores, no confidence percentages. The AI reads every data point from every layer and tells you what the data says.

---

## Trading Modes

| Mode | Timeframe | DTE | Use Case |
|------|-----------|-----|----------|
| SCALP | 5-minute bars | 0-2 days | Quick in-and-out trades |
| INTRADAY | 15-minute bars | 0-5 days | Same-day directional |
| SWING | Daily bars | 7-45 days | Multi-day trend trades |
| LEAPS | Daily bars | 180-720 days | Long-term positions |

Each mode automatically fetches the right timeframe data and focuses on the layers most relevant to that style.

---

## Setup

```bash
# Clone
git clone https://github.com/Sakhir786/tradepilot-mcp-serveradvance2.0.git
cd tradepilot-mcp-serveradvance2.0

# Install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your POLYGON_API_KEY

# Run
uvicorn main:app --host 0.0.0.0 --port 10000
```

Docs at: `http://localhost:10000/docs`

---

## API Endpoints

### Analysis Engine (`/engine18/`)

| Endpoint | What It Does |
|----------|-------------|
| `GET /engine18/analyze?symbol=SPY&mode=scalp` | Full 18-layer analysis |
| `GET /engine18/quick?symbol=SPY&mode=swing` | Fast signal check |
| `GET /engine18/scan?symbols=SPY,QQQ,AAPL&mode=scalp` | Multi-ticker scan |
| `GET /engine18/compare?symbols=SPY,QQQ&mode=swing` | Side-by-side comparison |
| `GET /engine18/layer/{number}?symbol=SPY` | Single layer analysis |
| `GET /engine18/layers` | List all 18 layers |
| `GET /engine18/playbooks` | List playbooks |
| `GET /engine18/health` | System health |

### Market Data (Polygon.io)

| Endpoint | What It Does |
|----------|-------------|
| `GET /candles?symbol=SPY&tf=day&limit=200` | OHLCV price data |
| `GET /options?symbol=SPY` | Options chain |
| `GET /news?symbol=SPY` | Latest news |
| `GET /ticker-details?symbol=SPY` | Company info |
| `GET /fundamentals?symbol=SPY` | Financial data |
| `GET /stock-snapshot?symbol=SPY` | Real-time snapshot |
| `GET /option-chain-snapshot?symbol=SPY` | Full options chain |

---

## The 18 Layers

### Technical Layers (1-10)
| Layer | Name | What It Analyzes |
|-------|------|-----------------|
| 1 | Momentum | RSI, MACD, Stochastic, CMF, ADX, Ichimoku |
| 2 | Volume | OBV, A/D Line, volume trend |
| 3 | Divergence | MACD + RSI divergences (regular + hidden) |
| 4 | Volume Strength | CVD, buying/selling pressure, EOM |
| 5 | Trend | SuperTrend, ATR, market regime, whipsaw detection |
| 6 | Structure | BOS, CHoCH, order blocks, fair value gaps (ICT) |
| 7 | Liquidity | Liquidity sweeps, ICT buy/sell side, grabs |
| 8 | Volatility Regime | ATR percentiles, regime classification |
| 9 | MTF Confirmation | Multi-timeframe SuperTrend alignment |
| 10 | Candle Intelligence | 15+ patterns, three white soldiers, inside bars |

### Price Action Layers (11-13)
| Layer | Name | What It Analyzes |
|-------|------|-----------------|
| 11 | Support/Resistance | Fractals, pivot points, MTF levels, confluence zones |
| 12 | VWAP | VWAP + bands, slope, crossovers, rejections |
| 13 | Volume Profile | POC, value area, buying/selling pressure at levels |

### Options Layers (14-17)
| Layer | Name | What It Analyzes |
|-------|------|-----------------|
| 14 | IV Analysis | IV rank, IV percentile, expected move, HV |
| 15 | Gamma/Max Pain | Max pain, gamma exposure, pin probability |
| 16 | Put/Call Ratio | PCR sentiment, z-score, band position |
| 17 | Greeks | Delta, gamma, theta, vega, best strike selection |

### Aggregation (18)
| Layer | Name | What It Does |
|-------|------|-------------|
| 18 | Brain | Aggregates all 17 layers into organized structure for AI consumption |

---

## Integration with TradingView MCP

TradePilot is designed to work alongside [TradingView MCP](https://github.com/LewisWJackson/tradingview-mcp-jackson) for a complete trading system:

- **TradePilot MCP** = The analytical brain (18-layer data analysis from Polygon.io)
- **TradingView MCP** = The live eyes and hands (real-time chart control, indicator reading, replay backtesting)
- **Claude** = The trader (reads data from both, makes trading decisions)

Both MCP servers connect to Claude Code. Claude calls TradePilot for deep analysis, calls TradingView for live chart data, and gives you the trade based on what ALL the data says.

---

## Environment Variables

```
POLYGON_API_KEY=your_api_key_here    # Required - get at polygon.io
PORT=10000                            # Server port (default 10000)
```

---

## Requirements

- Python 3.11+
- Polygon.io API key
- FastAPI, Pandas, NumPy, SciPy (see requirements.txt)

---

## License

MIT License - See LICENSE file.
