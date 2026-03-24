# Configuration for TradePilot data sources
# In production, use environment variables instead of hardcoding

import os
from dotenv import load_dotenv

load_dotenv()

# Data source: "polygon" (default) or "ibkr"
DATA_SOURCE = os.getenv("DATA_SOURCE", "polygon").lower()

# Polygon.io settings
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"
