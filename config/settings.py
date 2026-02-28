"""
Global settings and API configuration for the MEXC trading system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data/cache")
LOG_DIR = ROOT_DIR / os.getenv("LOG_DIR", "logs")
DB_PATH = ROOT_DIR / os.getenv("DB_PATH", "data/trading.db")

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── API Credentials ─────────────────────────────────────────────────────────
MEXC_API_KEY = os.getenv("MEXC_API_KEY", "")
MEXC_SECRET_KEY = os.getenv("MEXC_SECRET_KEY", "")

# ─── API Endpoints ────────────────────────────────────────────────────────────
BASE_URL = "https://api.mexc.com"
WS_URL = "wss://wbs-api.mexc.com/ws"

# ─── Trading Settings ────────────────────────────────────────────────────────
TRADING_MODE = os.getenv("TRADING_MODE", "paper")  # 'paper' or 'live'
DEFAULT_QUOTE_ASSET = os.getenv("DEFAULT_QUOTE_ASSET", "USDC")
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "2.0"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "10"))

# ─── Rate Limits ──────────────────────────────────────────────────────────────
RATE_LIMIT_PER_10S = 500  # per IP per 10 seconds
RATE_LIMIT_BUFFER = 0.8   # use 80% of limit

# ─── Kline Intervals ─────────────────────────────────────────────────────────
KLINE_INTERVALS = {
    "1m": "Min1",
    "5m": "Min5",
    "15m": "Min15",
    "30m": "Min30",
    "1h": "Min60",
    "4h": "Hour4",
    "8h": "Hour8",
    "1d": "Day1",
    "1w": "Week1",
    "1M": "Month1",
}

# API kline interval mapping (REST API uses different values)
API_KLINE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "4h",
    "8h": "8h",
    "1d": "1d",
    "1w": "1W",
    "1M": "1M",
}

# ─── Backtesting Defaults ────────────────────────────────────────────────────
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION = 0.0  # 0% for USDC pairs
DEFAULT_SLIPPAGE = 0.0005  # 0.05%

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
