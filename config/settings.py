from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else BASE_DIR / p


# ---------- paths ----------
DATA_DIR = _path("DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
LOG_DIR = DATA_DIR / "logs"
DB_PATH = _path("DB_PATH", "data/menu.db")

# ---------- scraper ----------
BASE_URL = os.getenv("BASE_URL", "https://foodnevercome.com")
ORDER_URL = f"{BASE_URL}/order"
USER_AGENT = os.getenv("USER_AGENT", "FoodMenuPipeline/1.0 (student project;)")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

# ---------- pipeline (retries for the scraper stage) ----------
SCRAPE_ATTEMPTS = int(os.getenv("SCRAPE_ATTEMPTS", "3"))
SCRAPE_RETRY_WAIT = int(os.getenv("SCRAPE_RETRY_WAIT", "30"))

# ---------- data quality / safety limits ----------
MAX_REJECT_RATIO = float(os.getenv("MAX_REJECT_RATIO", "0.20"))
MAX_REMOVED_RATIO = float(os.getenv("MAX_REMOVED_RATIO", "0.30"))
MIN_ACTIVE_FOR_GUARD = int(os.getenv("MIN_ACTIVE_FOR_GUARD", "10"))