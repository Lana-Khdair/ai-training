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
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "1"))

# ---------- vectorstore / vector store ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "menu_items")
QDRANT_URL = os.getenv("QDRANT_URL", "https://7cf72c1f-7e19-473b-b77d-fef40a3d3a42.australia-southeast1-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Mjg3MGYwNTYtYmQ1NC00NTE1LThjNDUtZTE5N2E1OWM0ZmUwIn0.dDIjazWkYoNlq4FI5a4f6SzFxDGF7T1yXefzKbEk_pI")