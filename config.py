"""
Central configuration for the ETL extraction engine.
All tunables live here so behavior can be changed without touching logic.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (used only as a fallback when deterministic extraction isn't enough) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.0
LLM_MAX_CHARS_PER_CHUNK = 9000          # keep well under context window per call
LLM_MAX_CHUNKS = 6                      # hard cap so one page can't explode into dozens of calls

# --- HTTP fetching ---
HTTP_TIMEOUT = 15
HTTP_RETRIES = 3
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 ETL-Bot/2.0"
)
MIN_STATIC_TEXT_CHARS = 400   # below this, we suspect the page needs JS rendering

# --- Browser fallback (Playwright, only used when static fetch isn't enough) ---
BROWSER_NAV_TIMEOUT_MS = 20000
BROWSER_WAIT_UNTIL = "networkidle"
BROWSER_HEADLESS = True

# --- Crawling / pagination ---
DEFAULT_MAX_PAGES = 1
MAX_PAGES_HARD_LIMIT = 20
CONCURRENT_PAGE_FETCHES = 5

# --- Output / caching ---
OUTPUT_DIR = os.getenv("ETL_OUTPUT_DIR", "data/output")
CACHE_DIR = os.getenv("ETL_CACHE_DIR", "data/.cache")
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes; avoids re-fetching the same URL while iterating

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
