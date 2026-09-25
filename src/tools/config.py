"""
Central configuration for the ETL extraction engine.
All tunables live here so behavior can be changed without touching logic.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (used only as a fallback when deterministic extraction isn't enough) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# No model ID is hardcoded here anymore. Groq periodically decommissions
# models (see https://console.groq.com/docs/deprecations) — a hardcoded
# default WILL eventually 404, which is exactly what happened with
# `llama-3.3-70b-versatile`/`llama-3.1-8b-instant`. Instead:
#   - GROQ_MODEL_OVERRIDE lets you pin an exact model via env var if you
#     want to (e.g. for cost/latency control), and is used as-is with no
#     validation.
#   - If unset, get_active_groq_model() (below) resolves a currently
#     working model at runtime via groq_model_resolver, caches it briefly,
#     and re-resolves automatically if that model stops working later.
GROQ_MODEL_OVERRIDE = os.getenv("GROQ_MODEL")
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"  # last-resort default if live resolution can't run at all
LLM_TEMPERATURE = 0.0
LLM_MAX_CHARS_PER_CHUNK = 9000          # keep well under context window per call
LLM_MAX_CHUNKS = 6                      # hard cap so one page can't explode into dozens of calls

_active_model_cache: dict[str, str] = {}


def get_active_groq_model() -> str:
    """Returns a Groq model id verified to work right now. Resolved lazily
    (only when an LLM call is actually about to happen) and memoized for
    the lifetime of the process, so a run that never needs the LLM
    fallback (e.g. a page fully covered by JSON-LD) pays zero cost for
    this and never touches the network for it."""
    if "model" in _active_model_cache:
        return _active_model_cache["model"]

    if GROQ_MODEL_OVERRIDE:
        _active_model_cache["model"] = GROQ_MODEL_OVERRIDE
        return GROQ_MODEL_OVERRIDE

    from langchain_groq import ChatGroq
    import groq_model_resolver

    def factory(model_id: str) -> ChatGroq:
        return ChatGroq(model=model_id, temperature=0.0, api_key=GROQ_API_KEY, max_retries=1, request_timeout=15)

    model = groq_model_resolver.resolve_groq_model(GROQ_API_KEY, factory, fallback=GROQ_FALLBACK_MODEL)
    _active_model_cache["model"] = model
    return model


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
