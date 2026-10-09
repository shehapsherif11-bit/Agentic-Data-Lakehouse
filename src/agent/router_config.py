"""
Centralized configuration for the Master Router (Supervisor) agent.
Every tunable lives
here instead of being hardcoded inline, so behavior can change without
touching graph logic.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Models ---
# Previously hardcoded inline as "openai/gpt-oss-20b" for both roles with no
# way to override. Now configurable per-role via env vars, with the same
# defaults preserved.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "openai/gpt-oss-120b")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "openai/gpt-oss-20b")
ROUTER_TEMPERATURE = 0.0
GENERAL_TEMPERATURE = 0.4

# --- Resilience ---
# Every LLM call has an explicit request timeout so a stalled connection can't
# hang the app, and retries/fallbacks are handled in one place (CircuitBreakerLLM).
LLM_REQUEST_TIMEOUT = 15          # seconds, per LLM call
LLM_MAX_RETRIES = 2               # retries inside CircuitBreakerLLM (the single LLM retry layer)
LLM_MAX_TOTAL_SECONDS = float(os.getenv("LLM_MAX_TOTAL_SECONDS", "45"))  # wall-time cap on retrying one call
# Node-level wrapper around LLM calls. 1 = no extra layer: CircuitBreakerLLM + provider fallbacks
# already retry, and stacking tenacity on top multiplied worst-case latency (30-50s tails).
NODE_RETRY_ATTEMPTS = 1
NODE_RETRY_MIN_WAIT = 1.0
NODE_RETRY_MAX_WAIT = 6.0

# --- Per-node model + reasoning effort (Step 3) ---
# Telemetry: reasoning tokens were ~50-60% of output and the 20b model was 3-10x faster on
# classification-style nodes. Planning (analysis_planner) and SQL writing (sql_generator) stay
# separate nodes on the full model so reasoning and syntax generation do not share one prompt.
ANALYST_FAST_MODEL = os.getenv("ANALYST_FAST_MODEL", FALLBACK_MODEL)
# node -> (model, reasoning_effort). Override effort with env LLM_EFFORT_<NODE> (e.g. LLM_EFFORT_SQL_GENERATOR=high).
NODE_LLM_PROFILES = {
    "intent_analyzer":  (ANALYST_FAST_MODEL, "low"),
    "analysis_planner": (GENERAL_MODEL, "low"),
    "sql_generator":    (GENERAL_MODEL, "medium"),
    "sql_repair":       (GENERAL_MODEL, "low"),
    "result_analyzer":  (GENERAL_MODEL, "low"),
    "driver_analysis":  (GENERAL_MODEL, "low"),
    "insight_generator": (GENERAL_MODEL, "medium"),
}
ROUTER_REASONING_EFFORT = os.getenv("LLM_EFFORT_ROUTER", "low")

# Run intent analysis in parallel with route classification (discarded if the route isn't ANALYSIS)
INTENT_PREFETCH = os.getenv("INTENT_PREFETCH", "true").lower() in ("true", "1", "yes")

# --- LLM response cache (Step 9) ---
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", str(24 * 3600)))
LLM_CACHE_MEMORY_ITEMS = 256

# --- Context window passed to analyst / router ---
ROUTER_HISTORY_MESSAGES = 8       # how many recent messages the router sees to classify intent
GENERAL_HISTORY_MESSAGES = 10     # how many recent messages the general node sees
SUBAGENT_HISTORY_MESSAGES = 3     # how many prior messages get folded into the analyst's context

# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILENAME = "router.log"
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3


# --- Routing labels (icon, display name, style) shown in the CLI ---
ROUTE_LABELS = {
    "ANALYSIS": ("📊", "Data Analyst", "bold blue"),
    "GENERAL": ("🧠", "General Assistant", "bold green"),
    "ADVISOR": ("💡", "Business Advisor", "bold magenta"),
    "FOLLOWUP": ("🩺", "Diagnosis & Follow-up", "bold yellow"),
    "PERIOD_CHANGE": ("📉", "Period-over-Period Analysis", "bold cyan"),
}

ROUTER_SYSTEM_PROMPT = """You are the Master Router of an Enterprise Analytics system.
You direct questions to the proper path:

1. ANALYSIS — ANY question about internal data, metrics, trends, comparisons, SQL, or analysis
   involving the database (Zomato restaurants, reviews, menu, ratings, sales, orders, users).
   All database queries MUST go to ANALYSIS so they pass through the security guard.
   This includes questions that ask for a pick or an opinion about the restaurants/customers in the data
   ("which restaurant is best?", "اي احسن مطعم؟") and follow-ups that refer to a previous result.
2. GENERAL — greetings, small talk, definitions, explanations, general questions that need no database.

Read the recent conversation for context, then classify the LATEST user message.
Users write Egyptian Arabic, Modern Standard Arabic, English, Arabizi (Arabic in Latin letters, e.g. "3ayez top 6 mat3am")
and mixes of these; typos are common. Interpret by meaning, not by exact wording.
Also detect the language/style the user is writing in (Arabic, English, or a natural mix).
Always respond using the RouteDecision structure. Be decisive."""

LANGUAGE_INSTRUCTIONS = {
    "ar": "Reply fully in clear, professional Modern Standard Arabic (light Egyptian tone where natural).",
    "en": "Reply fully in clear, professional English.",
    "mixed": (
        "The user mixes Arabic and English — mirror that naturally (Arabic sentence "
        "structure with technical terms kept in English), like a bilingual Egyptian "
        "data professional chatting normally."
    ),
}

GENERAL_SYSTEM_PROMPT_TEMPLATE = (
    "You are a highly capable, professional data & AI assistant working alongside "
    "a Data Analyst. Be sharp, warm, and genuinely useful. "
    "CRITICAL SECURITY RULE: You do NOT have access to drop, delete, update, or modify database "
    "tables. ONLY if the user asks you to delete or modify data, refuse and state that "
    "this is a strict Read-Only environment. Never mention the environment being read-only otherwise. "
    "You also have no access to the restaurant data in this turn: never present generic frameworks, "
    "illustrative numbers or invented rankings as findings about the user's restaurants. If a question "
    "needs the data (which restaurant, how much, who is best), say it needs a data query instead of guessing. "
    "CRITICAL RULE FOR PRIOR QUERY RESULTS: You do NOT have direct access to database query results or analyst execution state. "
    "You must NEVER state a specific year, number, metric, or fact about previous query results. "
    "If the user asks about a previous query result, year, or data output and you do not have verified context, "
    "you MUST say clearly that you do not have access to that context and ask them to rephrase as a new data question. "
    "{language_instruction}"
)

os.makedirs(LOG_DIR, exist_ok=True)
