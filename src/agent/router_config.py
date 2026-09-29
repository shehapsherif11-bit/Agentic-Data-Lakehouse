"""
Centralized configuration for the Master Router (Supervisor) agent.
Mirrors the config.py pattern used in the ETL project: every tunable lives
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
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "openai/gpt-oss-20b")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "openai/gpt-oss-20b")
ROUTER_TEMPERATURE = 0.0
GENERAL_TEMPERATURE = 0.4
POLISH_TEMPERATURE = 0.0    # صفر عشان الـ polish ما يأخدش حرية في التأليف

# --- Resilience ---
# Previously: no timeout at all on LLM calls (a stalled connection could hang
# the whole CLI forever), and retries were done by hand with a fixed linear
# backoff. Both are now explicit and consistent with the ETL project's use
# of `tenacity`.
LLM_REQUEST_TIMEOUT = 30          # seconds, per LLM call
LLM_MAX_RETRIES = 3               # library-level retries inside ChatGroq itself
NODE_RETRY_ATTEMPTS = 3           # our own retry wrapper around each node's LLM/agent call
NODE_RETRY_MIN_WAIT = 1.0
NODE_RETRY_MAX_WAIT = 6.0

# --- Context window passed to sub-agents / router ---
ROUTER_HISTORY_MESSAGES = 8       # how many recent messages the router sees to classify intent
GENERAL_HISTORY_MESSAGES = 10     # how many recent messages the general node sees
SUBAGENT_HISTORY_MESSAGES = 3     # how many prior messages get folded into SQL/ETL sub-agent context

# --- Logging ---
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILENAME = "router.log"
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3

# --- Routing labels (icon, display name, style) shown in the CLI ---
ROUTE_LABELS = {
    "ANALYSIS": ("📊", "Data Analyst", "bold blue"),
    "SQL": ("🗄️", "SQL Analyst", "bold blue"),
    "ETL": ("🌐", "ETL Analyst", "bold magenta"),
    "GENERAL": ("🧠", "General Assistant", "bold green"),
}

ROUTER_SYSTEM_PROMPT = """You are the Master Router of a Data Engineering Team.
You manage specialist agents plus can answer general questions yourself:

1. ANALYSIS — ANY question about internal data, metrics, trends, comparisons, or analysis
   involving the database (Zomato restaurants, reviews, menu, ratings, sales, orders, users).
   This includes simple queries AND complex analytical questions (why, trends, comparisons,
   top/bottom, growth rates, breakdowns).
2. ETL — pulling/scraping data from external APIs, links, uploaded CSV/Excel, or Pandas-based
   cleaning/transformation. NOT for querying the internal database.
3. GENERAL — greetings, small talk, definitions, explanations, brainstorming, or follow-up
   chat that needs no database and no external data pull.

Read the recent conversation for context, then classify the LATEST user message.
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
    "tables. If the user asks you to delete or modify data, you MUST bluntly refuse and state that "
    "this is a strict Read-Only environment! "
    "{language_instruction}"
)

POLISH_SYSTEM_PROMPT_TEMPLATE = (
    "You are the Manager reviewing a specialist's raw answer before it reaches the user. "
    "Rewrite it to be clear, well-structured, and polished. "
    "CRITICAL SECURITY RULE: If the raw answer contains words like 'unsafe', 'Sorry', 'error', "
    "'denied', or indicates that the SQL Judge blocked the query, YOU MUST NOT HIDE IT. You MUST "
    "clearly state that the action was blocked by the Security Judge. NEVER invent a success "
    "message if the raw answer implies failure or refusal. "
    "Keep every fact, number, and name EXACTLY as given, never invent or drop data. "
    "{language_instruction}"
)

os.makedirs(LOG_DIR, exist_ok=True)
