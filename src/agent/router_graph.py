"""
router_graph.py — the LangGraph wiring for the Master Router.

Kept separate from CLI/presentation concerns (see router.py) so the graph
itself can be built and tested without a terminal, Rich, or Arabic text
shaping in the loop.

Architecture (unchanged from the original design):

    START → router_node ──┬─ SQL     → sql_node     ──┐
                           ├─ ETL     → etl_node     ──┼─ polish_node → END
                           └─ GENERAL → general_node ──┘

Fixes applied vs. the original router.py:
  * Sub-agent imports are wrapped in try/except. If the SQL or ETL agent
    module fails to import (missing dependency, bad path, etc.) the router
    still boots — that one route just returns a clear error instead of the
    whole CLI refusing to start.
  * `sql_node`/`etl_node` no longer duplicate the same "build context from
    recent messages" logic — factored into `_recent_context()`.
  * Retries use `tenacity` (same library as the ETL project) instead of a
    hand-rolled linear-backoff loop, and every LLM call now has an explicit
    request timeout so a stalled connection can't hang the CLI forever.
  * All models, prompts, and history-window sizes come from `router_config`
    instead of being hardcoded inline.
"""
import logging
import logging.handlers
import os
from typing import Annotated, Literal, Sequence, TypedDict

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Works both as a flat top-level script (`import router_graph`) and as a
# package module (e.g. `src.agent.router_graph`, as app.py expects) —
# falls back to a flat import if there's no enclosing package.
try:
    from . import router_config as cfg
except ImportError:
    import router_config as cfg

# ==========================================
# Logging
# ==========================================
logger = logging.getLogger("router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(cfg.LOG_DIR, cfg.LOG_FILENAME),
        maxBytes=cfg.LOG_MAX_BYTES,
        backupCount=cfg.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_file_handler)

# ==========================================
# Sub-agents — imported defensively so a broken/missing specialist doesn't
# take down the whole router at startup.
# ==========================================
try:
    from src.agent.sql_agent import sql_analyst_agent
except Exception as e:  # noqa: BLE001 - deliberately broad; this is a best-effort import
    sql_analyst_agent = None
    logger.warning("SQL agent unavailable at import time: %s", e)

try:
    # Note: the ETL agent module (see etl_agent.py in the ETL project rewrite)
    # exposes its system prompt as `SYSTEM_PROMPT` (module-level constant,
    # uppercase). If you're wiring this against the original etl_agent.py,
    # adjust this import to whatever name that module actually exports.
    from src.agent.etl_agent import etl_analyst_agent, SYSTEM_PROMPT as etl_system_prompt
except Exception as e:  # noqa: BLE001
    etl_analyst_agent = None
    etl_system_prompt = ""
    logger.warning("ETL agent unavailable at import time: %s", e)

try:
    from src.agent.analyst_graph import analyst_agent
except Exception as e:  # noqa: BLE001
    analyst_agent = None
    logger.warning("Analyst agent unavailable at import time: %s", e)


# ==========================================
# LLMs
# ==========================================
def _make_llm(model: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=cfg.GROQ_API_KEY,
        request_timeout=cfg.LLM_REQUEST_TIMEOUT,
        max_retries=cfg.LLM_MAX_RETRIES,
    )


router_llm = _make_llm(cfg.ROUTER_MODEL, cfg.ROUTER_TEMPERATURE)
general_llm = _make_llm(cfg.GENERAL_MODEL, cfg.GENERAL_TEMPERATURE)
polish_llm = _make_llm(cfg.GENERAL_MODEL, cfg.POLISH_TEMPERATURE)


class RouteDecision(BaseModel):
    agent: Literal["ANALYSIS", "SQL", "ETL", "GENERAL"] = Field(
        description=(
            "ANALYSIS: any question about internal data, metrics, trends, comparisons, "
            "or analysis involving the database (Zomato). "
            "SQL: direct database queries (legacy fallback — prefer ANALYSIS). "
            "ETL: pulling/scraping data from external APIs, links, uploaded files, "
            "or Pandas-based cleaning/transformation. "
            "GENERAL: greetings, small talk, definitions, explanations, brainstorming, "
            "or follow-up chat that needs no database and no external data pull."
        )
    )
    confidence: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="One short sentence explaining the decision")
    language: Literal["ar", "en", "mixed"] = Field(
        description="The language/style of the user's latest message: Arabic, English, or a natural mix of both."
    )


# ==========================================
# Graph state
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    route: str
    confidence: float
    reasoning: str
    language: str
    raw_answer: str
    final_answer: str
    viz_html: str
    llm_call_count: int
    stage_latencies: dict
    analyst_memory: dict


# ==========================================
# Shared helpers
# ==========================================
def _retrying(label: str):
    """Tenacity retry decorator factory, parameterized by a log label."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(cfg.NODE_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=cfg.NODE_RETRY_MIN_WAIT, max=cfg.NODE_RETRY_MAX_WAIT),
        before_sleep=lambda retry_state: logger.warning(
            "%s failed (attempt %d/%d): %s",
            label, retry_state.attempt_number, cfg.NODE_RETRY_ATTEMPTS,
            retry_state.outcome.exception(),
        ),
    )


def _speaker(m: BaseMessage) -> str:
    return "User" if isinstance(m, HumanMessage) else "Assistant"


def _recent_context(messages: Sequence[BaseMessage], n: int, exclude_last: bool) -> str:
    """Render the last `n` messages as a 'Speaker: text' transcript. When
    `exclude_last` is True, the current/last message is left out (used by
    sub-agent nodes, which pass the current question separately)."""
    window = messages[-(n + 1):-1] if exclude_last else messages[-n:]
    return "\n".join(f"{_speaker(m)}: {m.content}" for m in window)


# ==========================================
# Nodes
# ==========================================
def router_node(state: AgentState) -> dict:
    user_question = state["messages"][-1].content
    structured_llm = router_llm.with_structured_output(RouteDecision)
    history_context = _recent_context(state["messages"], cfg.ROUTER_HISTORY_MESSAGES, exclude_last=False)

    messages = [
        SystemMessage(content=cfg.ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Recent conversation:\n{history_context}"),
    ]

    @_retrying("router classification")
    def _call():
        return structured_llm.invoke(messages)

    try:
        decision: RouteDecision = _call()
    except Exception as e:
        logger.error("Router fully failed, defaulting to GENERAL: %s", e)
        decision = RouteDecision(agent="GENERAL", confidence=0.3, reasoning="fallback after failure", language="en")

    logger.info(
        "Q='%s' | route=%s | confidence=%.2f | lang=%s | reason=%s",
        user_question, decision.agent, decision.confidence, decision.language, decision.reasoning,
    )

    return {
        "route": decision.agent,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "language": decision.language,
    }


def analysis_node(state: AgentState) -> dict:
    """Routes analytical questions through the full analytical pipeline."""
    if analyst_agent is None:
        logger.error("analysis_node called but the analyst agent failed to import.")
        return {"raw_answer": "[Analyst agent is unavailable: it failed to load at startup — check the logs.]"}

    question = state["messages"][-1].content
    lang = state.get("language", "en")
    try:
        # We must pass the actual message objects so the analyst_graph 
        # can maintain its own conversational state for follow-ups!
        analyst_input = {
            "messages": state["messages"][-cfg.SUBAGENT_HISTORY_MESSAGES:],
            "language": lang,
            "retry_count": 0,
            "current_step": 0,
        }
        
        # Inject previous state if available
        memory = state.get("analyst_memory", {})
        if memory:
            analyst_input.update(memory)
            
        result = analyst_agent.invoke(analyst_input)
        answer = result.get("final_answer", "No answer generated.")
        viz_html = result.get("viz_html", "")

        # Save context for next follow-up
        new_memory = {
            "previous_question": result.get("previous_question"),
            "previous_plan": result.get("analysis_plan"),
            "previous_results": result.get("query_results")
        }

        # Pass visualization HTML and telemetry to the outer state
        output = {
            "final_answer": answer, 
            "messages": [AIMessage(content=answer)],
            "viz_html": viz_html,
            "llm_call_count": result.get("llm_call_count", 0),
            "stage_latencies": result.get("stage_latencies", {}),
            "analyst_memory": new_memory
        }
        return output
    except Exception as e:
        logger.error("Analysis agent failed: %s", e)
        error_msg = f"Sorry, I encountered an internal error during the analysis: {e}"
        return {"final_answer": error_msg, "messages": [AIMessage(content=error_msg)]}


def sql_node(state: AgentState) -> dict:
    if sql_analyst_agent is None:
        logger.error("sql_node called but the SQL agent failed to import.")
        return {"raw_answer": "[SQL agent is unavailable: it failed to load at startup — check the logs.]"}

    question = state["messages"][-1].content
    history = _recent_context(state["messages"], cfg.SUBAGENT_HISTORY_MESSAGES, exclude_last=True)
    contextualized = (
        f"Context of our conversation:\n{history}\n\nBased on the context, please answer this: {question}"
        if history else question
    )

    @_retrying("SQL agent")
    def _call():
        result = sql_analyst_agent.invoke({"messages": [HumanMessage(content=contextualized)]})
        return result.get("final_answer", "No answer generated.")

    try:
        answer = _call()
    except Exception as e:
        logger.error("SQL agent failed: %s", e)
        answer = f"[SQL agent error: {e}]"
    return {"raw_answer": answer}


def etl_node(state: AgentState) -> dict:
    if etl_analyst_agent is None:
        logger.error("etl_node called but the ETL agent failed to import.")
        return {"raw_answer": "[ETL agent is unavailable: it failed to load at startup — check the logs.]"}

    question = state["messages"][-1].content
    history = _recent_context(state["messages"], cfg.SUBAGENT_HISTORY_MESSAGES, exclude_last=True)
    contextualized = (
        f"Context of our conversation:\n{history}\n\nBased on the context, please execute this request: {question}"
        if history else question
    )

    @_retrying("ETL agent")
    def _call():
        result = etl_analyst_agent.invoke({
            "messages": [
                SystemMessage(content=etl_system_prompt),
                HumanMessage(content=contextualized),
            ]
        })
        return result["messages"][-1].content

    try:
        answer = _call()
    except Exception as e:
        logger.error("ETL agent failed: %s", e)
        answer = f"[ETL agent error: {e}]"
    return {"raw_answer": answer}


def general_node(state: AgentState) -> dict:
    question = state["messages"][-1].content
    lang = state.get("language", "en")
    history_context = _recent_context(state["messages"], cfg.GENERAL_HISTORY_MESSAGES, exclude_last=False)

    system = cfg.GENERAL_SYSTEM_PROMPT_TEMPLATE.format(
        language_instruction=cfg.LANGUAGE_INSTRUCTIONS.get(lang, cfg.LANGUAGE_INSTRUCTIONS["en"])
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Conversation so far:\n{history_context}\n\nUser: {question}"),
    ]

    @_retrying("general answer")
    def _call():
        return general_llm.invoke(messages).content.strip()

    try:
        answer = _call()
    except Exception as e:
        logger.error("General node failed: %s", e)
        answer = f"[Assistant error: {e}]"

    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def polish_node(state: AgentState) -> dict:
    """Polishes a raw SQL/ETL answer without altering any fact, number, or name."""
    question = state["messages"][-1].content
    lang = state.get("language", "en")
    raw = state.get("raw_answer", "")

    # ❶ لو الإجابة الخام فاضية أو فيها error، رجّعها زي ما هي بدون "تحسين"
    if not raw or not raw.strip():
        fallback = "لم يتم العثور على إجابة من قاعدة البيانات." if lang == "ar" else "No answer was returned from the database."
        return {"final_answer": fallback, "messages": [AIMessage(content=fallback)]}

    raw_lower = raw.lower()
    if any(kw in raw_lower for kw in ["error", "unavailable", "failed to load", "no answer"]):
        return {"final_answer": raw, "messages": [AIMessage(content=raw)]}

    # ❷ تعليمات صارمة لمنع التأليف
    system = cfg.POLISH_SYSTEM_PROMPT_TEMPLATE.format(
        language_instruction=cfg.LANGUAGE_INSTRUCTIONS.get(lang, cfg.LANGUAGE_INSTRUCTIONS["en"])
    )
    system += (
        "\n\nSTRICT RULE: You MUST ONLY rephrase the specialist's raw answer. "
        "Do NOT add any new facts, numbers, names, or information that is not explicitly "
        "present in the raw answer. If the raw answer contains no data, say so clearly. "
        "NEVER make up or invent data."
    )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"User's question: {question}\n\nSpecialist's raw answer:\n{raw}"),
    ]

    # ❸ استخدام polish_llm (temperature=0.0) بدل general_llm
    @_retrying("polish")
    def _call():
        return polish_llm.invoke(messages).content.strip()

    try:
        final = _call()
    except Exception:
        final = raw  # never lose the raw answer just because polishing failed

    return {"final_answer": final, "messages": [AIMessage(content=final)]}


def route_decision(state: AgentState) -> str:
    return state["route"]


# ==========================================
# Graph builder
# ==========================================
def build_graph(checkpointer=None):
    """Builds and compiles the router graph. Pass a custom `checkpointer`
    (e.g. a SqliteSaver) for persistence across process restarts; defaults
    to an in-memory MemorySaver, which only lasts for the current run."""
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("sql", sql_node)
    graph.add_node("etl", etl_node)
    graph.add_node("general", general_node)
    graph.add_node("polish", polish_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "ANALYSIS": "analysis",
        "SQL": "sql",
        "ETL": "etl",
        "GENERAL": "general",
    })
    # ANALYSIS → END directly (the analyst pipeline generates its own final answer + viz)
    graph.add_edge("analysis", END)
    # Legacy SQL + ETL still go through polish
    graph.add_edge("sql", "polish")
    graph.add_edge("etl", "polish")
    graph.add_edge("polish", END)
    graph.add_edge("general", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
