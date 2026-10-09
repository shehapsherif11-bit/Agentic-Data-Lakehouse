"""
router_graph.py — the LangGraph wiring for the Master Router.

Kept separate from CLI/presentation concerns (see router.py) so the graph
itself can be built and tested without a terminal, Rich, or Arabic text
shaping in the loop.

Architecture:

    START → router_node ──┬─ ANALYSIS → analysis_node ──→ END
                          ├─ ADVISOR  → advisor_node  ──→ END   (opinion/decision, from retrieved rows)
                          └─ GENERAL  → general_node  ──→ END

If the analyst module fails to import (missing dependency, bad path, etc.) the
router still boots; the ANALYSIS route then returns a clear error instead of the
whole app refusing to start. All models, prompts and history-window sizes come
from `router_config`.
"""
import logging
import logging.handlers
import os
from typing import Annotated, Literal, Sequence, TypedDict

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agent.llm_factory import get_llm
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import Runnable

# Works both as a flat top-level script (`import router_graph`) and as a
# package module (e.g. `src.agent.router_graph`, as app.py expects) —
# falls back to a flat import if there's no enclosing package.
try:
    from . import router_config as cfg
    from .markdown_utils import sanitize_for_markdown
    from . import llm_cache
    from .concurrency import run_parallel
    from .query_intent import is_advisory_question, detect_language, looks_like_data_question
    from .advisor import build_advice
    from .followups import followup_kind, run_followup
    from .period_change import parse_period_change, run_period_change
except ImportError:
    import router_config as cfg
    from markdown_utils import sanitize_for_markdown
    import llm_cache
    from concurrency import run_parallel
    from query_intent import is_advisory_question, detect_language, looks_like_data_question
    from advisor import build_advice
    from followups import followup_kind, run_followup
    from period_change import parse_period_change, run_period_change

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
    from src.agent.analyst_graph import analyst_agent, compute_intent as _compute_intent, warm_up_db
    from src.agent.analyst_graph import execute_guarded as _execute_guarded
except Exception as e:  # noqa: BLE001
    analyst_agent = None
    _compute_intent = None
    _execute_guarded = None

    def warm_up_db(background: bool = True):  # analyst unavailable: nothing to warm
        return None

    logger.warning("Analyst agent unavailable at import time: %s", e)


try:
    from .telemetry import (
        telemetry_handler, set_current_stage, reset_current_stage,
        start_turn_telemetry, get_turn_telemetry
    )
except ImportError:
    from telemetry import (
        telemetry_handler, set_current_stage, reset_current_stage,
        start_turn_telemetry, get_turn_telemetry
    )

# ==========================================
# LLMs
# ==========================================
def _make_llm(model: str, temperature: float, reasoning_effort: str | None = None):
    llm = get_llm(model_name=model, temperature=temperature, request_timeout=cfg.LLM_REQUEST_TIMEOUT,
                  max_retries=cfg.LLM_MAX_RETRIES, reasoning_effort=reasoning_effort,
                  max_total_seconds=cfg.LLM_MAX_TOTAL_SECONDS)
    return llm.with_config(callbacks=[telemetry_handler])


# Routing is a 3-way classification: low reasoning effort is plenty and cuts reasoning tokens.
router_llm = _make_llm(cfg.ROUTER_MODEL, cfg.ROUTER_TEMPERATURE, cfg.ROUTER_REASONING_EFFORT)
general_llm = _make_llm(cfg.GENERAL_MODEL, cfg.GENERAL_TEMPERATURE)


class RouteDecision(BaseModel):
    agent: Literal["ANALYSIS", "GENERAL"] = Field(
        description=(
            "ANALYSIS: any question about internal data, metrics, trends, comparisons, SQL, "
            "or analysis involving the database (Zomato). Every database query MUST route here. "
            "GENERAL: greetings, small talk, definitions, explanations, brainstorming, "
            "or follow-up chat that needs no database and no external data pull."
        )
    )
    confidence: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="One short sentence explaining the decision")
    language: Literal["ar", "en", "mixed"] = Field(
        description="The language/style of the user's latest message: Arabic (incl. Arabizi), English, or a natural mix of both."
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
    llm_telemetry: list
    analyst_memory: dict
    evidence: list
    prefetched_intent: dict
    turn_evidence: list   # the queries THIS turn relied on (provenance panel); reset every turn


# ==========================================
# Shared helpers
# ==========================================
def _retrying(label: str, attempts: int | None = None):
    """Tenacity retry decorator factory, parameterized by a log label.
    LLM calls already retry/fail over inside CircuitBreakerLLM, so NODE_RETRY_ATTEMPTS defaults to 1
    (a pass-through)."""
    attempts = attempts or cfg.NODE_RETRY_ATTEMPTS
    return retry(
        reraise=True,
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=cfg.NODE_RETRY_MIN_WAIT, max=cfg.NODE_RETRY_MAX_WAIT),
        before_sleep=lambda retry_state: logger.warning(
            "%s failed (attempt %d/%d): %s",
            label, retry_state.attempt_number, attempts,
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
    return "\n".join(f"{_speaker(m)}: {_clip(m.content)}" for m in window)


HISTORY_CHARS = 600


def _clip(text, limit: int = HISTORY_CHARS) -> str:
    """Long assistant answers (tables, diagnoses) are clipped in the routing context: the router only needs the
    gist, and full tables pushed every request to ~3.5k tokens and into Groq's per-minute rate limit."""
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


import re


def _is_referential_question(text: str) -> bool:
    """
    Detects whether a user question is short, pronoun-heavy, or referential (lacking
    its own self-contained data intent), referencing prior query results.
    """
    if not text:
        return False
    t = text.strip().lower()
    words = re.findall(r"\w+", t)
    
    # Must be relatively concise (<= 15 words)
    if len(words) > 15:
        return False

    # Standalone domain entities that indicate self-contained intent
    standalone_domain_terms = {
        "restaurant", "restaurants", "city", "cities", "user", "users",
        "order", "orders", "cuisine", "cuisines", "customer", "customers",
        "مطعم", "مطاعم", "مدينة", "مدن", "عميل", "عملاء", "مستخدم", "مستخدمين", "زبون", "زبائن"
    }

    # Explicit referential indicators
    explicit_ref_markers = {
        "that", "this", "these", "those", "it", "they", "them", "its", "their",
        "previous", "prior", "earlier", "last", "above", "former", "result", "results",
        "ده", "دي", "دول", "ذلك", "تلك", "هذا", "هذه", "هؤلاء",
        "هو", "هي", "هما", "هم", "هن", "فيهم", "منهم", "عنه", "عنها",
        "بتاعه", "بتاعتها", "بتاعهم", "بتاعتهم",
        "السابق", "السابقة", "اللي فات", "الماضي", "النتيجه", "النتيجة"
    }

    has_ref_marker = any(w in explicit_ref_markers for w in words) or any(m in t for m in ("السابق", "السابقة", "اللي فات", "النتيجه", "النتيجة"))
    if has_ref_marker:
        return True

    # If it lacks standalone domain entities and is a short question with referential interrogatives
    has_domain_term = any(w in standalone_domain_terms for w in words)
    if not has_domain_term and len(words) <= 10:
        short_ref_interrogatives = {"which", "why", "who", "when", "how", "انهي", "أنهي", "اي", "أي", "مين", "كام", "ليه"}
        if any(w in short_ref_interrogatives for w in words):
            return True

    return False


# ==========================================
# Nodes
# ==========================================
def router_node(state: AgentState) -> dict:
    start_turn_telemetry()
    user_question = state["messages"][-1].content
    if parse_period_change(user_question):
        # "which week had the biggest drop ... and why": answered by code end to end, so no router / intent LLM call is made.
        return {"route": "PERIOD_CHANGE", "confidence": 1.0, "reasoning": "Deterministic period-over-period question",
                "language": detect_language(user_question), "prefetched_intent": {}, "viz_html": "", "llm_call_count": 0,
                "stage_latencies": {}, "llm_telemetry": [], "turn_evidence": []}
    token = set_current_stage("router")
    structured_llm = router_llm.with_structured_output(RouteDecision)
    history_context = _recent_context(state["messages"], cfg.ROUTER_HISTORY_MESSAGES, exclude_last=False)

    has_prior_evidence = bool(
        state.get("evidence") or 
        (state.get("analyst_memory") and state.get("analyst_memory").get("previous_results"))
    )
    is_referential = _is_referential_question(user_question)

    messages = [
        SystemMessage(content=cfg.ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Recent conversation:\n{history_context}"),
    ]

    def _classify():
        # Not wrapped in tenacity: CircuitBreakerLLM + provider fallbacks already retry.
        # with_structured_output() drops router_llm's with_config callbacks, so telemetry is attached
        # per call (this is why router calls never showed up in llm_telemetry.jsonl).
        call_cfg = {"callbacks": [telemetry_handler]}
        if not isinstance(router_llm, Runnable):  # test double: call straight through
            return structured_llm.invoke(messages, config=call_cfg)
        key_text = "\n".join(str(m.content) for m in messages)
        value, _hit = llm_cache.get_or_compute(
            "router", f"{cfg.ROUTER_MODEL}|{cfg.ROUTER_REASONING_EFFORT}", key_text,
            lambda: structured_llm.invoke(messages, config=call_cfg).model_dump(),
        )
        return RouteDecision(**value)

    # Speculatively compute the analyst's intent while the route is being classified. Both are
    # independent LLM calls on the same text, so the common ANALYSIS path loses one serial round-trip.
    # If the route turns out not to be ANALYSIS, the prefetched intent is simply discarded.
    prefetched_intent: dict = {}

    def _prefetch():
        try:
            return _compute_intent(state["messages"][-cfg.SUBAGENT_HISTORY_MESSAGES:])
        except Exception as e:  # noqa: BLE001
            logger.info("Intent prefetch failed (analyst will recompute): %s", e)
            return None

    try:
        if _compute_intent is not None and analyst_agent is not None and cfg.INTENT_PREFETCH:
            decision, prefetched = run_parallel([_classify, _prefetch])
            if decision.agent == "ANALYSIS" and prefetched:
                prefetched_intent = prefetched
        else:
            decision = _classify()
        if decision.agent == "GENERAL" and has_prior_evidence and is_referential:
            logger.info("Overriding GENERAL decision to ANALYSIS for referential follow-up with active prior evidence.")
            decision = RouteDecision(
                agent="ANALYSIS",
                confidence=0.95,
                reasoning="Referential follow-up routed to ANALYSIS with active prior evidence in context",
                language=decision.language
            )
    except Exception as e:
        logger.error("Router fully failed: %s", e)
        if has_prior_evidence and is_referential:
            logger.info("Router failed, but question is referential follow-up with prior evidence -> routing to ANALYSIS.")
            decision = RouteDecision(
                agent="ANALYSIS",
                confidence=0.95,
                reasoning="Fallback: referential follow-up routed to ANALYSIS with active prior evidence",
                language="ar" if any('\u0600' <= c <= '\u06FF' for c in user_question) else "en"
            )
        elif looks_like_data_question(user_question):
            # Router LLM down (429 / malformed tool call): a data question must still reach the guarded analyst —
            # the GENERAL model has no data and would invent tables and margins.
            decision = RouteDecision(agent="ANALYSIS", confidence=0.6, reasoning="Fallback: data terms in question",
                                     language=detect_language(user_question))
        else:
            lang_guess = detect_language(user_question)
            decision = RouteDecision(agent="GENERAL", confidence=0.3, reasoning="fallback after failure",
                                     language=lang_guess)
    finally:
        reset_current_stage(token)

    # Opinion / decision questions ("which is best? one name", "اي احسن مطعم استثمر فيه") go to the evidence-bound
    # advisor, whatever the LLM voted: the GENERAL model has no data and answers them with generic essays.
    route = decision.agent
    prior_evidence = state.get("evidence") or (state.get("analyst_memory") or {}).get("previous_results") or []
    # Follow-ups on the restaurants already on screen ("why so low? where is the problem?", "add their city and
    # rating") are answered by the deterministic diagnosis / enrichment, never by re-planning from scratch.
    kind = followup_kind(user_question, prior_evidence)
    if kind:
        route = "FOLLOWUP"
        decision = RouteDecision(agent="ANALYSIS", confidence=max(decision.confidence, 0.9),
                                 reasoning=f"Follow-up ({kind}) on the restaurants of the previous result",
                                 language=decision.language)
        prefetched_intent = {}
    elif is_advisory_question(user_question):
        route = "ADVISOR"
        decision = RouteDecision(agent="ANALYSIS", confidence=max(decision.confidence, 0.9),
                                 reasoning="Opinion/decision question answered from retrieved data",
                                 language=decision.language)
        prefetched_intent = {}

    logger.info(
        "Q='%s' | route=%s | confidence=%.2f | lang=%s | reason=%s",
        user_question, route, decision.confidence, decision.language, decision.reasoning,
    )

    # Per-turn fields are reset here: the checkpointed state would otherwise leak the PREVIOUS turn's chart,
    # latency and LLM-call numbers onto a turn (GENERAL / ADVISOR) that never produces them.
    return {
        "route": route,
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "language": decision.language,
        "prefetched_intent": prefetched_intent,
        "viz_html": "",
        "llm_call_count": 0,
        "stage_latencies": {},
        "llm_telemetry": [],
        "turn_evidence": [],
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
            "repair_attempts": 0,
            "current_step": 0,
        }
        if state.get("prefetched_intent"):
            analyst_input["prefetched_intent"] = state["prefetched_intent"]
        
        # Inject previous state if available
        memory = state.get("analyst_memory", {})
        if memory:
            for k, v in memory.items():
                if k not in ["repair_attempts", "error", "error_type", "safety_blocked", "sql_queries", "current_step"]:
                    analyst_input[k] = v
            
        result = analyst_agent.invoke(analyst_input)
        answer = result.get("final_answer", "No answer generated.")
        answer = sanitize_for_markdown(answer)
        viz_html = result.get("viz_html", "")

        # Save context for next follow-up. A turn that produced no rows (insufficient data, error, a
        # chart-only follow-up) must NOT wipe the last real result the user may still refer to.
        query_results = result.get("query_results") or []
        if query_results:
            new_memory = {
                "previous_question": result.get("previous_question"),
                "previous_plan": result.get("analysis_plan"),
                "previous_results": query_results,
                "previous_intent": result.get("intent"),
            }
            evidence = query_results
        else:
            new_memory = dict(memory) if memory else {}
            evidence = state.get("evidence", [])

        # Pass visualization HTML and telemetry to the outer state
        output = {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "viz_html": viz_html,
            "llm_call_count": result.get("llm_call_count", 0),
            "stage_latencies": result.get("stage_latencies", {}),
            "llm_telemetry": result.get("llm_telemetry") or get_turn_telemetry(),
            "analyst_memory": new_memory,
            "evidence": evidence,
            "turn_evidence": query_results,
        }
        return output
    except Exception as e:
        error_str = str(e).lower()
        logger.error("Analysis agent failed: %s", e)
        
        if "rate limit" in error_str or "429" in error_str or "timeout" in error_str:
            if lang == "ar":
                error_msg = "الخدمة مشغولة حالياً، يرجى المحاولة مرة أخرى بعد قليل."
            else:
                error_msg = "The service is busy, please try again in a moment."
        else:
            if lang == "ar":
                error_msg = "عذراً، واجهت خطأ داخلي أثناء التحليل. يرجى المحاولة مرة أخرى."
            else:
                error_msg = "Sorry, I encountered an internal error during the analysis. Please try again."
                
        return {"final_answer": error_msg, "messages": [AIMessage(content=error_msg)]}


def general_node(state: AgentState) -> dict:
    token = set_current_stage("general")
    question = state["messages"][-1].content
    lang = state.get("language", "en")
    history_context = _recent_context(state["messages"], cfg.GENERAL_HISTORY_MESSAGES, exclude_last=False)

    # Check if the question asks about a previous query result when no analyst evidence is available
    has_evidence = bool(state.get("evidence") or (state.get("analyst_memory") and state.get("analyst_memory").get("previous_results")))
    if not has_evidence and _is_referential_question(question):
        msg = (
            "عذراً، لا توجد نتائج سابقة مسجلة في هذه الجلسة للرجوع إليها. يرجى طرح سؤال جديد بخصوص البيانات المطلوبة."
            if lang == "ar"
            else "I do not have access to any previous query results in this session. Please rephrase as a new data question."
        )
        reset_current_stage(token)
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}

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
        answer = sanitize_for_markdown(answer)
    except Exception as e:
        logger.error("General node failed: %s", e)
        answer = f"[Assistant error: {e}]"
    finally:
        reset_current_stage(token)

    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def _fetch_baseline_ranking(lang: str):
    """No prior result to base an opinion on: retrieve the one the advisor needs (top restaurants by revenue)
    through the normal, guarded analyst pipeline. Returns the analyst result dict, or None on failure."""
    if analyst_agent is None:
        return None
    try:
        result = analyst_agent.invoke({
            "messages": [HumanMessage(content="Top 5 restaurants by revenue")],
            "language": lang, "repair_attempts": 0, "current_step": 0,
        })
        return result if result.get("query_results") else None
    except Exception as e:  # noqa: BLE001
        logger.error("Advisor baseline retrieval failed: %s", e)
        return None


def advisor_node(state: AgentState) -> dict:
    """Opinion / decision questions, answered ONLY from retrieved rows (see advisor.py)."""
    token = set_current_stage("advisor")
    question = state["messages"][-1].content
    lang = state.get("language") or detect_language(question)
    memory = state.get("analyst_memory") or {}
    evidence = state.get("evidence") or memory.get("previous_results") or []
    out: dict = {}
    try:
        previous_question = memory.get("previous_question")
        previous_intent = memory.get("previous_intent")
        viz_html = ""
        if not evidence:
            fetched = _fetch_baseline_ranking(lang)
            if fetched is None:
                answer = (
                    "معرفتش أجيب بيانات أبني عليها رأيي دلوقتي، وماينفعش أرشّح من غير أرقام. جرّب تاني بعد شوية."
                    if lang in ("ar", "mixed") else
                    "I could not retrieve data to base an opinion on, and I will not recommend without numbers. Please try again shortly."
                )
                return {"final_answer": answer, "messages": [AIMessage(content=answer)]}
            evidence = fetched["query_results"]
            previous_question = "Top 5 restaurants by revenue"
            previous_intent = fetched.get("intent")
            viz_html = fetched.get("viz_html", "")
            out["analyst_memory"] = {
                "previous_question": previous_question,
                "previous_plan": fetched.get("analysis_plan"),
                "previous_results": evidence,
                "previous_intent": previous_intent,
            }
            out["evidence"] = evidence
        answer = sanitize_for_markdown(build_advice(question, evidence, lang, previous_intent, previous_question))
        out.update({"final_answer": answer, "messages": [AIMessage(content=answer)], "viz_html": viz_html,
                    "turn_evidence": list(evidence)})
        return out
    finally:
        reset_current_stage(token)


def _to_evidence(items: list) -> list:
    """followups' {columns, rows, sql} -> Evidence objects (for the 'How I got this' provenance panel)."""
    import datetime
    import uuid
    from src.agent.analyst_state import Evidence
    out = []
    for it in items or []:
        cols = tuple(it.get("columns") or [])
        rows = tuple(tuple(r.get(c) for c in cols) if isinstance(r, dict) else tuple(r) for r in it.get("rows") or [])
        out.append(Evidence(query_id=str(uuid.uuid4()), executed_sql=it.get("sql", ""), columns=cols, rows=rows,
                            row_count=len(rows), truncated=False,
                            executed_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    return out


def followup_node(state: AgentState) -> dict:
    """Diagnosis / enrichment of the restaurants in the previous result (see followups.py, diagnostics.py).
    The previous result stays the conversation's reference evidence, so further follow-ups keep working."""
    token = set_current_stage("followup")
    question = state["messages"][-1].content
    lang = state.get("language") or detect_language(question)
    memory = state.get("analyst_memory") or {}
    evidence = state.get("evidence") or memory.get("previous_results") or []
    try:
        kind = followup_kind(question, evidence)
        if not kind or _execute_guarded is None:
            msg = ("مقدرتش أربط سؤالك بالنتيجة السابقة. اسألني سؤال بيانات جديد." if lang in ("ar", "mixed")
                   else "I could not link your question to the previous result. Please ask a new data question.")
            return {"final_answer": msg, "messages": [AIMessage(content=msg)]}
        result = run_followup(kind, question, evidence, lang, _execute_guarded)
        answer = sanitize_for_markdown(result["answer"])
        return {"final_answer": answer, "messages": [AIMessage(content=answer)],
                "turn_evidence": _to_evidence(result.get("evidence"))}
    except Exception as e:  # noqa: BLE001
        logger.exception("Follow-up node failed: %s", e)
        msg = ("حصل خطأ أثناء التحليل، ومش هخمّن من غير أرقام. جرّب تاني." if lang in ("ar", "mixed")
               else "Something went wrong during the analysis, and I will not guess without numbers. Please try again.")
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}
    finally:
        reset_current_stage(token)


def period_change_node(state: AgentState) -> dict:
    """Biggest week/month-over-week/month change + who contributed + measured drivers (see period_change.py)."""
    token = set_current_stage("period_change")
    question = state["messages"][-1].content
    lang = state.get("language") or detect_language(question)
    try:
        if _execute_guarded is None:
            raise RuntimeError("guarded executor unavailable")
        result = run_period_change(question, lang, _execute_guarded)
        answer = sanitize_for_markdown(result["answer"])
        return {"final_answer": answer, "messages": [AIMessage(content=answer)],
                "turn_evidence": _to_evidence(result.get("evidence"))}
    except Exception as e:  # noqa: BLE001
        logger.exception("Period-change node failed: %s", e)
        msg = ("حصل خطأ أثناء التحليل، ومش هخمّن من غير أرقام. جرّب تاني." if lang in ("ar", "mixed")
               else "Something went wrong during the analysis, and I will not guess without numbers. Please try again.")
        return {"final_answer": msg, "messages": [AIMessage(content=msg)]}
    finally:
        reset_current_stage(token)


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
    graph.add_node("general", general_node)
    graph.add_node("advisor", advisor_node)
    graph.add_node("followup", followup_node)
    graph.add_node("period_change", period_change_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_decision, {
        "ANALYSIS": "analysis",
        "GENERAL": "general",
        "ADVISOR": "advisor",
        "FOLLOWUP": "followup",
        "PERIOD_CHANGE": "period_change",
    })
    # ANALYSIS → END directly (the analyst pipeline generates its own final answer + viz)
    graph.add_edge("analysis", END)
    graph.add_edge("general", END)
    graph.add_edge("advisor", END)
    graph.add_edge("followup", END)
    graph.add_edge("period_change", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
