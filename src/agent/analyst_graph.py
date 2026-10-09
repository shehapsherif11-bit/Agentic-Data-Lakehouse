"""
analyst_graph.py — LangGraph analytical pipeline for the Agentic AI Data Analyst.

The full workflow:
  User Question → Intent Analyzer → Metric/Semantic Resolver → Data Sufficiency Check →
  Analysis Planner → SQL Generator → SQL Validator (structural) → SQL Safety Guard (security) →
  SQL Executor → Result Validator → Result Analyzer → Driver Analysis (conditional) →
  Analysis Completeness Check → Business Insight + Visualization

This module replaces the old single-query sql_agent.py for analytical questions.
The sql_agent.py is NOT deleted — it's still compiled and importable for backward
compatibility. This module reuses DatabricksUtil for database access and builds
on the same LangGraph patterns used in router_graph.py.
"""
import json
import logging
import re
import os

from src.agent.llm_factory import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

try:
    from . import router_config as cfg
    from .analyst_state import AnalystState, Evidence
    from .numeric_grounding import ungrounded_numbers, unbacked_narrative_claims
    from . import analyst_prompts as prompts
    from . import metrics_registry as metrics
    from .viz_engine import generate_chart
    from .markdown_utils import sanitize_for_markdown, format_markdown_table
    from . import llm_cache
    from .concurrency import run_parallel
    from .query_intent import extract_top_n, ranking_direction, is_singular_superlative, extract_dimension, DEFAULT_TOP_N, extract_derived_measures
    from .result_classifier import classify_rows, describe_empty
    from .ranking_text import build_ranking_answer
except ImportError:
    from query_intent import extract_top_n, ranking_direction, is_singular_superlative, extract_dimension, DEFAULT_TOP_N, extract_derived_measures
    from result_classifier import classify_rows, describe_empty
    from ranking_text import build_ranking_answer
    import router_config as cfg
    from analyst_state import AnalystState, Evidence
    from numeric_grounding import ungrounded_numbers, unbacked_narrative_claims
    import analyst_prompts as prompts
    import metrics_registry as metrics
    from viz_engine import generate_chart
    from markdown_utils import sanitize_for_markdown, format_markdown_table
    import llm_cache
    from concurrency import run_parallel

try:
    from .telemetry import (
        telemetry_handler, set_current_stage, reset_current_stage,
        get_turn_telemetry, start_turn_telemetry, log_executed_sql
    )
    from .token_budget import check_budget_available, record_tokens
except ImportError:
    from telemetry import (
        telemetry_handler, set_current_stage, reset_current_stage,
        get_turn_telemetry, start_turn_telemetry, log_executed_sql
    )
    from token_budget import check_budget_available, record_tokens

from src.utils.database import DatabricksUtil
from src.utils.pipeline_marker import read_marker
from src.agent.sql_safety_guard import check_sql_safety, check_multiple_queries as safety_check_batch, MAX_ROWS

import time
from functools import wraps

def track_performance(stage_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state: AnalystState):
            if stage_name == "intent_analyzer" and not state.get("llm_telemetry"):
                start_turn_telemetry()
            token = set_current_stage(stage_name)
            start = time.time()
            try:
                result = func(state)
            finally:
                latency = time.time() - start
                reset_current_stage(token)
            
            updates = result if result else {}
            
            latencies = state.get('stage_latencies', {})
            new_latencies = dict(latencies)
            new_latencies[stage_name] = round(latency, 2)
            updates['stage_latencies'] = new_latencies
            
            turn_recs = get_turn_telemetry()
            if turn_recs:
                updates['llm_telemetry'] = list(turn_recs)
                updates['llm_call_count'] = len(turn_recs)
            # No telemetry records means no LLM call was made (cache hit / deterministic path): never guess a count.
            elif 'llm_call_count' not in updates:
                updates['llm_call_count'] = state.get('llm_call_count', 0)

            return updates
        return wrapper
    return decorator

logger = logging.getLogger("analyst")

# ==========================================
# LLM — reuse the same factory from router_graph
# ==========================================
def _build_node_llm(model: str, effort: str | None):
    """Primary chain (Groq -> OpenRouter) for `model`, plus ONE cross-model Groq fallback.
    The cross-model hop skips OpenRouter so a single outage doesn't walk every provider twice."""
    common = dict(temperature=0.0, request_timeout=cfg.LLM_REQUEST_TIMEOUT, max_retries=cfg.LLM_MAX_RETRIES,
                  reasoning_effort=effort, max_total_seconds=cfg.LLM_MAX_TOTAL_SECONDS)
    primary = get_llm(model_name=model, **common).with_config(callbacks=[telemetry_handler])
    cross_model = cfg.FALLBACK_MODEL if model != cfg.FALLBACK_MODEL else cfg.GENERAL_MODEL
    secondary = get_llm(model_name=cross_model, enable_openrouter=False, **common).with_config(callbacks=[telemetry_handler])
    return primary.with_fallbacks([secondary])


# Default "main" chain. Kept as a module-level name: tests monkeypatch it to inject a fake LLM, and
# _llm_for() honours that override for every node.
_analyst_llm = _build_node_llm(cfg.GENERAL_MODEL, "medium")
_DEFAULT_ANALYST_LLM = _analyst_llm
_node_llms: dict = {}


def _llm_for(node: str):
    """Per-node model + reasoning effort (cfg.NODE_LLM_PROFILES), built lazily and memoized."""
    if _analyst_llm is not _DEFAULT_ANALYST_LLM:  # overridden (tests): use it for everything
        return _analyst_llm
    llm = _node_llms.get(node)
    if llm is None:
        model, effort = cfg.NODE_LLM_PROFILES.get(node, (cfg.GENERAL_MODEL, "medium"))
        effort = os.getenv(f"LLM_EFFORT_{node.upper()}", effort)
        llm = _node_llms[node] = _build_node_llm(model, effort)
    return llm


def _invoke_cached(node: str, prompt: str, is_valid=None) -> str:
    """Invoke `node`'s LLM and return the text content. Identical (node, model, prompt) requests are
    served from llm_cache; results failing `is_valid(content)` are never cached. Cache is bypassed
    when the LLM has been overridden (tests) so fakes never pollute the real cache."""
    llm = _llm_for(node)

    def compute() -> str:
        return llm.invoke(prompt).content

    if _analyst_llm is not _DEFAULT_ANALYST_LLM:
        return compute()
    profile = "|".join(str(p) for p in cfg.NODE_LLM_PROFILES.get(node, ("", "")))
    content, _hit = llm_cache.get_or_compute(node, profile, prompt, compute, is_valid or bool)
    return content


_db = DatabricksUtil()

MAX_REPAIRS = 3
MAX_RESULT_ROWS_FOR_LLM = 50  # لا نرسل أكتر من 50 صف للـ LLM عشان ما نملأش الـ context


# ==========================================
# Helper: parse JSON from LLM response
# ==========================================
def _parse_json(text: str) -> dict | list | None:
    """Extract JSON from an LLM response, stripping markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object or array in the text
        for pattern in [r"\{.*\}", r"\[.*\]"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
    return None


def _truncate_results(rows: list[dict], max_rows: int = MAX_RESULT_ROWS_FOR_LLM) -> str:
    """Convert result rows to a string for LLM consumption, truncated to max_rows."""
    if not rows:
        return "No results returned."
    truncated = rows[:max_rows]
    text = json.dumps(truncated, default=str, ensure_ascii=False)
    if len(rows) > max_rows:
        text += f"\n... ({len(rows)} total rows, showing first {max_rows})"
    return text


def _format_prompt(prompt_template: str, **kwargs) -> str:
    """Safely format prompts without tripping over JSON curly braces."""
    result = prompt_template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    return result


# ==========================================
# Node 1: Intent Analyzer
# ==========================================
def _normalize_derived_measures(intent: dict, question: str) -> None:
    """Split "percentage_of_total_revenue"-style metric names into base metric + derived measure, in place.
    Also reads the question itself: the LLM may omit the request or name it differently on each phrasing."""
    derived = set(extract_derived_measures(question))
    base_metrics = []
    for m in intent.get("metrics") or []:
        base, d = metrics.split_derived_metric(m)
        derived.update(d)
        if base and base not in base_metrics:
            base_metrics.append(base)
    intent["metrics"] = base_metrics
    intent["derived_measures"] = sorted(derived)


def compute_intent(messages) -> dict:
    """The LLM part of intent analysis, independent of graph state so the router can run it
    speculatively, in parallel with route classification (same `messages` slice the analyst gets)."""
    question = messages[-1].content

    # Get previous context
    # Clip long answers (tables, diagnoses): the intent only needs the gist, and full tables blow the token budget.
    messages_content = [f"{m.type}: {str(m.content)[:600]}" for m in messages[:-1]]
    chat_history = "\n".join(messages_content[-4:]) if len(messages_content) > 0 else "No previous history."

    prompt = _format_prompt(
        prompts.INTENT_ANALYZER_PROMPT,
        question=question,
        chat_history=chat_history
    )
    token = set_current_stage("intent_analyzer")
    try:
        content = _invoke_cached("intent_analyzer", prompt, is_valid=lambda c: isinstance(_parse_json(c), dict))
    finally:
        reset_current_stage(token)
    intent = _parse_json(content)

    if not isinstance(intent, dict) or not intent:
        # Fallback: treat as simple query
        intent = {
            "is_followup": False,
            "followup_type": None,
            "intent_type": "simple_query",
            "metrics": [],
            "dimensions": [],
            "filters": [],
            "time_period": None,
            "granularity": None,
            "comparisons": [],
            "is_driver_question": False,
        }

    intent["original_question"] = question
    # Deterministic parsing of the count / direction: the LLM may omit or mangle them.
    top_n = extract_top_n(question)
    if top_n:
        intent["top_n"] = top_n
    intent["sort_order"] = ranking_direction(question)
    intent["singular"] = is_singular_superlative(question)
    _normalize_derived_measures(intent, question)
    # The ranked entity must come from THIS question: the LLM sometimes carries the previous turn's dimension
    # over ("مين اكتر عميل محققلي ارباح" after a restaurant ranking was answered with KFC as a "customer").
    asked_dim = extract_dimension(question)
    if asked_dim and intent.get("intent_type") in ("ranking", "simple_query", "comparison"):
        dims = [str(d).lower() for d in (intent.get("dimensions") or [])]
        aliases = {"customer": ("customer", "customers", "customer_name", "customer_id", "user", "users", "user_id", "name", "client"),
                   "restaurant_name": ("restaurant_name", "restaurant", "restaurants", "brand"),
                   "city": ("city", "cities", "restaurant_city"), "cuisine": ("cuisine", "category")}[asked_dim]
        if not any(d in aliases for d in dims):
            intent["dimensions"] = [asked_dim]
            intent["is_followup"] = False
    return intent


@track_performance("intent_analyzer")
def intent_analyzer(state: AnalystState) -> dict:
    """تحليل نية المستخدم — ايه المقاييس والأبعاد المطلوبة وهل دا سؤال جديد ولا متابعة"""
    question = state["messages"][-1].content

    prefetched = state.get("prefetched_intent")
    if prefetched and prefetched.get("original_question") == question:
        intent = dict(prefetched)  # computed by the router in parallel with routing: no extra LLM wait
    else:
        intent = compute_intent(state["messages"])

    logger.info("Intent analyzed: followup=%s, type=%s, metrics=%s", intent.get("followup_type"), intent.get("intent_type"), intent.get("metrics"))

    # Pass previous state fields if it's a follow-up
    updates = {"intent": intent}
    if intent.get("is_followup"):
        updates["is_followup"] = True
        updates["followup_type"] = intent.get("followup_type")
        updates["previous_question"] = state.get("previous_question", "")
        updates["previous_plan"] = state.get("previous_plan", {})
        updates["previous_results"] = state.get("previous_results", [])
    else:
        updates["is_followup"] = False
        updates["followup_type"] = None
        # Since it's a new question, the current question becomes the new previous_question for future follow-ups
        updates["previous_question"] = question
        
    return updates

# ==========================================
# Node 2: Metric / Semantic Resolver
# ==========================================
@track_performance("metric_resolver")
def metric_resolver(state: AnalystState) -> dict:
    """حل المقاييس — هل كل metric موجود ولا محتاج يتحسب ولا مستحيل"""
    intent = state.get("intent", {})
    requested_metrics = intent.get("metrics") or []
    requested_dimensions = intent.get("dimensions") or []

    resolved = []
    for metric_name in requested_metrics:
        result = metrics.resolve_metric(metric_name)
        resolved.append(result)

    sufficiency = metrics.check_data_sufficiency(requested_metrics, requested_dimensions)

    return {
        "resolved_metrics": resolved,
        "data_sufficiency": sufficiency,
    }


# ==========================================
# Node 3: Data Sufficiency Gate
# ==========================================
_PROFIT_LIKE = {"profit", "profit_margin"}


def _profit_proxy(state: AnalystState) -> dict | None:
    """'Which restaurant/customer made me the least/most PROFIT?' — profit needs cost data that does not exist.
    Instead of a dead-end refusal, answer with the closest measurable metric (revenue) and SAY SO prominently.
    Only for plain ranking/lookup questions; a 'why did profit drop' question is still refused."""
    intent = dict(state.get("intent") or {})
    missing = (state.get("data_sufficiency") or {}).get("missing") or []
    if not missing or intent.get("is_driver_question") or intent.get("intent_type") not in ("ranking", "simple_query", "comparison"):
        return None
    if any(metrics.resolve_metric(m["name"]).get("name") not in _PROFIT_LIKE for m in missing):
        return None
    kept = [m for m in (intent.get("metrics") or []) if metrics.resolve_metric(m).get("name") not in _PROFIT_LIKE]
    if not any(metrics.resolve_metric(m).get("name") == "revenue" for m in kept):
        kept = ["revenue"] + kept
    intent["metrics"] = kept
    resolved = [metrics.resolve_metric(m) for m in intent["metrics"]]
    sufficiency = metrics.check_data_sufficiency(intent["metrics"], intent.get("dimensions") or [])
    if not sufficiency.get("sufficient"):
        return None
    ar = state.get("language", "en") in ("ar", "mixed")
    note = ("⚠️ **الربح مش متاح** في البيانات (مفيش تكلفة أو COGS). عرضتلك بدله **الإيراد** كأقرب مؤشر قابل للقياس — "
            "الإيراد مش ربح: إيراد عالي ممكن يكون وراه ربح قليل."
            if ar else
            "⚠️ **Profit is not available** in the data (no cost / COGS). I am showing **revenue** instead as the closest "
            "measurable indicator — revenue is not profit: high revenue can still come with thin margins.")
    return {"intent": intent, "resolved_metrics": resolved, "data_sufficiency": sufficiency, "proxy_note": note}


@track_performance("sufficiency_check")
def data_sufficiency_check(state: AnalystState) -> dict:
    """لو الداتا مش كافية — نقول للمستخدم بوضوح"""
    sufficiency = state.get("data_sufficiency", {})
    if not sufficiency.get("sufficient", True):
        proxy = _profit_proxy(state)
        if proxy:
            return proxy
        missing = sufficiency.get("missing", [])
        lang = state.get("language", "en")
        ar = lang in ("ar", "mixed")
        # The catalog is the source of truth for what CAN be computed (not just the metrics the user asked for).
        available_str = ", ".join(metrics.list_available_metrics())

        def _why(m):
            reason = m.get("reason") or ""
            if reason in ("", "Unknown metric", "complex"):
                reason = ("المقياس ده مش معرّف في كتالوج المقاييس ومفيش أعمدة في جداول Gold تكفي لحسابه"
                          if ar else "this metric is not defined in the metric catalog and the Gold tables lack the data to derive it")
            return reason

        missing_text = "\n".join(f"- {m['name']}: {_why(m)}" for m in missing)
        hint = metrics.alternatives_hint(missing, ar)
        if ar:
            msg = (
                "عذراً، لا يمكن حساب المقاييس المطلوبة لأن البيانات التالية غير متوفرة:\n" 
                f"{missing_text}\n\n" 
                f"المقاييس المتاحة حالياً: {available_str}"
            )
        else:
            msg = (
                "The requested analysis cannot be completed because the following data is not available:\n" 
                f"{missing_text}\n\n" 
                f"Available metrics that can be computed: {available_str}"
            )
        if hint:
            msg += f"\n\n{hint}"
        return {"final_answer": msg, "error": "data_insufficient"}
    return {}


# ==========================================
# Node 4: Analysis Planner
# ==========================================
@track_performance("analysis_planner")
def analysis_planner(state: AnalystState) -> dict:
    """التخطيط التحليلي — ايه الخطوات والاستعلامات المطلوبة"""
    intent = state.get("intent", {})
    resolved = state.get("resolved_metrics", [])

    # Build metric context for the planner
    metrics_context = []
    for m in resolved:
        defn = m.get("definition") or {}
        if m.get("status") == "derivable":
            metrics_context.append(
                f"- {m['name']}: {defn.get('display_name', m['name'])} = {defn.get('sql_expression', 'N/A')}"
            )
        elif m.get("status") == "missing":
            metrics_context.append(f"- {m['name']}: MISSING — {m.get('missing_reason', 'data not available')}")

    schema_context = metrics.get_schema_context_for_llm()

    # Determine which tables might be needed
    tables_needed = set()
    for m in resolved:
        defn = m.get("definition") or {}
        for tbl, _ in defn.get("required_columns", []):
            tables_needed.add(tbl)
    join_warnings = metrics.get_join_safety_context(list(tables_needed)) if tables_needed else ""

    prompt = _format_prompt(prompts.ANALYSIS_PLANNER_PROMPT, 
        intent_json=json.dumps(intent, ensure_ascii=False, default=str),
        metrics_context="\n".join(metrics_context) or "No specific metrics resolved.",
        schema_context=schema_context,
        join_warnings=join_warnings or "No specific join warnings.",
    )
    plan_text = _invoke_cached(
        "analysis_planner", prompt,
        is_valid=lambda c: isinstance(_parse_json(c), dict) and bool((_parse_json(c) or {}).get("steps")),
    )
    plan = _parse_json(plan_text)

    if not plan:
        plan = {
            "goal": intent.get("original_question", "Answer the user's question"),
            "steps": [{"description": "Execute a query to answer the question", "query_purpose": "main query"}],
            "needs_driver_analysis": intent.get("is_driver_question", False),
            "needs_temporal_comparison": bool(intent.get("comparisons")),
            "dimensions_to_investigate": intent.get("dimensions", []),
        }

    logger.info("Analysis plan: %d steps, driver=%s", len(plan.get("steps", [])), plan.get("needs_driver_analysis"))
    return {
        "analysis_plan": plan,
        "schema_context": schema_context,
        "join_warnings": join_warnings,
    }


# ==========================================
# Node 5: SQL Generator (generates ALL queries from the plan)
# ==========================================
@track_performance("sql_generator")
def sql_generator(state: AnalystState) -> dict:
    """توليد استعلامات SQL — واحد لكل خطوة في الخطة"""
    plan = state.get("analysis_plan", {})
    schema_context = state.get("schema_context", "")
    join_warnings = state.get("join_warnings", "")
    resolved = state.get("resolved_metrics", [])

    # Build metric definitions string
    metric_defs = []
    for m in resolved:
        defn = m.get("definition") or {}
        if defn.get("sql_expression"):
            metric_defs.append(f"- {m['name']}: {defn['sql_expression']}")
        elif defn.get("sql_template"):
            metric_defs.append(f"- {m['name']}: (temporal template available)")
    metric_defs_str = "\n".join(metric_defs) or "Use standard SQL aggregations as needed."

    import json
    plan_steps = json.dumps(plan.get("steps", []), ensure_ascii=False)
    
    prompt = _format_prompt(prompts.SQL_GENERATOR_PROMPT, 
        plan_steps=plan_steps,
        schema_context=schema_context,
        join_warnings=join_warnings,
        metric_definitions=metric_defs_str,
    )
    def _has_sql(c: str) -> bool:
        p = _parse_json(c)
        return isinstance(p, dict) and any((q.get("sql") or "").strip() for q in (p.get("queries") or []))

    response_text = _invoke_cached("sql_generator", prompt, is_valid=_has_sql)
    parsed = _parse_json(response_text)

    sql_queries = []
    if parsed and "queries" in parsed:
        for q in parsed["queries"]:
            sql_queries.append({
                "sql": q.get("sql", "").strip(),
                "purpose": q.get("purpose", ""),
                "validated": False,
                "validation_issues": [],
                "attempt": 0,
                "result": None,
                "error": None,
            })
    else:
        # Fallback if LLM didn't return JSON format correctly
        # Try to extract multiple SQL blocks if they exist
        blocks = re.findall(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if not blocks:
            # If no blocks, just take the whole thing
            blocks = [response_text.replace("```json", "").replace("```sql", "").replace("```", "").strip()]
            
        for idx, block in enumerate(blocks):
            purpose = "Main query" if idx == 0 else f"Breakdown dimension {idx}"
            sql_queries.append({
                "sql": block.strip(),
                "purpose": purpose,
                "validated": False,
                "validation_issues": [],
                "attempt": 0,
                "result": None,
                "error": None,
            })

    return {"sql_queries": sql_queries, "current_step": 0, "repair_attempts": 0}


# ==========================================
# Node 5b: Deterministic SQL Generator (Fast Path)
# ==========================================
@track_performance("deterministic_sql_generator")
def deterministic_sql_generator(state: AnalystState) -> dict:
    intent = state.get("intent", {})
    resolved_metrics = state.get("resolved_metrics", [])
    dims = intent.get("dimensions", [])
    
    if not resolved_metrics:
        return {"sql_queries": []}
        
    metric_obj = resolved_metrics[0]
    metric_def = metric_obj.get("definition", {})
    metric_expr = metric_def.get("sql_expression", "")
    
    # Simple mapping of dimensions to tables/columns. Customers are grouped by user_id AND name: ~29% of
    # customer names are shared by several people, so grouping by name alone would merge different customers.
    users = ("workspace.zomato_gold.dim_users", ("user_id", "name"))
    dim_col_map = {
        "city": ("workspace.zomato_gold.dim_resturant", ("city",)),
        "restaurant_name": ("workspace.zomato_gold.dim_resturant", ("restaurant_name",)),
        "restaurant": ("workspace.zomato_gold.dim_resturant", ("restaurant_name",)),
        "cuisine": ("workspace.zomato_gold.dim_resturant", ("cuisine",)),
        "category": ("workspace.zomato_gold.dim_resturant", ("cuisine",)),
        "customer_segment": ("workspace.zomato_gold.dim_users", ("customer_segment",)),
        "customer": users, "customers": users, "customer_name": users, "customer_id": users,
        "user": users, "users": users, "user_id": users, "user_name": users, "name": users, "client": users,
        "date": ("workspace.zomato_gold.dim_date", ("full_date",)),
        "month": ("workspace.zomato_gold.dim_date", ("month_name",)),
        "year": ("workspace.zomato_gold.dim_date", ("year",))
    }

    tables_needed = {metric_def.get("source_table")}
    select_cols = []
    group_cols = []

    dim_col_full = None
    if dims:
        dim = dims[0]
        mapped = dim_col_map.get(dim.lower().strip())
        if mapped and metric_def.get("source_table") == "workspace.zomato_gold.fact_orders":
            t, cols = mapped
            tables_needed.add(t)
            alias = metrics.get_table_alias(t)
            for n, c in enumerate(cols, 1):
                select_cols.append(f"{alias}.{c}")
                group_cols.append(str(n))
            dim_col_full = select_cols[0]
        else:
            # Unknown dimension (or an item-level metric): the LLM planner handles it (see after_fast_path).
            return {"sql_queries": []}

    # Construct FROM and JOINs
    from_clause = ""
    # Usually fact_orders is the base
    base_table = "workspace.zomato_gold.fact_orders" if "workspace.zomato_gold.fact_orders" in tables_needed else list(tables_needed)[0]
    base_alias = metrics.get_table_alias(base_table)
    
    from_clause = f"FROM {base_table} AS {base_alias}"
    
    joins = []
    if base_table == "workspace.zomato_gold.fact_orders":
        if "workspace.zomato_gold.dim_resturant" in tables_needed:
            joins.append("JOIN workspace.zomato_gold.dim_resturant AS dr ON fo.restaurant_id = dr.restaurant_id")
        if "workspace.zomato_gold.dim_users" in tables_needed:
            joins.append("JOIN workspace.zomato_gold.dim_users AS du ON fo.user_id = du.user_id")
        if "workspace.zomato_gold.dim_date" in tables_needed:
            joins.append("JOIN workspace.zomato_gold.dim_date AS dd ON fo.date_id = dd.date_id")
            
    for j in joins:
        from_clause += f" {j}"

    metric_alias = metric_obj.get("name") or "metric_value"   # e.g. revenue — a readable column header
    select_cols.append(f"{metric_expr} AS {metric_alias}")
    if "share_of_total" in (intent.get("derived_measures") or []):
        if metric_obj.get("name") not in metrics.ADDITIVE_METRICS or not group_cols:
            # A share of an average/ratio (or of a single un-grouped number) is meaningless: let the planner decide.
            return {"sql_queries": []}
        # Window over the aggregate: evaluated before LIMIT, so the share is against ALL groups, not just the top N.
        select_cols.append(f"ROUND(100.0 * {metric_expr} / NULLIF(SUM({metric_expr}) OVER (), 0), 2) AS pct_of_total")
    select_clause = "SELECT " + ", ".join(select_cols)
    
    group_clause = f" GROUP BY {', '.join(group_cols)}" if group_cols else ""
    
    order_clause = ""
    limit_clause = ""

    if intent.get("intent_type") == "ranking" and dim_col_full:
        # N and direction come from the question itself (deterministic), never a hardcoded LIMIT.
        question = intent.get("original_question", "")
        top_n = intent.get("top_n") or extract_top_n(question) or DEFAULT_TOP_N
        direction = intent.get("sort_order") or ranking_direction(question)
        direction = "ASC" if str(direction).upper() == "ASC" else "DESC"
        order_clause = f" ORDER BY {metric_alias} {direction}, 1 ASC"
        limit_clause = f" LIMIT {int(top_n)}"

    sql = f"{select_clause} {from_clause}{group_clause}{order_clause}{limit_clause}"
    
    return {
        "sql_queries": [{"sql": sql, "purpose": f"Deterministic generation for {intent.get('intent_type')}", "validated": True, "safety_blocked": False}],
        "current_step": 0,
        "repair_attempts": 0
    }

# ==========================================
# Node 6: SQL Validator
# ==========================================


# ==========================================
# Node 6b: SQL Safety Guard (deterministic — separate from analytical validation)
# ==========================================
@track_performance("sql_safety_guard")
def sql_safety_guard(state: AnalystState) -> dict:
    """Deterministic security layer — blocks destructive/write SQL before Databricks execution.
    
    This is intentionally SEPARATE from the analytical sql_validator.
    It enforces:
      - Read-only (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN only)
      - No INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/MERGE/EXEC/...
      - No multiple statements (injection defense)
      - No suspicious patterns (UNION injection, cloud paths, xp_cmdshell, ...)
      - Table references must be in the allowed schemas whitelist
      - Fail-closed: if we can't verify, we reject
    """
    sql_queries = list(state.get("sql_queries", []))
    
    checked = safety_check_batch(sql_queries)
    
    blocked_count = 0
    for i, q in enumerate(checked):
        if not q.get("safety_passed", False):
            sql_queries[i] = {
                **sql_queries[i],
                "validated": False,
                "error": q.get("error", "SECURITY BLOCKED"), "error_type": "guard",
                "safety_blocked": True,
            }
            blocked_count += 1
        else:
            # Must merge back the injected limit and the cleaned SQL, and clear errors
            sql_queries[i] = {
                **sql_queries[i], 
                "sql": q.get("sql", sql_queries[i].get("sql")),
                "limit_injected": q.get("limit_injected", False),
                "error": None, 
                "error_type": None,
                "safety_blocked": False,
                "validated": True
            }
    
    if blocked_count:
        logger.warning("SQL Safety Guard blocked %d / %d queries.", blocked_count, len(sql_queries))
    
    return {"sql_queries": sql_queries}


# ==========================================
# Node 7: SQL Executor
# ==========================================
@track_performance("sql_executor")
def sql_executor(state: AnalystState) -> dict:
    """Executes validated queries on Databricks behind the result cache.

    Results keep the order of `sql_queries` (cache hits are NOT moved ahead of freshly-executed
    queries). `cache_hit` is True only when EVERY executed query was served from cache; per-turn
    hit/miss counts are returned as cache_hits / cache_misses."""
    sql_queries = list(state.get("sql_queries", []))
    query_results = list(state.get("query_results", []))
    question = (state.get("intent") or {}).get("original_question", "")

    import datetime
    import uuid
    try:
        from . import query_cache
    except ImportError:
        import query_cache

    validated = [(i, q) for i, q in enumerate(sql_queries)
                 if q.get("validated", False) and not q.get("safety_blocked", False)]
    slots: list = [None] * len(validated)   # Evidence per validated query, in order
    to_run: list = []                        # (slot, original_index, query) still needing execution
    claimed: list = []                       # SQL we hold an in-flight claim on (released in `finally`)
    hits = misses = 0

    try:
        for slot, (orig_idx, q) in enumerate(validated):
            cached, leader = query_cache.lookup_or_claim(q["sql"])
            if cached:
                hits += 1
                limit_injected = q.get("limit_injected", False)

                # Rows in cache are list of dicts/lists, convert back to tuple of tuples
                cols = cached.get("columns", [])
                raw_rows = cached.get("rows", [])
                if len(raw_rows) > 0 and isinstance(raw_rows[0], dict):
                    tuple_rows = tuple(tuple(row.get(col) for col in cols) for row in raw_rows)
                else:
                    tuple_rows = tuple(tuple(row) for row in raw_rows)

                slots[slot] = Evidence(
                    query_id=str(uuid.uuid4()),
                    executed_sql=q["sql"],
                    columns=tuple(cols),
                    rows=tuple_rows,
                    row_count=cached.get("row_count", 0),
                    truncated=limit_injected and cached.get("row_count", 0) == MAX_ROWS,
                    executed_at=cached.get("executed_at", datetime.datetime.utcnow().isoformat() + "Z")
                )
                logger.info(f"CACHE HIT for query purpose: {q.get('purpose')}")
                log_executed_sql(question, q["sql"], q.get("purpose", ""), cache_hit=True, row_count=cached.get("row_count"))
            else:
                misses += 1
                if leader:
                    claimed.append(q["sql"])
                to_run.append((slot, orig_idx, q))
                logger.info(f"CACHE MISS for query purpose: {q.get('purpose')}")

        if not to_run and not query_results and not any(slots):
            return {"query_results": [], "error": "No validated queries to execute."}

        if to_run:
            t_exec = time.time()
            results = _db.execute_queries([{"sql": q["sql"], "purpose": q["purpose"]} for _, _, q in to_run])
            exec_s = round(time.time() - t_exec, 2)   # batch wall time (queries may run in parallel)

            for (slot, orig_idx, q), result in zip(to_run, results):
                if result.get("success"):
                    cols = result.get("columns", [])
                    raw_rows = result.get("rows", [])
                    tuple_rows = tuple(tuple(row.get(col) for col in cols) for row in raw_rows)
                    executed_at = datetime.datetime.utcnow().isoformat() + "Z"
                    row_count = result.get("row_count", 0)

                    log_executed_sql(question, q["sql"], q.get("purpose", ""), cache_hit=False, row_count=row_count, seconds=exec_s)
                    query_cache.set_cached_result(
                        sql=result.get("sql", ""),
                        columns=cols,
                        rows=raw_rows,
                        row_count=row_count,
                        executed_at=executed_at
                    )

                    slots[slot] = Evidence(
                        query_id=str(uuid.uuid4()),
                        executed_sql=result.get("sql", ""),
                        columns=tuple(cols),
                        rows=tuple_rows,
                        row_count=row_count,
                        truncated=q.get("limit_injected", False) and row_count == MAX_ROWS,
                        executed_at=executed_at
                    )
                else:
                    error_msg = result.get("error", "Unknown execution error")
                    sql_queries[orig_idx] = {
                        **sql_queries[orig_idx],
                        "error": error_msg,
                        # error_type drives after_execution's routing (repair vs. fatal connection/timeout)
                        "error_type": result.get("error_type"),
                        "attempt": q.get("attempt", 0) + 1,
                    }
                    logger.warning("SQL execution failed for '%s': %s", result.get("purpose"), error_msg)
                    log_executed_sql(question, q["sql"], q.get("purpose", ""), cache_hit=False, seconds=exec_s, error=error_msg)
    finally:
        for sql in claimed:
            query_cache.release_claim(sql)

    query_results.extend(ev for ev in slots if ev is not None)
    logger.info("Query cache this turn: %d hit(s), %d miss(es)", hits, misses)
    return {
        "sql_queries": sql_queries,
        "query_results": query_results,
        "cache_hit": bool(validated) and misses == 0,
        "cache_hits": hits,
        "cache_misses": misses,
    }


# ==========================================
# Node 8: SQL Repair (conditional — triggered on errors)
# ==========================================
@track_performance("sql_repair")
def sql_repair(state: AnalystState) -> dict:
    """Repairs failed SQL queries"""
    sql_queries = list(state.get("sql_queries", []))
    schema_context = state.get("schema_context", "")
    repair_attempts = state.get("repair_attempts", 0)

    failed = [(i, q) for i, q in enumerate(sql_queries) if q.get("error")]

    def _repair_one(q):
        prompt = _format_prompt(prompts.SQL_REPAIR_PROMPT,
            sql_query=q.get("sql", ""),
            errors=q["error"],
            schema_context=schema_context,
        )
        # Not cached: the repair prompt embeds a transient execution error.
        return _llm_for("sql_repair").invoke(prompt).content.strip()

    # Independent failed queries are repaired concurrently instead of one LLM round-trip each.
    fixed_list = run_parallel([(lambda q=q: _repair_one(q)) for _, q in failed])

    for (i, q), fixed_sql in zip(failed, fixed_list):
        fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
        sql_queries[i] = {
            **q,
            "sql": fixed_sql,
            "validated": False,
            "safety_blocked": False,
            "error": None,
            "attempt": q.get("attempt", 0) + 1
        }
        logger.info("Repaired SQL for step '%s' (attempt %d)", q.get("purpose", ""), q.get("attempt", 1))

    return {"sql_queries": sql_queries, "repair_attempts": repair_attempts + 1}


# ==========================================
# Node 9: Result Validator (B-4 Reconciliation Validator)
# ==========================================
@track_performance("result_validator")
def result_validator(state: AnalystState) -> dict:
    """Deterministic Result Validator:
    1. Verifies row count.
    2. Verifies that derived metrics columns (delta, pct_change, sales_prev, contribution) are not 100% NULL.
    3. For period comparison, verifies that total_delta is non-zero when individual deltas exist.
    If invalid, populates error on queries to trigger sql_repair!
    """
    query_results = state.get("query_results", [])
    sql_queries = list(state.get("sql_queries", []))

    if not query_results:
        return {"result_validation": {"valid": False, "issues": ["No query results to validate."], "warnings": [], "severity": "critical"}}

    issues = []
    warnings = []
    
    for idx, r in enumerate(query_results):
        row_count = r.row_count if isinstance(r, Evidence) else r.get("row_count", 0)
        cols = r.columns if isinstance(r, Evidence) else r.get("columns", [])
        rows = [dict(zip(r.columns, row)) for row in r.rows] if isinstance(r, Evidence) else r.get("rows", [])
        
        if row_count == 0:
            purpose = "Query" if isinstance(r, Evidence) else r.get("purpose", "")
            warnings.append(f"Query '{purpose}' returned 0 rows.")
            continue

        # B-4: Check for all-null derived columns
        derived_cols_to_check = [
            "delta", "sales_delta", "sales_prev", "prior_sales",
            "pct_change", "percent_change", "contribution_to_change", "contribution"
        ]
        present_derived = [c for c in cols if c.lower() in derived_cols_to_check]
        
        for c in present_derived:
            values = [row.get(c) for row in rows]
            if values and all(v is None for v in values):
                err_msg = (
                    f"Reconciliation Failure: Derived metric column '{c}' is 100% NULL across all {row_count} rows. "
                    f"The calculation or window function failed (e.g. filtered before LAG or missing partition rows)."
                )
                issues.append(err_msg)
                if idx < len(sql_queries):
                    sql_queries[idx] = {**sql_queries[idx], "error": err_msg}

        # B-4: Period comparison reconciliation
        cols_lower = [c.lower() for c in cols]
        if "total_delta" in cols_lower and "delta" in cols_lower:
            tot_delta = rows[0].get("total_delta")
            if tot_delta is None or (tot_delta == 0 and any(row.get("delta", 0) != 0 for row in rows)):
                err_msg = "Reconciliation Failure: total_delta is zero or NULL while individual entities have non-zero deltas."
                issues.append(err_msg)
                if idx < len(sql_queries):
                    sql_queries[idx] = {**sql_queries[idx], "error": err_msg}

    severity = "ok"
    is_valid = True
    if warnings:
        severity = "warning"
    if issues:
        severity = "critical"
        is_valid = False

    validation = {"valid": is_valid, "issues": issues, "warnings": warnings, "severity": severity}
    out = {"result_validation": validation}
    if not is_valid:
        out["sql_queries"] = sql_queries
    return out


def _build_driver_prompt(query_results: list, main_finding: str) -> str:
    dimensional_results = []
    for r in query_results:
        purpose = ("Query" if isinstance(r, Evidence) else r.get("purpose", "")).lower()
        r_rows = [dict(zip(r.columns, row)) for row in r.rows] if isinstance(r, Evidence) else r.get("rows", [])
        rows_text = _truncate_results(r_rows)
        dimensional_results.append(f"Breakdown: {purpose}\nResults:\n{rows_text}")

    return _format_prompt(prompts.DRIVER_ANALYSIS_PROMPT,
        main_finding=main_finding,
        dimensional_results="\n\n---\n\n".join(dimensional_results) if dimensional_results else "No breakdown available.",
    )


def _merge_driver_result(analysis: dict, drivers: dict | list | None) -> dict:
    if isinstance(drivers, dict) and drivers:
        return {**analysis, "drivers": drivers.get("drivers", []), "driver_summary": drivers.get("summary", "")}
    return {**analysis, "drivers": [], "driver_summary": "Driver analysis completed."}


# ==========================================
# Node 10: Result Analyzer (Deterministic Fast-Path)
# ==========================================
@track_performance("result_analyzer")
def result_analyzer(state: AnalystState) -> dict:
    """تحليل النتائج — استخراج النتائج والأرقام الرئيسية بشكل حتمي لتوفير التوكنز والوقت"""
    query_results = state.get("query_results", [])
    plan = state.get("analysis_plan", {})
    intent = state.get("intent", {})

    if not query_results:
        return {"analysis_result": {"findings": "No data returned.", "evidence": "No rows", "key_numbers": {}, "trend_direction": None}}

    # Deterministic extraction of key numbers across query results
    r0 = query_results[0]
    rows = [dict(zip(r0.columns, row)) for row in r0.rows] if isinstance(r0, Evidence) else r0.get("rows", [])
    cols = r0.columns if isinstance(r0, Evidence) else r0.get("columns", [])
    row_count = len(rows)

    key_numbers = {"row_count": row_count}
    numeric_cols = [c for c in cols if any(isinstance(r.get(c), (int, float)) for r in rows)]
    
    for c in numeric_cols:
        vals = [r.get(c) for r in rows if isinstance(r.get(c), (int, float))]
        if vals:
            key_numbers[f"{c}_total"] = sum(vals)
            key_numbers[f"{c}_min"] = min(vals)
            key_numbers[f"{c}_max"] = max(vals)

    # Fast deterministic path: avoids expensive LLM call for standard/ranking queries
    if intent.get("intent_type") in ("simple_query", "ranking", "comparison") and len(query_results) == 1:
        top_entity = rows[0].get("restaurant_name", rows[0].get("city", "Top Entity")) if rows else "None"
        findings = f"Successfully analyzed {row_count} rows. Leading record: {top_entity}."
        return {
            "analysis_result": {
                "findings": findings,
                "evidence": str(rows[:5]),
                "key_numbers": key_numbers,
                "trend_direction": "down" if key_numbers.get("delta_total", 0) < 0 else "up" if key_numbers.get("delta_total", 0) > 0 else None,
            }
        }

    # For complex multi-query questions, invoke LLM
    results_context = []
    for r in query_results:
        r_rows = [dict(zip(r.columns, row)) for row in r.rows] if isinstance(r, Evidence) else r.get("rows", [])
        rows_text = _truncate_results(r_rows)
        purpose = "Query" if isinstance(r, Evidence) else r.get("purpose", "N/A")
        results_context.append(f"Query Purpose: {purpose}\nResults:\n{rows_text}")

    prompt = _format_prompt(prompts.RESULT_ANALYZER_PROMPT, 
        results_context="\n\n---\n\n".join(results_context),
        goal=plan.get("goal", "Answer the user's question"),
    )
    analyzer_call = lambda: _llm_for("result_analyzer").invoke(prompt).content

    # When a 'why' question will need LLM driver analysis next (no deterministic delta columns), the two
    # LLM calls are independent: run them concurrently and let driver_analysis skip itself afterwards.
    deterministic_drivers = bool(rows) and ("delta" in rows[0] or "drop_amount" in rows[0])
    driver_prompt = None
    if not deterministic_drivers and after_analysis(state) == "driver":
        driver_prompt = _build_driver_prompt(query_results, plan.get("goal", intent.get("original_question", "")))

    if driver_prompt:
        analysis_text, driver_text = run_parallel([
            analyzer_call,
            lambda: _llm_for("driver_analysis").invoke(driver_prompt).content,
        ])
    else:
        analysis_text, driver_text = analyzer_call(), None

    analysis = _parse_json(analysis_text)

    if not analysis:
        analysis = {
            "findings": "Analysis completed.",
            "evidence": str(rows[:5]),
            "key_numbers": key_numbers,
            "trend_direction": None,
        }

    if driver_text is not None:
        analysis = _merge_driver_result(analysis, _parse_json(driver_text))

    return {"analysis_result": analysis}


# ==========================================
# Node 11: Driver / Root-Cause Analysis (conditional)
# ==========================================
@track_performance("driver_analysis")
def driver_analysis(state: AnalystState) -> dict:
    """تحليل الأسباب — استخراج المساهمات الرئيسية"""
    analysis = state.get("analysis_result", {})
    query_results = state.get("query_results", [])

    if not query_results:
        return {"analysis_result": {**analysis, "drivers": [], "driver_summary": "No data available."}}

    r0 = query_results[0]
    rows = [dict(zip(r0.columns, row)) for row in r0.rows] if isinstance(r0, Evidence) else r0.get("rows", [])
    
    # If the SQL already computed contribution_to_change and delta, extract drivers deterministically!
    if rows and ("delta" in rows[0] or "drop_amount" in rows[0]):
        drivers = []
        ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD_PCT", "10.0"))
        
        for r in rows[:5]:
            brand = r.get("restaurant_name", "Unknown")
            delta_val = r.get("delta", r.get("drop_amount", 0))
            contrib = r.get("contribution_to_change", r.get("contribution", 0))
            
            # Simple Python threshold anomaly detection
            pct_change = r.get("pct_change", r.get("percent_change", 0))
            is_anomaly = False
            if isinstance(pct_change, (int, float)) and abs(pct_change) > ANOMALY_THRESHOLD:
                is_anomaly = True
                
            drivers.append({
                "dimension": "restaurant_name",
                "value": brand,
                "impact": f"{delta_val:,.0f} ({contrib*100:.1f}%)" if isinstance(contrib, (int, float)) else f"{delta_val:,.0f}",
                "direction": "negative" if delta_val < 0 else "positive",
                "is_anomaly": is_anomaly
            })
        
        tot_delta = rows[0].get("total_delta", sum(r.get("delta", 0) for r in rows))
        summary = f"Top {len(drivers)} contributors explain a combined decline against a total ecosystem change of {tot_delta:,.0f}."
        
        anomaly_count = sum(1 for d in drivers if d.get("is_anomaly"))
        if anomaly_count > 0:
            summary += f" Detected {anomaly_count} anomalous movements exceeding {ANOMALY_THRESHOLD}% threshold."
            
        return {"analysis_result": {**analysis, "drivers": drivers, "driver_summary": summary}}

    # Already produced concurrently by result_analyzer (LLM path)
    if "drivers" in analysis:
        return {}

    # Fallback to LLM driver analysis if columns are non-standard
    prompt = _build_driver_prompt(query_results, analysis.get("findings", ""))
    response = _llm_for("driver_analysis").invoke(prompt)
    return {"analysis_result": _merge_driver_result(analysis, _parse_json(response.content))}


# ==========================================
# Node 12: Analysis Completeness Check (Deterministic)
# ==========================================
_SHARE_COLUMN = re.compile(r"share|pct|percent|contribution|proportion|نسبة|حصة", re.IGNORECASE)


def _missing_derived_columns(intent: dict, query_results: list) -> list[str]:
    """Derived columns the user asked for ('share_of_total') that no executed query actually returned."""
    if "share_of_total" not in (intent.get("derived_measures") or []) or not query_results:
        return []
    for r in query_results:
        cols = r.columns if isinstance(r, Evidence) else r.get("columns", [])
        if any(_SHARE_COLUMN.search(str(c)) for c in cols):
            return []
    return ["percentage share of total was requested but is not in the query result"]


@track_performance("completeness_check")
def completeness_check(state: AnalystState) -> dict:
    """التحقق من اكتمال التحليل بشكل حتمي لتوفير التوكنز والوقت"""
    validation = state.get("result_validation", {})
    query_results = state.get("query_results", [])
    
    missing = _missing_derived_columns(state.get("intent") or {}, query_results)
    if missing:
        return {"completeness": {"complete": False, "answered_question": "A requested column is missing from the result.",
                                 "missing_aspects": missing}}

    if validation.get("valid", True) and query_results:
        return {"completeness": {"complete": True, "answered_question": "Analysis completed successfully.", "missing_aspects": []}}
    
    return {"completeness": {"complete": False, "answered_question": "Issues encountered during execution.", "missing_aspects": validation.get("issues", [])}}


# ==========================================
# Node 13: Visualization Planner
# ==========================================
_COVERAGE_SQL = (
    "SELECT MIN(dd.full_date) AS min_date, MAX(dd.full_date) AS max_date "
    "FROM workspace.zomato_gold.fact_orders fo JOIN workspace.zomato_gold.dim_date dd ON fo.date_id = dd.date_id"
)
_coverage_cache: dict = {}


def _data_coverage() -> dict | None:
    """First/last order date in the warehouse (cached 1h). None on any failure — callers must cope."""
    cached = _coverage_cache.get("v")
    # Stale after an hour OR once a pipeline run finished after it was cached (new data moves the last order date).
    if cached and time.time() - cached[0] < 3600 and read_marker() <= cached[0]:
        return cached[1]
    try:
        res = _db.execute_queries([{"sql": _COVERAGE_SQL, "purpose": "data coverage"}])
        if res and res[0].get("success") and res[0].get("rows"):
            row = res[0]["rows"][0]
            value = {"min_date": row.get("min_date"), "max_date": row.get("max_date")}
            if value["min_date"] is not None:
                _coverage_cache["v"] = (time.time(), value)
                return value
    except Exception as e:  # noqa: BLE001 - coverage is an extra; never break the answer
        logger.info("Data coverage lookup failed: %s", e)
    return None


def _explain_empty_results(query_results: list, intent: dict, lang: str) -> str | None:
    """If EVERY executed query is empty (no rows, or one all-NULL/0 aggregate row) return an honest explanation."""
    if not query_results:
        return None
    first = None
    for r in query_results:
        cols = list(r.columns) if isinstance(r, Evidence) else list(r.get("columns", []))
        rows = [dict(zip(r.columns, row)) for row in r.rows] if isinstance(r, Evidence) else r.get("rows", [])
        info = classify_rows(cols, rows)
        if info["kind"] == "ok":
            return None
        first = first or (info["kind"], cols, rows)
    kind, cols, rows = first
    return describe_empty(kind, cols, rows, _data_coverage(), lang, intent.get("time_period"))


# ==========================================
# Node 14: Business Insight Generator (final answer)
# ==========================================
@track_performance("insight_generator")
def insight_generator(state: AnalystState) -> dict:
    """توليد الإجابة النهائية والتخطيط البياني معاً لتوفير الوقت"""
    intent = state.get("intent", {})
    analysis = state.get("analysis_result", {})
    query_results = state.get("query_results", [])
    lang = state.get("language", "en")

    # CHANGE_VISUALIZATION / CLARIFICATION follow-ups skip SQL entirely: they must work on the rows the
    # user is already looking at, not on an empty result set.
    if not query_results and state.get("is_followup") and state.get("followup_type") in ("CHANGE_VISUALIZATION", "CLARIFICATION"):
        query_results = list(state.get("previous_results") or [])
        if not query_results:
            msg = ("مفيش نتيجة سابقة أعدّل عليها أو أشرحها. اسألني سؤال بيانات الأول." if lang in ("ar", "mixed")
                   else "There is no earlier result to modify or explain. Please ask a data question first.")
            return {"final_answer": msg, "viz_html": "", "messages": [AIMessage(content=msg)]}

    # Empty / NULL-only results are explained deterministically (zero vs NULL vs no rows) — never narrated by the LLM.
    empty_answer = _explain_empty_results(query_results, intent, lang)
    if empty_answer:
        if state.get("proxy_note"):
            empty_answer = state["proxy_note"] + "\n\n" + empty_answer
        empty_answer = sanitize_for_markdown(empty_answer)
        return {"final_answer": empty_answer, "viz_html": "", "viz_spec": {"should_visualize": False},
                "messages": [AIMessage(content=empty_answer)]}

    # Build driver analysis text
    drivers = analysis.get("drivers", [])
    driver_text = ""
    if drivers:
        driver_text = analysis.get("driver_summary", "")
        if isinstance(drivers, list) and drivers:
            driver_lines = []
            for d in drivers:
                driver_lines.append(f"- {d.get('dimension', 'N/A')}: {d.get('value', 'N/A')} (impact: {d.get('impact', 'N/A')}, direction: {d.get('direction', 'N/A')})")
            driver_text += "\n" + "\n".join(driver_lines)

    # Build evidence object
    evidence_object = {
        "question": intent.get("original_question", ""),
        "sql_executed": len(query_results) > 0,
        "queries_executed": len(query_results),
        "results": [
            {
                "query_id": r.query_id if hasattr(r, 'query_id') else "N/A",
                "row_count": r.row_count if hasattr(r, 'row_count') else r.get("row_count", 0),
                "truncated": r.truncated if hasattr(r, 'truncated') else False,
                "columns": r.columns if hasattr(r, 'columns') else r.get("columns", []),
                "final_sql_output": [dict(zip(r.columns, row)) for row in r.rows][:50] if hasattr(r, 'rows') else r.get("rows", [])[:50]
            } for r in query_results
        ],
        "findings": analysis.get("findings", ""),
        "key_numbers": analysis.get("key_numbers", {}),
    }

    # Find the most appropriate result for visualization (usually the first one with multiple rows)
    viz_target_result = None
    for r in query_results:
        rc = r.row_count if isinstance(r, Evidence) else r.get("row_count", 0)
        if rc > 1:
            viz_target_result = r
            break
    if not viz_target_result and query_results:
        viz_target_result = query_results[0]

    # B-4 Table-First Path: Rankings and multi-row list queries must NOT be narrated by LLM
    question_text = intent.get("original_question", "")
    is_ranking = (
        intent.get("intent_type") == "ranking"
        or (not intent.get("is_driver_question")
            and re.search(r"\b(top|highest|lowest|bottom)\b|اكتر|أفضل|أكثر", question_text.lower()) is not None)
    )
    target_rc = viz_target_result.row_count if isinstance(viz_target_result, Evidence) else viz_target_result.get("row_count", 0) if viz_target_result else 0
    if is_ranking and viz_target_result and target_rc > 1:
        rows = [dict(zip(viz_target_result.columns, row)) for row in viz_target_result.rows] if isinstance(viz_target_result, Evidence) else viz_target_result.get("rows", [])
        cols = list(viz_target_result.columns) if isinstance(viz_target_result, Evidence) else list(viz_target_result.get("columns", []))

        bottom = str(intent.get("sort_order", "DESC")).upper() == "ASC"
        final_table_answer = sanitize_for_markdown(
            build_ranking_answer(rows, cols, intent, lang, proxy_note=state.get("proxy_note")))
        if _missing_derived_columns(intent, [viz_target_result]):
            # Never present a table as the full answer when a requested column is absent: say so instead of dropping it silently.
            final_table_answer += ("\n\n⚠️ ملحوظة: العمود المطلوب (نسبة المساهمة من الإجمالي) مش موجود في نتيجة الاستعلام، فمعرضتوش بدل ما أقدّره."
                                   if lang in ("ar", "mixed") else
                                   "\n\n⚠️ Note: the requested percentage-of-total column is not in the query result, so it is not shown rather than estimated.")
        rows = rows[:max(15, intent.get("top_n") or 0)]   # chart only what is shown

        # Automatic visualization spec
        viz_spec = {"should_visualize": False}
        viz_html = None
        if len(cols) >= 2 and len(rows) > 1:
            numeric_cols = [c for c in cols if c != "user_id" and any(isinstance(r.get(c), (int, float)) for r in rows)]
            text_cols = [c for c in cols if c not in numeric_cols and c != "user_id"]
            x_col = text_cols[0] if text_cols else cols[0]
            y_col = numeric_cols[0] if numeric_cols else cols[1]
            viz_spec = {
                "should_visualize": True,
                "chart_type": "bar",
                "x_axis": x_col,
                "y_axis": y_col,
                "title": f"{'Bottom' if bottom else 'Top'} {len(rows)} by {y_col.replace('_', ' ').title()}"
            }
            try:
                viz_html = generate_chart(viz_spec, rows)
            except Exception as e:
                logger.error("Chart generation failed: %s", e)

        return {
            "evidence_object": evidence_object,
            "final_answer": final_table_answer,
            "viz_spec": viz_spec,
            "viz_html": viz_html or "",
            "messages": [AIMessage(content=final_table_answer)]
        }

    results_summary = "No results available for visualization."
    if viz_target_result:
        rows = [dict(zip(viz_target_result.columns, row)) for row in viz_target_result.rows] if isinstance(viz_target_result, Evidence) else viz_target_result.get("rows", [])
        cols = viz_target_result.columns if isinstance(viz_target_result, Evidence) else viz_target_result.get("columns", [])
        rc = viz_target_result.row_count if isinstance(viz_target_result, Evidence) else viz_target_result.get("row_count", 0)
        rows_sample = _truncate_results(rows, max_rows=20)
        results_summary = f"Columns: {cols}\nRow Count: {rc}\nSample:\n{rows_sample}"

    lang_instruction = cfg.LANGUAGE_INSTRUCTIONS.get(lang, cfg.LANGUAGE_INSTRUCTIONS.get("en", "Reply in English."))

    # B-5 Honest Narrative rules for 'why' / driver analysis
    b5_instructions = ""
    if intent.get("is_driver_question") or intent.get("intent_type") == "driver_analysis":
        if query_results:
            r0 = query_results[0]
            rows0 = [dict(zip(r0.columns, row)) for row in r0.rows] if isinstance(r0, Evidence) else r0.get("rows", [])
            if rows0 and "delta" in rows0[0] and "total_delta" in rows0[0]:
                tot_delta = rows0[0].get("total_delta", 0)
                days_cur = rows0[0].get("days_with_data_cur", 30)
                days_prev = rows0[0].get("days_with_data_prev", 31)
                top_5_delta = sum(r.get("delta", 0) for r in rows0[:5])
                concentration = abs(top_5_delta / tot_delta) if tot_delta else 0.0

                b5_instructions = (
                    f"\n\nCRITICAL REPORTING CONSTRAINTS (B-5):\n"
                    f"- Total Ecosystem Delta: {tot_delta:,.0f}\n"
                    f"- Current active days: {days_cur} | Previous active days: {days_prev}\n"
                    f"- Top 5 entities concentration: {concentration*100:.1f}%\n"
                )
                if concentration < 0.20:
                    b5_instructions += (
                        f"- CONCENTRATION IS LOW ({concentration*100:.1f}% < 20%): The top 5 decliners explain only "
                        f"{concentration*100:.1f}% of the total decline. You MUST state clearly that the decline is "
                        f"broadly distributed across the restaurant ecosystem and NOT attributable primarily to any single brand. "
                        f"Do NOT present the top decliner as 'the cause'.\n"
                    )
                if days_cur != days_prev:
                    b5_instructions += (
                        f"- CALENDAR DIFFERENCE: The current period had {days_cur} days whereas the prior period had {days_prev} days. "
                        f"You MUST explicitly cite this 1-day difference ({days_prev} vs {days_cur} days) and compare average daily sales "
                        f"(avg_daily_cur vs avg_daily_prev) to explain how much of the drop is simply due to a shorter month.\n"
                    )

    prompt = _format_prompt(prompts.INSIGHT_GENERATOR_PROMPT, 
        question=intent.get("original_question", ""),
        evidence_object=json.dumps(evidence_object, ensure_ascii=False, default=str)[:3000],
        driver_analysis=(driver_text or "No driver analysis performed.") + b5_instructions,
        results_summary=results_summary,
        language_instruction=lang_instruction,
    )
    
    data_for_grounding = []
    for r in query_results:
        if isinstance(r, Evidence):
            data_for_grounding.extend([dict(zip(r.columns, row)) for row in r.rows])
        else:
            data_for_grounding.extend(r.get("rows", []))
            
    response = _llm_for("insight_generator").invoke(prompt)
    parsed = _parse_json(response.content)
    
    if parsed and "insight" in parsed:
        ungrounded = ungrounded_numbers(parsed["insight"], data_for_grounding)
        narrative_violations = unbacked_narrative_claims(parsed["insight"], data_for_grounding)
        hallucination_issues = ungrounded + narrative_violations
        if hallucination_issues:
            logger.warning(f"Hallucination detected in insight: {hallucination_issues}. Retrying once.")
            retry_prompt = (
                prompt + "\n\nCRITICAL WARNING: Your previous answer contained these ungrounded numbers or unbacked claims:\n" 
                + "\n".join(f"- {issue}" for issue in hallucination_issues) 
                + "\nYou MUST NOT invent numbers, and NEVER claim entities represent the 'bulk' or 'majority' without explicit >50% share proof."
            )
            response2 = _llm_for("insight_generator").invoke(retry_prompt)
            parsed2 = _parse_json(response2.content)
            
            if parsed2 and "insight" in parsed2:
                parsed = parsed2
                ungrounded2 = ungrounded_numbers(parsed["insight"], data_for_grounding)
                narrative_violations2 = unbacked_narrative_claims(parsed["insight"], data_for_grounding)
                if ungrounded2 or narrative_violations2:
                    logger.error(f"Hallucination persisted: {ungrounded2 + narrative_violations2}. Using fallback template.")
                    if lang == "ar":
                        parsed["insight"] = f"\u062a\u0645 \u0625\u0631\u062c\u0627\u0639 {evidence_object['results'][0]['row_count'] if evidence_object['results'] else 0} \u0635\u0641\u0648\u0641 \u0645\u0646 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a. \u064a\u0631\u062c\u0649 \u0645\u0631\u0627\u062c\u0639\u0629 \u0627\u0644\u062a\u0641\u0627\u0635\u064a\u0644 \u0641\u064a \u0642\u0633\u0645 '\u0643\u064a\u0641 \u062d\u0635\u0644\u062a \u0639\u0644\u0649 \u0647\u0630\u0627'."
                    else:
                        parsed["insight"] = f"The data returned {evidence_object['results'][0]['row_count'] if evidence_object['results'] else 0} rows. Please view the 'How I got this' expander for details."
                        
        final_answer = parsed["insight"]
        viz_spec = parsed.get("viz_spec", {"should_visualize": False})
    else:
        # Fallback if parsing fails
        final_answer = response.content
        viz_spec = {"should_visualize": False}
        
    viz_html = None
    if viz_spec.get("should_visualize") and viz_target_result:
        viz_data = [dict(zip(viz_target_result.columns, row)) for row in viz_target_result.rows] if isinstance(viz_target_result, Evidence) else viz_target_result.get("rows", [])
        try:
            chart_html = generate_chart(viz_spec, viz_data)
            if chart_html:
                viz_html = chart_html
        except Exception as e:
            logger.error(f"Viz generation failed: {e}")

    # Item 8: Tabular formatting for answers containing 3+ related numeric values
    # For narrator-generated answers ("why" questions, trends, comparisons):
    # any numbers the narrative references as a set must ALSO be rendered as a small Markdown table below the prose.
    if query_results:
        target_res = viz_target_result or query_results[0]
        data_rows = [dict(zip(target_res.columns, row)) for row in target_res.rows] if isinstance(target_res, Evidence) else target_res.get("rows", [])
        if len(data_rows) >= 3 and "|" not in final_answer:
            table_md = format_markdown_table(data_rows, max_rows=10, lang=lang)
            if table_md:
                final_answer = f"{final_answer.strip()}\n\n{table_md}"

    if state.get("proxy_note"):
        final_answer = state["proxy_note"] + "\n\n" + final_answer

    # Item 6: Apply centralized Markdown / LaTeX sanitization
    final_answer = sanitize_for_markdown(final_answer)

    return {"evidence_object": evidence_object, "final_answer": final_answer, "viz_spec": viz_spec, "viz_html": viz_html, "messages": [AIMessage(content=final_answer)]}


# ==========================================
# Routing functions (conditional edges)
# ==========================================
def after_sufficiency(state: AnalystState) -> str:
    """بعد فحص كفاية البيانات — نكمل ولا نوقف"""
    if state.get("error") == "data_insufficient":
        return "end"
    return "continue"



def after_guard(state: AnalystState) -> str:
    """Routes after safety guard: if guard fails, go to repair or error."""
    sql_queries = state.get("sql_queries", [])
    repair_attempts = state.get("repair_attempts", 0)

    has_guard_errors = any(q.get("error_type") == "guard" for q in sql_queries)
    if has_guard_errors:
        if repair_attempts < MAX_REPAIRS:
            return "repair"
        else:
            return "end_with_error"
    return "execute"

def after_execution(state: AnalystState) -> str:
    """Routes after execution: if genuine SQL errors, go to repair."""
    sql_queries = state.get("sql_queries", [])
    repair_attempts = state.get("repair_attempts", 0)

    has_sql_errors = any(q.get("error_type") == "sql" for q in sql_queries)
    has_fatal_errors = any(q.get("error_type") in ["connection", "timeout"] for q in sql_queries)
    
    if has_fatal_errors:
        return "end_with_error"
        
    if has_sql_errors:
        if repair_attempts < MAX_REPAIRS:
            return "repair"
        else:
            return "end_with_error"
            
    query_results = state.get("query_results", [])
    if not query_results:
        return "end_with_error"

    return "validate"


def after_result_validation(state: AnalystState) -> str:
    """بعد فحص النتائج — لو critical أو غير صالح نبعت للـ repair لو لسه في محاولات"""
    validation = state.get("result_validation", {})
    if not validation.get("valid", True) or validation.get("severity") == "critical":
        repair_attempts = state.get("repair_attempts", 0)
        if repair_attempts < MAX_REPAIRS:
            return "repair"
        return "end_with_error"
    return "analyze"


def after_analysis(state: AnalystState) -> str:
    """بعد التحليل — لو سؤال why محتاج driver analysis"""
    intent = state.get("intent", {})
    plan = state.get("analysis_plan", {})
    
    # Fast path: skip driver analysis for simple queries
    if intent.get("intent_type") in ["simple_query", "ranking"]:
        return "completeness"
        
    if plan.get("needs_driver_analysis", False) or intent.get("is_driver_question", False):
        return "driver"
        
    return "completeness"


def after_completeness(state: AnalystState) -> str:
    """بعد فحص الاكتمال — دايما نكمل للـ visualization"""
    return "viz"


# ==========================================
# Error end node
# ==========================================
def error_end(state: AnalystState) -> dict:
    """نهاية بخطأ — نرجع رسالة واضحة"""
    lang = state.get("language", "en")
    sql_queries = state.get("sql_queries", [])
    errors = [q.get("error", "") for q in sql_queries if q.get("error")]

    types = {q.get("error_type") for q in sql_queries if q.get("error")}
    if errors:
        logger.error("Analysis ended with errors: %s", errors)   # technical details stay in the logs
    ar = lang in ("ar", "mixed")
    if types & {"connection", "timeout"}:
        reason = ("قاعدة البيانات مش بترد دلوقتي (اتصال/مهلة). جرّب تاني بعد دقيقة." if ar
                  else "The database is not responding right now (connection/timeout). Please retry in a minute.")
    elif "guard" in types:
        reason = ("الاستعلام اللي اتكوّن مكانش آمن (قراءة فقط) فاترفض. جرّب تصيغ السؤال بالمقياس والبُعد، مثلاً: «أعلى 5 مطاعم من حيث عدد الطلبات»."
                  if ar else "The generated query did not pass the read-only safety guard. Try phrasing it as metric + dimension, e.g. “top 5 restaurants by number of orders”.")
    elif errors:
        reason = ("الاستعلام فشل حتى بعد 3 محاولات تصليح. جرّب تبسّط السؤال أو تحدد المقياس والفترة."
                  if ar else "The query kept failing after 3 repair attempts. Try simplifying the question or stating the metric and period.")
    else:
        reason = ("مقدرتش أكوّن استعلام للسؤال ده. حدد المقياس (إيراد، طلبات، تقييم...) والبُعد (مطعم، مدينة، عميل...)، مثلاً: «مين أكتر عميل من حيث الإيراد»."
                  if ar else "I could not build a query for this question. Name the metric (revenue, orders, rating...) and the dimension (restaurant, city, customer...), e.g. “top customer by revenue”.")
    msg = ("عذراً، لم أتمكن من إكمال التحليل. " if ar else "Sorry, I could not complete the analysis. ") + reason

    msg = sanitize_for_markdown(msg)
    return {"final_answer": msg, "messages": [AIMessage(content=msg)]}


# ==========================================
# Build the analyst graph
# ==========================================
def build_analyst_graph():
    """Builds and compiles the full analytical pipeline graph."""
    graph = StateGraph(AnalystState)

    # Add all nodes
    graph.add_node("intent_analyzer", intent_analyzer)
    graph.add_node("metric_resolver", metric_resolver)
    graph.add_node("sufficiency_check", data_sufficiency_check)
    graph.add_node("analysis_planner", analysis_planner)
    graph.add_node("deterministic_sql_generator", deterministic_sql_generator)
    graph.add_node("sql_generator", sql_generator)
    graph.add_node("sql_safety_guard", sql_safety_guard)
    graph.add_node("sql_executor", sql_executor)
    graph.add_node("sql_repair", sql_repair)
    graph.add_node("result_validator", result_validator)
    graph.add_node("result_analyzer", result_analyzer)
    graph.add_node("driver_analysis", driver_analysis)
    graph.add_node("completeness_check", completeness_check)
    
    graph.add_node("insight_generator", insight_generator)
    graph.add_node("error_end", error_end)

    # Wire the edges
    graph.set_entry_point("intent_analyzer")

    # Conditional: after intent -> if CHANGE_VISUALIZATION or CLARIFICATION -> skip to insight_generator
    def after_intent(state: AnalystState) -> str:
        if state.get("is_followup") and state.get("followup_type") in ("CHANGE_VISUALIZATION", "CLARIFICATION"):
            return "insight"
        return "metrics"
        
    graph.add_conditional_edges("intent_analyzer", after_intent, {
        "metrics": "metric_resolver",
        "insight": "insight_generator"
    })
    
    graph.add_edge("metric_resolver", "sufficiency_check")

    # Conditional: sufficiency → (fast_path | continue to planner | end with message)
    def after_sufficiency(state: AnalystState) -> str:
        if not state.get("data_sufficiency", {}).get("sufficient", True):
            return "end"
        
        intent = state.get("intent", {})
        itype = intent.get("intent_type", "")
        is_driver = intent.get("is_driver_question", False)
        metrics = intent.get("metrics") or []
        dims = intent.get("dimensions") or []
        filters = intent.get("filters") or []
        comparisons = intent.get("comparisons") or []
        
        # FAST PATH CRITERIA:
        # - Not a 'why' question
        # - Single step intent type (simple_query, ranking, simple breakdown)
        # - Max 1 metric, max 1 dimension
        # - No complex comparisons or complex filtering
        if not is_driver and itype in ("simple_query", "ranking") and len(metrics) == 1 and len(dims) <= 1 and not comparisons and not filters and not intent.get("time_period")                 and set(intent.get("derived_measures") or []) <= {"share_of_total"}:
            return "fast_path"
            
        return "continue"

    graph.add_conditional_edges("sufficiency_check", after_sufficiency, {
        "fast_path": "deterministic_sql_generator",
        "continue": "analysis_planner",
        "end": END,
    })

    # Linear: planner → sql gen → sql validate → SAFETY GUARD → sql execute
    graph.add_edge("analysis_planner", "sql_generator")
    graph.add_edge("sql_generator", "sql_safety_guard")
    # The fast path returns no query for a dimension it cannot map: fall back to the LLM planner instead of
    # ending with "could not complete the analysis" (that is what "مين اكتر عميل اشتري مني" hit).
    graph.add_conditional_edges(
        "deterministic_sql_generator",
        lambda s: "guard" if s.get("sql_queries") else "planner",
        {"guard": "sql_safety_guard", "planner": "analysis_planner"},
    )
    
    graph.add_conditional_edges("sql_safety_guard", after_guard, {
        "execute": "sql_executor",
        "repair": "sql_repair",
        "end_with_error": "error_end",
    })
        
    # Conditional: after execution → (repair | validate results | error end)
    graph.add_conditional_edges("sql_executor", after_execution, {
        "repair": "sql_repair",
        "validate": "result_validator",
        "end_with_error": "error_end",
    })

    # Repair loop → back to validator
    graph.add_edge("sql_repair", "sql_safety_guard")

    # Conditional: after result validation → (analyze | repair | error end)
    graph.add_conditional_edges("result_validator", after_result_validation, {
        "analyze": "result_analyzer",
        "repair": "sql_repair",
        "end_with_error": "error_end",
    })

    # Conditional: after analysis → (driver analysis | completeness check)
    graph.add_conditional_edges("result_analyzer", after_analysis, {
        "driver": "driver_analysis",
        "completeness": "completeness_check",
    })

    # Driver → completeness
    graph.add_edge("driver_analysis", "completeness_check")

    # Completeness → insight
    graph.add_conditional_edges("completeness_check", after_completeness, {
        "viz": "insight_generator",
    })

    # Final nodes → END
    graph.add_edge("insight_generator", END)
    graph.add_edge("error_end", END)

    return graph.compile()


# Module-level compiled graph
analyst_agent = build_analyst_graph()


def execute_guarded(queries: list[dict]) -> list[dict]:
    """Run code-built read-only queries (follow-up diagnosis / enrichment) through the SAME safety guard and
    result cache as the analyst pipeline. Returns [{success, columns, rows, error, sql}] in input order."""
    try:
        from . import query_cache
    except ImportError:
        import query_cache

    checked = safety_check_batch(queries)
    out: list = [None] * len(checked)
    to_run = []
    for i, q in enumerate(checked):
        if not q.get("safety_passed"):
            logger.error("Follow-up SQL blocked by the guard: %s", q.get("error"))
            out[i] = {"success": False, "error": q.get("error"), "columns": [], "rows": [], "sql": q.get("sql")}
            continue
        cached, _leader = query_cache.lookup_or_claim(q["sql"])
        if cached:
            cols = cached.get("columns", [])
            rows = [r if isinstance(r, dict) else dict(zip(cols, r)) for r in cached.get("rows", [])]
            out[i] = {"success": True, "columns": cols, "rows": rows, "error": None, "sql": q["sql"]}
        else:
            to_run.append((i, q))
    try:
        if to_run:
            results = _db.execute_queries([{"sql": q["sql"], "purpose": q.get("purpose", "")} for _, q in to_run])
            for (i, q), res in zip(to_run, results):
                if res.get("success"):
                    import datetime
                    query_cache.set_cached_result(sql=q["sql"], columns=res.get("columns", []), rows=res.get("rows", []),
                                                  row_count=res.get("row_count", 0),
                                                  executed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                out[i] = {"success": bool(res.get("success")), "columns": res.get("columns", []),
                          "rows": res.get("rows", []), "error": res.get("error"), "sql": q["sql"]}
    finally:
        for _, q in to_run:
            query_cache.release_claim(q["sql"])
    return out


def warm_up_db(background: bool = True):
    """Pay Databricks cold-start (warehouse wake-up, first connection, Gold schema introspection) at app
    startup instead of on the first user question. Never raises."""
    return _db.warm_up(background=background)
