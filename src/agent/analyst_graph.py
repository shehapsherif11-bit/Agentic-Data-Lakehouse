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
except ImportError:
    import router_config as cfg
    from analyst_state import AnalystState, Evidence
    from numeric_grounding import ungrounded_numbers, unbacked_narrative_claims
    import analyst_prompts as prompts
    import metrics_registry as metrics
    from viz_engine import generate_chart
    from markdown_utils import sanitize_for_markdown, format_markdown_table

try:
    from .telemetry import (
        telemetry_handler, set_current_stage, reset_current_stage,
        get_turn_telemetry, start_turn_telemetry
    )
    from .token_budget import check_budget_available, record_tokens
except ImportError:
    from telemetry import (
        telemetry_handler, set_current_stage, reset_current_stage,
        get_turn_telemetry, start_turn_telemetry
    )
    from token_budget import check_budget_available, record_tokens

from src.utils.database import DatabricksUtil
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
            elif 'llm_call_count' not in updates:
                is_llm_node = stage_name not in ['metric_resolver', 'sufficiency_check', 'sql_executor', 'sql_validator', 'sql_safety_guard', 'result_validator']
                llm_count = state.get('llm_call_count', 0)
                if is_llm_node:
                    updates['llm_call_count'] = llm_count + 1
                
            return updates
        return wrapper
    return decorator

logger = logging.getLogger("analyst")

# ==========================================
# LLM — reuse the same factory from router_graph
# ==========================================
_base_analyst_llm = get_llm(model_name=cfg.GENERAL_MODEL, temperature=0.0, request_timeout=cfg.LLM_REQUEST_TIMEOUT, max_retries=cfg.LLM_MAX_RETRIES).with_config(callbacks=[telemetry_handler])
_fallback_llm = get_llm(model_name=cfg.FALLBACK_MODEL, temperature=0.0, request_timeout=cfg.LLM_REQUEST_TIMEOUT, max_retries=cfg.LLM_MAX_RETRIES).with_config(callbacks=[telemetry_handler])
_analyst_llm = _base_analyst_llm.with_fallbacks([_fallback_llm])

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
@track_performance("intent_analyzer")
def intent_analyzer(state: AnalystState) -> dict:
    """تحليل نية المستخدم — ايه المقاييس والأبعاد المطلوبة وهل دا سؤال جديد ولا متابعة"""
    question = state["messages"][-1].content
    
    # Get previous context
    messages_content = [f"{m.type}: {m.content}" for m in state["messages"][:-1]]
    chat_history = "\n".join(messages_content[-4:]) if len(messages_content) > 0 else "No previous history."
    
    prompt = _format_prompt(
        prompts.INTENT_ANALYZER_PROMPT, 
        question=question,
        chat_history=chat_history
    )
    response = _analyst_llm.invoke(prompt)
    intent = _parse_json(response.content)

    if not intent:
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
    requested_metrics = intent.get("metrics", [])
    requested_dimensions = intent.get("dimensions", [])

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
@track_performance("sufficiency_check")
def data_sufficiency_check(state: AnalystState) -> dict:
    """لو الداتا مش كافية — نقول للمستخدم بوضوح"""
    sufficiency = state.get("data_sufficiency", {})
    if not sufficiency.get("sufficient", True):
        missing = sufficiency.get("missing", [])
        missing_text = "\n".join(f"- {m['name']}: {m['reason']}" for m in missing)
        lang = state.get("language", "en")
        
        available_list = sufficiency.get('available', [])
        available_str = ", ".join(m for m in available_list)
        
        if lang == "ar":
            msg = (
                f"عذراً، لا يمكن حساب المقاييس المطلوبة لأن البيانات التالية غير متوفرة:\n"
                f"{missing_text}\n\n"
                f"البيانات المتوفرة حالياً يمكنها حساب: {available_str}"
            )
        else:
            msg = (
                f"The requested analysis cannot be completed because the following data is not available:\n"
                f"{missing_text}\n\n"
                f"Available metrics that can be computed: {available_str}"
            )
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
    response = _analyst_llm.invoke(prompt)
    plan = _parse_json(response.content)

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
    response = _analyst_llm.invoke(prompt)
    parsed = _parse_json(response.content)
    
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
        blocks = re.findall(r'```sql\s*(.*?)\s*```', response.content, re.DOTALL | re.IGNORECASE)
        if not blocks:
            # If no blocks, just take the whole thing
            blocks = [response.content.replace("```json", "").replace("```sql", "").replace("```", "").strip()]
            
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
    """تنفيذ الاستعلامات على Databricks"""
    sql_queries = list(state.get("sql_queries", []))
    query_results = []

    queries_to_run = [
        {"sql": q["sql"], "purpose": q["purpose"]}
        for q in sql_queries
        if q.get("validated", False) and not q.get("safety_blocked", False)
    ]

    if not queries_to_run:
        return {"query_results": [], "error": "No validated queries to execute."}

    results = _db.execute_queries(queries_to_run)

    for i, result in enumerate(results):
        if result.get("success"):
            # Find matching query to get limit_injected
            limit_injected = False
            for q in sql_queries:
                if q.get("purpose") == result.get("purpose", ""):
                    limit_injected = q.get("limit_injected", False)
                    break
                    
            import datetime
            import uuid
            
            # Convert rows from list of dicts to tuple of tuples
            cols = result.get("columns", [])
            tuple_rows = tuple(tuple(row.get(col) for col in cols) for row in result.get("rows", []))
            
            query_results.append(Evidence(
                query_id=str(uuid.uuid4()),
                executed_sql=result.get("sql", ""),
                columns=tuple(cols),
                rows=tuple_rows,
                row_count=result.get("row_count", 0),
                truncated=limit_injected and result.get("row_count", 0) == MAX_ROWS,
                executed_at=datetime.datetime.utcnow().isoformat() + "Z"
            ))
        else:
            # Track SQL errors for potential retry
            error_msg = result.get("error", "Unknown execution error")
            # Update the corresponding sql_query with the error
            for j, q in enumerate(sql_queries):
                if q["purpose"] == result.get("purpose", ""):
                    sql_queries[j] = {**q, "error": error_msg, "attempt": q.get("attempt", 0) + 1}
                    break
            logger.warning("SQL execution failed for '%s': %s", result.get("purpose"), error_msg)

    return {"sql_queries": sql_queries, "query_results": query_results}


# ==========================================
# Node 8: SQL Repair (conditional — triggered on errors)
# ==========================================
@track_performance("sql_repair")
def sql_repair(state: AnalystState) -> dict:
    """Repairs failed SQL queries"""
    sql_queries = list(state.get("sql_queries", []))
    schema_context = state.get("schema_context", "")
    repair_attempts = state.get("repair_attempts", 0)

    for i, q in enumerate(sql_queries):
        if q.get("error"):
            prompt = _format_prompt(prompts.SQL_REPAIR_PROMPT, 
                sql_query=q.get("sql", ""),
                errors=q["error"],
                schema_context=schema_context,
            )
            response = _analyst_llm.invoke(prompt)
            fixed_sql = response.content.strip()
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
    response = _analyst_llm.invoke(prompt)
    analysis = _parse_json(response.content)

    if not analysis:
        analysis = {
            "findings": "Analysis completed.",
            "evidence": str(rows[:5]),
            "key_numbers": key_numbers,
            "trend_direction": None,
        }

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
        for r in rows[:5]:
            brand = r.get("restaurant_name", "Unknown")
            delta_val = r.get("delta", r.get("drop_amount", 0))
            contrib = r.get("contribution_to_change", r.get("contribution", 0))
            drivers.append({
                "dimension": "restaurant_name",
                "value": brand,
                "impact": f"{delta_val:,.0f} ({contrib*100:.1f}%)" if isinstance(contrib, (int, float)) else f"{delta_val:,.0f}",
                "direction": "negative" if delta_val < 0 else "positive"
            })
        
        tot_delta = rows[0].get("total_delta", sum(r.get("delta", 0) for r in rows))
        summary = f"Top {len(drivers)} contributors explain a combined decline against a total ecosystem change of {tot_delta:,.0f}."
        return {"analysis_result": {**analysis, "drivers": drivers, "driver_summary": summary}}

    # Fallback to LLM driver analysis if columns are non-standard
    dimensional_results = []
    for r in query_results:
        purpose = ("Query" if isinstance(r, Evidence) else r.get("purpose", "")).lower()
        r_rows = [dict(zip(r.columns, row)) for row in r.rows] if isinstance(r, Evidence) else r.get("rows", [])
        rows_text = _truncate_results(r_rows)
        dimensional_results.append(f"Breakdown: {purpose}\nResults:\n{rows_text}")

    prompt = _format_prompt(prompts.DRIVER_ANALYSIS_PROMPT, 
        main_finding=analysis.get("findings", ""),
        dimensional_results="\n\n---\n\n".join(dimensional_results) if dimensional_results else "No breakdown available.",
    )
    response = _analyst_llm.invoke(prompt)
    drivers = _parse_json(response.content)

    if drivers:
        analysis_with_drivers = {**analysis, "drivers": drivers.get("drivers", []), "driver_summary": drivers.get("summary", "")}
    else:
        analysis_with_drivers = {**analysis, "drivers": [], "driver_summary": "Driver analysis completed."}

    return {"analysis_result": analysis_with_drivers}


# ==========================================
# Node 12: Analysis Completeness Check (Deterministic)
# ==========================================
@track_performance("completeness_check")
def completeness_check(state: AnalystState) -> dict:
    """التحقق من اكتمال التحليل بشكل حتمي لتوفير التوكنز والوقت"""
    validation = state.get("result_validation", {})
    query_results = state.get("query_results", [])
    
    if validation.get("valid", True) and query_results:
        return {"completeness": {"complete": True, "answered_question": "Analysis completed successfully.", "missing_aspects": []}}
    
    return {"completeness": {"complete": False, "answered_question": "Issues encountered during execution.", "missing_aspects": validation.get("issues", [])}}


# ==========================================
# Node 13: Visualization Planner
# ==========================================
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
    is_ranking = (
        intent.get("intent_type") == "ranking"
        or any(w in intent.get("original_question", "").lower() for w in ["top", "اكتر", "أفضل", "أكثر", "highest", "lowest"])
    )
    target_rc = viz_target_result.row_count if isinstance(viz_target_result, Evidence) else viz_target_result.get("row_count", 0) if viz_target_result else 0
    if is_ranking and viz_target_result and target_rc > 1:
        rows = [dict(zip(viz_target_result.columns, row)) for row in viz_target_result.rows] if isinstance(viz_target_result, Evidence) else viz_target_result.get("rows", [])
        cols = list(viz_target_result.columns) if isinstance(viz_target_result, Evidence) else list(viz_target_result.get("columns", []))
        
        # Build clean markdown table with right-aligned numeric columns
        table_md = format_markdown_table(rows, cols, lang=lang)
        intro = f"**Top {len(rows)} Results:**\n\n" if lang == "en" else f"**أفضل {len(rows)} نتائج:**\n\n"
        final_table_answer = sanitize_for_markdown(f"{intro}{table_md}")

        # Automatic visualization spec
        viz_spec = {"should_visualize": False}
        viz_html = None
        if len(cols) >= 2 and len(rows) > 1:
            numeric_cols = [c for c in cols if any(isinstance(r.get(c), (int, float)) for r in rows)]
            text_cols = [c for c in cols if c not in numeric_cols]
            x_col = text_cols[0] if text_cols else cols[0]
            y_col = numeric_cols[0] if numeric_cols else cols[1]
            viz_spec = {
                "should_visualize": True,
                "chart_type": "bar",
                "x_axis": x_col,
                "y_axis": y_col,
                "title": f"Top {len(rows)} by {y_col.replace('_', ' ').title()}"
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
            
    response = _analyst_llm.invoke(prompt)
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
            response2 = _analyst_llm.invoke(retry_prompt)
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

    if lang == "ar":
        msg = "عذراً، لم أتمكن من إكمال التحليل."
        if errors:
            msg += f"\nأخطاء التنفيذ:\n" + "\n".join(f"- {e}" for e in errors)
    else:
        msg = "Sorry, I could not complete the analysis."
        if errors:
            msg += f"\nExecution errors:\n" + "\n".join(f"- {e}" for e in errors)

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

    # Conditional: sufficiency → (continue to planner | end with message)
    graph.add_conditional_edges("sufficiency_check", after_sufficiency, {
        "continue": "analysis_planner",
        "end": END,
    })

    # Linear: planner → sql gen → sql validate → SAFETY GUARD → sql execute
    graph.add_edge("analysis_planner", "sql_generator")
    graph.add_edge("sql_generator", "sql_safety_guard")
    
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
