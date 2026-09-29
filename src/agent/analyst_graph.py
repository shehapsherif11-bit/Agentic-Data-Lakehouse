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

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

try:
    from . import router_config as cfg
    from .analyst_state import AnalystState
    from . import analyst_prompts as prompts
    from . import metrics_registry as metrics
    from .viz_engine import generate_chart
except ImportError:
    import router_config as cfg
    from analyst_state import AnalystState
    import analyst_prompts as prompts
    import metrics_registry as metrics
    from viz_engine import generate_chart

from src.utils.database import DatabricksUtil
from src.agent.sql_safety_guard import check_sql_safety, check_multiple_queries as safety_check_batch

import time
from functools import wraps

def track_performance(stage_name):
    def decorator(func):
        @wraps(func)
        def wrapper(state: AnalystState):
            start = time.time()
            result = func(state)
            latency = time.time() - start
            
            # Count how many LLM calls happened by looking if it's an LLM node
            is_llm_node = stage_name not in ['metric_resolver', 'sufficiency_check', 'sql_executor', 'sql_validator', 'sql_safety_guard', 'result_validator']
            
            updates = result if result else {}
            
            llm_count = state.get('llm_call_count', 0)
            latencies = state.get('stage_latencies', {})
            
            new_latencies = dict(latencies)
            new_latencies[stage_name] = round(latency, 2)
            
            updates['stage_latencies'] = new_latencies
            if is_llm_node:
                updates['llm_call_count'] = llm_count + 1
                
            return updates
        return wrapper
    return decorator

logger = logging.getLogger("analyst")

# ==========================================
# LLM — reuse the same factory from router_graph
# ==========================================
_analyst_llm = ChatGroq(
    model=cfg.GENERAL_MODEL,
    temperature=0.0,
    api_key=cfg.GROQ_API_KEY,
    request_timeout=cfg.LLM_REQUEST_TIMEOUT,
    max_retries=cfg.LLM_MAX_RETRIES,
)

_db = DatabricksUtil()

MAX_SQL_RETRIES = 2
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
        if lang == "ar":
            msg = (
                f"عذراً، لا يمكن حساب المقاييس المطلوبة لأن البيانات التالية غير متوفرة:\n"
                f"{missing_text}\n\n"
                f"البيانات المتوفرة حالياً يمكنها حساب: {', '.join(m['name'] for m in sufficiency.get('available', []))}"
            )
        else:
            msg = (
                f"The requested analysis cannot be completed because the following data is not available:\n"
                f"{missing_text}\n\n"
                f"Available metrics that can be computed: {', '.join(m['name'] for m in sufficiency.get('available', []))}"
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

    return {"sql_queries": sql_queries, "current_step": 0, "retry_count": 0}


# ==========================================
# Node 6: SQL Validator
# ==========================================
@track_performance("sql_validator")
def sql_validator(state: AnalystState) -> dict:
    """Static Python SQL Validator (replaces LLM to save time & rate limits)"""
    sql_queries = list(state.get("sql_queries", []))
    
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'GRANT', 'REVOKE', 'TRUNCATE', 'ALTER']
    
    for i, q in enumerate(sql_queries):
        if q.get("validated"):
            continue
            
        sql_upper = q["sql"].upper()
        issues = []
        
        # 1. Safety check
        for kw in dangerous_keywords:
            if re.search(r'\b' + kw + r'\b', sql_upper):
                issues.append(f"Contains dangerous keyword: {kw}")
                
        # 2. Very basic check (must have SELECT and FROM)
        if "SELECT" not in sql_upper:
            issues.append("Missing SELECT statement")
        if "FROM" not in sql_upper:
            issues.append("Missing FROM statement")
            
        if issues:
            sql_queries[i] = {**q, "validated": False, "validation_issues": issues, "error": "; ".join(issues)}
        else:
            sql_queries[i] = {**q, "validated": True, "validation_issues": []}

    return {"sql_queries": sql_queries}


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
            # Mark as failed with a clear error so the repair loop can see it
            sql_queries[i] = {
                **sql_queries[i],
                "validated": False,
                "error": f"SECURITY BLOCKED: {q['safety_reason']}",
                "safety_blocked": True,
            }
            blocked_count += 1
        else:
            sql_queries[i] = {**sql_queries[i], "safety_blocked": False}
    
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
            query_results.append({
                "purpose": result.get("purpose", ""),
                "columns": result.get("columns", []),
                "rows": result.get("rows", []),
                "row_count": result.get("row_count", 0),
            })
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
    """إصلاح استعلامات SQL الفاشلة"""
    sql_queries = list(state.get("sql_queries", []))
    schema_context = state.get("schema_context", "")
    retry_count = state.get("retry_count", 0)

    repaired = False
    for i, q in enumerate(sql_queries):
        if q.get("error") and q.get("attempt", 0) <= MAX_SQL_RETRIES:
            prompt = _format_prompt(prompts.SQL_REPAIR_PROMPT, 
                sql_query=q["sql"],
                errors=q["error"],
                schema_context=schema_context,
            )
            response = _analyst_llm.invoke(prompt)
            fixed_sql = response.content.strip()
            fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()

            sql_queries[i] = {
                **q,
                "sql": fixed_sql,
                "validated": False,  # Will need re-validation
                "error": None,
            }
            repaired = True
            logger.info("Repaired SQL for step '%s' (attempt %d)", q["purpose"], q.get("attempt", 0))

    return {"sql_queries": sql_queries, "retry_count": retry_count + 1}


# ==========================================
# Node 9: Result Validator
# ==========================================
@track_performance("result_validator")
def result_validator(state: AnalystState) -> dict:
    """Static Python Result Validator (Replaces LLM to save time)"""
    query_results = state.get("query_results", [])

    if not query_results:
        return {"result_validation": {"valid": False, "issues": ["No query results to validate."], "warnings": [], "severity": "critical"}}

    issues = []
    warnings = []
    for r in query_results:
        if r.get("row_count", 0) == 0:
            warnings.append(f"Query '{r.get('purpose')}' returned 0 rows.")
            
    severity = "warning" if warnings else "ok"
    if issues:
        severity = "critical"

    validation = {"valid": len(issues) == 0, "issues": issues, "warnings": warnings, "severity": severity}
    return {"result_validation": validation}


# ==========================================
# Node 10: Result Analyzer
# ==========================================
@track_performance("result_analyzer")
def result_analyzer(state: AnalystState) -> dict:
    """تحليل النتائج — استخراج النتائج والأرقام الرئيسية"""
    query_results = state.get("query_results", [])
    plan = state.get("analysis_plan", {})

    results_context = []
    for r in query_results:
        rows_text = _truncate_results(r.get("rows", []))
        results_context.append(f"Query Purpose: {r.get('purpose', 'N/A')}\nResults:\n{rows_text}")

    prompt = _format_prompt(prompts.RESULT_ANALYZER_PROMPT, 
        results_context="\n\n---\n\n".join(results_context),
        goal=plan.get("goal", "Answer the user's question"),
    )
    response = _analyst_llm.invoke(prompt)
    analysis = _parse_json(response.content)

    if not analysis:
        analysis = {
            "findings": "Analysis completed but structured parsing failed.",
            "evidence": str(query_results[0].get("rows", [])[:5]) if query_results else "No data",
            "key_numbers": {},
            "trend_direction": None,
        }

    return {"analysis_result": analysis}


# ==========================================
# Node 11: Driver / Root-Cause Analysis (conditional)
# ==========================================
@track_performance("driver_analysis")
def driver_analysis(state: AnalystState) -> dict:
    """تحليل الأسباب — ليه المقياس تغير؟"""
    analysis = state.get("analysis_result", {})
    query_results = state.get("query_results", [])

    # The driver queries should be among the query_results (planned by the analysis_planner)
    dimensional_results = []
    for r in query_results:
        purpose = r.get("purpose", "").lower()
        if any(kw in purpose for kw in ["breakdown", "driver", "dimension", "by "]):
            rows_text = _truncate_results(r.get("rows", []))
            dimensional_results.append(f"Breakdown: {r.get('purpose')}\nResults:\n{rows_text}")

    if not dimensional_results:
        # Use all results as dimensional data
        for r in query_results[1:]:  # Skip the first (main metric) result
            rows_text = _truncate_results(r.get("rows", []))
            dimensional_results.append(f"Breakdown: {r.get('purpose')}\nResults:\n{rows_text}")

    prompt = _format_prompt(prompts.DRIVER_ANALYSIS_PROMPT, 
        main_finding=analysis.get("findings", ""),
        dimensional_results="\n\n---\n\n".join(dimensional_results) if dimensional_results else "No dimensional breakdown available.",
    )
    response = _analyst_llm.invoke(prompt)
    drivers = _parse_json(response.content)

    if drivers:
        analysis_with_drivers = {**analysis, "drivers": drivers.get("drivers", []), "driver_summary": drivers.get("summary", "")}
    else:
        analysis_with_drivers = {**analysis, "drivers": [], "driver_summary": "Driver analysis could not be completed."}

    return {"analysis_result": analysis_with_drivers}


# ==========================================
# Node 12: Analysis Completeness Check
# ==========================================
@track_performance("completeness_check")
def completeness_check(state: AnalystState) -> dict:
    """التحقق من اكتمال التحليل — هل الإجابة كاملة؟"""
    intent = state.get("intent", {})
    analysis = state.get("analysis_result", {})

    analysis_summary = json.dumps(analysis, ensure_ascii=False, default=str)

    prompt = _format_prompt(prompts.COMPLETENESS_CHECK_PROMPT, 
        question=intent.get("original_question", ""),
        analysis_summary=analysis_summary[:3000],  # Truncate to avoid context overflow
    )
    response = _analyst_llm.invoke(prompt)
    completeness = _parse_json(response.content)

    if not completeness:
        completeness = {"complete": True, "answered_question": "Analysis completed.", "missing_aspects": []}

    return {"completeness": completeness}


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
                "purpose": r.get("purpose", ""),
                "row_count": r.get("row_count", 0),
                "columns": r.get("columns", []),
                # Pass the exact output, up to 50 rows. The LLM is banned from aggregating this.
                "final_sql_output": r.get("rows", [])[:50] 
            } for r in query_results
        ],
        "findings": analysis.get("findings", ""),
        "key_numbers": analysis.get("key_numbers", {}),
    }

    # Find the most appropriate result for visualization (usually the first one with multiple rows)
    viz_target_result = None
    for r in query_results:
        if r.get("row_count", 0) > 1:
            viz_target_result = r
            break
    if not viz_target_result and query_results:
        viz_target_result = query_results[0]
        
    results_summary = "No results available for visualization."
    if viz_target_result:
        rows_sample = _truncate_results(viz_target_result.get("rows", []), max_rows=20)
        results_summary = f"Columns: {viz_target_result.get('columns', [])}\nRow Count: {viz_target_result.get('row_count', 0)}\nSample:\n{rows_sample}"

    lang_instruction = cfg.LANGUAGE_INSTRUCTIONS.get(lang, cfg.LANGUAGE_INSTRUCTIONS.get("en", "Reply in English."))

    prompt = _format_prompt(prompts.INSIGHT_GENERATOR_PROMPT, 
        question=intent.get("original_question", ""),
        evidence_object=json.dumps(evidence_object, ensure_ascii=False, default=str)[:4000],
        driver_analysis=driver_text or "No driver analysis performed.",
        results_summary=results_summary,
        language_instruction=lang_instruction,
    )
    response = _analyst_llm.invoke(prompt)
    
    parsed = _parse_json(response.content)
    
    if parsed and "insight" in parsed:
        final_answer = parsed["insight"]
        viz_spec = parsed.get("viz_spec", {"should_visualize": False})
    else:
        # Fallback if parsing fails
        final_answer = response.content.replace("```json", "").replace("```", "").strip()
        viz_spec = {"should_visualize": False}

    # Generate the chart if needed
    viz_html = ""
    if viz_spec.get("should_visualize") and viz_target_result:
        chart_html = generate_chart(viz_spec, viz_target_result.get("rows", []))
        if chart_html:
            viz_html = chart_html

    return {"evidence_object": evidence_object, "final_answer": final_answer, "viz_spec": viz_spec, "viz_html": viz_html, "messages": [AIMessage(content=final_answer)]}


# ==========================================
# Routing functions (conditional edges)
# ==========================================
def after_sufficiency(state: AnalystState) -> str:
    """بعد فحص كفاية البيانات — نكمل ولا نوقف"""
    if state.get("error") == "data_insufficient":
        return "end"
    return "continue"


def after_execution(state: AnalystState) -> str:
    """بعد التنفيذ — نتحقق لو فيه أخطاء محتاجة إصلاح"""
    sql_queries = state.get("sql_queries", [])
    retry_count = state.get("retry_count", 0)

    # Check if any queries failed and can be retried
    has_errors = any(q.get("error") and q.get("attempt", 0) <= MAX_SQL_RETRIES for q in sql_queries)
    if has_errors and retry_count < MAX_SQL_RETRIES:
        return "repair"

    # Check if we got any results at all
    query_results = state.get("query_results", [])
    if not query_results:
        return "end_with_error"

    return "validate"


def after_result_validation(state: AnalystState) -> str:
    """بعد فحص النتائج — لو critical نوقف"""
    validation = state.get("result_validation", {})
    if validation.get("severity") == "critical":
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
    graph.add_node("sql_validator", sql_validator)
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

    # Conditional: after intent -> if CHANGE_VISUALIZATION -> skip to insight_generator
    def after_intent(state: AnalystState) -> str:
        if state.get("is_followup") and state.get("followup_type") == "CHANGE_VISUALIZATION":
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
    graph.add_edge("sql_generator", "sql_validator")
    graph.add_edge("sql_validator", "sql_safety_guard")
    graph.add_edge("sql_safety_guard", "sql_executor")

    # Conditional: after execution → (repair | validate results | error end)
    graph.add_conditional_edges("sql_executor", after_execution, {
        "repair": "sql_repair",
        "validate": "result_validator",
        "end_with_error": "error_end",
    })

    # Repair loop → back to validator
    graph.add_edge("sql_repair", "sql_validator")

    # Conditional: after result validation → (analyze | error end)
    graph.add_conditional_edges("result_validator", after_result_validation, {
        "analyze": "result_analyzer",
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
