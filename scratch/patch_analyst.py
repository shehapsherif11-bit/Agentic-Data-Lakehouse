import re
import os

file_path = r'd:\Data ENG Project\src\agent\analyst_graph.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update sql_generator
old_sql_gen = """    sql_queries = []
    for step in plan.get("steps", []):
        prompt = _format_prompt(prompts.SQL_GENERATOR_PROMPT, 
            step_description=step.get("description", ""),
            query_purpose=step.get("query_purpose", ""),
            schema_context=schema_context,
            join_warnings=join_warnings,
            metric_definitions=metric_defs_str,
        )
        response = _analyst_llm.invoke(prompt)
        sql = response.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()

        sql_queries.append({
            "sql": sql,
            "purpose": step.get("query_purpose", ""),
            "validated": False,
            "validation_issues": [],
            "attempt": 0,
            "result": None,
            "error": None,
        })

    return {"sql_queries": sql_queries, "current_step": 0, "retry_count": 0}"""

new_sql_gen = """    import json
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
        sql_queries.append({
            "sql": response.content.replace("```json", "").replace("```sql", "").replace("```", "").strip(),
            "purpose": "Main query",
            "validated": False,
            "validation_issues": [],
            "attempt": 0,
            "result": None,
            "error": None,
        })

    return {"sql_queries": sql_queries, "current_step": 0, "retry_count": 0}"""
content = content.replace(old_sql_gen, new_sql_gen)

# 2. Update sql_validator (deterministic)
old_sql_val = """def sql_validator(state: AnalystState) -> dict:
    \"\"\"التحقق من صحة SQL — فحص الجداول والأعمدة والـ joins\"\"\"
    sql_queries = list(state.get("sql_queries", []))
    schema_context = state.get("schema_context", "")

    for i, q in enumerate(sql_queries):
        if q.get("validated"):
            continue

        prompt = _format_prompt(prompts.SQL_VALIDATOR_PROMPT, 
            sql_query=q["sql"],
            schema_context=schema_context,
        )
        response = _analyst_llm.invoke(prompt)
        validation = _parse_json(response.content)

        if validation:
            if validation.get("valid", True):
                sql_queries[i] = {**q, "validated": True, "validation_issues": []}
            else:
                issues = validation.get("issues", [])
                fixed_sql = validation.get("fixed_sql")
                if fixed_sql:
                    sql_queries[i] = {**q, "sql": fixed_sql, "validated": True, "validation_issues": issues}
                else:
                    # Mark valid anyway to let the database engine catch any real errors
                    # and trigger the repair loop, rather than silently dropping the query.
                    sql_queries[i] = {**q, "validated": True, "validation_issues": issues}
        else:
            # If validation parsing fails, proceed optimistically
            sql_queries[i] = {**q, "validated": True, "validation_issues": []}

    return {"sql_queries": sql_queries}"""

new_sql_val = """def sql_validator(state: AnalystState) -> dict:
    \"\"\"Static Python SQL Validator (replaces LLM to save time & rate limits)\"\"\"
    sql_queries = list(state.get("sql_queries", []))
    
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'GRANT', 'REVOKE', 'TRUNCATE', 'ALTER']
    
    for i, q in enumerate(sql_queries):
        if q.get("validated"):
            continue
            
        sql_upper = q["sql"].upper()
        issues = []
        
        # 1. Safety check
        for kw in dangerous_keywords:
            if re.search(r'\\b' + kw + r'\\b', sql_upper):
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

    return {"sql_queries": sql_queries}"""
content = content.replace(old_sql_val, new_sql_val)

# 3. Update Result Validator (merge into Result Analyzer logic)
old_res_val = """def result_validator(state: AnalystState) -> dict:
    \"\"\"التحقق من نتائج الاستعلام — هل النتيجة منطقية؟\"\"\"
    query_results = state.get("query_results", [])

    if not query_results:
        return {"result_validation": {"valid": False, "issues": ["No query results to validate."], "warnings": [], "severity": "critical"}}

    # Build a summary of results for the LLM
    summaries = []
    for r in query_results:
        rows_sample = _truncate_results(r.get("rows", []), max_rows=10)
        summaries.append(f"Query Purpose: {r.get('purpose', 'N/A')}\\nRow Count: {r.get('row_count', 0)}\\nSample:\\n{rows_sample}")

    prompt = _format_prompt(prompts.RESULT_VALIDATOR_PROMPT, 
        results_summary="\\n\\n---\\n\\n".join(summaries)
    )
    response = _analyst_llm.invoke(prompt)
    validation = _parse_json(response.content)

    if not validation:
        validation = {"valid": True, "issues": [], "warnings": [], "severity": "ok"}

    return {"result_validation": validation}"""

new_res_val = """def result_validator(state: AnalystState) -> dict:
    \"\"\"Static Python Result Validator (Replaces LLM to save time)\"\"\"
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
    return {"result_validation": validation}"""
content = content.replace(old_res_val, new_res_val)

# 4. Insight Generator to parse viz as well
old_insight_gen = """def insight_generator(state: AnalystState) -> dict:
    \"\"\"توليد الإجابة النهائية — insight مبني على الأدلة\"\"\"
    intent = state.get("intent", {})
    analysis = state.get("analysis_result", {})
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
            driver_text += "\\n" + "\\n".join(driver_lines)

    lang_instruction = cfg.LANGUAGE_INSTRUCTIONS.get(lang, cfg.LANGUAGE_INSTRUCTIONS.get("en", "Reply in English."))

    prompt = _format_prompt(prompts.INSIGHT_GENERATOR_PROMPT, 
        question=intent.get("original_question", ""),
        findings=json.dumps(analysis, ensure_ascii=False, default=str)[:3000],
        driver_analysis=driver_text or "No driver analysis performed.",
        language_instruction=lang_instruction,
    )
    response = _analyst_llm.invoke(prompt)
    final_answer = response.content.strip()

    return {"final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}"""

new_insight_gen = """def insight_generator(state: AnalystState) -> dict:
    \"\"\"توليد الإجابة النهائية والتخطيط البياني معاً لتوفير الوقت\"\"\"
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
            driver_text += "\\n" + "\\n".join(driver_lines)

    # Main result for Viz
    main_result = query_results[0] if query_results else {}
    rows_sample = _truncate_results(main_result.get("rows", []), max_rows=20)
    results_summary = f"Columns: {main_result.get('columns', [])}\\nRow Count: {main_result.get('row_count', 0)}\\nSample:\\n{rows_sample}"

    lang_instruction = cfg.LANGUAGE_INSTRUCTIONS.get(lang, cfg.LANGUAGE_INSTRUCTIONS.get("en", "Reply in English."))

    prompt = _format_prompt(prompts.INSIGHT_GENERATOR_PROMPT, 
        question=intent.get("original_question", ""),
        findings=json.dumps(analysis, ensure_ascii=False, default=str)[:3000],
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
    if viz_spec.get("should_visualize") and main_result:
        chart_html = generate_chart(viz_spec, main_result.get("rows", []))
        if chart_html:
            viz_html = chart_html

    return {"final_answer": final_answer, "viz_spec": viz_spec, "viz_html": viz_html, "messages": [AIMessage(content=final_answer)]}"""
content = content.replace(old_insight_gen, new_insight_gen)

# Remove viz_planner node definition
viz_planner_start = content.find("def viz_planner(")
if viz_planner_start != -1:
    viz_planner_end = content.find("# ==========================================\n# Node 14:", viz_planner_start)
    if viz_planner_end != -1:
        content = content[:viz_planner_start] + content[viz_planner_end:]

# Update the graph building
content = content.replace('graph.add_node("viz_planner", viz_planner)', '')
content = content.replace('''    # Completeness → viz
    graph.add_conditional_edges("completeness_check", after_completeness, {
        "viz": "viz_planner",
    })

    # Viz → insight
    graph.add_edge("viz_planner", "insight_generator")''', 
    '''    # Completeness → insight
    graph.add_conditional_edges("completeness_check", after_completeness, {
        "viz": "insight_generator",
    })''')

# Reduce MAX_SQL_RETRIES
content = content.replace("MAX_SQL_RETRIES = 3", "MAX_SQL_RETRIES = 2")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied successfully.")
