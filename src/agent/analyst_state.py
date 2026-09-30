from typing import Annotated, Any, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Evidence:
    query_id: str
    executed_sql: str
    columns: Tuple[str, ...]
    rows: Tuple[Tuple, ...]
    row_count: int
    truncated: bool
    executed_at: str

from langgraph.graph.message import add_messages

class AnalystState(TypedDict):
    # Conversation
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    
    # Conversational Follow-up Tracking
    is_followup: bool
    followup_type: str  # NEW_QUESTION, MODIFY_ANALYSIS, CHANGE_VIZ, etc.
    previous_question: str
    previous_plan: dict
    previous_results: list
    
    # Intent Analysis
    intent: dict  # {type, metrics, dimensions, filters, time_period, granularity, comparisons, is_followup, is_driver_question, original_question}
    
    # Metric Resolution 
    resolved_metrics: list  # [{name, status, definition, missing_reason}]
    data_sufficiency: dict  # {sufficient, available, missing, warnings}
    
    # Analysis Plan
    analysis_plan: dict  # {goal, steps: [{description, query_purpose}], needs_driver_analysis, needs_temporal_comparison}
    
    # Schema + Grain
    schema_context: str
    join_warnings: str
    
    # SQL Pipeline (multi-query support)
    sql_queries: list  # [{sql, purpose, validated, validation_issues, attempt, result, error}]
    current_step: int
    
    # Results
    query_results: list  # [{purpose, columns, rows, row_count}]
    result_validation: dict  # {valid, issues, warnings}
    
    # Analysis
    analysis_result: dict  # {findings, evidence, key_numbers, drivers}
    completeness: dict  # {complete, answered_question, missing_aspects, recommendation}
    
    # Visualization
    viz_spec: dict  # {should_visualize, chart_type, x_axis, y_axis, title, reason}
    viz_html: str
    
    # Output
    evidence_object: dict
    final_answer: str
    
    # Control
    repair_attempts: int
    error: str
    
    # Performance Tracking
    llm_call_count: int
    stage_latencies: dict  # {stage_name: latency_seconds}
    llm_telemetry: list    # [{node, provider, model, was_fallback, retries, prompt_tokens, completion_tokens, reasoning_tokens, wall_time, timestamp}]
