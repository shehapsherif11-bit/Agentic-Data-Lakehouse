"""
telemetry.py — Per-LLM-call telemetry and audit logging for the AI Data Analyst.

Captures:
- node: which LangGraph node executed the LLM call
- provider: 'groq' | 'openrouter' | etc.
- model: model identifier used
- was_fallback: boolean flag indicating if fallback model or retry answered
- retries: number of attempts/retries
- prompt_tokens: tokens in the prompt
- completion_tokens: tokens generated
- reasoning_tokens: tokens used for reasoning / chain-of-thought (if provided)
- wall_time: elapsed time in seconds for the call
- timestamp: UTC ISO timestamp

Writes append-only to `eval/llm_telemetry.jsonl` and keeps turn-level records
for rendering in the UI.
"""
import os
import time
import json
import logging
import threading
from datetime import datetime, timezone
import contextvars
from typing import List, Dict, Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
try:
    from . import router_config as cfg
except ImportError:
    import router_config as cfg

logger = logging.getLogger("analyst.telemetry")

# Context variables for per-request tracking
_current_stage_var = contextvars.ContextVar("current_stage", default="unknown")
_turn_records_var = contextvars.ContextVar("turn_records", default=None)

_audit_lock = threading.Lock()
AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "eval",
    "llm_telemetry.jsonl"
)
SQL_LOG_PATH = os.path.join(os.path.dirname(AUDIT_LOG_PATH), "executed_sql.jsonl")

def set_current_stage(stage_name: str):
    return _current_stage_var.set(stage_name)

def reset_current_stage(token):
    _current_stage_var.reset(token)

def get_current_stage() -> str:
    return _current_stage_var.get()

def start_turn_telemetry() -> List[Dict[str, Any]]:
    records = []
    _turn_records_var.set(records)
    return records

def get_turn_telemetry() -> List[Dict[str, Any]]:
    records = _turn_records_var.get()
    if records is None:
        records = []
        _turn_records_var.set(records)
    return records

def _append_jsonl(path: str, record: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _audit_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.warning(f"Failed to append to {os.path.basename(path)}: {e}")


def _write_audit_record(record: Dict[str, Any]):
    _append_jsonl(AUDIT_LOG_PATH, record)


def log_executed_sql(question: str, sql: str, purpose: str, *, cache_hit: bool, row_count: Optional[int] = None,
                     seconds: Optional[float] = None, error: Optional[str] = None):
    """One line per query the executor ran or served from cache, so empty/wrong results can be audited later."""
    _append_jsonl(SQL_LOG_PATH, {
        "timestamp": datetime.now(timezone.utc).isoformat(), "question": question, "purpose": purpose,
        "sql": sql, "cache_hit": cache_hit, "row_count": row_count, "seconds": seconds, "error": error,
    })

def _expected_model(node: str) -> Optional[str]:
    """The model a node is *configured* to use (None if unknown)."""
    profile = getattr(cfg, "NODE_LLM_PROFILES", {}).get(node)
    if profile:
        return profile[0]
    if node == "router":
        return getattr(cfg, "ROUTER_MODEL", None)
    if node == "general":
        return getattr(cfg, "GENERAL_MODEL", None)
    return None


def _was_fallback(node: str, model: str, provider: str, retries: int) -> bool:
    """True only if a backup actually answered: a retry was needed, a different model than the node's
    configured one responded, or the answer came from the OpenRouter backup. A node deliberately
    configured for the 20b model (e.g. intent_analyzer) is NOT a fallback when it answers."""
    if retries > 0:
        return True
    if "openrouter" in str(provider).lower():
        return True
    expected = _expected_model(node)
    return bool(expected) and model not in ("unknown", expected)


class TelemetryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler to capture per-LLM-call telemetry."""
    
    def __init__(self):
        super().__init__()
        self._starts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        with self._lock:
            self._starts[str(run_id)] = {
                "start_time": time.time(),
                "class_name": str(((serialized or {}).get("id") or [""])[-1]),
                "invocation_params": kwargs.get("invocation_params", {}),
                "tags": tags or [],
                "metadata": metadata or {}
            }

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        with self._lock:
            self._starts.pop(str(run_id), None)  # don't leak start records for failed calls
        logger.warning(f"[Telemetry] LLM error in node '{get_current_stage()}': {error}")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        end_time = time.time()
        with self._lock:
            start_info = self._starts.pop(str(run_id), {})
            node = get_current_stage()
        # CircuitBreakerLLM stamps the attempt number of THIS call into the config metadata, so the
        # count is per call (the old per-stage counter was never reset and inflated across turns).
        retries = int((start_info.get("metadata") or {}).get("retry_count", 0) or 0)
        
        start_time = start_info.get("start_time", end_time)
        wall_time = end_time - start_time
        
        meta = {}
        usage = {}
        if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
            meta.update(response.llm_output)
            
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, 'message', None)
                if msg:
                    msg_meta = getattr(msg, 'response_metadata', None) or {}
                    meta.update(msg_meta)
                    usage = getattr(msg, 'usage_metadata', None) or {}
                    break
                gen_info = getattr(gen, 'generation_info', None) or {}
                if gen_info:
                    meta.update(gen_info)
                    break
        
        token_usage = meta.get("token_usage", {})
        prompt_tokens = usage.get("input_tokens") or token_usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("output_tokens") or token_usage.get("completion_tokens", 0)
        
        details = token_usage.get("completion_tokens_details") or usage.get("output_token_details") or {}
        reasoning_tokens = details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0
        
        model = meta.get("model_name") or meta.get("model") or start_info.get("invocation_params", {}).get("model_name", "unknown")
        
        raw_provider = meta.get("model_provider", "")
        class_name = start_info.get("class_name", "")
        if class_name == "ChatOpenAI":      # OpenRouter is reached through the OpenAI-compatible client
            provider = "openrouter"
        elif class_name == "ChatGroq":
            provider = "groq"
        elif raw_provider:
            provider = raw_provider
        elif "groq" in str(model).lower() or getattr(cfg, "GROQ_API_KEY", None):
            provider = "groq"
        else:
            provider = "openrouter"
            
        was_fallback = _was_fallback(node, model, provider, retries)
        finish_reason = meta.get("finish_reason")
        
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": node,
            "provider": provider,
            "model": model,
            "was_fallback": was_fallback,
            "retries": retries,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "wall_time": round(wall_time, 3),
            "finish_reason": finish_reason,
        }
        
        # Write to JSONL audit log
        _write_audit_record(record)
        
        # Save to current turn records
        turn_recs = get_turn_telemetry()
        turn_recs.append(record)
        
        logger.info(
            f"[Telemetry] node={node} | provider={provider} | model={model} | "
            f"fallback={was_fallback} | retries={retries} | "
            f"tokens={prompt_tokens}->{completion_tokens} (reasoning={reasoning_tokens}) | "
            f"time={wall_time:.3f}s"
        )

# Global shared handler instance
telemetry_handler = TelemetryCallbackHandler()

def format_telemetry_table(records: List[Dict[str, Any]]) -> str:
    """Format a list of telemetry records into a Markdown table."""
    if not records:
        return "No LLM calls recorded for this step."
    
    headers = ["Node", "Provider", "Model", "Fallback", "Retries", "Prompt Tokens", "Completion Tokens", "Reasoning Tokens", "Latency (s)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in records:
        row = [
            str(r.get("node", "-")),
            str(r.get("provider", "-")),
            str(r.get("model", "-")),
            "Yes" if r.get("was_fallback") else "No",
            str(r.get("retries", 0)),
            str(r.get("prompt_tokens", 0)),
            str(r.get("completion_tokens", 0)),
            str(r.get("reasoning_tokens", 0)),
            f"{r.get('wall_time', 0.0):.3f}"
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
