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

def _write_audit_record(record: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with _audit_lock:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        logger.warning(f"Failed to append to telemetry audit file: {e}")

class TelemetryCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler to capture per-LLM-call telemetry."""
    
    def __init__(self):
        super().__init__()
        self._starts: Dict[str, Dict[str, Any]] = {}
        self._retries_counter: Dict[str, int] = {}
        self._lock = threading.Lock()

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, tags=None, metadata=None, **kwargs):
        with self._lock:
            self._starts[str(run_id)] = {
                "start_time": time.time(),
                "invocation_params": kwargs.get("invocation_params", {}),
                "tags": tags or [],
                "metadata": metadata or {}
            }

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        node = get_current_stage()
        with self._lock:
            self._retries_counter[node] = self._retries_counter.get(node, 0) + 1
        logger.warning(f"[Telemetry] LLM error in node '{node}': {error}")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        end_time = time.time()
        with self._lock:
            start_info = self._starts.pop(str(run_id), {})
            node = get_current_stage()
            retries = self._retries_counter.get(node, 0)
        
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
        if raw_provider:
            provider = raw_provider
        elif "groq" in str(model).lower() or getattr(cfg, "GROQ_API_KEY", None):
            provider = "groq"
        else:
            provider = "openrouter"
            
        fallback_model = getattr(cfg, "FALLBACK_MODEL", "openai/gpt-oss-20b")
        was_fallback = (retries > 0 or model == fallback_model)
        
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
            "wall_time": round(wall_time, 3)
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
