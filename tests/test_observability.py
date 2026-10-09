"""P0-1: tests must not touch the real eval/ logs, executed SQL is auditable, and the LLM call count is real."""
import json
import os
from unittest.mock import MagicMock

from src.agent import telemetry
from src.agent import analyst_graph as ag


def test_tests_never_write_to_real_eval_logs():
    real = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")
    assert not os.path.abspath(telemetry.AUDIT_LOG_PATH).startswith(real)
    assert not os.path.abspath(telemetry.SQL_LOG_PATH).startswith(real)


def test_log_executed_sql_writes_one_json_line():
    telemetry.log_executed_sql("q?", "SELECT 1", "p", cache_hit=False, row_count=0, seconds=1.5)
    with open(telemetry.SQL_LOG_PATH, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    assert rec["sql"] == "SELECT 1" and rec["row_count"] == 0 and rec["cache_hit"] is False and rec["question"] == "q?"


def test_executor_logs_miss_then_hit(monkeypatch):
    db = MagicMock()
    db.execute_queries.return_value = [{"success": True, "columns": ["x"], "rows": [{"x": 1}], "row_count": 1,
                                        "sql": "SELECT 1 AS x", "purpose": "t"}]
    monkeypatch.setattr(ag, "_db", db)
    state = {"intent": {"original_question": "q?"},
             "sql_queries": [{"sql": "SELECT 1 AS x", "purpose": "t", "validated": True}]}
    ag.sql_executor(state)
    ag.sql_executor(state)
    assert db.execute_queries.call_count == 1  # second run served from the query cache
    with open(telemetry.SQL_LOG_PATH, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f]
    assert [r["cache_hit"] for r in recs] == [False, True]


def test_call_count_is_zero_without_llm_calls():
    """A deterministic/cached node must not invent an LLM call (the UI used to show '5 LLM Calls' for 0)."""
    telemetry.start_turn_telemetry()
    wrapped = ag.track_performance("result_analyzer")(lambda s: {})
    out = wrapped({"llm_call_count": 0})
    assert out["llm_call_count"] == 0


def test_call_count_counts_telemetry_records():
    recs = telemetry.start_turn_telemetry()
    recs.append({"node": "x"}); recs.append({"node": "y"})
    wrapped = ag.track_performance("result_analyzer")(lambda s: {})
    assert wrapped({})["llm_call_count"] == 2
