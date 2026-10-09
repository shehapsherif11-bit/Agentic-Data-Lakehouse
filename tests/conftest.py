"""
Every test runs against throw-away caches. Without this, a test that mocks the warehouse with fake rows writes
them into the REAL query cache under a realistic SQL key, and the app then serves those fake rows to users
for up to an hour (this happened: "اي اقل مطعم محقق ارباح" answered with test rows "A — 10").
"""
from collections import OrderedDict

import pytest

from src.agent import llm_cache, query_cache, telemetry


@pytest.fixture(autouse=True)
def _isolated_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(query_cache, "CACHE_DIR", str(tmp_path / "query_cache"))
    monkeypatch.setattr(llm_cache, "CACHE_DIR", str(tmp_path / "llm_cache"))
    monkeypatch.setattr(llm_cache, "_memory", OrderedDict())
    # Tests must never append to the real eval/ logs: fake LLM/SQL rows skew every token and latency analysis.
    monkeypatch.setattr(telemetry, "AUDIT_LOG_PATH", str(tmp_path / "llm_telemetry.jsonl"))
    monkeypatch.setattr(telemetry, "SQL_LOG_PATH", str(tmp_path / "executed_sql.jsonl"))
    query_cache.clear_memory()
    yield
    query_cache.clear_memory()
