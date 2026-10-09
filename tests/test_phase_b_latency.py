import asyncio
import threading
import time

import pytest
from langchain_core.runnables import RunnableLambda

from src.agent import llm_cache, llm_factory
from src.agent.concurrency import run_parallel
from src.agent.llm_factory import CircuitBreakerLLM, CircuitBreakerOpenException, _BreakerState


# ---------- CircuitBreakerLLM (Steps 4, 8) ----------

class Flaky:
    """Fails `fail_times` with the given message, then succeeds."""
    def __init__(self, fail_times, msg="429 rate limit"):
        self.fail_times, self.msg, self.calls = fail_times, msg, 0

    def invoke(self, input, config=None, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise Exception(self.msg)
        return "ok"

    async def ainvoke(self, input, config=None, **kw):
        return self.invoke(input, config, **kw)


def _breaker(llm, **kw):
    kw.setdefault("initial_backoff", 0.01)
    kw.setdefault("max_backoff", 0.02)
    return CircuitBreakerLLM(llm, **kw)


def test_retries_transient_then_succeeds():
    llm = Flaky(2)
    assert _breaker(llm, max_retries=3).invoke("x") == "ok"
    assert llm.calls == 3


def test_total_deadline_stops_retrying_early():
    llm = Flaky(99)
    b = _breaker(llm, max_retries=10, initial_backoff=0.5, max_backoff=0.5, max_total_seconds=0.6)
    start = time.time()
    with pytest.raises(Exception, match="429"):
        b.invoke("x")
    assert time.time() - start < 1.5
    assert llm.calls < 10


def test_breaker_state_shared_with_derived_runnables():
    state = _BreakerState()
    b = _breaker(Flaky(99, "503 service unavailable"), max_retries=0, breaker_threshold=2, state=state)
    child = b._derive(Flaky(99, "503 service unavailable"))
    for _ in range(2):
        with pytest.raises(Exception):
            child.invoke("x")
    assert state.failure_count == 2
    with pytest.raises(CircuitBreakerOpenException):
        b.invoke("x")  # parent sees the child's failures


def test_request_errors_do_not_trip_breaker():
    state = _BreakerState()
    b = _breaker(Flaky(99, "400 bad request: invalid schema"), max_retries=0, breaker_threshold=2, state=state)
    for _ in range(4):
        with pytest.raises(Exception):
            b.invoke("x")
    assert state.failure_count == 0


def test_ainvoke_is_native_and_does_not_block_the_loop():
    llm = Flaky(1)
    b = _breaker(llm, max_retries=2, initial_backoff=0.2, max_backoff=0.2)

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        t = asyncio.create_task(ticker())
        result = await b.ainvoke("x")
        t.cancel()
        return result, ticks

    result, ticks = asyncio.run(main())
    assert result == "ok"
    assert ticks >= 3  # loop kept running during the 0.2s backoff (time.sleep would give ~0)


# ---------- llm_cache (Step 9) ----------

@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(llm_cache.cfg, "LLM_CACHE_ENABLED", True)
    llm_cache._memory.clear()
    yield llm_cache
    llm_cache._memory.clear()


def test_cache_hit_skips_compute(cache):
    calls = []
    compute = lambda: calls.append(1) or "answer"
    assert cache.get_or_compute("n", "m", "prompt", compute) == ("answer", False)
    assert cache.get_or_compute("n", "m", "prompt", compute) == ("answer", True)
    assert len(calls) == 1
    cache._memory.clear()  # force the disk layer
    assert cache.get_or_compute("n", "m", "prompt", compute) == ("answer", True)
    assert len(calls) == 1


def test_different_prompt_or_model_misses(cache):
    cache.get_or_compute("n", "m", "p1", lambda: "a")
    assert cache.get_or_compute("n", "m", "p2", lambda: "b") == ("b", False)
    assert cache.get_or_compute("n", "other-model", "p1", lambda: "c") == ("c", False)


def test_invalid_results_are_not_cached(cache):
    calls = []
    compute = lambda: calls.append(1) or ""
    cache.get_or_compute("n", "m", "p", compute, is_valid=bool)
    cache.get_or_compute("n", "m", "p", compute, is_valid=bool)
    assert len(calls) == 2


def test_ttl_expiry(cache, monkeypatch):
    cache.get_or_compute("n", "m", "p", lambda: "old")
    monkeypatch.setattr(llm_cache.cfg, "LLM_CACHE_TTL_SECONDS", -1)
    assert cache.get_or_compute("n", "m", "p", lambda: "new") == ("new", False)


def test_identical_concurrent_requests_are_coalesced(cache):
    calls = []

    def slow():
        calls.append(1)
        time.sleep(0.3)
        return "shared"

    out = run_parallel([lambda: cache.get_or_compute("n", "m", "same", slow) for _ in range(4)], max_workers=4)
    assert len(calls) == 1
    assert all(v == "shared" for v, _ in out)


def test_disabled_cache_always_computes(cache, monkeypatch):
    monkeypatch.setattr(llm_cache.cfg, "LLM_CACHE_ENABLED", False)
    calls = []
    for _ in range(2):
        cache.get_or_compute("n", "m", "p", lambda: calls.append(1) or "x")
    assert len(calls) == 2


# ---------- run_parallel (Steps 6, 7) ----------

def test_run_parallel_is_concurrent_and_ordered():
    start = time.time()
    out = run_parallel([lambda i=i: (time.sleep(0.3), i)[1] for i in range(4)], max_workers=4)
    assert out == [0, 1, 2, 3]
    assert time.time() - start < 0.8  # serial would be 1.2s


def test_run_parallel_propagates_first_error_after_all_finish():
    done = []

    def boom():
        raise ValueError("bad")

    def ok():
        time.sleep(0.1)
        done.append(1)

    with pytest.raises(ValueError):
        run_parallel([boom, ok])
    assert done == [1]


def test_run_parallel_propagates_contextvars():
    import contextvars
    var = contextvars.ContextVar("v", default="unset")
    var.set("parent")
    assert run_parallel([var.get, var.get]) == ["parent", "parent"]


# ---------- sql_repair fan-out + analyst prefetch (Steps 6, 7) ----------

def test_sql_repair_repairs_failed_queries_concurrently(monkeypatch):
    import src.agent.analyst_graph as ag

    class SlowLLM:
        def invoke(self, prompt, *a, **k):
            time.sleep(0.3)
            return type("R", (), {"content": "```sql\nSELECT 1\n```"})()

    monkeypatch.setattr(ag, "_analyst_llm", SlowLLM())
    state = {"sql_queries": [{"sql": f"bad{i}", "purpose": f"p{i}", "error": "boom"} for i in range(3)] +
                            [{"sql": "fine", "purpose": "ok", "error": None}],
             "schema_context": "", "repair_attempts": 0}
    start = time.time()
    out = ag.sql_repair(state)
    assert time.time() - start < 0.8
    assert [q["sql"] for q in out["sql_queries"]] == ["SELECT 1"] * 3 + ["fine"]
    assert all(q["error"] is None for q in out["sql_queries"])
    assert out["repair_attempts"] == 1


def test_intent_analyzer_uses_prefetched_intent_without_llm_call(monkeypatch):
    import src.agent.analyst_graph as ag
    from langchain_core.messages import HumanMessage

    class Boom:
        def invoke(self, *a, **k):
            raise AssertionError("LLM must not be called when intent is prefetched")

    monkeypatch.setattr(ag, "_analyst_llm", Boom())
    intent = {"intent_type": "ranking", "metrics": ["revenue"], "original_question": "top cities", "is_followup": False}
    out = ag.intent_analyzer({"messages": [HumanMessage(content="top cities")], "prefetched_intent": intent})
    assert out["intent"]["intent_type"] == "ranking"


def test_intent_prefetch_ignored_when_question_differs(monkeypatch):
    import src.agent.analyst_graph as ag
    from langchain_core.messages import HumanMessage

    class Fake:
        def invoke(self, *a, **k):
            return type("R", (), {"content": '{"intent_type": "simple_query", "metrics": ["aov"]}'})()

    monkeypatch.setattr(ag, "_analyst_llm", Fake())
    stale = {"intent_type": "ranking", "original_question": "old question"}
    out = ag.intent_analyzer({"messages": [HumanMessage(content="new question")], "prefetched_intent": stale})
    assert out["intent"]["metrics"] == ["aov"]


def test_router_runs_classification_and_intent_prefetch_in_parallel(monkeypatch):
    import src.agent.router_graph as rg
    from langchain_core.messages import HumanMessage

    class Structured:
        def invoke(self, messages, config=None):
            time.sleep(0.4)
            return rg.RouteDecision(agent="ANALYSIS", confidence=0.9, reasoning="r", language="en")

    class RouterLLM:  # non-Runnable test double -> cache bypassed, direct call
        def with_structured_output(self, _):
            return Structured()

    def slow_intent(messages):
        time.sleep(0.4)
        return {"intent_type": "ranking", "original_question": messages[-1].content}

    monkeypatch.setattr(rg, "router_llm", RouterLLM())
    monkeypatch.setattr(rg, "_compute_intent", slow_intent)
    monkeypatch.setattr(rg, "analyst_agent", object())

    start = time.time()
    out = rg.router_node({"messages": [HumanMessage(content="top cities?")]})
    assert time.time() - start < 0.7  # serial would be ~0.8s
    assert out["route"] == "ANALYSIS"
    assert out["prefetched_intent"]["original_question"] == "top cities?"


def test_router_discards_prefetched_intent_for_non_analysis_route(monkeypatch):
    import src.agent.router_graph as rg
    from langchain_core.messages import HumanMessage

    class Structured:
        def invoke(self, messages, config=None):
            return rg.RouteDecision(agent="GENERAL", confidence=0.9, reasoning="r", language="en")

    class RouterLLM:
        def with_structured_output(self, _):
            return Structured()

    monkeypatch.setattr(rg, "router_llm", RouterLLM())
    monkeypatch.setattr(rg, "_compute_intent", lambda m: {"original_question": "hi"})
    monkeypatch.setattr(rg, "analyst_agent", object())
    out = rg.router_node({"messages": [HumanMessage(content="hi")]})
    assert out["route"] == "GENERAL" and out["prefetched_intent"] == {}


# ---------- telemetry (Step 14) ----------

def test_configured_20b_node_is_not_flagged_as_fallback():
    from src.agent import telemetry, router_config as cfg
    node, (model, _effort) = "intent_analyzer", cfg.NODE_LLM_PROFILES["intent_analyzer"]
    assert telemetry._was_fallback(node, model, "groq", 0) is False


def test_fallback_flagged_when_other_model_or_provider_or_retry():
    from src.agent import telemetry, router_config as cfg
    planner_model = cfg.NODE_LLM_PROFILES["analysis_planner"][0]
    assert telemetry._was_fallback("analysis_planner", planner_model, "groq", 0) is False
    assert telemetry._was_fallback("analysis_planner", cfg.FALLBACK_MODEL, "groq", 0) is True   # cross-model fallback answered
    assert telemetry._was_fallback("analysis_planner", planner_model, "openrouter", 0) is True  # OpenRouter answered
    assert telemetry._was_fallback("analysis_planner", planner_model, "groq", 2) is True        # needed retries


def test_retry_count_is_per_call_not_accumulated_across_turns():
    from langchain_core.outputs import LLMResult, ChatGeneration
    from langchain_core.messages import AIMessage
    from src.agent import telemetry
    h = telemetry.TelemetryCallbackHandler()
    telemetry.start_turn_telemetry()
    tok = telemetry.set_current_stage("sql_generator")
    try:
        for i, retries in enumerate([2, 0]):
            run_id = f"run{i}"
            h.on_llm_start({"id": ["x", "ChatGroq"]}, ["p"], run_id=run_id, metadata={"retry_count": retries})
            h.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="x"))]]), run_id=run_id)
    finally:
        telemetry.reset_current_stage(tok)
    recs = telemetry.get_turn_telemetry()
    assert [r["retries"] for r in recs] == [2, 0]  # second call must not inherit the first call's retries
