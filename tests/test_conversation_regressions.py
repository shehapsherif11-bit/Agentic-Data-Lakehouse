"""
Regression tests for the failure transcript:
  * "top 6 restaurants" returned 5 rows (hardcoded LIMIT 5)
  * "which restaurant is best? one name" re-ran the ranking / produced a generic essay with no memory
  * NULL revenue + 0 orders narrated as 'no data' without explaining NULL vs 0 vs missing months
  * "customer retention" refused with an EMPTY list of available metrics
  * a failed / empty turn wiped the stored result the user was still referring to
Every displayed number is asserted against the mocked database rows it must come from.
"""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.agent.analyst_graph as ag
import src.agent.router_graph as rg
from src.agent.advisor import build_advice
from src.agent.analyst_state import Evidence
from src.agent.query_intent import (
    detect_language, extract_top_n, is_advisory_question, ranking_direction,
)
from src.agent.result_classifier import classify_rows

TOP5 = [("KFC", 42466314.0), ("Domino's Pizza", 35671213.0), ("Pizza Hut", 32128653.0),
        ("Behrouz Biryani", 25922775.0), ("Subway", 25483225.0)]
TOP6 = TOP5 + [("Burger King", 21000000.0)]


class FakeLLM:
    def __init__(self, responses):
        self.responses, self.calls, self.prompts = list(responses), 0, []

    def invoke(self, prompt, *a, **k):
        self.prompts.append(prompt)
        res = self.responses[self.calls] if self.calls < len(self.responses) else "{}"
        self.calls += 1
        return type("R", (), {"content": res})()


class MockDB:
    """Records every SQL it sees and returns canned rows; the data-coverage query gets its own answer."""
    def __init__(self, rows, columns, coverage=("2022-01-01", "2022-12-31")):
        self.rows, self.columns, self.coverage, self.sqls = rows, columns, coverage, []

    def execute_queries(self, queries):
        out = []
        for q in queries:
            self.sqls.append(q["sql"])
            if "MIN(dd.full_date)" in q["sql"]:
                rows = [{"min_date": self.coverage[0], "max_date": self.coverage[1]}]
                out.append({"purpose": q.get("purpose"), "success": True, "rows": rows, "row_count": 1,
                            "columns": ["min_date", "max_date"], "sql": q["sql"]})
            else:
                out.append({"purpose": q.get("purpose"), "success": True, "rows": self.rows,
                            "row_count": len(self.rows), "columns": self.columns, "sql": q["sql"]})
        return out


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    ag._coverage_cache.clear()
    from src.agent import query_cache
    monkeypatch.setattr(query_cache, "lookup_or_claim", lambda sql: (None, False), raising=True)
    monkeypatch.setattr(query_cache, "set_cached_result", lambda **kw: None, raising=True)


def run_analyst(monkeypatch, question, intent_json, db, lang="en", extra_llm=()):
    llm = FakeLLM([intent_json, *extra_llm])
    monkeypatch.setattr(ag, "_analyst_llm", llm)
    monkeypatch.setattr(ag, "_db", db)
    graph = ag.build_analyst_graph()
    return graph.invoke({"messages": [HumanMessage(content=question)], "language": lang,
                         "repair_attempts": 0, "current_step": 0}), llm


RANKING_INTENT = ('{"is_followup": false, "intent_type": "ranking", "metrics": ["revenue"], '
                  '"dimensions": ["restaurant_name"], "filters": [], "time_period": null, "is_driver_question": false}')


# ------------------------------------------------------------------ deterministic parsing
@pytest.mark.parametrize("text,expected", [
    ("i want top 6 resturants in revenue", 6),
    ("top six restaurants", 6),
    ("show me the 7 best restaurants", 7),
    ("أفضل 6 مطاعم من حيث الإيراد", 6),
    ("اعلى ٦ مطاعم", 6),
    ("عايز اعلى ستة مطاعم", 6),
    ("give me one name", 1),
    ("اديني اسم واحد", 1),
    ("show revenue by city", None),
    ("top restaurants", None),
    ("top 100000 restaurants", None),
])
def test_extract_top_n(text, expected):
    assert extract_top_n(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("top 5 restaurants", "DESC"),
    ("bottom 3 restaurants by revenue", "ASC"),
    ("lowest revenue cities", "ASC"),
    ("اقل 3 مطاعم في الايراد", "ASC"),
    ("اعلى 3 مطاعم", "DESC"),
])
def test_ranking_direction(text, expected):
    assert ranking_direction(text) == expected


@pytest.mark.parametrize("text", [
    "من وجهه نظرك انت اي احسن واحد من المطاعم ال عندك اديني اسم واحد",
    "اي احسن مطعم استثمر فيه",
    "In your opinion, which restaurant is best? Give me one name.",
    "should I invest in KFC?",
])
def test_advisory_detected(text):
    assert is_advisory_question(text)


@pytest.mark.parametrize("text", [
    "i want top 6 resturants in revenue",
    "أفضل 5 مطاعم",
    "What is the total revenue and number of orders for the last month?",
    "Which year was that for?",
])
def test_plain_data_questions_are_not_advisory(text):
    assert not is_advisory_question(text)


def test_detect_language():
    assert detect_language("اي احسن مطعم") == "ar"
    assert detect_language("top restaurants") == "en"
    assert detect_language("عايز top 6 restaurants by revenue") == "mixed"


# ------------------------------------------------------------------ top-N end to end
def test_top_6_returns_six_rows_and_exact_limit(monkeypatch):
    rows = [{"restaurant_name": n, "revenue": v} for n, v in TOP6]
    db = MockDB(rows, ["restaurant_name", "revenue"])
    result, _ = run_analyst(monkeypatch, "i want top 6 resturants in revenue", RANKING_INTENT, db)

    main_sql = db.sqls[0]
    assert "LIMIT 6" in main_sql and "LIMIT 5" not in main_sql
    assert "ORDER BY revenue DESC" in main_sql
    answer = result["final_answer"]
    assert "Top 6" in answer
    for name, value in TOP6:                       # every displayed number comes from the DB rows
        assert name.replace("'", "'") in answer
        assert f"{value:,.0f}" in answer
    assert answer.count("\n|") >= 8                # header + separator + 6 rows


def test_bottom_n_uses_ascending_order(monkeypatch):
    rows = [{"restaurant_name": n, "revenue": v} for n, v in TOP5[::-1][:3]]
    db = MockDB(rows, ["restaurant_name", "revenue"])
    result, _ = run_analyst(monkeypatch, "bottom 3 restaurants by revenue",
                            RANKING_INTENT, db)
    assert "ORDER BY revenue ASC" in db.sqls[0] and "LIMIT 3" in db.sqls[0]
    assert "Bottom 3" in result["final_answer"]


def test_fewer_rows_than_requested_is_disclosed(monkeypatch):
    rows = [{"restaurant_name": n, "revenue": v} for n, v in TOP5[:3]]
    db = MockDB(rows, ["restaurant_name", "revenue"])
    result, _ = run_analyst(monkeypatch, "top 6 restaurants by revenue", RANKING_INTENT, db)
    assert "You asked for 6, but only 3" in result["final_answer"]


def test_follow_up_with_a_different_n_is_not_reused(monkeypatch):
    """Contradictory context: the previous turn asked for 5; the new question asks for 3."""
    rows = [{"restaurant_name": n, "revenue": v} for n, v in TOP5[:3]]
    db = MockDB(rows, ["restaurant_name", "revenue"])
    run_analyst(monkeypatch, "top 5 restaurants by revenue", RANKING_INTENT, db)
    db2 = MockDB(rows, ["restaurant_name", "revenue"])
    run_analyst(monkeypatch, "top 3 restaurants by revenue", RANKING_INTENT, db2)
    assert "LIMIT 5" in db.sqls[0] and "LIMIT 3" in db2.sqls[0]


def test_top_two_still_renders_a_table(monkeypatch):
    rows = [{"restaurant_name": n, "revenue": v} for n, v in TOP5[:2]]
    db = MockDB(rows, ["restaurant_name", "revenue"])
    result, _ = run_analyst(monkeypatch, "top 2 restaurants by revenue", RANKING_INTENT, db)
    assert "KFC" in result["final_answer"] and "Domino" in result["final_answer"]


def test_time_period_prevents_fast_path_that_would_drop_it(monkeypatch):
    intent = ('{"is_followup": false, "intent_type": "ranking", "metrics": ["revenue"], "dimensions": '
              '["restaurant_name"], "filters": [], "time_period": "2022", "is_driver_question": false}')
    rows = [{"restaurant_name": "KFC", "revenue": 10.0}, {"restaurant_name": "Subway", "revenue": 5.0}]
    db = MockDB(rows, ["restaurant_name", "revenue"])
    sql = ("```sql\nSELECT dr.restaurant_name, SUM(fo.sales_amount) AS revenue FROM workspace.zomato_gold.fact_orders fo "
           "JOIN workspace.zomato_gold.dim_date dd ON fo.date_id = dd.date_id "
           "JOIN workspace.zomato_gold.dim_resturant dr ON fo.restaurant_id = dr.restaurant_id "
           "WHERE dd.year = 2022 GROUP BY 1 ORDER BY 2 DESC LIMIT 5\n```")
    run_analyst(monkeypatch, "top 5 restaurants in 2022", intent, db, extra_llm=['{}', sql])
    assert "dd.year = 2022" in db.sqls[0]          # planner/SQL-generator path, filter kept


# ------------------------------------------------------------------ zero vs NULL vs empty
SIMPLE_INTENT = ('{"is_followup": false, "intent_type": "simple_query", "metrics": ["revenue", "total_orders"], '
                 '"dimensions": [], "filters": [], "time_period": "last month", "is_driver_question": false}')
SIMPLE_SQL = ("```sql\nSELECT SUM(fo.sales_amount) AS total_revenue, COUNT(DISTINCT fo.order_id) AS total_orders "
              "FROM workspace.zomato_gold.fact_orders fo\n```")


def test_null_revenue_zero_orders_is_explained_not_narrated(monkeypatch):
    db = MockDB([{"total_revenue": None, "total_orders": 0}], ["total_revenue", "total_orders"])
    result, llm = run_analyst(monkeypatch, "What is the total revenue and number of orders for the last month?",
                              SIMPLE_INTENT, db, extra_llm=['{}', SIMPLE_SQL])
    answer = result["final_answer"]
    assert "no sales are recorded" in answer
    assert "NULL" in answer and "does not mean revenue was actually zero" in answer
    assert "2022-01-01" in answer and "2022-12-31" in answer        # real data coverage, from the DB
    assert "latest month that actually has data" in answer            # offers, never silently substitutes
    assert llm.calls <= 3                                             # no LLM narrative call for an empty result


def test_empty_result_arabic(monkeypatch):
    db = MockDB([{"total_revenue": None, "total_orders": 0}], ["total_revenue", "total_orders"])
    result, _ = run_analyst(monkeypatch, "ايه اجمالي الايراد وعدد الطلبات الشهر اللي فات؟", SIMPLE_INTENT, db,
                            lang="ar", extra_llm=['{}', SIMPLE_SQL])
    assert "مفيش مبيعات مسجلة" in result["final_answer"]
    assert "2022-12-31" in result["final_answer"]


def test_zero_rows_vs_ok_rows():
    assert classify_rows(["a"], [])["kind"] == "no_rows"
    assert classify_rows(["a", "b"], [{"a": None, "b": 0}])["kind"] == "empty_aggregate"
    assert classify_rows(["a", "b"], [{"a": 5, "b": 0}])["kind"] == "ok"
    assert classify_rows(["n", "v"], [{"n": "KFC", "v": 0}])["kind"] == "ok"      # a real entity with 0 is data
    assert classify_rows(["a", "b"], [{"a": 1, "b": None}, {"a": 2, "b": None}])["null_columns"] == ["b"]


def test_no_rows_message(monkeypatch):
    db = MockDB([], ["restaurant_name", "revenue"])
    result, _ = run_analyst(monkeypatch, "top 5 restaurants in 1999", RANKING_INTENT, db,
                            extra_llm=['{}', SIMPLE_SQL])
    assert "no rows" in result["final_answer"] or "no matching" in result["final_answer"]


# ------------------------------------------------------------------ unsupported metrics
RETENTION_INTENT = ('{"is_followup": false, "intent_type": "driver_analysis", "metrics": ["customer_retention"], '
                    '"dimensions": [], "filters": [], "time_period": null, "is_driver_question": true}')


def test_retention_refusal_lists_real_alternatives_en(monkeypatch):
    db = MockDB([], [])
    result, _ = run_analyst(monkeypatch, "Why did customer retention drop among high-value segments?",
                            RETENTION_INTENT, db)
    answer = result["final_answer"]
    assert "customer_retention" in answer
    assert "Available metrics that can be computed: Revenue" in answer      # no longer an empty list
    assert "Total Orders" in answer and "K-Means" in answer
    assert db.sqls == []                                                    # nothing executed, nothing substituted


def test_retention_refusal_arabic(monkeypatch):
    db = MockDB([], [])
    result, _ = run_analyst(monkeypatch, "ليه retention العملاء قل؟", RETENTION_INTENT, db, lang="ar")
    assert "المقاييس المتاحة حالياً: Revenue" in result["final_answer"]
    assert "الاحتفاظ بالعملاء" in result["final_answer"]


def test_profit_is_still_refused_with_alternatives(monkeypatch):
    intent = RETENTION_INTENT.replace("customer_retention", "profit")
    result, _ = run_analyst(monkeypatch, "what is the profit?", intent, MockDB([], []))
    assert "cost" in result["final_answer"].lower()
    assert "Available alternatives" in result["final_answer"]


# ------------------------------------------------------------------ failed SQL
def test_failed_sql_does_not_fabricate_numbers(monkeypatch):
    class DownDB:
        def execute_queries(self, queries):
            return [{"purpose": "x", "success": False, "error": "connection reset", "error_type": "connection",
                     "rows": [], "row_count": 0, "columns": []} for _ in queries]
    result, _ = run_analyst(monkeypatch, "top 5 restaurants by revenue", RANKING_INTENT, DownDB())
    assert "could not complete" in result["final_answer"]
    assert not any(ch.isdigit() for ch in result["final_answer"].replace("connection reset", ""))


# ------------------------------------------------------------------ advisor
def _evidence(pairs, cols=("restaurant_name", "revenue")):
    return [Evidence(query_id="q1", executed_sql="SELECT 1", columns=cols, rows=tuple(pairs),
                     row_count=len(pairs), truncated=False, executed_at="2026-01-01T00:00:00Z")]


def test_advisor_picks_from_previous_results_and_states_criterion():
    answer = build_advice("In your opinion, which restaurant is best? Give me one name.",
                          _evidence(TOP5), "en", {"metrics": ["revenue"]}, "top 5 restaurants by revenue")
    assert "My pick: KFC" in answer
    assert "Decision criterion" in answer and "highest **Revenue**" in answer
    for name, value in TOP5:
        assert name in answer and f"{value:,.0f}" in answer
    # derived numbers are plain arithmetic on the stored values
    gap = (42466314 - 35671213) / 35671213 * 100
    assert f"{gap:.1f}%" in answer
    assert "does not prove" in answer and "profit" in answer.lower()
    assert "Monthly revenue trend" in answer


def test_advisor_never_calls_revenue_an_investment_recommendation():
    en = build_advice("which restaurant should I invest in?", _evidence(TOP5), "en", {"metrics": ["revenue"]})
    assert "not an investment recommendation" in en
    ar = build_advice("اي احسن مطعم استثمر فيه", _evidence(TOP5), "ar", {"metrics": ["revenue"]})
    assert "مش توصية استثمارية" in ar and "KFC" in ar
    assert "الإيراد بيقيس الحجم" in ar


def test_advisor_respects_requested_count_and_lower_is_better():
    ev = _evidence([("A", 30.0), ("B", 20.0), ("C", 45.0)], cols=("restaurant_name", "avg_delivery_time"))
    answer = build_advice("which is best? give me one name", ev, "en", {"metrics": ["avg_delivery_time"]})
    assert "My pick: B" in answer and "lowest" in answer


def test_advisor_without_comparable_data_is_honest():
    assert "no earlier result" in build_advice("which is best?", [], "en").lower()
    one = build_advice("which is best?", _evidence([("KFC", 5.0)]), "en")
    assert "only one entity" in one


# ------------------------------------------------------------------ multi-turn through the real router graph
class RouteLLM:
    """router_llm stand-in: always votes ANALYSIS, so the ADVISOR route below can only come from the deterministic override."""
    def __init__(self):
        self.structured = MagicMock()
        self.structured.invoke.side_effect = lambda *a, **k: rg.RouteDecision(
            agent="ANALYSIS", confidence=0.9, reasoning="x", language="ar")

    def with_structured_output(self, _):
        return self.structured


def _build(monkeypatch, analyst_results):
    monkeypatch.setattr(rg, "router_llm", RouteLLM())
    monkeypatch.setattr(rg, "_compute_intent", None)
    calls = []

    class Analyst:
        def invoke(self, state, config=None, **k):
            calls.append(state["messages"][-1].content)
            return analyst_results.pop(0)
    monkeypatch.setattr(rg, "analyst_agent", Analyst())
    return rg.build_graph(), calls


def _turn(graph, cfg, text):
    return graph.invoke({"messages": [HumanMessage(content=text)]}, config=cfg)


def test_multi_turn_ranking_then_unsupported_then_opinion_keeps_memory(monkeypatch):
    ranking = {"final_answer": "table", "query_results": _evidence(TOP5), "viz_html": "<div>chart</div>",
               "previous_question": "i want top 5 restaurants", "intent": {"metrics": ["revenue"]}}
    refusal = {"final_answer": "cannot compute customer_retention", "query_results": None}
    graph, calls = _build(monkeypatch, [ranking, refusal])
    cfg = {"configurable": {"thread_id": "t1"}}

    s1 = _turn(graph, cfg, "i want top 5 restaurants in revenue")
    assert s1["route"] == "ANALYSIS"

    s2 = _turn(graph, cfg, "Why did customer retention drop among high-value segments?")
    assert s2["analyst_memory"]["previous_results"], "a refused turn must not wipe the last real result"
    assert [list(r) for r in s2["evidence"][0].rows] == [list(r) for r in TOP5]   # checkpoint turns tuples into lists

    s3 = _turn(graph, cfg, "من وجهه نظرك انت اي احسن واحد من المطاعم ال عندك اديني اسم واحد")
    assert s3["route"] == "ADVISOR"
    assert "KFC" in s3["final_answer"] and "42,466,314" in s3["final_answer"]
    assert "معيار الاختيار" in s3["final_answer"]               # states the decision criterion
    assert s3["viz_html"] == ""                                  # no stale chart from turn 1
    assert s3["llm_call_count"] == 0 and s3["stage_latencies"] == {}
    assert len(calls) == 2                                       # the opinion did NOT re-run the analyst


def test_opinion_without_prior_results_fetches_data_first(monkeypatch):
    ranking = {"final_answer": "table", "query_results": _evidence(TOP5), "viz_html": "<div>c</div>",
               "intent": {"metrics": ["revenue"]}}
    graph, calls = _build(monkeypatch, [ranking])
    s = _turn(graph, {"configurable": {"thread_id": "t2"}}, "اي احسن مطعم استثمر فيه")
    assert s["route"] == "ADVISOR"
    assert calls == ["Top 5 restaurants by revenue"]
    assert "KFC" in s["final_answer"] and "مش توصية استثمارية" in s["final_answer"]
    assert s["analyst_memory"]["previous_results"]               # stored for further follow-ups


def test_opinion_when_data_cannot_be_fetched_does_not_guess(monkeypatch):
    graph, _ = _build(monkeypatch, [{"final_answer": "err", "query_results": None}])
    s = _turn(graph, {"configurable": {"thread_id": "t3"}}, "In your opinion which restaurant is best?")
    assert "ماينفعش أرشّح من غير أرقام" in s["final_answer"]      # router stub reports language=ar
    assert "KFC" not in s["final_answer"]


def test_general_turn_does_not_inherit_previous_chart(monkeypatch):
    ranking = {"final_answer": "table", "query_results": _evidence(TOP5), "viz_html": "<div>chart</div>",
               "llm_call_count": 4, "stage_latencies": {"x": 1.0}}
    graph, _ = _build(monkeypatch, [ranking])
    monkeypatch.setattr(rg, "general_llm", MagicMock(invoke=lambda m: AIMessage(content="hello")))
    cfg = {"configurable": {"thread_id": "t4"}}
    # force the first turn through the analyst
    monkeypatch.setattr(rg, "router_llm", MagicMock(with_structured_output=lambda _: MagicMock(
        invoke=lambda *a, **k: rg.RouteDecision(agent="ANALYSIS", confidence=0.9, reasoning="x", language="en"))))
    s1 = _turn(graph, cfg, "top 5 restaurants by revenue")
    assert s1["viz_html"] == "<div>chart</div>"
    monkeypatch.setattr(rg, "router_llm", MagicMock(with_structured_output=lambda _: MagicMock(
        invoke=lambda *a, **k: rg.RouteDecision(agent="GENERAL", confidence=0.9, reasoning="x", language="en"))))
    s2 = _turn(graph, cfg, "thanks!")
    assert s2["route"] == "GENERAL" and s2["viz_html"] == "" and s2["llm_call_count"] == 0


# ------------------------------------------------------------------ chart-only follow-up keeps its data
def test_change_visualization_followup_uses_previous_results(monkeypatch):
    insight = ('{"insight": "Here is the bar chart.", "viz_spec": {"should_visualize": true, "chart_type": "bar", '
               '"x_axis": "restaurant_name", "y_axis": "revenue", "title": "Revenue"}}')
    llm = FakeLLM([insight])
    monkeypatch.setattr(ag, "_analyst_llm", llm)
    out = ag.insight_generator({
        "messages": [HumanMessage(content="make it a bar chart")], "language": "en",
        "intent": {"original_question": "make it a bar chart", "intent_type": "simple_query"},
        "is_followup": True, "followup_type": "CHANGE_VISUALIZATION",
        "previous_results": _evidence(TOP5), "query_results": [],
    })
    assert out["viz_html"], "the chart must be built from the previous rows"
    assert "KFC" in llm.prompts[0] and "42466314" in llm.prompts[0]   # the LLM saw the real previous rows


def test_change_visualization_without_any_previous_result_is_honest():
    out = ag.insight_generator({
        "messages": [HumanMessage(content="make it a bar chart")], "language": "en",
        "intent": {"original_question": "make it a bar chart"}, "is_followup": True,
        "followup_type": "CHANGE_VISUALIZATION", "previous_results": [], "query_results": [],
    })
    assert "no earlier result" in out["final_answer"].lower()
