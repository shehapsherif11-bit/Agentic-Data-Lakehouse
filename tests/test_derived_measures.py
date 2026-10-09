"""P0-2: "% of total" requests must not be dropped (Q4) or refused (Q7), and must be phrased-independent."""
import pytest

from src.agent import analyst_graph as ag
from src.agent import metrics_registry as metrics
from src.agent.query_intent import extract_derived_measures


@pytest.mark.parametrize("q", [
    "Show me the top 10 restaurants by total revenue. Include each restaurant's name, total revenue, and percentage contribution to overall revenue.",
    "top 5 restaurants and their share of revenue",
    "what is the revenue share of each city",
    "اعرض اعلى 10 مطاعم مع حصة كل مطعم من الايراد",
    "اعلى 5 مطاعم ونسبة مساهمة كل واحد من الاجمالي",
])
def test_share_requests_detected(q):
    assert extract_derived_measures(q) == ["share_of_total"]


@pytest.mark.parametrize("q", [
    "top 3 restaurants by revenue",
    "ايه نسبة الخصم في اكتر مطعم",          # discount RATE metric, not a share request
    "which restaurants contributed most to the decline",
])
def test_plain_questions_have_no_derived_measure(q):
    assert extract_derived_measures(q) == []


@pytest.mark.parametrize("name,base", [
    ("percentage_of_total_revenue", "revenue"),
    ("revenue_share", "revenue"),
    ("pct contribution revenue", "revenue"),
])
def test_derived_metric_names_resolve_to_base(name, base):
    assert metrics.split_derived_metric(name) == (base, ["share_of_total"])


def test_real_metrics_and_unknown_are_untouched():
    for name in ("revenue", "total_orders", "churn", "profit"):
        assert metrics.split_derived_metric(name)[1] == []


def _intent(question, llm_metrics):
    i = {"metrics": list(llm_metrics), "dimensions": ["restaurant_name"], "intent_type": "ranking", "original_question": question}
    ag._normalize_derived_measures(i, question)
    return i


def test_q4_and_q7_phrasings_produce_same_resolution():
    q = "top 10 restaurants by revenue with percentage contribution to overall revenue"
    a = _intent(q, ["revenue", "percentage_contribution"])
    b = _intent(q, ["revenue", "percentage_of_total_revenue"])
    c = _intent(q, ["percentage_of_total_revenue"])
    for i in (a, b, c):
        assert i["metrics"] == ["revenue"] and i["derived_measures"] == ["share_of_total"]
        assert metrics.check_data_sufficiency(i["metrics"], i["dimensions"])["sufficient"]


def _gen(intent, metric="revenue"):
    state = {"intent": intent, "resolved_metrics": [metrics.resolve_metric(metric)]}
    return ag.deterministic_sql_generator(state)["sql_queries"]


def test_fast_path_sql_has_share_window_before_limit():
    i = _intent("top 10 restaurants by revenue with share of total", ["revenue"])
    i["top_n"] = 10
    (q,) = _gen(i)
    sql = q["sql"]
    assert "OVER ()" in sql and "AS pct_of_total" in sql
    assert sql.index("OVER ()") < sql.index("LIMIT 10")
    # window over the grouped aggregate: SUM(SUM(x)) OVER (), computed before LIMIT
    assert "SUM(SUM(fo.sales_amount)) OVER ()" in sql


def test_share_of_non_additive_metric_goes_to_planner():
    i = _intent("top 10 restaurants by aov with share of total", ["aov"])
    assert _gen(i, "aov") == []


def test_plain_ranking_sql_unchanged():
    i = _intent("top 3 restaurants by revenue", ["revenue"]); i["top_n"] = 3
    (q,) = _gen(i)
    assert "OVER" not in q["sql"] and "pct_of_total" not in q["sql"]


SHARE_Q = ("Show me the top 10 restaurants by total revenue, ranked from highest to lowest. Include each "
           "restaurant's name, total revenue, and percentage contribution to overall revenue.")


def _share_rows():
    total = 1000.0
    return [{"restaurant_name": n, "revenue": v, "pct_of_total": round(100 * v / total, 2)}
            for n, v in (("KFC", 300.0), ("Domino's Pizza", 200.0), ("Pizza Hut", 100.0))]


@pytest.mark.parametrize("llm_metrics", [
    '["revenue", "percentage_contribution"]',        # Q4: the LLM invented a second metric, which used to be dropped silently
    '["revenue", "percentage_of_total_revenue"]',    # Q7: the same, but refused as "not in the catalog"
    '["percentage_of_total_revenue"]',
])
def test_share_question_end_to_end_one_llm_call_with_share_column(monkeypatch, llm_metrics):
    from tests.test_conversation_regressions import MockDB, run_analyst
    intent = ('{"is_followup": false, "intent_type": "ranking", "metrics": %s, "dimensions": ["restaurant_name"], '
              '"filters": [], "time_period": null, "is_driver_question": false}' % llm_metrics)
    db = MockDB(_share_rows(), ["restaurant_name", "revenue", "pct_of_total"])
    result, llm = run_analyst(monkeypatch, SHARE_Q, intent, db)

    assert result.get("error") != "data_insufficient", result["final_answer"]
    assert "pct_of_total" in db.sqls[0] and "LIMIT 10" in db.sqls[0]
    assert llm.calls == 1                                   # intent only: SQL is deterministic, no narration
    answer = result["final_answer"]
    assert "30" in answer and "KFC" in answer               # share values come from the DB rows
    assert "not in the query result" not in answer


def test_missing_share_column_is_disclosed_not_dropped(monkeypatch):
    from tests.test_conversation_regressions import MockDB, run_analyst
    intent = ('{"is_followup": false, "intent_type": "ranking", "metrics": ["revenue"], "dimensions": ["restaurant_name"], '
              '"filters": [], "time_period": null, "is_driver_question": false}')
    rows = [{"restaurant_name": "KFC", "revenue": 300.0}, {"restaurant_name": "Pizza Hut", "revenue": 100.0}]
    db = MockDB(rows, ["restaurant_name", "revenue"])    # warehouse result lacks the share column
    result, _ = run_analyst(monkeypatch, SHARE_Q, intent, db)
    assert "not in the query result" in result["final_answer"]


def test_completeness_flags_missing_share_column():
    i = {"derived_measures": ["share_of_total"]}
    ok = [{"columns": ["restaurant_name", "revenue", "pct_of_total"], "rows": []}]
    bad = [{"columns": ["restaurant_name", "revenue"], "rows": []}]
    assert ag._missing_derived_columns(i, ok) == []
    assert ag._missing_derived_columns(i, bad)
    assert ag._missing_derived_columns({}, bad) == []
