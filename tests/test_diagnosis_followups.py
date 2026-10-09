"""
Regression tests for the second failure transcript:
  * "زود عليهم المدينه والتقييم" re-planned from scratch and returned "Top 10000" unrelated rows
  * "ليه محقق العدد ده بس؟ المشكلة فين؟" / "اي السبب؟" answered "the data has no reason" — no root cause, no fix
  * "مين اكتر عميل اشتري مني؟" ended with a bare "could not complete the analysis"
  * "اي اقل مطعم محقق ارباح؟" was a dead end instead of a disclosed revenue proxy
  * "اي اقل مطعم ...؟" listed 10 rows without naming THE answer (and its tie)
  * names with apostrophes (McDonald's, Domino's) silently matched nothing
"""
import datetime
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

import src.agent.analyst_graph as ag
import src.agent.router_graph as rg
from src.agent import diagnostics as dx
from src.agent.analyst_state import Evidence
from src.agent.followups import followup_kind, run_followup, target_names
from src.agent.query_intent import is_singular_superlative
from src.agent.ranking_text import build_ranking_answer
from src.agent.sql_safety_guard import guard_sql

BOTTOM = [("K2H Meals", 12), ("SAI DURGA CHAT CENTRE", 12), ("Banglar Rannaghar", 13)]
TOP = [("KFC", 42466314.0), ("Domino's Pizza", 35671213.0), ("McDonald's", 20492809.0)]


def ev(pairs, metric):
    return [Evidence(query_id="q", executed_sql="SELECT 1", columns=("restaurant_name", metric),
                     rows=tuple(pairs), row_count=len(pairs), truncated=False, executed_at="t")]


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize("q,kind", [
    ("زود عليهم المدينه بتاعه كل مطعم والتقييم بتاعه", "enrich"),
    ("add their city and rating", "enrich"),
    ("ليه محقق العدد ده بس المشكله ممكن تكون فين ؟", "diagnose"),
    ("اي السبب برده ؟", "diagnose"),
    ("why?", "diagnose"),
    ("what should I do to fix K2H Meals?", "diagnose"),
    ("طب اعمل ايه عشان احسنهم", "diagnose"),
    ("Why did customer retention drop among high-value segments?", None),
    ("Why did sales decline in 2025?", None),
    ("عاوز افضل 9 مطاعم", None),
])
def test_followup_kind(q, kind):
    assert followup_kind(q, ev(BOTTOM, "total_orders")) == kind


def test_no_followup_without_restaurant_evidence():
    assert followup_kind("ليه؟", []) is None
    customers = [Evidence("q", "s", ("user_id", "name", "revenue"), ((1, "A", 5.0),), 1, False, "t")]
    assert followup_kind("ليه؟", customers) is None


def test_arabic_why_does_not_match_inside_other_words():
    # 'عليهم' contains the letters of 'ليه' — it must not turn an enrichment into a diagnosis
    assert followup_kind("زود عليهم التقييم", ev(BOTTOM, "total_orders")) == "enrich"


def test_target_names_prefers_named_restaurant():
    assert target_names("why does K2H Meals have so few orders?", ev(BOTTOM, "total_orders")) == ["K2H Meals"]
    assert target_names("ليه؟", ev(BOTTOM, "total_orders")) == [n for n, _ in BOTTOM]


# ------------------------------------------------------------------ SQL literal safety
def test_apostrophe_survives_the_guard():
    sql, _ = guard_sql(f"SELECT a FROM workspace.zomato_gold.t WHERE n IN ({dx.sql_literal(chr(77) + 'cDonald' + chr(39) + 's')})")
    assert "CONCAT" not in sql and "McDonald\\'s" in sql
    # the old escape really is broken in Databricks: two literals concatenated, apostrophe lost
    bad, _ = guard_sql("SELECT a FROM workspace.zomato_gold.t WHERE n = 'McDonald''s'")
    assert "CONCAT" in bad


# ------------------------------------------------------------------ fake warehouse
END = datetime.datetime(2026, 6, 30)
PLATFORM = {"p10": 28.0, "p25": 33.0, "median_orders": 40.0, "p75": 60.0, "p90": 138.0, "branches": 148541,
            "orders_per_customer": 1.05, "aov": 483.0, "cancel_rate": 0.06, "refund_rate": 0.025,
            "avg_delivery_min": 31.1, "avg_customer_rating": 4.15, "discount_rate": 0.088}
QUALITY = {"branches": 148541, "brands": 112818, "rated": 61441, "with_cost": 0, "with_rating_count": 0}


def target_row(name, orders, *, branches=1, cancel=0.06, delivery=31.0, rating=4.15, listing=None, discount=0.088,
               aov=483.0, city="Narhe,Pune", cuisine="Maharashtrian", last6=None, prev6=None, customers=None):
    return {"restaurant_name": name, "branches": branches, "cities": 1, "city": city, "cuisine": cuisine,
            "listing_rating": listing, "orders": orders, "customers": customers or orders, "revenue": orders * aov,
            "aov": aov, "cancel_rate": cancel, "refund_rate": 0.02, "avg_delivery_min": delivery,
            "avg_customer_rating": rating, "discount_rate": discount, "first_order": datetime.datetime(2024, 4, 6),
            "last_order": datetime.datetime(2026, 5, 16), "data_end": END,
            "orders_last_6m": last6 if last6 is not None else orders // 4,
            "orders_prev_6m": prev6 if prev6 is not None else orders // 4}


class FakeWarehouse:
    def __init__(self, targets, enrich=None, fail=False):
        self.targets, self.enrich_rows, self.fail, self.sqls = targets, enrich or [], fail, []

    def __call__(self, queries):
        out = []
        for q in queries:
            sql = q["sql"]
            self.sqls.append(sql)
            if self.fail:
                out.append({"success": False, "error": "boom", "rows": [], "columns": []})
            elif "WITH bounds" in sql:
                out.append({"success": True, "rows": self.targets, "columns": list(self.targets[0]) if self.targets else []})
            elif "COUNT(cost)" in sql:
                out.append({"success": True, "rows": [QUALITY], "columns": list(QUALITY)})
            elif "AS grp" in sql and "dr.city" in sql:
                out.append({"success": True, "rows": [{"grp": "Narhe,Pune", "branches": 50, "median_orders": 41.0,
                                                       "aov": 470.0, "cancel_rate": 0.06, "refund_rate": 0.025,
                                                       "avg_delivery_min": 31.0, "avg_customer_rating": 4.1,
                                                       "discount_rate": 0.09}], "columns": []})
            elif "AS grp" in sql:
                out.append({"success": True, "rows": [], "columns": []})
            elif "PERCENTILE" in sql:
                out.append({"success": True, "rows": [PLATFORM], "columns": list(PLATFORM)})
            elif "MAX_BY(b.city" in sql:
                out.append({"success": True, "rows": self.enrich_rows, "columns": list(self.enrich_rows[0]) if self.enrich_rows else []})
            else:
                raise AssertionError(f"unexpected SQL: {sql[:80]}")
        return out


# ------------------------------------------------------------------ enrichment
ENRICH = [
    {"restaurant_name": "KFC", "top_city": "Mayur Vihar,Delhi", "cities": 309, "branches": 309, "avg_rating": 3.94, "rated_branches": 306, "cuisine": "Burger"},
    {"restaurant_name": "Domino's Pizza", "top_city": "Karappakam,Chennai", "cities": 442, "branches": 442, "avg_rating": 4.21, "rated_branches": 409, "cuisine": "Pizza"},
]


def test_enrich_keeps_same_restaurants_order_and_values():
    wh = FakeWarehouse([], enrich=ENRICH)
    out = run_followup("enrich", "زود عليهم المدينه والتقييم", ev(TOP, "revenue"), "ar", wh)
    a = out["answer"]
    assert "Top 10000" not in a and "10000" not in a
    assert a.index("KFC") < a.index("Domino") < a.index("McDonald")         # previous order preserved
    assert "42,466,314" in a and "35,671,213" in a                           # previous metric values, not recomputed
    assert "Mayur Vihar,Delhi" in a and "3.94" in a and "4.21" in a
    assert "McDonald's" in a and "مالقيتش" in a                              # missing enrichment row is disclosed
    assert len(wh.sqls) == 1 and "McDonald\\'s" in wh.sqls[0]


# ------------------------------------------------------------------ diagnosis
def test_diagnosis_finds_measured_root_causes_and_actions():
    targets = [target_row("K2H Meals", 30, cancel=0.20, delivery=45.0, rating=3.6)]
    wh = FakeWarehouse(targets)
    out = run_followup("diagnose", "ليه محقق العدد ده بس المشكله ممكن تكون فين ؟", ev([("K2H Meals", 30)], "total_orders"), "ar", wh)
    a = out["answer"]
    assert "التشخيص: K2H Meals" in a
    assert "30 طلب" in a and "الوسيط 40" in a and "أقل 25%" in a           # the problem, measured
    assert "ارتفاع الطلبات الفاشلة" in a and "22.0%" in a and "8.5%" in a   # cause + evidence vs benchmark (city)
    assert "التوصيل أبطأ" in a and "تقييم العملاء للطلبات أقل" in a
    assert "### الحل المقترح" in a and "الهدف" in a                          # concrete action with a target
    assert "ارتباط مش سببية" in a
    assert "(cost) فاضي في 100%" in a                                       # measured data-quality limit
    assert "الطلبات لكل فرع" in a                                           # decomposition names the weak factor


def test_diagnosis_with_nothing_abnormal_says_so_and_points_to_demand():
    targets = [target_row("Plain Place", 12, listing=4.0, rating=4.15)]
    a = run_followup("diagnose", "why?", ev([("Plain Place", 12)], "total_orders"), "en", FakeWarehouse(targets))["answer"]
    assert "demand and visibility" in a
    assert "low-confidence" in a or "Order counts are very small" in a


def test_diagnosis_of_top_restaurants_explains_strength():
    targets = [target_row("KFC", 61731, branches=309, aov=688.0)]
    a = run_followup("diagnose", "why is KFC on top?", ev(TOP, "revenue"), "en", FakeWarehouse(targets))["answer"]
    assert "Why performance is high" in a
    assert "number of branches" in a                                        # 309 branches vs ~1.3 per brand


def test_diagnosis_numbers_come_from_the_warehouse():
    targets = [target_row("K2H Meals", 12, cancel=0.0, delivery=25.9, rating=3.57, discount=0.127, aov=214.0)]
    a = run_followup("diagnose", "ليه؟", ev([("K2H Meals", 12)], "total_orders"), "ar", FakeWarehouse(targets))["answer"]
    for s in ("| K2H Meals | 12 |", "25.9", "3.57", "12.7%", "214"):
        assert s in a


def test_diagnosis_failure_does_not_guess():
    out = run_followup("diagnose", "ليه؟", ev(BOTTOM, "total_orders"), "ar", FakeWarehouse([], fail=True))
    assert "مش هخمّن" in out["answer"] and "K2H" not in out["answer"]


# ------------------------------------------------------------------ router integration
def test_router_follow_up_flow(monkeypatch):
    monkeypatch.setattr(rg, "router_llm", MagicMock(with_structured_output=lambda _: MagicMock(
        invoke=lambda *a, **k: rg.RouteDecision(agent="GENERAL", confidence=0.3, reasoning="x", language="ar"))))
    monkeypatch.setattr(rg, "_compute_intent", None)
    calls = []

    class Analyst:
        def invoke(self, state, config=None, **k):
            calls.append(state["messages"][-1].content)
            return {"final_answer": "table", "query_results": ev(TOP, "revenue"), "intent": {"metrics": ["revenue"]}}
    monkeypatch.setattr(rg, "analyst_agent", Analyst())
    wh = FakeWarehouse([target_row("KFC", 61731, branches=309, aov=688.0)], enrich=ENRICH)
    monkeypatch.setattr(rg, "_execute_guarded", wh)

    graph = rg.build_graph()
    cfg = {"configurable": {"thread_id": "dx1"}}
    # turn 1 forced through the analyst
    monkeypatch.setattr(rg, "router_llm", MagicMock(with_structured_output=lambda _: MagicMock(
        invoke=lambda *a, **k: rg.RouteDecision(agent="ANALYSIS", confidence=0.9, reasoning="x", language="ar"))))
    graph.invoke({"messages": [HumanMessage(content="عاوز افضل 3 مطاعم")]}, config=cfg)

    s2 = graph.invoke({"messages": [HumanMessage(content="زود عليهم المدينه والتقييم")]}, config=cfg)
    assert s2["route"] == "FOLLOWUP" and "Mayur Vihar" in s2["final_answer"]
    assert s2["turn_evidence"] and "MAX_BY" in s2["turn_evidence"][0].executed_sql

    s3 = graph.invoke({"messages": [HumanMessage(content="اي السبب ؟")]}, config=cfg)
    assert s3["route"] == "FOLLOWUP" and "التشخيص" in s3["final_answer"]

    s4 = graph.invoke({"messages": [HumanMessage(content="من وجهة نظرك اي احسن واحد فيهم؟ اسم واحد")]}, config=cfg)
    assert s4["route"] == "ADVISOR" and "KFC" in s4["final_answer"]      # still based on the original ranking
    assert calls == ["عاوز افضل 3 مطاعم"]                                  # no re-planning for follow-ups


def test_router_failure_sends_data_questions_to_the_analyst_not_general(monkeypatch):
    """Groq 429 on the router used to fall back to GENERAL, which then invented SQL and a '30% margin'."""
    failing = MagicMock()
    failing.invoke.side_effect = Exception("Error code: 429 - rate_limit_exceeded")
    monkeypatch.setattr(rg, "router_llm", MagicMock(with_structured_output=lambda _: failing))
    monkeypatch.setattr(rg, "_compute_intent", None)
    out = rg.router_node({"messages": [HumanMessage(content="مين اكتر عميل محققلي ارباح ؟")]})
    assert out["route"] == "ANALYSIS" and out["language"] == "ar"
    out = rg.router_node({"messages": [HumanMessage(content="hello there!")]})
    assert out["route"] == "GENERAL"


def test_router_context_clips_long_answers():
    from langchain_core.messages import AIMessage
    long = "| KFC | 1 |\n" * 500
    ctx = rg._recent_context([HumanMessage(content="q"), AIMessage(content=long)], 8, exclude_last=False)
    assert len(ctx) < 800 and "[truncated]" in ctx


# ------------------------------------------------------------------ ranking answers
def test_singular_superlative_detection():
    assert is_singular_superlative("اي اقل مطعم محقق اوردرات")
    assert is_singular_superlative("مين اكتر عميل اشتري مني ؟")
    assert is_singular_superlative("which restaurant has the most orders?")
    assert not is_singular_superlative("عاوز افضل 9 مطاعم")
    assert not is_singular_superlative("top 5 restaurants")


def test_singular_answer_names_leader_with_ties():
    rows = [{"restaurant_name": n, "total_orders": v} for n, v in BOTTOM]
    a = build_ranking_answer(rows, ["restaurant_name", "total_orders"],
                             {"singular": True, "sort_order": "ASC"}, "ar")
    assert a.startswith("الأقل في total orders: **K2H Meals** و **SAI DURGA CHAT CENTRE** — 12 (تعادل بين 2)")
    assert "ليه؟" in a                                                       # proactive next step


def test_oversized_result_is_not_called_top_10000():
    rows = [{"restaurant_name": f"R{i}", "revenue": float(10000 - i)} for i in range(10000)]
    a = build_ranking_answer(rows, ["restaurant_name", "revenue"], {"sort_order": "DESC"}, "ar")
    assert "10000 نتائج" not in a and "عرض أول 15 من 10,000 صف" in a
    assert "| R14 |" in a and "| R15 |" not in a


def test_customer_ranking_uses_user_id_and_name(monkeypatch):
    from tests.test_conversation_regressions import MockDB, run_analyst
    intent = ('{"is_followup": false, "intent_type": "ranking", "metrics": ["revenue"], "dimensions": ["customer"], '
              '"filters": [], "time_period": null, "is_driver_question": false}')
    rows = [{"user_id": 7, "name": "Asha", "revenue": 9000.0}, {"user_id": 9, "name": "Asha", "revenue": 8000.0}]
    db = MockDB(rows, ["user_id", "name", "revenue"])
    result, _ = run_analyst(monkeypatch, "مين اكتر عميل اشتري مني ؟", intent, db, lang="ar")
    assert "du.user_id, du.name" in db.sqls[0] and "GROUP BY 1, 2" in db.sqls[0]
    assert "**Asha** — 9,000" in result["final_answer"]                     # two different people named Asha kept apart
    assert "could not complete" not in result["final_answer"]


def test_unknown_dimension_falls_back_to_planner(monkeypatch):
    from tests.test_conversation_regressions import MockDB, run_analyst
    intent = ('{"is_followup": false, "intent_type": "ranking", "metrics": ["revenue"], "dimensions": ["occupation"], '
              '"filters": [], "time_period": null, "is_driver_question": false}')
    sql = ("```sql\nSELECT du.occupation, SUM(fo.sales_amount) AS revenue FROM workspace.zomato_gold.fact_orders fo "
           "JOIN workspace.zomato_gold.dim_users du ON fo.user_id = du.user_id GROUP BY 1 ORDER BY 2 DESC LIMIT 5\n```")
    db = MockDB([{"occupation": "Student", "revenue": 5.0}, {"occupation": "Employee", "revenue": 3.0}], ["occupation", "revenue"])
    result, _ = run_analyst(monkeypatch, "top occupations by revenue", intent, db, extra_llm=["{}", sql])
    assert "du.occupation" in db.sqls[0] and "Student" in result["final_answer"]


def test_no_query_error_is_actionable_not_bare():
    out = ag.error_end({"language": "ar", "sql_queries": []})
    assert "حدد المقياس" in out["final_answer"]
    out = ag.error_end({"language": "en", "sql_queries": [{"error": "SECURITY BLOCKED: x", "error_type": "guard"}]})
    assert "SECURITY BLOCKED" not in out["final_answer"] and "safety guard" in out["final_answer"]


def test_question_dimension_overrides_carried_over_llm_dimension(monkeypatch):
    """'مين اكتر عميل محققلي ارباح' was answered with KFC: the intent LLM carried 'restaurant' over from context."""
    from src.agent.query_intent import extract_dimension
    assert extract_dimension("مين اكتر عميل محققلي ارباح ؟") == "customer"
    assert extract_dimension("اي اقل مطعم محقق اوردرات") == "restaurant_name"
    assert extract_dimension("revenue by city") == "city"
    stale = ('{"is_followup": true, "intent_type": "ranking", "metrics": ["profit"], "dimensions": ["restaurant_name"], '
             '"filters": [], "time_period": null, "is_driver_question": false}')
    monkeypatch.setattr(ag, "_analyst_llm", type("L", (), {"invoke": lambda self, p: type("R", (), {"content": stale})()})())
    intent = ag.compute_intent([HumanMessage(content="مين اكتر عميل محققلي ارباح ؟")])
    assert intent["dimensions"] == ["customer"] and intent["is_followup"] is False


def test_advisor_reports_ties_instead_of_picking_at_random():
    from src.agent.advisor import build_advice
    e = ev([("Amer Sai Fast Food", 15), ("BLACK MUGS CAFE", 15), ("K2H Meals", 12)], "total_orders")
    a = build_advice("من وجهة نظرك اي احسن واحد فيهم؟ اديني اسم واحد", e, "ar", {"metrics": ["total_orders"]})
    assert a.startswith("## تعادل: Amer Sai Fast Food / BLACK MUGS CAFE")
    assert "0.0%" not in a and "الترتيب بـ Total Orders" in a and "بالإيراد" not in a


# ------------------------------------------------------------------ profit proxy
def test_profit_ranking_uses_disclosed_revenue_proxy(monkeypatch):
    from tests.test_conversation_regressions import MockDB, run_analyst
    intent = ('{"is_followup": false, "intent_type": "ranking", "metrics": ["profit"], "dimensions": ["restaurant_name"], '
              '"filters": [], "time_period": null, "is_driver_question": false}')
    rows = [{"restaurant_name": "A", "revenue": 10.0}, {"restaurant_name": "B", "revenue": 20.0}]
    db = MockDB(rows, ["restaurant_name", "revenue"])
    result, _ = run_analyst(monkeypatch, "اي اقل مطعم محقق ارباح ؟", intent, db, lang="ar")
    assert "SUM(fo.sales_amount)" in db.sqls[0] and "ASC" in db.sqls[0]
    a = result["final_answer"]
    assert a.startswith("⚠️ **الربح مش متاح**") and "الإيراد مش ربح" in a
    assert "**A** — 10" in a
