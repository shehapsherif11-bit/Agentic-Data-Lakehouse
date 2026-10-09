"""P0-3: "biggest week/month drop" is answered by code: correct period, exact arithmetic, evidence-backed, honest limits."""
import datetime as dt

import pytest
from langchain_core.messages import HumanMessage

from src.agent import period_change as pc
from src.agent import router_graph as rg
from src.agent.sql_safety_guard import check_multiple_queries

Q5 = ("Which week had the biggest week-over-week decrease in revenue, and what are the most likely data-supported reasons "
      "behind the decrease? Show the revenue for both weeks, the absolute and percentage change, and supporting evidence.")
Q6 = ("For the week with the largest revenue drop, identify the 5 restaurants that contributed most to the decline. Show each "
      "restaurant's revenue in both weeks, absolute change, percentage change, and share of the total decline.")


# ------------------------------------------------------------------ parser
@pytest.mark.parametrize("q,unit,direction,metric,n", [
    (Q5, "week", "down", "revenue", 5),
    (Q6, "week", "down", "revenue", 5),
    ("which month had the largest drop in orders?", "month", "down", "orders", 5),
    ("biggest month over month increase in revenue", "month", "up", "revenue", 5),
    ("اي اكبر اسبوع نزلت فيه المبيعات", "week", "down", "revenue", 5),
    ("which week had the worst decline, show the top 3 restaurants behind it", "week", "down", "revenue", 3),
])
def test_parse_positive(q, unit, direction, metric, n):
    assert pc.parse_period_change(q) == {"unit": unit, "direction": direction, "metric": metric, "top_n": n}


@pytest.mark.parametrize("q", [
    "top 3 restaurants by revenue",
    "why did revenue drop last month",            # relative period: the planner resolves 'last month'
    "revenue drop in March 2025",                 # a named period is not an 'extreme period' question
    "which week is best for sales",               # no change direction
    "why so low? what should I do?",
    "",
])
def test_parse_negative(q):
    assert pc.parse_period_change(q) is None


# ------------------------------------------------------------------ SQL
@pytest.mark.parametrize("unit", ["week", "month"])
@pytest.mark.parametrize("direction", ["down", "up"])
@pytest.mark.parametrize("metric", ["revenue", "orders"])
def test_generated_sql_passes_the_safety_guard(unit, direction, metric):
    queries = [
        {"sql": pc.extreme_period_sql(unit, direction, metric), "purpose": "a"},
        {"sql": pc.contributors_sql(unit, direction, metric, "2025-03-03", "2025-02-24", 5), "purpose": "b"},
        {"sql": pc.drivers_sql(unit, "2025-03-03", "2025-02-24"), "purpose": "c"},
    ]
    assert all(q.get("safety_passed") for q in check_multiple_queries(queries)), check_multiple_queries(queries)


def test_extreme_period_sql_excludes_partial_periods_and_requires_consecutive_periods():
    wk = pc.extreme_period_sql("week", "down", "revenue")
    assert "p.p_start >= b.d0" in wk and "DATE_ADD(p.p_start, 6) <= b.d1" in wk     # whole Monday-Sunday weeks only
    assert "DATEDIFF(c.p_start, c.prev_start) = 7" in wk and "DATE_TRUNC('week'" in wk
    assert "ORDER BY x.abs_change ASC" in wk
    mo = pc.extreme_period_sql("month", "up", "orders")
    assert "LAST_DAY(p.p_start) <= b.d1" in mo and "ADD_MONTHS(c.prev_start, 1) = c.p_start" in mo
    assert "ORDER BY x.abs_change DESC" in mo and "COUNT(DISTINCT fo.order_id)" in mo


def test_contributor_share_is_against_the_total_change_before_limit():
    sql = pc.contributors_sql("week", "down", "revenue", "2025-03-03", "2025-02-24", 5)
    assert "SUM(cur_value - prev_value) OVER () AS total_change" in sql
    assert sql.index("total_change") < sql.index("LIMIT 5")
    assert "GROUP BY dr.restaurant_name" in sql and "restaurant_id" not in sql.split("GROUP BY")[1]   # brand grain
    assert "DATE'2025-03-03'" in sql and "DATE_ADD(DATE'2025-03-03', 7)" in sql


# ------------------------------------------------------------------ execution + rendering
EXTREME = {"period_start": dt.date(2025, 3, 3), "prev_period_start": dt.date(2025, 2, 24), "current_value": 800.0,
           "previous_value": 1000.0, "abs_change": -200.0, "pct_change": -20.0, "baseline_avg": 700.0, "pct_stddev": 4.0,
           "periods_compared": 120, "data_start": dt.date(2024, 1, 1), "data_end": dt.date(2026, 6, 30)}
CONTRIB = [{"restaurant_name": "KFC", "prev_value": 300.0, "cur_value": 150.0, "abs_change": -150.0, "pct_change": -50.0,
            "share_of_total_change": 75.0, "brands_down": 30, "brands": 50},
           {"restaurant_name": "Subway", "prev_value": 100.0, "cur_value": 70.0, "abs_change": -30.0, "pct_change": -30.0,
            "share_of_total_change": 15.0, "brands_down": 30, "brands": 50}]
DRIVERS = [{"period": "current", "revenue": 800.0, "orders": 10, "aov": 80.0, "active_branches": 40, "customers": 9, "failed_rate": 0.2},
           {"period": "previous", "revenue": 1000.0, "orders": 10, "aov": 100.0, "active_branches": 45, "customers": 10, "failed_rate": 0.1}]


class FakeExec:
    def __init__(self, extreme_rows=None, fail_on=None):
        self.calls, self.extreme_rows, self.fail_on = [], [EXTREME] if extreme_rows is None else extreme_rows, fail_on

    def __call__(self, queries):
        out = []
        for q in queries:
            self.calls.append(q["sql"])
            purpose = q["purpose"]
            if self.fail_on and self.fail_on in purpose:
                out.append({"success": False, "error": "boom", "columns": [], "rows": []})
            elif "extreme" in purpose:
                out.append({"success": True, "columns": list(EXTREME), "rows": self.extreme_rows})
            elif "contribution" in purpose:
                out.append({"success": True, "columns": list(CONTRIB[0]), "rows": CONTRIB})
            else:
                out.append({"success": True, "columns": list(DRIVERS[0]), "rows": DRIVERS})
        return out


def test_q5_answer_has_both_periods_changes_evidence_and_limits():
    ex = FakeExec()
    res = pc.run_period_change(Q5, "en", ex)
    a = res["answer"]
    assert "2025-03-03 → 2025-03-09" in a and "2025-02-24 → 2025-03-02" in a        # both weeks, Monday-Sunday
    assert "800" in a and "1,000" in a and "-200" in a and "-20.00%" in a
    assert "KFC" in a and "75.0%" in a and "30 of 50 brands declined" in a and "Share of total decline" in a
    assert "association, not causation" in a
    assert "above the average" in a                                                  # prev week 1000 vs baseline 700: regression to the mean
    assert len(ex.calls) == 3 and len(res["evidence"]) == 3
    assert res["evidence"][0]["rows"][0]["restaurant_name"] == "KFC"                 # restaurants first, for "why?" follow-ups


def test_small_extreme_is_reported_as_ordinary_variation_not_a_cause():
    """The live data's worst week was -1.35% (sigma ~0.6%): saying 'here is why it dropped' would be inventing a story."""
    calm = dict(EXTREME, pct_change=-6.0, pct_stddev=3.0)       # z = 2.0 over 120 comparisons
    a = pc.run_period_change(Q5, "en", FakeExec(extreme_rows=[calm]))["answer"]
    assert "expected by chance alone" in a and "2.0x" in a


def test_large_extreme_is_flagged_as_unusual():
    wild = dict(EXTREME, pct_change=-20.0, pct_stddev=4.0)      # z = 5.0
    a = pc.run_period_change(Q5, "en", FakeExec(extreme_rows=[wild]))["answer"]
    assert "stands out from ordinary variation" in a and "5.0x" in a


def test_few_comparisons_make_no_variability_claim():
    a = pc.run_period_change(Q5, "en", FakeExec(extreme_rows=[dict(EXTREME, periods_compared=5)]))["answer"]
    assert "expected by chance" not in a and "stands out" not in a


def test_revenue_split_is_exact():
    """R1 - R0 = (o1 - o0) * a0 + o1 * (a1 - a0): here orders are flat, so the whole -200 is the basket-size effect."""
    a = pc.run_period_change(Q5, "en", FakeExec())["answer"]
    assert "**order-volume effect** +0 + **basket-size effect** -200 = -200" in a


def test_q6_uses_requested_top_n_in_sql():
    ex = FakeExec()
    pc.run_period_change(Q6, "en", ex)
    assert "LIMIT 5" in ex.calls[1]


def test_arabic_answer_is_arabic():
    a = pc.run_period_change("اي اكبر اسبوع نزلت فيه المبيعات", "ar", FakeExec())["answer"]
    assert "ارتباط مش سببية" in a and "أكبر انخفاض" in a


def test_no_consecutive_full_periods_is_reported_not_invented():
    res = pc.run_period_change(Q5, "en", FakeExec(extreme_rows=[]))
    assert "does not contain two consecutive full weeks" in res["answer"]
    assert "800" not in res["answer"]


def test_failed_query_does_not_fabricate_numbers():
    res = pc.run_period_change(Q5, "en", FakeExec(fail_on="extreme"))
    assert "will not guess" in res["answer"] and "800" not in res["answer"]


def test_contributor_failure_still_returns_the_measured_period_change():
    res = pc.run_period_change(Q5, "en", FakeExec(fail_on="contribution"))
    assert "800" in res["answer"] and "Biggest contributors" not in res["answer"]


def test_dates_interpolated_into_sql_are_validated():
    bad = dict(EXTREME, period_start="2025-03-03'; DROP TABLE x; --")
    ex = FakeExec(extreme_rows=[bad])
    res = pc.run_period_change(Q5, "en", ex)
    assert len(ex.calls) == 1 and "will not guess" in res["answer"]       # never reached the follow-up queries


# ------------------------------------------------------------------ routing: zero LLM calls
class ExplodingLLM:
    def __getattr__(self, name):
        raise AssertionError("the router LLM must not be called for a period-over-period question")


def test_router_routes_to_period_change_without_any_llm(monkeypatch):
    monkeypatch.setattr(rg, "router_llm", ExplodingLLM())
    out = rg.router_node({"messages": [HumanMessage(content=Q5)]})
    assert out["route"] == "PERIOD_CHANGE" and out["llm_call_count"] == 0 and out["language"] == "en"


def test_full_graph_turn_for_q5_makes_zero_llm_calls(monkeypatch):
    monkeypatch.setattr(rg, "router_llm", ExplodingLLM())
    monkeypatch.setattr(rg, "_execute_guarded", FakeExec())
    g = rg.build_graph()
    out = g.invoke({"messages": [HumanMessage(content=Q5)]}, config={"configurable": {"thread_id": "t1"}})
    assert out["route"] == "PERIOD_CHANGE" and "KFC" in out["final_answer"] and out["llm_call_count"] == 0
    assert len(out["turn_evidence"]) == 3


@pytest.mark.parametrize("v,expected", [
    (dt.date(2025, 3, 3), "2025-03-03"), (dt.datetime(2025, 3, 3, 0, 0), "2025-03-03"), ("2025-03-03", "2025-03-03"),
    ("2025-03-03 00:00:00", "2025-03-03"), ("2025-03-03T00:00:00", "2025-03-03"),
    ("2025-03-03'; DROP TABLE x", None), ("2025-03-03 AND 1=1", None), ("03/03/2025", None), (None, None),
])
def test_strict_date_validation(v, expected):
    assert pc._d(v) == expected


def test_ordinary_ranking_question_is_not_hijacked():
    assert pc.parse_period_change("Show me the top 10 restaurants by total revenue") is None
