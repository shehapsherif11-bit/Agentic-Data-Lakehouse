"""P1-4: the diagnosis answers the metric that was ranked (orders, not revenue), and never over-claims at tiny sample sizes."""
import pytest

from src.agent import diagnostics as dx
from src.agent.followups import ranked_metric, run_followup
from tests.test_diagnosis_followups import PLATFORM, FakeWarehouse, ev, target_row

# Q3 replay: three restaurants with 12-13 orders; two have a small basket (the old answer blamed it for FEW ORDERS)
Q3_TARGETS = [target_row("K2H Meals", 12, aov=214.0, cancel=0.0),
              target_row("SAI DURGA CHAT CENTRE", 12, aov=149.0, cancel=0.0),
              target_row("Banglar Rannaghar", 13, aov=541.0, cancel=0.077)]
BOTTOM = [("K2H Meals", 12), ("SAI DURGA CHAT CENTRE", 12), ("Banglar Rannaghar", 13)]


def diagnose(metric_col, targets=Q3_TARGETS, lang="en"):
    return run_followup("diagnose", "why?", ev(BOTTOM, metric_col), lang, FakeWarehouse(targets))["answer"]


# ------------------------------------------------------------------ F4: metric awareness
@pytest.mark.parametrize("col,expected", [("total_orders", "orders"), ("Total Orders", "orders"), ("order_count", "orders"),
                                          ("revenue", "revenue"), ("aov", "revenue"), ("avg_order_value", "revenue"),
                                          ("orders_per_customer", "revenue"), ("avg_rating", "revenue")])
def test_ranked_metric_detection(col, expected):
    assert ranked_metric(ev(BOTTOM, col)) == expected


def test_ranked_metric_defaults_to_revenue_without_a_table():
    assert ranked_metric([]) == "revenue"


def test_orders_diagnosis_never_blames_the_basket():
    a = diagnose("total_orders")
    assert "orders = branches × orders per branch" in a
    assert "Small basket" not in a and "upsell" not in a.lower()           # the old answer's strongest signal and first action
    assert "Basket" not in a.split("### The numbers")[0]                  # no basket column in the decomposition table
    assert "Main factor" in a and "orders per branch" in a


def test_revenue_diagnosis_still_uses_the_basket():
    a = diagnose("revenue")
    assert "revenue = branches × orders per branch × average order value" in a
    assert "Small basket" in a


def test_orders_driver_ignores_the_basket_factor():
    r = dx.analyse_entity(Q3_TARGETS[0], PLATFORM, None, None, metric="orders")
    assert r["driver"] in ("branches", "orders_per_branch")
    assert dx.analyse_entity(Q3_TARGETS[0], PLATFORM, None, None, metric="revenue")["factors"]["aov"] < 0.7


def test_high_price_is_still_a_valid_orders_cause():
    t = target_row("Pricey", 20, aov=900.0, cancel=0.0)
    keys = {f["key"] for f in dx.analyse_entity(t, PLATFORM, None, None, metric="orders")["findings"]}
    assert "high_price" in keys            # a price barrier can reduce the NUMBER of orders


# ------------------------------------------------------------------ F5: confidence and small samples
def test_wilson_lower_bound():
    assert dx.wilson_lower(2, 12) < 0.085          # 2 failed of 12 is compatible with the platform rate
    assert dx.wilson_lower(60, 300) > 0.085        # 20% of 300 clearly is not
    assert dx.wilson_lower(0, 0) == 0.0


def test_two_failed_orders_out_of_twelve_is_not_reported_as_a_cause():
    t = target_row("Tiny", 12, cancel=0.1667)      # the Q3 "16.7% failed" case
    keys = {f["key"] for f in dx.analyse_entity(t, PLATFORM, None, None)["findings"]}
    assert "failed_orders" not in keys


def test_failed_orders_at_a_real_sample_size_are_still_reported():
    t = target_row("Big", 300, cancel=0.20)
    keys = {f["key"] for f in dx.analyse_entity(t, PLATFORM, None, None)["findings"]}
    assert "failed_orders" in keys


def test_failed_orders_threshold_at_exactly_30_orders_uses_plain_rule():
    t = target_row("Edge", 30, cancel=0.20)
    keys = {f["key"] for f in dx.analyse_entity(t, PLATFORM, None, None)["findings"]}
    assert "failed_orders" in keys


def test_peer_comparison_is_context_not_a_high_confidence_root_cause():
    a = diagnose("total_orders")
    causes = a.split("### Likely root causes")[1].split("###")[0] if "### Likely root causes" in a else ""
    assert "same city" not in causes.lower()
    assert "describes the gap, does not explain it" in a


def test_no_high_confidence_claim_in_a_tiny_sample_answer():
    a = diagnose("total_orders")
    assert "Confidence: high" not in a
    assert "low-confidence" in a or "Order counts are very small" in a


def test_arabic_orders_diagnosis():
    a = diagnose("total_orders", lang="ar")
    assert "عدد الطلبات = الفروع × الطلبات لكل فرع" in a and "upsell" not in a.lower()
    assert "وصف للفجوة، مش تفسير لها" in a


def test_orders_diagnosis_does_not_list_basket_as_checked_and_normal():
    a = diagnose("total_orders")
    checked = a.split("Checked and normal (not the cause):**")[1].split("\n")[0]
    assert "basket value" not in checked
