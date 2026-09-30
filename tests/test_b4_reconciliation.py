import unittest
from src.agent.analyst_graph import result_validator, after_result_validation
from src.agent.analyst_state import AnalystState, Evidence

class TestB4Reconciliation(unittest.TestCase):
    def test_all_null_derived_columns_fail_validator(self):
        """Simulate R1 original failure shape where sales_prev and sales_delta were 100% NULL."""
        # Evidence with NULL derived metrics
        ev = Evidence(
            query_id="q-null-test",
            executed_sql="SELECT restaurant_name, sales_current, sales_prev, sales_delta FROM ...",
            columns=("restaurant_name", "sales_current", "sales_prev", "sales_delta"),
            rows=(
                ("KFC", 1921496.0, None, None),
                ("Domino's Pizza", 1620782.0, None, None),
                ("Pizza Hut", 1471770.0, None, None)
            ),
            row_count=3,
            truncated=False,
            executed_at="2026-09-30T22:00:00Z"
        )
        state: AnalystState = {
            "query_results": [ev],
            "sql_queries": [{"sql": ev.executed_sql, "purpose": "decline drivers", "error": None}],
            "repair_attempts": 0
        }

        res = result_validator(state)
        val = res["result_validation"]
        self.assertFalse(val["valid"], "Validator must fail when derived columns are 100% NULL")
        self.assertEqual(val["severity"], "critical")
        self.assertTrue(any("sales_prev" in issue for issue in val["issues"]))
        self.assertTrue(any("sales_delta" in issue for issue in val["issues"]))

        # Check edge routing
        state.update(res)
        route = after_result_validation(state)
        self.assertEqual(route, "repair", "Failed validation should route to repair when attempts remain")

    def test_total_delta_reconciliation_zero_fails(self):
        """Period comparison where individual deltas exist but total_delta is 0 or NULL must fail."""
        ev = Evidence(
            query_id="q-recon-fail",
            executed_sql="SELECT restaurant_name, delta, total_delta FROM ...",
            columns=("restaurant_name", "delta", "total_delta"),
            rows=(
                ("KFC", -162598.0, 0.0),
                ("Domino's Pizza", -126194.0, 0.0)
            ),
            row_count=2,
            truncated=False,
            executed_at="2026-09-30T22:00:00Z"
        )
        state: AnalystState = {
            "query_results": [ev],
            "sql_queries": [{"sql": ev.executed_sql, "purpose": "period comparison", "error": None}],
            "repair_attempts": 0
        }
        res = result_validator(state)
        val = res["result_validation"]
        self.assertFalse(val["valid"])
        self.assertTrue(any("total_delta is zero" in issue for issue in val["issues"]))

    def test_valid_period_comparison_passes(self):
        """Valid period comparison with non-null metrics and matching totals passes."""
        ev = Evidence(
            query_id="q-valid-pass",
            executed_sql="SELECT restaurant_name, sales_cur, sales_prev, delta, total_delta FROM ...",
            columns=("restaurant_name", "sales_cur", "sales_prev", "delta", "total_delta"),
            rows=(
                ("KFC", 1921496.0, 2084094.0, -162598.0, -10001732.0),
                ("Domino's Pizza", 1620782.0, 1746976.0, -126194.0, -10001732.0)
            ),
            row_count=2,
            truncated=False,
            executed_at="2026-09-30T22:00:00Z"
        )
        state: AnalystState = {
            "query_results": [ev],
            "sql_queries": [{"sql": ev.executed_sql, "purpose": "period comparison", "error": None}],
            "repair_attempts": 0
        }
        res = result_validator(state)
        val = res["result_validation"]
        self.assertTrue(val["valid"])
        self.assertEqual(val["severity"], "ok")

        state.update(res)
        route = after_result_validation(state)
        self.assertEqual(route, "analyze")


if __name__ == "__main__":
    unittest.main()
