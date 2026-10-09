import os
import unittest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage
from src.agent.router_graph import build_graph, RouteDecision
from src.agent.sql_safety_guard import guard_sql, SQLGuardError
from src.agent.analyst_graph import sql_safety_guard
from src.agent.analyst_state import AnalystState
import src.agent.router_config as router_cfg


class TestB0Security(unittest.TestCase):
    def test_no_sql_route_in_graph(self):
        """Verify that the legacy un-guarded 'SQL' route no longer exists in compiled graph."""
        graph = build_graph()
        # Verify node names in graph
        nodes = graph.nodes.keys()
        self.assertNotIn("sql", nodes, "Legacy 'sql' node must NOT exist in router graph")
        self.assertIn("analysis", nodes)
        self.assertNotIn("etl", nodes, "ETL route was removed")
        self.assertNotIn("polish", nodes)
        self.assertIn("general", nodes)

    def test_drop_table_attempt_blocked_by_guard(self):
        """Verify that DROP TABLE is blocked by sql_safety_guard."""
        state: AnalystState = {
            "sql_queries": [
                {
                    "sql": "DROP TABLE workspace.zomato_gold.fact_orders",
                    "purpose": "malicious drop",
                    "validated": False,
                    "validation_issues": [],
                    "attempt": 0,
                    "result": None,
                    "error": None,
                    "limit_injected": False,
                    "error_type": None,
                    "safety_blocked": False
                }
            ]
        }
        res = sql_safety_guard(state)
        q = res["sql_queries"][0]
        self.assertTrue(q["safety_blocked"])
        self.assertEqual(q["error_type"], "guard")
        self.assertIn("Root statement must be SELECT or UNION", q["error"])

    def test_system_schema_attempt_blocked_by_guard(self):
        """Verify that system.information_schema is blocked by sql_safety_guard."""
        state: AnalystState = {
            "sql_queries": [
                {
                    "sql": "SELECT table_name FROM system.information_schema.tables",
                    "purpose": "inspect metadata",
                    "validated": False,
                    "validation_issues": [],
                    "attempt": 0,
                    "result": None,
                    "error": None,
                    "limit_injected": False,
                    "error_type": None,
                    "safety_blocked": False
                }
            ]
        }
        res = sql_safety_guard(state)
        q = res["sql_queries"][0]
        self.assertTrue(q["safety_blocked"])
        self.assertEqual(q["error_type"], "guard")
        self.assertIn("is not in the allowed schema", q["error"])


if __name__ == "__main__":
    unittest.main()
