import os
import unittest
from unittest.mock import patch, MagicMock

from langchain_core.messages import HumanMessage
from src.agent.router_graph import build_graph, etl_node, RouteDecision
from src.agent.sql_safety_guard import guard_sql, SQLGuardError
from src.agent.analyst_graph import sql_safety_guard
from src.agent.analyst_state import AnalystState
from src.tools.etl_tools import extract_from_api, _is_safe_url
import src.agent.router_config as router_cfg


class TestB0Security(unittest.TestCase):
    def test_no_sql_route_in_graph(self):
        """Verify that the legacy un-guarded 'SQL' route no longer exists in compiled graph."""
        graph = build_graph()
        # Verify node names in graph
        nodes = graph.nodes.keys()
        self.assertNotIn("sql", nodes, "Legacy 'sql' node must NOT exist in router graph")
        self.assertIn("analysis", nodes)
        self.assertIn("etl", nodes)
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

    def test_etl_feature_flag_disabled_by_default(self):
        """Verify etl_node refuses execution when ENABLE_ETL_AGENT is False."""
        with patch.object(router_cfg, "ENABLE_ETL_AGENT", False):
            state = {
                "messages": [HumanMessage(content="scrape https://example.com")],
                "language": "en"
            }
            res = etl_node(state)
            self.assertIn("disabled for security reasons", res["raw_answer"])

    def test_etl_feature_flag_disabled_arabic(self):
        """Verify etl_node provides bilingual Arabic message when disabled."""
        with patch.object(router_cfg, "ENABLE_ETL_AGENT", False):
            state = {
                "messages": [HumanMessage(content="استخرج بيانات من الرابط")],
                "language": "ar"
            }
            res = etl_node(state)
            self.assertIn("معطل حالياً لأسباب أمنية", res["raw_answer"])

    def test_ssrf_protection_aws_metadata(self):
        """Verify 169.254.169.254 metadata endpoint is blocked."""
        is_safe, reason = _is_safe_url("http://169.254.169.254/latest/meta-data")
        self.assertFalse(is_safe)
        self.assertIn("SSRF Protection", reason)

    def test_ssrf_protection_loopback(self):
        """Verify localhost / 127.0.0.1 is blocked."""
        is_safe, reason = _is_safe_url("http://127.0.0.1:8000/api")
        self.assertFalse(is_safe)
        self.assertIn("SSRF Protection", reason)

    def test_ssrf_protection_private_network(self):
        """Verify 10.x and 192.168.x are blocked."""
        is_safe, reason = _is_safe_url("http://192.168.1.1/secret")
        self.assertFalse(is_safe)
        self.assertIn("SSRF Protection", reason)

    def test_ssrf_protection_invalid_scheme(self):
        """Verify file:// and gopher:// are blocked."""
        is_safe, reason = _is_safe_url("file:///etc/passwd")
        self.assertFalse(is_safe)
        self.assertIn("Disallowed scheme", reason)

    def test_extract_from_api_tool_blocks_ssrf(self):
        """Verify extract_from_api tool returns security error for SSRF target."""
        res = extract_from_api.invoke({"url": "http://127.0.0.1:8080/test", "output_path": "data/test.csv"})
        self.assertIn("ERROR: Security violation", res)

    def test_extract_from_api_path_traversal(self):
        """Verify extract_from_api blocks path traversal output paths."""
        res = extract_from_api.invoke({"url": "https://example.com/api", "output_path": "../../secret.csv"})
        self.assertIn("Path traversal detected", res)


if __name__ == "__main__":
    unittest.main()
