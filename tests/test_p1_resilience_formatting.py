"""
tests/test_p1_resilience_formatting.py

Regression test suite for:
1. Markdown / LaTeX sanitization (neutralizing $, backslashes, underscores in identifiers, raw HTML).
2. Tabular data formatting (3+ numeric values formatted as Markdown table with right-aligned columns, single values unforced).
3. Referential follow-up routing (routing referential questions with prior evidence to ANALYSIS, refusing to hallucinate in GENERAL).
"""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.markdown_utils import sanitize_for_markdown, format_markdown_table
from src.agent.router_graph import _is_referential_question, router_node, general_node
from src.agent.analyst_state import Evidence


class TestMarkdownSanitization(unittest.TestCase):
    def test_dollar_amount_with_comma_and_underscore(self):
        """
        User requirement 6:
        Add a test with a dollar amount, an underscore in a name, and a number with
        a comma next to a $ sign, asserting the rendered text is not interpreted as LaTeX.
        """
        raw_text = "The total sales for restaurant kfc_east_branch reached $1,921,496, while subway_main had $850,000."
        sanitized = sanitize_for_markdown(raw_text)

        # 1. Underscores in identifiers must be escaped
        self.assertIn("kfc\\_east\\_branch", sanitized)
        self.assertIn("subway\\_main", sanitized)

        # 2. Dollar amounts must be escaped so KaTeX math delimiters are not activated
        self.assertIn("\\$1,921,496", sanitized)
        self.assertIn("\\$850,000", sanitized)

        # 3. Assert no unescaped single or double dollar signs exist
        import re
        unescaped_dollars = re.findall(r"(?<!\\)\$", sanitized)
        self.assertEqual(len(unescaped_dollars), 0, "All dollar signs must be escaped to prevent LaTeX parsing.")

    def test_angle_brackets_not_treated_as_html_tags(self):
        """Angle brackets next to numbers (<5%, <20%) must not break formatting as malformed HTML."""
        raw_text = "All restaurants contributed <5% to the net decline, specifically <20% for top 5 combined."
        sanitized = sanitize_for_markdown(raw_text)
        self.assertIn("&lt;5%", sanitized)
        self.assertIn("&lt;20%", sanitized)

    def test_code_blocks_preserved(self):
        """Code blocks with underscores and dollar signs should remain intact."""
        raw_text = "Run this: `SELECT $column FROM fact_orders` and check."
        sanitized = sanitize_for_markdown(raw_text)
        self.assertIn("`SELECT $column FROM fact_orders`", sanitized)


class TestTabularFormatting(unittest.TestCase):
    def test_three_plus_numeric_values_render_as_table(self):
        """
        User requirement 8:
        Any answer containing 3+ related numeric values must render as a proper Markdown table
        with right-aligned numeric columns (---:) and thousands separators.
        """
        sample_rows = [
            {"restaurant_name": "KFC", "sales_cur": 1921496.0, "sales_prev": 2084094.0, "delta": -162598.0, "pct_change": -7.80, "contribution_to_change": 0.0163},
            {"restaurant_name": "Domino's Pizza", "sales_cur": 1620782.0, "sales_prev": 1746976.0, "delta": -126194.0, "pct_change": -7.22, "contribution_to_change": 0.0126},
            {"restaurant_name": "Pizza Hut", "sales_cur": 1471770.0, "sales_prev": 1581232.0, "delta": -109462.0, "pct_change": -6.92, "contribution_to_change": 0.0109},
        ]
        table_md = format_markdown_table(sample_rows)

        # Check table headers and alignment
        self.assertIn("| Restaurant Name | Current Sales | Prior Sales | Change | % Change | % of Net Change |", table_md)
        self.assertIn("| :--- | ---: | ---: | ---: | ---: | ---: |", table_md)

        # Check formatted numbers with thousands separators and fixed percentages
        self.assertIn("1,921,496", table_md)
        self.assertIn("-162,598", table_md)
        self.assertIn("-7.80%", table_md)
        self.assertIn("1.63%", table_md)

    def test_single_value_does_not_force_table(self):
        """Single-value or < 3 row outputs should NOT generate a table."""
        sample_rows = [{"total_revenue": 42466314}]
        table_md = format_markdown_table(sample_rows)
        self.assertEqual(table_md, "")


class TestReferentialRoutingRegression(unittest.TestCase):
    def test_referential_detection_arabic_and_english(self):
        """Test identification of referential follow-up queries."""
        self.assertTrue(_is_referential_question("النتيجه السابقه ال انت طلعتها لانهي سنه ؟"))
        self.assertTrue(_is_referential_question("دي انهي سنه بالظبط ؟"))
        self.assertTrue(_is_referential_question("Which year was that previous result for?"))
        self.assertTrue(_is_referential_question("What year was that?"))
        self.assertTrue(_is_referential_question("Why did it drop?"))

        # Independent queries should NOT be classified as referential
        self.assertFalse(_is_referential_question("What are the top 5 restaurants by total sales in 2026?"))

    @patch("src.agent.router_graph.router_llm")
    def test_followup_with_prior_evidence_routes_to_analysis(self, mock_router_llm):
        """
        User requirement 5:
        If analyst_state holds a prior Evidence Object and question is referential,
        route to ANALYSIS with evidence in context, even if router LLM fails or votes GENERAL.
        """
        # Simulate router LLM failing or returning GENERAL
        mock_invoker = MagicMock()
        mock_invoker.invoke.side_effect = Exception("Groq 429 TPD limit reached")
        mock_router_llm.with_structured_output.return_value = mock_invoker

        fake_evidence = [
            Evidence(
                query_id="q1",
                executed_sql="SELECT dd.year, dd.month_number FROM fact_orders...",
                columns=("restaurant_name", "sales_cur", "sales_prev"),
                rows=(("KFC", 1921496, 2084094),),
                row_count=1,
                truncated=False,
                executed_at="2026-10-01T13:50:00Z"
            )
        ]

        # Arabic test case
        state_ar = {
            "messages": [
                HumanMessage(content="Why did sales decline?"),
                AIMessage(content="Sales fell by 10M..."),
                HumanMessage(content="النتيجه السابقه ال انت طلعتها لانهي سنه ؟")
            ],
            "evidence": fake_evidence,
            "analyst_memory": {"previous_results": fake_evidence}
        }
        res_ar = router_node(state_ar)
        self.assertEqual(res_ar["route"], "ANALYSIS")
        self.assertGreaterEqual(res_ar["confidence"], 0.90)

        # English test case
        state_en = {
            "messages": [
                HumanMessage(content="Why did sales decline?"),
                AIMessage(content="Sales fell by 10M..."),
                HumanMessage(content="Which year was that for?")
            ],
            "evidence": fake_evidence,
            "analyst_memory": {"previous_results": fake_evidence}
        }
        res_en = router_node(state_en)
        self.assertEqual(res_en["route"], "ANALYSIS")

    def test_general_node_declines_without_evidence_never_fabricates(self):
        """
        User requirement 5:
        If General Assistant is reached with no evidence available, it must decline
        saying context is not available, never fabricating a year like '2022'.
        """
        state_no_evidence_ar = {
            "messages": [HumanMessage(content="النتيجه السابقه ال انت طلعتها لانهي سنه ؟")],
            "language": "ar",
            "evidence": [],
            "analyst_memory": {}
        }
        res_ar = general_node(state_no_evidence_ar)
        self.assertIn("لا توجد نتائج", res_ar["final_answer"])
        self.assertNotIn("2022", res_ar["final_answer"])

        state_no_evidence_en = {
            "messages": [HumanMessage(content="Which year was the previous result for?")],
            "language": "en",
            "evidence": [],
            "analyst_memory": {}
        }
        res_en = general_node(state_no_evidence_en)
        self.assertIn("do not have access to any previous query results", res_en["final_answer"])
        self.assertNotIn("2022", res_en["final_answer"])


if __name__ == "__main__":
    unittest.main()
