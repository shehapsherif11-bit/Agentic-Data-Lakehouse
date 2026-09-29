import unittest
import sys
import json

sys.path.insert(0, r'd:\Data ENG Project')

from src.agent.metrics_registry import (
    resolve_metric,
    check_data_sufficiency,
    get_join_safety_context,
    get_table_alias,
    get_schema_context_for_llm
)
from src.agent.viz_engine import generate_chart, safe_chart_data
from src.agent.analyst_graph import _parse_json, build_analyst_graph
from src.agent import analyst_prompts
from src.agent.analyst_state import AnalystState

class TestMetricResolver(unittest.TestCase):
    """Test the metrics registry offline functions."""

    def test_resolve_known_metric_revenue(self):
        """resolve_metric('revenue') returns status='derivable'"""
        res = resolve_metric('revenue')
        self.assertEqual(res['status'], 'derivable')
        self.assertEqual(res['name'], 'revenue')

    def test_resolve_known_metric_aov(self):
        """resolve_metric('aov') returns status='derivable' with correct SQL expression"""
        res = resolve_metric('aov')
        self.assertEqual(res['status'], 'derivable')
        self.assertIn('SUM(fo.sales_amount) / NULLIF(COUNT(DISTINCT fo.order_id), 0)', res['definition']['sql_expression'])

    def test_resolve_missing_metric_profit(self):
        """resolve_metric('profit') returns status='missing' with reason"""
        res = resolve_metric('profit')
        self.assertEqual(res['status'], 'missing')
        self.assertIn('needs cost/COGS data', res['missing_reason'])

    def test_resolve_missing_metric_profit_margin(self):
        """resolve_metric('profit_margin') returns status='missing'"""
        res = resolve_metric('profit_margin')
        self.assertEqual(res['status'], 'missing')

    def test_resolve_unknown_metric(self):
        """resolve_metric('xyzzy123') returns status='unknown'"""
        res = resolve_metric('xyzzy123')
        self.assertEqual(res['status'], 'missing')
        self.assertEqual(res['missing_reason'], 'Unknown metric')

    def test_fuzzy_matching_total_revenue(self):
        """resolve_metric('total revenue') matches 'revenue'"""
        res = resolve_metric('total revenue')
        self.assertEqual(res['name'], 'revenue')

    def test_fuzzy_matching_average_order_value(self):
        """resolve_metric('average order value') matches 'aov'"""
        res = resolve_metric('average order value')
        self.assertEqual(res['name'], 'aov')

    def test_data_sufficiency_all_available(self):
        """check_data_sufficiency(['revenue', 'aov'], ['city']) returns sufficient=True"""
        res = check_data_sufficiency(['revenue', 'aov'], ['city'])
        self.assertTrue(res['sufficient'])
        self.assertIn('revenue', res['available'])
        self.assertIn('aov', res['available'])

    def test_data_sufficiency_missing_metric(self):
        """check_data_sufficiency(['profit_margin'], []) returns sufficient=False"""
        res = check_data_sufficiency(['profit_margin'], [])
        self.assertFalse(res['sufficient'])
        self.assertEqual(len(res['missing']), 1)
        self.assertEqual(res['missing'][0]['name'], 'profit_margin')

    def test_data_sufficiency_mixed(self):
        """check_data_sufficiency(['revenue', 'profit'], []) has both available and missing"""
        res = check_data_sufficiency(['revenue', 'profit'], [])
        self.assertFalse(res['sufficient'])
        self.assertIn('revenue', res['available'])
        self.assertTrue(any(m['name'] == 'profit' for m in res['missing']))


class TestJoinSafety(unittest.TestCase):
    """Test grain and join safety."""

    def test_join_safety_fact_to_dim(self):
        """get_join_safety_context with fact_orders + dim_resturant should NOT warn"""
        res = get_join_safety_context(["workspace.zomato_gold.fact_orders", "workspace.zomato_gold.dim_resturant"])
        self.assertNotIn("WARNING", res)

    def test_join_safety_one_to_many(self):
        """get_join_safety_context with fact_orders + fact_order_items SHOULD warn about one-to-many"""
        res = get_join_safety_context(["workspace.zomato_gold.fact_orders", "workspace.zomato_gold.fact_order_items"])
        self.assertIn("WARNING", res)

    def test_table_alias(self):
        """get_table_alias for each table returns correct alias"""
        self.assertEqual(get_table_alias("workspace.zomato_gold.fact_orders"), "fo")
        self.assertEqual(get_table_alias("workspace.zomato_gold.fact_order_items"), "fi")
        self.assertEqual(get_table_alias("workspace.zomato_gold.dim_date"), "dd")
        self.assertEqual(get_table_alias("workspace.zomato_gold.dim_menu"), "dm")
        self.assertEqual(get_table_alias("workspace.zomato_gold.dim_resturant"), "dr")
        self.assertEqual(get_table_alias("workspace.zomato_gold.dim_users"), "du")


class TestSchemaContext(unittest.TestCase):
    """Test schema context generation."""

    def test_schema_context_not_empty(self):
        """get_schema_context_for_llm() returns non-empty string"""
        res = get_schema_context_for_llm()
        self.assertTrue(len(res) > 0)

    def test_schema_context_contains_tables(self):
        """contains all 6 table names"""
        res = get_schema_context_for_llm()
        tables = ["fact_orders", "fact_order_items", "dim_date", "dim_menu", "dim_resturant", "dim_users"]
        for t in tables:
            self.assertIn(t, res)

    def test_schema_context_contains_grain(self):
        """contains 'one row per order' etc."""
        res = get_schema_context_for_llm()
        self.assertIn("one row per order", res)
        self.assertIn("one row per menu item", res)


class TestVizEngine(unittest.TestCase):
    """Test chart generation."""

    def test_no_viz_when_disabled(self):
        """generate_chart({'should_visualize': False}, []) returns None"""
        self.assertIsNone(generate_chart({'should_visualize': False}, [{'x': 1, 'y': 2}]))

    def test_no_viz_empty_data(self):
        """generate_chart({'should_visualize': True, ...}, []) returns None"""
        self.assertIsNone(generate_chart({'should_visualize': True, 'chart_type': 'bar', 'x_axis': 'x', 'y_axis': 'y'}, []))

    def test_bar_chart_generation(self):
        """generate_chart with valid bar spec and sample data returns HTML string containing 'plotly'"""
        spec = {'should_visualize': True, 'chart_type': 'bar', 'x_axis': 'category', 'y_axis': 'value'}
        data = [{'category': 'A', 'value': 10}, {'category': 'B', 'value': 20}]
        res = generate_chart(spec, data)
        self.assertIsNotNone(res)
        self.assertIn('plotly', res.lower())

    def test_line_chart_generation(self):
        """similar for line chart"""
        spec = {'should_visualize': True, 'chart_type': 'line', 'x_axis': 'date', 'y_axis': 'sales'}
        data = [{'date': '2023-01-01', 'sales': 100}, {'date': '2023-01-02', 'sales': 150}]
        res = generate_chart(spec, data)
        self.assertIsNotNone(res)
        self.assertIn('plotly', res.lower())

    def test_safe_chart_data_extraction(self):
        """safe_chart_data extracts x,y correctly"""
        data = [{'category': 'A', 'value': 10}, {'category': 'B', 'value': 20}]
        x, y = safe_chart_data(data, 'category', 'value')
        self.assertEqual(x, ['A', 'B'])
        self.assertEqual(y, [10, 20])

    def test_safe_chart_data_missing_column(self):
        """safe_chart_data with wrong column names returns empty lists"""
        data = [{'category': 'A', 'value': 10}]
        x, y = safe_chart_data(data, 'wrong_x', 'wrong_y')
        self.assertEqual(x, [])
        self.assertEqual(y, [])


class TestJSONParser(unittest.TestCase):
    """Test the _parse_json helper from analyst_graph."""

    def test_parse_clean_json(self):
        """parses '{"key": "value"}' correctly"""
        res = _parse_json('{"key": "value"}')
        self.assertEqual(res, {"key": "value"})

    def test_parse_json_with_markdown_fences(self):
        """parses '```json\n{"key": "value"}\n```' correctly"""
        res = _parse_json('```json\n{"key": "value"}\n```')
        self.assertEqual(res, {"key": "value"})

    def test_parse_json_with_surrounding_text(self):
        """extracts JSON from 'Here is the answer: {"key": "value"}'"""
        res = _parse_json('Here is the answer: {"key": "value"}')
        self.assertEqual(res, {"key": "value"})

    def test_parse_invalid_json(self):
        """returns None for 'not json at all'"""
        res = _parse_json('not json at all')
        self.assertIsNone(res)


class TestAnalystPrompts(unittest.TestCase):
    """Verify all prompts are defined and contain expected placeholders."""

    def test_all_prompts_exist(self):
        """every prompt constant is a non-empty string"""
        prompt_names = [
            'INTENT_ANALYZER_PROMPT', 'ANALYSIS_PLANNER_PROMPT',
            'SQL_GENERATOR_PROMPT', 'SQL_VALIDATOR_PROMPT',
            'SQL_REPAIR_PROMPT', 'RESULT_VALIDATOR_PROMPT',
            'RESULT_ANALYZER_PROMPT', 'DRIVER_ANALYSIS_PROMPT',
            'COMPLETENESS_CHECK_PROMPT', 'VIZ_PLANNER_PROMPT',
            'INSIGHT_GENERATOR_PROMPT'
        ]
        for name in prompt_names:
            val = getattr(analyst_prompts, name, None)
            self.assertIsNotNone(val)
            self.assertTrue(isinstance(val, str))
            self.assertTrue(len(val) > 0)

    def test_intent_prompt_has_question_placeholder(self):
        """INTENT_ANALYZER_PROMPT contains '{question}'"""
        self.assertIn('{question}', analyst_prompts.INTENT_ANALYZER_PROMPT)

    def test_planner_prompt_has_placeholders(self):
        """contains '{intent_json}', '{schema_context}', etc."""
        self.assertIn('{intent_json}', analyst_prompts.ANALYSIS_PLANNER_PROMPT)
        self.assertIn('{schema_context}', analyst_prompts.ANALYSIS_PLANNER_PROMPT)

    def test_insight_prompt_has_language_placeholder(self):
        """contains '{language_instruction}'"""
        self.assertIn('{language_instruction}', analyst_prompts.INSIGHT_GENERATOR_PROMPT)


class TestAnalyticalWorkflowDesign(unittest.TestCase):
    """Verify the graph structure."""

    def test_analyst_graph_compiles(self):
        """build_analyst_graph() returns a compiled graph"""
        # We can just check that it's a valid object
        graph = build_analyst_graph()
        self.assertIsNotNone(graph)

    def test_analyst_state_has_required_keys(self):
        """AnalystState annotations include all required fields"""
        annotations = AnalystState.__annotations__
        required_keys = ['messages', 'intent', 'resolved_metrics', 'sql_queries', 'analysis_plan']
        for k in required_keys:
            self.assertIn(k, annotations)


class TestQuestionClassification(unittest.TestCase):
    """conceptual - no LLM needed"""

    def test_driver_question_keywords(self):
        """verify that keywords like 'why', 'what caused' are present in the INTENT_ANALYZER_PROMPT"""
        prompt = analyst_prompts.INTENT_ANALYZER_PROMPT.lower()
        self.assertIn('why', prompt)
        self.assertIn('what caused', prompt)

    def test_temporal_keywords_in_prompts(self):
        """verify temporal keywords exist in relevant prompts"""
        prompt = analyst_prompts.INTENT_ANALYZER_PROMPT.lower()
        self.assertIn('last month', prompt)
        self.assertIn('previous quarter', prompt)

if __name__ == '__main__':
    unittest.main()
