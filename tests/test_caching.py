import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import sys
import time
import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import query_cache
from src.agent.analyst_graph import sql_executor
from src.agent.analyst_state import AnalystState

class TestQueryCache(unittest.TestCase):
    
    def setUp(self):
        # Ensure we have a clean test cache dir
        self.test_cache_dir = os.path.join(os.path.dirname(query_cache.__file__), ".test_query_cache")
        query_cache.CACHE_DIR = self.test_cache_dir
        query_cache.clear_memory()
        query_cache.reset_stats()
        if not os.path.exists(self.test_cache_dir):
            os.makedirs(self.test_cache_dir)
            
    def tearDown(self):
        # Clean up test cache
        import shutil
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)

    def test_cache_set_and_get(self):
        sql = "SELECT * FROM fact_orders LIMIT 10"
        columns = ["order_id", "total_amount"]
        rows = [{"order_id": 1, "total_amount": 100}]
        row_count = 1
        executed_at = datetime.datetime.utcnow().isoformat() + "Z"
        
        query_cache.set_cached_result(sql, columns, rows, row_count, executed_at)
        
        cached = query_cache.get_cached_result(sql)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["columns"], columns)
        self.assertEqual(cached["row_count"], row_count)
        self.assertEqual(cached["executed_at"], executed_at)
        
    def test_cache_invalidation_by_dag_marker(self):
        """Real path: the marker file written by the DAG invalidates older cache entries."""
        marker = os.path.join(self.test_cache_dir, "last_ai_run.txt")
        sql = "SELECT * FROM fact_orders"
        with patch.object(query_cache, "AI_RUN_MARKER_PATH", marker):
            # No marker yet: only TTL applies
            query_cache.set_cached_result(sql, [], [], 0, "now")
            self.assertIsNotNone(query_cache.get_cached_result(sql))

            # DAG finishes after the entry was cached -> stale
            future = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
            with open(marker, "w") as f:
                f.write(future.strftime("%Y-%m-%dT%H:%M:%SZ"))
            self.assertIsNone(query_cache.get_cached_result(sql))

    def test_invalidating_timestamp_missing_marker(self):
        with patch.object(query_cache, "AI_RUN_MARKER_PATH", os.path.join(self.test_cache_dir, "nope.txt")):
            self.assertEqual(query_cache._get_invalidating_timestamp({"fact_orders"}), 0.0)

    @patch('src.agent.analyst_graph._db.execute_queries')
    def test_sql_executor_uses_cache(self, mock_execute):
        # Set up cache
        sql = "SELECT 1"
        query_cache.set_cached_result(sql, ["1"], [{"1": 1}], 1, "now")
        
        state = AnalystState(
            sql_queries=[{"sql": sql, "purpose": "test", "validated": True}],
            query_results=[],
        )
        
        result_state = sql_executor(state)
        
        # Database should not be called
        mock_execute.assert_not_called()
        self.assertTrue(result_state.get("cache_hit"))
        self.assertEqual(len(result_state["query_results"]), 1)

if __name__ == '__main__':
    unittest.main()
