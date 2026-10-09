import unittest
import time
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.utils.database import DatabricksUtil


class TestDBHang(unittest.TestCase):
    def setUp(self):
        DatabricksUtil._idle = []
        DatabricksUtil._total = 0

    @patch('databricks.sql.connect')
    def test_connection_timeout_does_not_block(self, mock_connect):
        # Connect hangs far longer than the timeout; the caller must return at ~timeout, not at 5s.
        def hanging_connect(*args, **kwargs):
            time.sleep(5)
            return MagicMock()
        mock_connect.side_effect = hanging_connect

        db = DatabricksUtil()
        db.connect_timeout = 1

        start = time.time()
        with self.assertRaises(Exception) as ctx:
            db.get_connection()
        elapsed = time.time() - start

        self.assertIn("Timed out acquiring", str(ctx.exception))
        self.assertLess(elapsed, 3, f"get_connection blocked {elapsed:.1f}s past its 1s timeout")

        # The pool lock is free and the reserved slot was returned (a hung connect must not leak capacity)
        lock_acquired = DatabricksUtil._pool_cond.acquire(blocking=False)
        self.assertTrue(lock_acquired, "Pool lock was not released after a connection timeout!")
        if lock_acquired:
            DatabricksUtil._pool_cond.release()
        self.assertEqual(DatabricksUtil._total, 0)


if __name__ == '__main__':
    unittest.main()
