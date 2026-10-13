"""
demo_database.py - the backend of the public live demo.

Same interface as DatabricksUtil (execute_queries, warm_up, get_cached_schema), but the data is a small frozen
sample of the Gold layer (Parquet files) loaded into an in-memory DuckDB. No warehouse, no credentials, no cost.

Everything above this layer is unchanged: the agent still writes Databricks-dialect SQL against
workspace.zomato_gold.*, the AST guard still checks it, and the query cache still stores the result. This
module only changes where the already-guarded SQL runs. It is selected with DEMO_MODE=1 (see analyst_graph.py).
"""
import json
import logging
import os
import threading
import time
from pathlib import Path

import duckdb
import sqlglot
import sqlglot.expressions as exp

from src.utils.concurrency import run_parallel

logger = logging.getLogger("demo_database")

CATALOG = "workspace"
SCHEMA = "zomato_gold"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "demo" / "sample_data"


def to_duckdb(sql: str) -> str:
    """Databricks SQL -> DuckDB SQL. Needed for dialect details such as string escaping (the agent writes
    'McDonald\\'s' for Databricks, DuckDB needs 'McDonald''s') and for DAYOFWEEK, which counts from 1 = Sunday
    in Databricks but from 0 = Sunday in DuckDB; sqlglot does not translate that difference."""
    tree = sqlglot.parse_one(sql, read="databricks")
    for node in list(tree.find_all(exp.DayOfWeek)):
        node.replace(exp.Paren(this=exp.Add(this=node.copy(), expression=exp.Literal.number(1))))
    return tree.sql(dialect="duckdb")


class DemoDuckDBUtil:
    _load_lock = threading.Lock()
    _con = None            # one in-memory database per process
    _data_dir = None
    _manifest: dict = {}

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or os.getenv("DEMO_DATA_DIR") or DEFAULT_DATA_DIR)
        self.catalog = CATALOG
        self.schema = SCHEMA
        self.statement_timeout = int(os.getenv("STATEMENT_TIMEOUT_SECONDS", "30"))
        self.pool_size = max(1, int(os.getenv("DB_POOL_SIZE", "3")))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> duckdb.DuckDBPyConnection:
        cls = self.__class__
        if cls._con is not None and cls._data_dir == self.data_dir:
            return cls._con
        with cls._load_lock:
            if cls._con is not None and cls._data_dir == self.data_dir:
                return cls._con
            files = sorted(self.data_dir.glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No sample data found in {self.data_dir}")
            started = time.time()
            con = duckdb.connect(":memory:")
            # Databricks sorts NULLs first on ASC and last on DESC. DuckDB's default puts them last on both.
            con.execute("SET GLOBAL default_null_order = 'nulls_first_on_asc_last_on_desc'")
            # Databricks warehouses default to UTC; "the date of an order" must not depend on the server's zone.
            con.execute("SET GLOBAL TimeZone = 'UTC'")
            con.execute(f"ATTACH ':memory:' AS {CATALOG}")
            con.execute(f"CREATE SCHEMA {CATALOG}.{SCHEMA}")
            for f in files:
                con.execute(
                    f"CREATE TABLE {CATALOG}.{SCHEMA}.{f.stem} AS SELECT * FROM read_parquet(?)", [f.as_posix()]
                )
            manifest_path = self.data_dir / "manifest.json"
            cls._manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            cls._con, cls._data_dir = con, self.data_dir
            logger.info("Demo sample loaded: %d tables in %.1fs", len(files), time.time() - started)
            return con

    def manifest(self) -> dict:
        """What the sample is (size, date range, how it was drawn). Shown in the UI so nobody mistakes it for
        the full dataset."""
        self._ensure_loaded()
        return dict(self.__class__._manifest)

    def warm_up(self, background: bool = True):
        def _go():
            try:
                self._ensure_loaded()
            except Exception as e:  # noqa: BLE001
                logger.warning("Demo warm-up failed (will retry lazily on first query): %s", e)

        if not background:
            _go()
            return None
        t = threading.Thread(target=_go, name="demo-warmup", daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def get_cached_schema(self) -> dict:
        """{table: {columns: [{name, type}], sample: [3 rows]}} - the same shape DatabricksUtil returns."""
        try:
            con = self._ensure_loaded().cursor()
            tables = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = ? AND table_schema = ? "
                "ORDER BY table_name", [CATALOG, SCHEMA]).fetchall()]
            out = {}
            for t in tables:
                cols = con.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_catalog = ? AND table_schema = ? AND table_name = ? ORDER BY ordinal_position",
                    [CATALOG, SCHEMA, t]).fetchall()
                cur = con.execute(f"SELECT * FROM {CATALOG}.{SCHEMA}.{t} LIMIT 3")
                names = [d[0] for d in cur.description]
                out[t] = {
                    "columns": [{"name": n, "type": ty} for n, ty in cols],
                    "sample": [dict(zip(names, row)) for row in cur.fetchall()],
                }
            return out
        except Exception as e:  # noqa: BLE001
            return {"error": f"Error fetching schema: {e}"}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def _run_one(self, query_info: dict) -> dict:
        """Run one already-guarded query. Never raises: failures are reported in the result, with the same
        error_type values the analyst graph already routes on ('sql' -> repair, 'timeout' -> stop)."""
        sql = query_info.get("sql", "")
        result = {
            "purpose": query_info.get("purpose", ""), "sql": sql, "columns": [], "rows": [],
            "row_count": 0, "error": None, "error_type": None, "success": False,
        }
        cursor = None
        timer = None
        try:
            translated = to_duckdb(sql)
            cursor = self._ensure_loaded().cursor()
            timer = threading.Timer(self.statement_timeout, cursor.interrupt)
            timer.start()
            cursor.execute(translated)
            if cursor.description:
                cols = [d[0] for d in cursor.description]
                rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                result.update(columns=cols, rows=rows, row_count=len(rows))
            result["success"] = True
        except duckdb.InterruptException as e:
            result.update(error=f"SQL or Timeout Error: {e}", error_type="timeout")
        except (duckdb.Error, sqlglot.errors.SqlglotError) as e:
            result.update(error=f"SQL Error: {e}", error_type="sql")
        except Exception as e:  # noqa: BLE001
            result.update(error=f"Demo database error: {e}", error_type="connection")
        finally:
            if timer is not None:
                timer.cancel()
            if cursor is not None:
                cursor.close()
        return result

    def execute_queries(self, queries: list[dict]) -> list[dict]:
        """Runs independent queries concurrently. Results keep the input order."""
        return run_parallel([(lambda q=q: self._run_one(q)) for q in queries], max_workers=self.pool_size)
