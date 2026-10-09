import concurrent.futures
import difflib
import logging
import os
import threading
import time

import databricks.sql
from dotenv import load_dotenv

from src.utils.concurrency import run_parallel
from src.utils.pipeline_marker import read_marker

load_dotenv()

logger = logging.getLogger("database")


class _Lease:
    """A pooled connection checked out of the pool. Use as a context manager:

        with db.get_connection() as conn:   # `conn` is the raw Databricks connection
            ...

    On exit the connection goes back to the pool, unless the block failed with something other than a
    plain SQL error (socket/timeout/auth...), in which case it is discarded because its state is unknown."""

    def __init__(self, owner_cls, conn):
        self.conn = conn
        self._owner = owner_cls
        self._done = False

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        broken = exc_type is not None and not DatabricksUtil._is_sql_error_text(str(exc))
        self.release(broken=broken)

    def discard(self):
        self.release(broken=True)

    def release(self, broken: bool = False):
        if not self._done:
            self._done = True
            self._owner._release(self.conn, broken)

    def __getattr__(self, name):
        if name == "conn":  # not yet set (e.g. during unpickling/copy)
            raise AttributeError(name)
        return getattr(self.conn, name)


class DatabricksUtil:
    # --- Gold schema cache (class-level, shared by every instance) ---
    _schema_cache: dict = {}
    _schema_cache_time: float = 0
    SCHEMA_CACHE_TTL = int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "3600"))
    _schema_lock = threading.Lock()

    # --- Connection pool (class-level) ---
    # `_total` counts open connections plus ones being created, so the pool never exceeds pool_size.
    _pool_cond = threading.Condition(threading.Lock())
    _idle: list = []
    _total = 0

    def __init__(self):
        self.host = os.getenv("DATABRICKS_HOST")
        self.http_path = os.getenv("DATABRICKS_HTTP_PATH")
        self.token = os.getenv("DATABRICKS_TOKEN")
        # We focus on the Gold layer: it is the clean, analysis-ready data.
        self.catalog = "workspace"
        self.schema = "zomato_gold"
        self.statement_timeout = int(os.getenv("STATEMENT_TIMEOUT_SECONDS", "30"))
        self.connect_timeout = int(os.getenv("CONNECT_TIMEOUT_SECONDS", "30"))
        self.pool_size = max(1, int(os.getenv("DB_POOL_SIZE", "3")))
        self.acquire_timeout = int(os.getenv("DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "60"))

    # ------------------------------------------------------------------
    # Pool
    # ------------------------------------------------------------------
    def _do_connect(self):
        return databricks.sql.connect(
            server_hostname=self.host,
            http_path=self.http_path,
            access_token=self.token,
            _socket_timeout=self.connect_timeout
        )

    def _connect_bounded(self):
        """Open one connection, giving up after connect_timeout. No `with` block on the executor:
        its __exit__ would wait for the hung connect and defeat the timeout."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._do_connect)
        try:
            return future.result(timeout=self.connect_timeout)
        except concurrent.futures.TimeoutError:
            raise Exception(f"Timed out acquiring Databricks connection ({self.connect_timeout}s limit)")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def get_connection(self) -> _Lease:
        """Check a connection out of the pool (reusing an idle one, opening a new one while under
        pool_size, otherwise waiting). New connections are opened OUTSIDE the pool lock, so a slow or
        hung connect never blocks callers that could use an idle connection."""
        cls = self.__class__
        deadline = time.time() + self.acquire_timeout
        with cls._pool_cond:
            while True:
                while cls._idle:
                    conn = cls._idle.pop()
                    if getattr(conn, "open", True) is not False:
                        return _Lease(cls, conn)
                    cls._total -= 1  # server closed it while idle
                if cls._total < self.pool_size:
                    cls._total += 1  # reserve the slot, then connect outside the lock
                    break
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise Exception(f"Timed out waiting for a Databricks connection from the pool ({self.acquire_timeout}s)")
                cls._pool_cond.wait(remaining)
        try:
            conn = self._connect_bounded()
        except BaseException:
            with cls._pool_cond:
                cls._total -= 1
                cls._pool_cond.notify()
            raise
        return _Lease(cls, conn)

    @classmethod
    def _release(cls, conn, broken: bool):
        keep = not broken and getattr(conn, "open", True) is not False
        with cls._pool_cond:
            if keep:
                cls._idle.append(conn)
            else:
                cls._total -= 1
            cls._pool_cond.notify()
        if not keep:
            try:
                conn.close()
            except Exception:
                pass

    @classmethod
    def _drain_idle(cls):
        """Drop every idle connection (used after a connection-level error: siblings are likely stale too)."""
        with cls._pool_cond:
            stale, cls._idle = cls._idle, []
            cls._total -= len(stale)
            cls._pool_cond.notify_all()
        for conn in stale:
            try:
                conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Schema (cached, introspected in parallel)
    # ------------------------------------------------------------------
    def _schema_cache_fresh(self) -> bool:
        cls = self.__class__
        if not cls._schema_cache:
            return False
        if time.time() - cls._schema_cache_time >= cls.SCHEMA_CACHE_TTL:
            return False
        # A newer successful pipeline run means the Gold tables may have changed.
        return read_marker() <= cls._schema_cache_time

    def _fetch_schema(self) -> dict:
        qualified = f"{self.catalog}.{self.schema}"
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SHOW TABLES IN {qualified}")
                cols = [desc[0] for desc in cursor.description]
                table_names = [dict(zip(cols, row))["tableName"] for row in cursor.fetchall()]

        def introspect(table_name: str):
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"DESCRIBE {qualified}.{table_name}")
                    desc_cols = [desc[0] for desc in cursor.description]
                    columns = [
                        {"name": c["col_name"], "type": c["data_type"]}
                        for c in (dict(zip(desc_cols, row)) for row in cursor.fetchall())
                        if c["col_name"] and not c["col_name"].startswith("#")  # skip internal comment rows
                    ]
                    cursor.execute(f"SELECT * FROM {qualified}.{table_name} LIMIT 3")
                    sample_cols = [desc[0] for desc in cursor.description]
                    sample = [dict(zip(sample_cols, row)) for row in cursor.fetchall()]
            return table_name, {"columns": columns, "sample": sample}

        # One DESCRIBE + sample per table, several tables at once (bounded by the pool).
        results = run_parallel([(lambda t=t: introspect(t)) for t in table_names], max_workers=self.pool_size)
        return dict(results)

    def get_cached_schema(self) -> dict:
        """Returns {table_name: {columns: [{name, type}], sample: [...]}}.
        Served from cache until the TTL expires or the pipeline marker shows a newer dbt run.
        Concurrent callers share a single refresh."""
        cls = self.__class__
        if self._schema_cache_fresh():
            return cls._schema_cache
        with cls._schema_lock:
            if self._schema_cache_fresh():  # another thread refreshed while we waited
                return cls._schema_cache
            started = time.time()
            try:
                schema_dict = self._fetch_schema()
            except Exception as e:
                return {"error": f"Error fetching schema: {e}"}
            cls._schema_cache = schema_dict
            cls._schema_cache_time = started  # stamped at fetch START so a run finishing mid-fetch still invalidates
            return schema_dict

    def get_schema_details(self) -> str:
        """Human/LLM-readable description of every Gold table: columns, types and 3 sample rows."""
        schema = self.get_cached_schema()
        if "error" in schema:
            return schema["error"]
        context = f"Database Schema Details for {self.catalog}.{self.schema}:\n\n"
        for table_name, data in schema.items():
            context += f"Table Name: {self.catalog}.{self.schema}.{table_name}\n"
            context += "Columns and Data Types:\n"
            for col in data["columns"]:
                context += f" - {col['name']}: {col['type']}\n"
            context += "Sample Data (Top 3 rows):\n"
            for row in data["sample"]:
                context += f" {row}\n"
            context += "-" * 40 + "\n\n"
        return context

    def warm_up(self, background: bool = True):
        """Pay the cold-start cost (warehouse wake-up, first connection, schema introspection) at app
        startup instead of on the first user question. Never raises."""
        def _go():
            try:
                with self.get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")  # wakes a stopped/serverless warehouse
                self.get_cached_schema()
                logger.info("Databricks warm-up complete.")
            except Exception as e:  # noqa: BLE001
                logger.warning("Databricks warm-up failed (will retry lazily on first query): %s", e)

        if not background:
            _go()
            return None
        t = threading.Thread(target=_go, name="db-warmup", daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def execute_sql(self, query: str) -> str:
        """Runs SQL the Agent wrote and returns the result as a string."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    cols = [desc[0] for desc in cursor.description]
                    results = [dict(zip(cols, row)) for row in cursor.fetchall()]
                    return str(results)
        except Exception as e:
            return f"SQL Execution Error: {e}"

    def validate_columns(self, table_name: str, columns: list[str]) -> dict:
        """Validate that columns exist in the given table.
        Returns: {valid: bool, invalid_columns: list[str], suggestions: dict[str, str]}
        Suggestions maps invalid column to closest valid column name."""
        schema = self.get_cached_schema()
        if "error" in schema:
            return {"valid": False, "invalid_columns": columns, "suggestions": {}, "error": schema["error"]}

        if table_name not in schema:
            return {"valid": False, "invalid_columns": columns, "suggestions": {}, "error": f"Table {table_name} not found"}

        valid_columns = [col["name"] for col in schema[table_name]["columns"]]
        invalid_columns = []
        suggestions = {}

        for col in columns:
            if col not in valid_columns:
                invalid_columns.append(col)
                matches = difflib.get_close_matches(col, valid_columns, n=1)
                if matches:
                    suggestions[col] = matches[0]

        return {
            "valid": len(invalid_columns) == 0,
            "invalid_columns": invalid_columns,
            "suggestions": suggestions
        }

    @staticmethod
    def _is_sql_error_text(error_str: str) -> bool:
        error_str = error_str.lower()
        sql_keywords = ["parse", "syntax", "column", "table", "not found", "unresolved", "analysisexception"]
        return any(kw in error_str for kw in sql_keywords)

    def _is_sql_error(self, error_str: str) -> bool:
        return self._is_sql_error_text(error_str)

    def _run_one(self, query_info: dict) -> dict:
        """Execute a single query on its own pooled connection. Never raises: failures are reported in the result."""
        sql = query_info.get("sql", "")
        result_item = {
            "purpose": query_info.get("purpose", ""),
            "sql": sql,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": None,
            "error_type": None,
            "success": False
        }

        def attempt():
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql)
                    if cursor.description:
                        cols = [desc[0] for desc in cursor.description]
                        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
                        result_item["columns"] = cols
                        result_item["rows"] = rows
                        result_item["row_count"] = len(rows)
                    result_item["success"] = True

        try:
            attempt()
        except Exception as e:
            error_str = str(e)
            if self._is_sql_error(error_str):
                result_item["error"] = f"SQL Error: {error_str}"
                result_item["error_type"] = "sql"
            else:
                # Connection-level failure: the lease already discarded the broken connection. Idle
                # siblings are probably just as stale, so drop them and retry once on a fresh one.
                try:
                    self._drain_idle()
                    attempt()
                except Exception as retry_e:
                    retry_err_str = str(retry_e)
                    if self._is_sql_error(retry_err_str) or "timeout" in retry_err_str.lower():
                        result_item["error"] = f"SQL or Timeout Error: {retry_err_str}"
                        result_item["error_type"] = "sql" if self._is_sql_error(retry_err_str) else "timeout"
                    else:
                        result_item["error"] = "Database Connection Error (Please try again). Details: " + retry_err_str
                        result_item["error_type"] = "connection"
        return result_item

    def execute_queries(self, queries: list[dict]) -> list[dict]:
        """Runs independent queries concurrently (bounded by the pool). Results keep the input order."""
        return run_parallel([(lambda q=q: self._run_one(q)) for q in queries], max_workers=self.pool_size)

# Run this file directly to check the connection and print the schema
if __name__ == "__main__":
    db = DatabricksUtil()
    print("Fetching Schema Details... Please wait.")
    schema_info = db.get_schema_details()
    print(schema_info)
