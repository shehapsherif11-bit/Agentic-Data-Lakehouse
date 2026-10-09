import threading
import time

import pytest

from src.agent import query_cache
from src.utils.database import DatabricksUtil


# ---------- fakes ----------

class FakeCursor:
    def __init__(self, conn):
        self.conn, self.description, self._rows = conn, None, []

    def __enter__(self): return self
    def __exit__(self, *a): pass

    def execute(self, sql):
        self.conn.log.append(sql)
        if self.conn.delay:
            time.sleep(self.conn.delay)
        if self.conn.fail_next:
            self.conn.fail_next -= 1
            raise Exception("socket closed")
        up = sql.upper()
        if up.startswith("SHOW TABLES"):
            self.description, self._rows = [("tableName",)], [(t,) for t in self.conn.tables]
        elif up.startswith("DESCRIBE"):
            self.description, self._rows = [("col_name",), ("data_type",)], [("id", "int"), ("# Partition", "")]
        elif "BADCOL" in up:
            raise Exception("AnalysisException: column not found")
        else:
            self.description, self._rows = [("v",)], [(sql,)]

    def fetchall(self): return self._rows


class FakeConn:
    def __init__(self, factory):
        self.factory, self.open = factory, True
        self.log, self.delay, self.fail_next, self.tables = factory.log, factory.delay, 0, factory.tables

    def cursor(self): return FakeCursor(self)
    def close(self): self.open = False


class Factory:
    def __init__(self, delay=0.0, tables=("t1", "t2", "t3", "t4")):
        self.log, self.delay, self.tables, self.created = [], delay, tables, []

    def __call__(self, *a, **k):
        c = FakeConn(self)
        c.delay = self.delay          # picks up delay changes made after the pool was created
        self.created.append(c)
        return c


def _reset_db_state():
    DatabricksUtil._idle, DatabricksUtil._total = [], 0
    DatabricksUtil._schema_cache, DatabricksUtil._schema_cache_time = {}, 0


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "3")
    _reset_db_state()
    factory = Factory()
    monkeypatch.setattr("databricks.sql.connect", factory)
    monkeypatch.setattr("src.utils.database.read_marker", lambda *a: 0.0)
    d = DatabricksUtil()
    d.factory = factory
    yield d
    _reset_db_state()


# ---------- Step 11: pool ----------

def test_connection_is_reused_not_reopened(db):
    for _ in range(5):
        with db.get_connection() as conn:
            conn.cursor().execute("SELECT 1")
    assert len(db.factory.created) == 1


def test_pool_never_exceeds_size_and_waiters_get_released_connections(db):
    db.acquire_timeout = 5
    leases = [db.get_connection() for _ in range(3)]
    got = []
    t = threading.Thread(target=lambda: got.append(db.get_connection()))
    t.start()
    time.sleep(0.2)
    assert not got and len(db.factory.created) == 3   # 4th caller blocks, no 4th connection
    leases[0].release()
    t.join(2)
    assert got and len(db.factory.created) == 3        # woke up with the released connection


def test_pool_acquire_times_out_when_exhausted(db):
    db.acquire_timeout = 0.3
    held = [db.get_connection() for _ in range(3)]
    with pytest.raises(Exception, match="pool"):
        db.get_connection()
    for h in held:
        h.release()


def test_execute_queries_run_concurrently_and_keep_order(db):
    db.factory.delay = 0.3
    queries = [{"sql": f"SELECT {i}", "purpose": f"p{i}"} for i in range(3)]
    start = time.time()
    out = db.execute_queries(queries)
    assert time.time() - start < 0.8                      # serial would be ~0.9s
    assert [r["purpose"] for r in out] == ["p0", "p1", "p2"]
    assert [r["rows"][0]["v"] for r in out] == ["SELECT 0", "SELECT 1", "SELECT 2"]
    assert all(r["success"] for r in out)


def test_sql_error_is_reported_and_connection_kept(db):
    out = db.execute_queries([{"sql": "SELECT BADCOL", "purpose": "x"}])
    assert out[0]["error_type"] == "sql" and not out[0]["success"]
    with db.get_connection():
        pass
    assert len(db.factory.created) == 1                    # SQL errors don't poison the connection


def test_connection_error_discards_connection_and_retries_on_a_fresh_one(db):
    with db.get_connection() as conn:                      # seed the pool with one connection that will fail
        conn.fail_next = 1
    out = db.execute_queries([{"sql": "SELECT 1", "purpose": "x"}])
    assert out[0]["success"] is True
    assert len(db.factory.created) == 2 and db.factory.created[0].open is False


def test_persistent_connection_error_is_classified(db, monkeypatch):
    def always_fail(*a, **k):
        raise Exception("connection reset by peer")
    monkeypatch.setattr("databricks.sql.connect", always_fail)
    out = db.execute_queries([{"sql": "SELECT 1", "purpose": "x"}])
    assert out[0]["error_type"] == "connection"
    assert DatabricksUtil._total == 0


# ---------- Step 12: schema cache + warm-up ----------

def test_schema_introspection_is_parallel_and_correct(db):
    db.factory.delay = 0.2
    start = time.time()
    schema = db.get_cached_schema()
    elapsed = time.time() - start
    assert set(schema) == {"t1", "t2", "t3", "t4"}
    assert schema["t1"]["columns"] == [{"name": "id", "type": "int"}]    # '#' rows dropped
    assert elapsed < 1.2   # 1 SHOW + 4 tables x 2 queries serially = 1.8s


def test_schema_is_cached_then_invalidated_by_pipeline_marker(db, monkeypatch):
    db.get_cached_schema()
    n = len(db.factory.log)
    db.get_cached_schema()
    assert len(db.factory.log) == n                                       # cache hit: no queries
    monkeypatch.setattr("src.utils.database.read_marker", lambda *a: time.time() + 10)
    db.get_cached_schema()
    assert len(db.factory.log) > n                                        # newer dbt run -> refreshed


def test_concurrent_schema_refreshes_share_one_fetch(db):
    db.factory.delay = 0.1
    threads = [threading.Thread(target=db.get_cached_schema) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for s in db.factory.log if s.upper().startswith("SHOW TABLES")) == 1


def test_warm_up_opens_connection_and_loads_schema_in_background(db):
    t = db.warm_up(background=True)
    t.join(3)
    assert DatabricksUtil._schema_cache and any(s == "SELECT 1" for s in db.factory.log)


def test_warm_up_never_raises(monkeypatch, db):
    def boom(*a, **k):
        raise Exception("warehouse down")
    monkeypatch.setattr("databricks.sql.connect", boom)
    db.warm_up(background=False)   # must swallow


# ---------- Step 13: query cache ----------

@pytest.fixture
def qc(tmp_path, monkeypatch):
    monkeypatch.setattr(query_cache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(query_cache, "AI_RUN_MARKER_PATH", str(tmp_path / "marker.txt"))
    query_cache.clear_memory()
    query_cache.reset_stats()
    yield query_cache
    query_cache.clear_memory()


def test_hit_served_from_memory_without_touching_disk(qc, tmp_path):
    qc.set_cached_result("SELECT 1", ["a"], [{"a": 1}], 1, "now")
    for f in tmp_path.glob("*.json"):
        f.unlink()
    assert qc.get_cached_result("SELECT 1")["row_count"] == 1             # LRU layer
    assert qc.get_stats()["hits"] == 1


def test_stats_count_hits_misses_and_rate(qc):
    qc.get_cached_result("SELECT 2")
    qc.set_cached_result("SELECT 2", [], [], 0, "now")
    qc.get_cached_result("SELECT 2")
    s = qc.get_stats()
    assert (s["hits"], s["misses"], s["stores"]) == (1, 1, 1) and s["hit_rate"] == 0.5


def test_write_is_atomic_no_tmp_files_left(qc, tmp_path):
    qc.set_cached_result("SELECT 3", ["a"], [{"a": 1}], 1, "now")
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_concurrent_readers_never_see_partial_files(qc):
    rows = [{"a": i} for i in range(2000)]
    qc.set_cached_result("SELECT 4", ["a"], rows, 2000, "now")
    errors = []

    def writer():
        for _ in range(30):
            qc.set_cached_result("SELECT 4", ["a"], rows, 2000, "now")

    def reader():
        for _ in range(60):
            qc.clear_memory()                                              # force the disk path
            r = qc.get_cached_result("SELECT 4")
            if r is not None and r["row_count"] != 2000:
                errors.append(r)

    ts = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert errors == []


def test_corrupt_file_is_a_miss_and_is_removed(qc, tmp_path):
    qc.set_cached_result("SELECT 5", [], [], 0, "now")
    qc.clear_memory()
    f = next(tmp_path.glob("*.json"))
    f.write_text("{not json")
    assert qc.get_cached_result("SELECT 5") is None and not f.exists()


def test_eviction_enforces_entry_cap_and_drops_expired(qc, tmp_path, monkeypatch):
    monkeypatch.setattr(qc, "MAX_ENTRIES", 3)
    for i in range(6):
        qc.set_cached_result(f"SELECT {i}", [], [], 0, "now")
        time.sleep(0.01)
    qc._maybe_evict(force=True)
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert qc.get_cached_result("SELECT 5") is not None                    # newest survives
    monkeypatch.setattr(qc, "TTL_SECONDS", -1)
    qc._maybe_evict(force=True)
    assert list(tmp_path.glob("*.json")) == []                             # all expired -> deleted


def test_pipeline_marker_invalidates(qc, tmp_path):
    qc.set_cached_result("SELECT 6", [], [], 0, "now")
    assert qc.get_cached_result("SELECT 6") is not None
    (tmp_path / "marker.txt").write_text("2999-01-01T00:00:00Z")
    assert qc.get_cached_result("SELECT 6") is None


def test_identical_concurrent_queries_are_coalesced(qc):
    executed, results = [], []

    def worker():
        cached, leader = qc.lookup_or_claim("SELECT slow")
        if cached is None and leader:
            try:
                executed.append(1)
                time.sleep(0.3)
                qc.set_cached_result("SELECT slow", ["a"], [{"a": 1}], 1, "now")
            finally:
                qc.release_claim("SELECT slow")
            results.append("ran")
        else:
            results.append(cached["row_count"] if cached else "unclaimed")

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(executed) == 1
    assert sorted(map(str, results)) == ["1", "1", "1", "ran"]


def test_duplicate_sql_in_one_batch_does_not_deadlock(qc):
    _, leader1 = qc.lookup_or_claim("SELECT dup")
    start = time.time()
    cached, leader2 = qc.lookup_or_claim("SELECT dup")                     # same thread owns the claim
    assert leader1 and not leader2 and cached is None and time.time() - start < 1
    qc.release_claim("SELECT dup")


# ---------- sql_executor integration ----------

def _executor_state(sqls):
    return {"sql_queries": [{"sql": s, "purpose": f"p{i}", "validated": True} for i, s in enumerate(sqls)],
            "query_results": []}


def test_executor_keeps_query_order_and_cache_hit_means_all_hit(qc, monkeypatch):
    import src.agent.analyst_graph as ag
    qc.set_cached_result("SELECT 'b'", ["x"], [{"x": "b"}], 1, "now")      # only the 2nd query is cached

    class DB:
        def execute_queries(self, qs):
            return [{"purpose": q["purpose"], "sql": q["sql"], "success": True, "columns": ["x"],
                     "rows": [{"x": q["sql"]}], "row_count": 1} for q in qs]

    monkeypatch.setattr(ag, "_db", DB())
    out = ag.sql_executor(_executor_state(["SELECT 'a'", "SELECT 'b'", "SELECT 'c'"]))
    assert [e.executed_sql for e in out["query_results"]] == ["SELECT 'a'", "SELECT 'b'", "SELECT 'c'"]
    assert out["cache_hit"] is False and (out["cache_hits"], out["cache_misses"]) == (1, 2)

    out2 = ag.sql_executor(_executor_state(["SELECT 'a'", "SELECT 'b'", "SELECT 'c'"]))
    assert out2["cache_hit"] is True and out2["cache_misses"] == 0


def test_executor_records_error_type_so_sql_failures_route_to_repair(qc, monkeypatch):
    import src.agent.analyst_graph as ag

    class DB:
        def execute_queries(self, qs):
            return [{"purpose": q["purpose"], "sql": q["sql"], "success": False,
                     "error": "SQL Error: column not found", "error_type": "sql"} for q in qs]

    monkeypatch.setattr(ag, "_db", DB())
    out = ag.sql_executor(_executor_state(["SELECT nope"]))
    assert out["sql_queries"][0]["error_type"] == "sql"
    assert ag.after_execution({**out, "repair_attempts": 0}) == "repair"
