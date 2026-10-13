"""The demo backend must behave like DatabricksUtil for everything the analyst graph relies on."""
import duckdb
import pytest

pytest.importorskip("duckdb")

from src.agent.sql_safety_guard import guard_sql  # noqa: E402
from src.utils.demo_database import DemoDuckDBUtil, to_duckdb  # noqa: E402

G = "workspace.zomato_gold"


@pytest.fixture
def db(tmp_path, monkeypatch):
    con = duckdb.connect(":memory:")
    con.execute("""CREATE TABLE r AS SELECT * FROM (VALUES
        (1, 'KFC', 'Cairo'), (2, 'McDonald''s', 'Cairo'), (3, 'McDonald''s', 'Giza')) t(restaurant_id, restaurant_name, city)""")
    con.execute("""CREATE TABLE o AS SELECT * FROM (VALUES
        (10, 1, TIMESTAMP '2025-01-05 10:00:00', 100.0), (11, 2, TIMESTAMP '2025-01-06 11:00:00', 50.0),
        (12, 3, TIMESTAMP '2025-02-07 12:00:00', 70.0), (13, 3, TIMESTAMP '2025-02-08 12:00:00', 30.0))
        t(order_id, restaurant_id, order_timestamp, sales_amount)""")
    con.execute(f"COPY r TO '{(tmp_path / 'dim_resturant.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY o TO '{(tmp_path / 'fact_orders.parquet').as_posix()}' (FORMAT parquet)")
    (tmp_path / "manifest.json").write_text('{"sample": "test"}', encoding="utf-8")
    monkeypatch.setattr(DemoDuckDBUtil, "_con", None)
    monkeypatch.setattr(DemoDuckDBUtil, "_data_dir", None)
    return DemoDuckDBUtil(data_dir=tmp_path)


def run(db, sql, purpose="t"):
    safe, _ = guard_sql(sql)  # the demo only ever receives guarded SQL
    return db.execute_queries([{"sql": safe, "purpose": purpose}])[0]


def test_result_has_the_shape_the_graph_expects(db):
    res = run(db, f"SELECT restaurant_name, COUNT(*) AS n FROM {G}.dim_resturant GROUP BY 1 ORDER BY 2 DESC, 1")
    assert res["success"] and res["error"] is None and res["error_type"] is None
    assert res["columns"] == ["restaurant_name", "n"]
    assert res["rows"] == [{"restaurant_name": "McDonald's", "n": 2}, {"restaurant_name": "KFC", "n": 1}]
    assert res["row_count"] == 2 and res["purpose"] == "t"


def test_databricks_escaped_apostrophe_matches(db):
    # The agent writes 'McDonald\'s' (Databricks). Raw DuckDB would not read that as one literal.
    sql = f"SELECT COUNT(*) AS n FROM {G}.dim_resturant WHERE restaurant_name = 'McDonald\\'s'"
    assert run(db, sql)["rows"] == [{"n": 2}]
    assert "''" in to_duckdb("SELECT 'McDonald\\'s'")


def test_period_style_sql_runs(db):
    sql = (f"SELECT DATE_TRUNC('month', order_timestamp) AS m, SUM(sales_amount) AS rev FROM {G}.fact_orders "
           f"GROUP BY 1 ORDER BY 1")
    res = run(db, sql)
    assert res["success"], res["error"]
    assert [r["rev"] for r in res["rows"]] == [150.0, 100.0]


def test_window_and_share_sql_runs(db):
    sql = (f"SELECT restaurant_id, SUM(sales_amount) AS rev, ROUND(100 * SUM(sales_amount) / SUM(SUM(sales_amount)) OVER (), 2) AS share "
           f"FROM {G}.fact_orders GROUP BY 1 ORDER BY 2 DESC, 1")
    res = run(db, sql)
    assert res["success"], res["error"]
    assert res["rows"][0]["share"] == pytest.approx(100 * 100 / 250, abs=0.01)


def test_unknown_column_is_a_sql_error_so_the_graph_repairs_it(db):
    res = run(db, f"SELECT nope FROM {G}.fact_orders")
    assert not res["success"] and res["error_type"] == "sql"
    assert res["error"].startswith("SQL Error")


def test_unknown_table_is_a_sql_error(db):
    res = run(db, f"SELECT a FROM {G}.not_a_table")
    assert not res["success"] and res["error_type"] == "sql"


def test_runaway_query_times_out(db):
    db.statement_timeout = 1
    res = db.execute_queries([{"sql": "SELECT COUNT(*) FROM range(1000000000) a, range(1000000000) b", "purpose": "slow"}])[0]
    assert not res["success"] and res["error_type"] == "timeout"


def test_queries_run_in_parallel_and_keep_order(db):
    qs = [{"sql": guard_sql(f"SELECT {i} AS v FROM {G}.dim_resturant LIMIT 1")[0], "purpose": str(i)} for i in range(6)]
    out = db.execute_queries(qs)
    assert [r["rows"][0]["v"] for r in out] == list(range(6))


def test_schema_shape_matches_databricks_util(db):
    schema = db.get_cached_schema()
    assert set(schema) == {"dim_resturant", "fact_orders"}
    assert {"name": "restaurant_name", "type": "VARCHAR"} in schema["dim_resturant"]["columns"]
    assert len(schema["fact_orders"]["sample"]) == 3


def test_manifest_is_exposed(db):
    assert db.manifest() == {"sample": "test"}


def test_missing_sample_data_is_reported_not_crashed(tmp_path, monkeypatch):
    monkeypatch.setattr(DemoDuckDBUtil, "_con", None)
    monkeypatch.setattr(DemoDuckDBUtil, "_data_dir", None)
    res = DemoDuckDBUtil(data_dir=tmp_path).execute_queries([{"sql": f"SELECT 1 FROM {G}.t", "purpose": "x"}])[0]
    assert not res["success"] and "No sample data" in res["error"]


# --- Spark/Databricks vs DuckDB behaviour the demo must reproduce -------------------------------------------
def scalar(db, expr):
    res = run(db, f"SELECT {expr} AS v FROM {G}.dim_resturant LIMIT 1")
    assert res["success"], res["error"]
    return res["rows"][0]["v"]


def test_dayofweek_counts_from_sunday_equals_one_like_databricks(db):
    # Databricks: Sunday=1 ... Saturday=7. Raw DuckDB: Sunday=0 ... Saturday=6.
    assert scalar(db, "DAYOFWEEK(DATE '2025-06-15')") == 1   # Sunday
    assert scalar(db, "DAYOFWEEK(DATE '2025-06-11')") == 4   # Wednesday
    assert scalar(db, "DAYOFWEEK(DATE '2025-06-14')") == 7   # Saturday
    assert scalar(db, "CASE WHEN DAYOFWEEK(DATE '2025-06-14') IN (1, 7) THEN 'Weekend' ELSE 'Weekday' END") == "Weekend"


def test_week_starts_on_monday_like_databricks(db):
    import datetime
    assert scalar(db, "CAST(DATE_TRUNC('week', DATE '2025-06-11') AS DATE)") == datetime.date(2025, 6, 9)


def test_datediff_is_end_minus_start_in_days(db):
    assert scalar(db, "DATEDIFF(DATE '2025-06-14', DATE '2025-06-11')") == 3


def test_nulls_sort_first_ascending_and_last_descending_like_databricks(db):
    asc = run(db, "SELECT t.x FROM (VALUES (1), (NULL), (2)) AS t(x) ORDER BY t.x")
    desc = run(db, "SELECT t.x FROM (VALUES (1), (NULL), (2)) AS t(x) ORDER BY t.x DESC")
    assert [r["x"] for r in asc["rows"]] == [None, 1, 2]
    assert [r["x"] for r in desc["rows"]] == [2, 1, None]
