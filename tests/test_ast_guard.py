import pytest
from src.agent.sql_safety_guard import guard_sql, SQLGuardError

REJECT_CASES = [
    ("SELECT 1; DROP TABLE x", "multi_statement"),
    ("/* x */ DROP TABLE workspace.zomato_gold.t", "hidden_drop"),
    ("INSERT INTO workspace.zomato_gold.t SELECT 1", "insert"),
    ("SELECT a FROM read_files('s3://x')", "read_files_tvf"),
    ("SELECT a FROM system.information_schema.tables", "system_schema"),
    ("SELECT a FROM workspace.other_schema.t", "other_schema"),
    ("SELECT t.* FROM workspace.zomato_gold.t t", "t_star"),
    ("SELECT a FROM workspace.zomato_gold.t1 UNION SELECT b FROM other.t2", "union_bad_schema"),
    ("SELECT a FROM workspace.zomato_gold.t WHERE a IN (SELECT b FROM bad_schema.t2)", "where_subquery_bad_schema"),
    ("", "empty"),
    ("   ", "whitespace")
]

@pytest.mark.parametrize("sql, name", REJECT_CASES, ids=[x[1] for x in REJECT_CASES])
def test_guard_rejects(sql, name):
    with pytest.raises(SQLGuardError):
        guard_sql(sql)

ACCEPT_CASES = [
    ("SELECT a FROM WORKSPACE.Zomato_Gold.t", "SELECT a FROM WORKSPACE.Zomato_Gold.t LIMIT 10000", "mixed_case"),
    ("SELECT a FROM `workspace`.`zomato_gold`.`t`", "SELECT a FROM workspace.zomato_gold.t LIMIT 10000", "backticks"),
    ("SELECT a, ROW_NUMBER() OVER(PARTITION BY b ORDER BY c) FROM workspace.zomato_gold.t", "SELECT a, ROW_NUMBER() OVER (PARTITION BY b ORDER BY c) FROM workspace.zomato_gold.t LIMIT 10000", "window_function"),
    ("WITH c1 AS (SELECT a FROM workspace.zomato_gold.t1), c2 AS (SELECT a FROM c1) SELECT a FROM c2", "WITH c1 AS (SELECT a FROM workspace.zomato_gold.t1), c2 AS (SELECT a FROM c1) SELECT a FROM c2 LIMIT 10000", "cte_chain"),
    ("SELECT a FROM workspace.zomato_gold.t LIMIT 5", "SELECT a FROM workspace.zomato_gold.t LIMIT 5", "existing_limit")
]

@pytest.mark.parametrize("sql, expected_substr, name", ACCEPT_CASES, ids=[x[2] for x in ACCEPT_CASES])
def test_guard_accepts(sql, expected_substr, name):
    result, limit_injected = guard_sql(sql)
    assert isinstance(result, str)
    if "LIMIT 5" in sql:
        assert "LIMIT 5" in result
        assert "10000" not in result
        assert limit_injected is False
    else:
        assert limit_injected is True


# ---------------------------------------------------------------------------
# Red-team suite. Every case below is an attack or a trick the guard must stop.
# ---------------------------------------------------------------------------
G = "workspace.zomato_gold"

# sqlglot parses functions it does not know as exp.Anonymous. These used to pass because the blocklist
# compared Func.sql_name(), which is "ANONYMOUS" for every one of them.
DANGEROUS_CALLS = [
    "ai_query('m', 'x')",
    "http_request(conn => 'c', method => 'GET', path => '/')",
    "remote_query('c', query => 'select 1')",
    "read_files('s3://x')",
    "input_file_name()",
    "ai_gen('hi')",
    "ai_classify('x', array('a', 'b'))",
]

DANGEROUS_POSITIONS = {
    "select_list": "SELECT {call} AS z FROM " + G + ".t",
    "nested_in_function": "SELECT upper({call}) AS z FROM " + G + ".t",
    "where": "SELECT a FROM " + G + ".t WHERE {call} = 'x'",
    "subquery": "SELECT a FROM " + G + ".t WHERE a IN (SELECT {call} FROM " + G + ".t)",
    "cte": "WITH c AS (SELECT {call} AS z FROM " + G + ".t) SELECT z FROM c",
}


@pytest.mark.parametrize("position", DANGEROUS_POSITIONS)
@pytest.mark.parametrize("call", DANGEROUS_CALLS)
def test_dangerous_function_is_blocked_everywhere(call, position):
    with pytest.raises(SQLGuardError):
        guard_sql(DANGEROUS_POSITIONS[position].format(call=call))


def test_dangerous_function_is_blocked_when_backticked_or_mixed_case():
    for sql in (f"SELECT `ai_query`('m', a) FROM {G}.t", f"SELECT Ai_Query('m', a) FROM {G}.t"):
        with pytest.raises(SQLGuardError):
            guard_sql(sql)


def test_normal_functions_are_not_blocked():
    sql = f"SELECT SUM(a), ROUND(AVG(b), 2), DATE_TRUNC('month', d), COALESCE(c, 0) FROM {G}.t GROUP BY 3"
    safe, _ = guard_sql(sql)
    assert "SUM(a)" in safe


def test_column_named_like_ai_function_is_not_blocked():
    # ai_customer_segments is a real Gold table; only function CALLS are blocked.
    safe, _ = guard_sql(f"SELECT ai_score FROM {G}.ai_customer_segments")
    assert "ai_customer_segments" in safe


# A CTE name must not hide a qualified table that lives in another schema.
CTE_BYPASS_CASES = [
    "WITH s AS (SELECT 1 AS a) SELECT a FROM workspace.zomato_silver.s",
    "WITH orders AS (SELECT 1 AS a) SELECT a FROM workspace.zomato_silver.orders",
    "WITH t AS (SELECT 1 AS a) SELECT a FROM other_catalog.zomato_gold.t",
    f"WITH t AS (SELECT a FROM {G}.t) SELECT x.a FROM t JOIN workspace.zomato_silver.t x ON 1 = 1",
]


@pytest.mark.parametrize("sql", CTE_BYPASS_CASES)
def test_cte_name_cannot_hide_a_table_in_another_schema(sql):
    with pytest.raises(SQLGuardError):
        guard_sql(sql)


def test_legitimate_cte_reference_still_works():
    safe, _ = guard_sql(f"WITH fact_orders AS (SELECT a FROM {G}.t) SELECT a FROM fact_orders")
    assert "fact_orders" in safe


# Statements that are not a plain SELECT, and other ways to reach outside the Gold schema.
OTHER_ATTACKS = [
    f"TRUNCATE TABLE {G}.fact_orders",
    f"GRANT SELECT ON TABLE {G}.fact_orders TO x",
    f"MERGE INTO {G}.t USING {G}.s ON t.a = s.a WHEN MATCHED THEN DELETE",
    f"ALTER TABLE {G}.t ADD COLUMNS (x INT)",
    f"CREATE TABLE x AS SELECT a FROM {G}.t",
    f"CREATE OR REPLACE VIEW v AS SELECT a FROM {G}.t",
    f"COPY INTO {G}.t FROM 's3://x'",
    "SET spark.sql.shuffle.partitions=1",
    "USE CATALOG main",
    "CALL main.default.p()",
    f"OPTIMIZE {G}.fact_orders",
    f"VACUUM {G}.fact_orders",
    f"EXPLAIN SELECT a FROM {G}.t",
    "SHOW TABLES IN workspace.zomato_silver",
    f"SELECT a FROM {G}.t; SELECT a FROM {G}.t",
    f"WITH x AS (SELECT 1) INSERT INTO {G}.t SELECT * FROM x",
    f"SELECT a FROM {G}.t, workspace.zomato_silver.s",
    f"SELECT a FROM {G}.t JOIN workspace.zomato_silver.s ON 1 = 1",
    f"SELECT a FROM {G}.t JOIN read_files('x') f ON 1 = 1",
    f"SELECT a FROM {G}.t WHERE a IN (SELECT x FROM read_files('s3://x'))",
    "SELECT a FROM IDENTIFIER('workspace.zomato_silver.s')",
    f"SELECT a FROM x.{G}.t",
    "SELECT a FROM delta.`s3://bucket/path`",
    "SELECT a FROM zomato_gold.t",
    "SELECT a FROM (SELECT a FROM workspace.zomato_silver.s) q",
    f"SELECT * FROM {G}.fact_orders",
    f"SELECT t.* FROM {G}.fact_orders t",
]


@pytest.mark.parametrize("sql", OTHER_ATTACKS)
def test_other_attacks_are_blocked(sql):
    with pytest.raises(SQLGuardError):
        guard_sql(sql)


def test_comment_cannot_smuggle_a_second_statement():
    # The comment is rewritten as a block comment, so the DROP is inert text, not a statement.
    safe, _ = guard_sql(f"SELECT a FROM {G}.t -- ; DROP TABLE {G}.t")
    assert "DROP TABLE" not in safe.split("*/")[-1].upper()


# Every result is capped at MAX_ROWS.
@pytest.mark.parametrize("limit", [10001, 99999999, 10**12])
def test_oversized_limit_is_lowered(limit):
    safe, flagged = guard_sql(f"SELECT a FROM {G}.t LIMIT {limit}")
    assert safe.endswith("LIMIT 10000")
    assert flagged is True


def test_oversized_limit_is_lowered_on_union():
    safe, flagged = guard_sql(f"SELECT a FROM {G}.t UNION ALL SELECT a FROM {G}.u LIMIT 500000")
    assert safe.endswith("LIMIT 10000")
    assert flagged is True


@pytest.mark.parametrize("limit", [1, 5, 9999, 10000])
def test_limit_within_cap_is_left_alone(limit):
    safe, flagged = guard_sql(f"SELECT a FROM {G}.t LIMIT {limit}")
    assert safe.endswith(f"LIMIT {limit}")
    assert flagged is False


@pytest.mark.parametrize("limit", ["(SELECT 99999999)", "5 + 99999999", "-1", "1.5"])
def test_unreadable_limit_is_rejected(limit):
    with pytest.raises(SQLGuardError):
        guard_sql(f"SELECT a FROM {G}.t LIMIT {limit}")
