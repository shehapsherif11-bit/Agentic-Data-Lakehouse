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
