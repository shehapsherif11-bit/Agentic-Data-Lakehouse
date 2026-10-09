"""P1-5: nothing the agent caches may outlive a data refresh."""
import os
import re
import time

from src.agent import analyst_graph as ag

DAG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "airflow", "dags", "zomato_dag.py")


class CountingDB:
    def __init__(self):
        self.calls = 0

    def execute_queries(self, queries):
        self.calls += 1
        return [{"success": True, "rows": [{"min_date": "2024-01-01", "max_date": f"2026-06-{self.calls:02d}"}], "row_count": 1}]


def test_data_coverage_is_cached_until_the_pipeline_marker_moves(monkeypatch):
    ag._coverage_cache.clear()
    db = CountingDB()
    monkeypatch.setattr(ag, "_db", db)
    monkeypatch.setattr(ag, "read_marker", lambda *a: 0.0)
    first = ag._data_coverage()
    assert ag._data_coverage() == first and db.calls == 1          # cached while nothing changed

    monkeypatch.setattr(ag, "read_marker", lambda *a: time.time() + 5)   # a pipeline run finished after we cached it
    second = ag._data_coverage()
    assert db.calls == 2 and second["max_date"] != first["max_date"]     # re-read: "data covers up to ..." is not stale


def _task_command(task_id: str) -> str:
    src = open(DAG, encoding="utf-8").read()
    m = re.search(rf'task_id="{task_id}",\s*bash_command=(.+?)(?:,\n|\n\s*\))', src, re.S)
    assert m, f"task {task_id} not found"
    return m.group(1)


def test_marker_is_written_after_the_core_build_and_after_the_ai_build():
    """If the AI steps fail, the Gold facts were already rewritten by dbt_build_core: cached answers must be invalidated then."""
    for task in ("dbt_build_core", "dbt_build_ai"):
        cmd = _task_command(task)
        assert "&& {MARK_DATA_CHANGED}" in cmd, f"{task} does not write the cache-invalidation marker after its dbt build"
    src = open(DAG, encoding="utf-8").read()
    assert "last_ai_run.txt" in src.split("MARK_DATA_CHANGED = ")[1].split("\n")[0]    # the file src/utils/pipeline_marker.py reads
