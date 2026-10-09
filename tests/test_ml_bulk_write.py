import importlib.util
import os

import pandas as pd

_PATH = os.path.join(os.path.dirname(__file__), "..", "airflow", "dags", "AI", "ml_build_segments.py")
spec = importlib.util.spec_from_file_location("ml_build_segments", _PATH)
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)


class RecordingCursor:
    def __init__(self):
        self.statements = []
        self.uploaded = None

    def execute(self, sql):
        self.statements.append(" ".join(sql.split()))
        if sql.strip().upper().startswith("PUT"):
            local = sql.split("'")[1]
            self.uploaded = pd.read_parquet(local)      # the file must exist and be valid Parquet at PUT time


def test_segments_are_loaded_with_one_parquet_put_and_one_atomic_ctas(tmp_path):
    df = pd.DataFrame({"user_id": range(25_000), "customer_segment": ["Loyal"] * 25_000, "extra": 1})
    cur = RecordingCursor()
    ml.bulk_write_segments(cur, df, str(tmp_path))

    kinds = [s.split()[0] for s in cur.statements]
    assert kinds == ["CREATE", "PUT", "CREATE", "REMOVE"]                       # 4 statements for 25k rows, no INSERT loop
    assert not any(s.startswith("INSERT") or s.startswith("DROP") for s in cur.statements)
    assert "CREATE OR REPLACE TABLE workspace.zomato_gold.ai_customer_segments" in cur.statements[2]
    assert "FROM parquet.`/Volumes/workspace/zomato_gold/staging/ai_customer_segments.parquet`" in cur.statements[2]
    assert list(cur.uploaded.columns) == ["user_id", "customer_segment"] and len(cur.uploaded) == 25_000
    assert str(cur.uploaded["user_id"].dtype) == "int32"
