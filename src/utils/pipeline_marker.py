"""
pipeline_marker.py — reads the timestamp the Airflow DAG writes after each successful dbt_build_ai run
(metadata/last_ai_run.txt). Anything cached before that moment (query results, the Gold schema) is stale.
A local file read: no network, safe on request paths.
"""
import datetime
import os

DEFAULT_MARKER_PATH = os.getenv(
    "AI_RUN_MARKER_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "metadata", "last_ai_run.txt"),
)


def read_marker(path: str = DEFAULT_MARKER_PATH) -> float:
    """Epoch seconds of the last successful pipeline run, or 0.0 if the marker is missing/unreadable."""
    try:
        with open(path, "r") as f:
            raw = f.read().strip()
    except OSError:
        return 0.0
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0
