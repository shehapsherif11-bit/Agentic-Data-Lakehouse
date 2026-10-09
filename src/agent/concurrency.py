"""Re-export: the implementation lives in src.utils.concurrency so the DB layer can use it without
depending on the agent package."""
from src.utils.concurrency import run_parallel  # noqa: F401
