"""
Entry point of the public live demo (Streamlit Community Cloud: main file = demo/streamlit_app.py).

It switches the app to DEMO_MODE (a frozen data sample in DuckDB, no Databricks) and then runs the normal
app.py, so the demo and the full project share one UI and one agent.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DEMO_MODE", "1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
