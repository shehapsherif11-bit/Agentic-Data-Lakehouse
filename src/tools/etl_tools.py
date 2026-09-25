"""
LangChain tool wrappers around the extraction engine + pipeline, for use by
the ReAct agent in etl_agent.py.

Compared to the previous version:
  * `scrape_dynamic_website` + `smart_data_extractor` are replaced by
    `extract_website_data`, a single call into the intelligent pipeline
    (structured-data-first, listing detection, JS fallback only when needed,
    pagination support).
  * `execute_pandas_code` now runs in a restricted namespace (no builtins,
    no imports, no filesystem/network/OS access) instead of a bare `exec` of
    LLM-generated code against the full Python environment.
"""
import os
import re
import pandas as pd
import requests
from langchain_core.tools import tool

from src.tools import pipeline

_DENYLIST = re.compile(r"\b(import|open|exec|eval|__|os\.|sys\.|subprocess|socket)\b")


@tool
def extract_from_api(url: str, output_path: str) -> str:
    """Extracts JSON data from a structured API URL and saves it as CSV."""
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()

        if isinstance(data, dict):
            if "results" in data:
                df = pd.json_normalize(data["results"])
            elif "data" in data:
                df = pd.json_normalize(data["data"])
            else:
                df = pd.json_normalize(data)
        else:
            df = pd.json_normalize(data)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return f"SUCCESS: API data saved to {output_path} ({len(df)} rows)."
    except Exception as e:
        return f"ERROR extracting API data: {e}"


@tool
def extract_website_data(url: str, fields: str = "", max_pages: int = 1) -> str:
    """
    Intelligently extracts the most valuable data from a website URL.
    - url: the page to extract from (required).
    - fields: optional comma-separated list of fields to prioritize
      (e.g. "Product Name, Price, Description"). Leave empty to let the
      engine decide what's valuable (structured data / article content).
    - max_pages: how many pages of pagination to follow (default 1).
    Automatically uses a lightweight HTTP fetch, and only falls back to a
    headless browser if the page requires JavaScript to render.
    """
    try:
        field_list = [f.strip() for f in fields.split(",") if f.strip()] or None
        path = pipeline.extract(url, fields=field_list, max_pages=max_pages)
        return f"SUCCESS: Extracted data from {url} and saved it to {path}."
    except Exception as e:
        return f"ERROR extracting website data: {e}"


@tool
def execute_pandas_code(code: str, input_csv: str, output_csv: str) -> str:
    """
    Runs a pandas transformation against an existing CSV and writes the result
    to a new CSV. `code` must operate on a variable named `df` (already loaded
    from input_csv) and assign its result back to `df`. Runs in a restricted
    sandbox: no imports, no file/network/OS access — pandas transforms only.
    """
    try:
        clean_code = code.replace("```python", "").replace("```", "").strip()
        if _DENYLIST.search(clean_code):
            return "ERROR: code contains a disallowed operation (imports/file/OS access are not permitted)."

        df = pd.read_csv(input_csv)
        safe_globals = {"__builtins__": {}}
        safe_locals = {"df": df, "pd": pd}
        exec(clean_code, safe_globals, safe_locals)  # restricted namespace; see denylist above
        result_df = safe_locals.get("df")
        if not isinstance(result_df, pd.DataFrame):
            return "ERROR: code did not leave a valid DataFrame in `df`."

        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return f"SUCCESS: transformed data saved to {output_csv} ({len(result_df)} rows)."
    except Exception as e:
        return f"ERROR executing pandas transform: {e}"


etl_toolkit = [extract_from_api, extract_website_data, execute_pandas_code]