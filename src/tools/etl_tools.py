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
    - fields: comma-separated list of fields to extract, e.g. "Product Name, Price, Category".
      ALWAYS pass this whenever the user names or implies specific data points to extract —
      translate/normalize their wording into clear English labels first, even if they asked in
      Arabic or another language. Leave empty ONLY when the user explicitly wants "whatever is
      useful" with nothing specific named.
    - max_pages: how many pages of pagination to follow (default 1).
    Automatically uses a lightweight HTTP fetch, and only falls back to a headless browser if the
    page requires JavaScript to render — or refuses outright and reports the failure honestly if the
    site returns a bot-protection / CDN-challenge page (Akamai, Cloudflare, etc.) instead of real
    content, rather than saving the block page as if it were data.
    """
    try:
        field_list = [f.strip() for f in fields.split(",") if f.strip()] or None
        path = pipeline.extract(url, fields=field_list, max_pages=max_pages)

        fallback_note = _check_for_generic_fallback(path)
        if fallback_note:
            return f"PARTIAL RESULT (saved to {path}) — {fallback_note} Consider this a likely failure to find the requested data, not a full success."
        return f"SUCCESS: Extracted data from {url} and saved it to {path}."
    except Exception as e:
        return f"ERROR extracting website data: {e}"


def _check_for_generic_fallback(path: str) -> str | None:
    """Peeks at the written output to see whether pipeline.py had to fall
    back to generic page metadata (see pipeline.py's `_note` field) rather
    than finding the data actually requested, so this tool can report that
    honestly instead of a bare SUCCESS the agent might relay uncritically."""
    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path, nrows=1)
            if "_note" in df.columns:
                return str(df["_note"].iloc[0])
        elif path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                import json
                row = json.loads(f.readline())
                if "_note" in row:
                    return row["_note"]
        elif path.endswith(".json"):
            import json
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
                if rows and isinstance(rows[0], dict) and "_note" in rows[0]:
                    return rows[0]["_note"]
    except Exception:
        pass
    return None


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