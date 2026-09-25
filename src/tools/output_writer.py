"""
Normalizes extracted records into a clean, deduplicated dataset and picks
the right output format (CSV for flat tabular data, JSON/JSONL when records
carry nested structures that CSV can't represent well).
"""
import json
import logging
import os
import re
import time
from urllib.parse import urlparse

import pandas as pd

import config

logger = logging.getLogger("etl.output")


def _slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "site"


def build_output_path(source_url: str, fmt: str) -> str:
    domain = _slugify(urlparse(source_url).netloc.replace("www.", ""))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{domain}_extract_{stamp}.{fmt}"
    return os.path.join(config.OUTPUT_DIR, filename)


def _is_flat(rows: list[dict]) -> bool:
    for row in rows:
        for v in row.values():
            if isinstance(v, (dict, list)):
                return False
    return True


def dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def normalize(rows: list[dict]) -> list[dict]:
    """Union all keys across rows so every record has the same columns,
    without fabricating values for records that are missing a field."""
    all_keys: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)
    return [{k: row.get(k, None) for k in all_keys} for row in rows]


def write(rows: list[dict], source_url: str, fmt: str | None = None) -> str:
    if not rows:
        raise ValueError("No rows to write — extraction produced no data.")

    rows = dedupe(rows)
    rows = normalize(rows)

    if fmt is None:
        fmt = "csv" if _is_flat(rows) else "jsonl"

    path = build_output_path(source_url, fmt)

    if fmt == "csv":
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    elif fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logger.info("wrote %d rows to %s", len(rows), path)
    return path