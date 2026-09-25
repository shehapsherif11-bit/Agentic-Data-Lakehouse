"""
Top-level orchestration: URL(s) in, clean structured file out.

Decision flow per page:
  1. Fetch (static first, headless browser only if the page needs JS).
  2. Look for JSON-LD / structured data first — free, deterministic, exact.
  3. If explicit fields were requested, or no usable structured data exists,
     look for a repeated "listing" pattern (product grid, cards, table rows)
     and run field extraction over each item.
  4. If nothing repeats, fall back to single-record "main content" extraction
     (article/body text + metadata) — no LLM call needed unless the caller
     asked for specific fields.
  5. Optionally follow pagination up to max_pages, merging + deduping rows.
"""
import asyncio
import logging
from typing import Optional

from src.tools import fetchers
from src.tools import content_extraction as ce
from src.tools import llm_extractor
from src.tools import output_writer
from src.tools import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("etl.pipeline")


def _rows_from_structured(structured: list[dict], fields: Optional[list[str]]) -> list[dict]:
    rows = []
    for item in structured:
        clean = {k: v for k, v in item.items() if not str(k).startswith("@") and not isinstance(v, dict)}
        if fields:
            row = {f: clean.get(f, "N/A") for f in fields}
            if any(v != "N/A" for v in row.values()):  # keep only if we actually found something requested
                rows.append(row)
        elif clean:
            rows.append(clean)
    return rows


def _extract_from_page(html: str, url: str, fields: Optional[list[str]]) -> list[dict]:
    structured = ce.extract_structured_data(html)
    rows = _rows_from_structured(structured, fields)
    if rows:
        logger.info("using structured data (JSON-LD/meta) for %s: %d rows", url, len(rows))
        return rows

    blocks = ce.find_repeated_blocks(html)
    if blocks:
        logger.info("detected a repeated listing pattern on %s: %d items", url, len(blocks))
        target_fields = fields or ["title", "description", "price", "link"]
        text_blob = "\n\n---ITEM---\n\n".join(b["text"] for b in blocks)
        rows = llm_extractor.extract_fields(text_blob, target_fields)
        for row, block in zip(rows, blocks):  # recover hrefs the LLM can't see reliably
            link_key = "link" if "link" in row else ("url" if "url" in row else None)
            if link_key and block.get("href"):
                row[link_key] = block["href"]
        return rows

    main = ce.extract_main_content(html, url)
    if fields:
        text = main.get("text", "") or ""
        rows = llm_extractor.extract_fields(text, fields)
        if rows:
            return rows
    record = {k: v for k, v in main.items() if v not in (None, "", [])}
    return [record] if record else []


async def _run(url: str, fields: Optional[list[str]], max_pages: int) -> list[dict]:
    max_pages = max(1, min(max_pages, config.MAX_PAGES_HARD_LIMIT))
    all_rows: list[dict] = []
    seen_urls: set[str] = set()
    current_url = url

    for page_num in range(1, max_pages + 1):
        if current_url in seen_urls:
            break
        seen_urls.add(current_url)

        result = await fetchers.fetch(current_url)
        rows = _extract_from_page(result.html, current_url, fields)
        for r in rows:
            r.setdefault("_source_url", current_url)
        all_rows.extend(rows)
        logger.info("page %d/%d done (%s): %d rows so far", page_num, max_pages, current_url, len(all_rows))

        if page_num == max_pages:
            break
        next_url = ce.find_next_page(result.html, current_url)
        if not next_url or next_url == current_url:
            break
        current_url = next_url

    return all_rows


def extract(
    url: str,
    fields: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    output_format: Optional[str] = None,
    max_pages: int = config.DEFAULT_MAX_PAGES,
) -> str:
    """
    Main entry point.
      url            - required, the page (or first page of a listing) to extract from
      fields         - optional list of field names to prioritize; if omitted the
                        engine infers what's valuable (structured data / article content)
      output_path    - optional explicit path; auto-generated (descriptive, timestamped) if omitted
      output_format  - optional 'csv' | 'json' | 'jsonl'; auto-selected from the data shape if omitted
      max_pages      - how many pages of pagination to follow (default 1 = no pagination)
    """
    rows = asyncio.run(_run(url, fields, max_pages))
    if not rows:
        raise ValueError(f"No extractable data found at {url}.")

    if output_path:
        rows = output_writer.normalize(output_writer.dedupe(rows))
        fmt = output_format or (output_path.rsplit(".", 1)[-1] if "." in output_path else "csv")
        if fmt == "csv":
            import pandas as pd
            pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2, default=str)
        return output_path

    return output_writer.write(rows, url, fmt=output_format)