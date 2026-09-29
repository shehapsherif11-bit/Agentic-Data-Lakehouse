"""
Top-level orchestration: URL(s) in, clean structured file out.

Decision flow per page:
  0. Reject bot-protection / CDN-challenge pages outright (retrying once
     with a full browser) rather than silently extracting the block page
     as if it were real content — see content_extraction.BlockedPageError.
  1. Fetch (static first, headless browser only if the page needs JS).
  2. Look for JSON-LD structured data first — a deliberate declaration by
     the site author (Product/Offer/ItemList/...), free, deterministic,
     exact. Generic OpenGraph/meta tags are NOT trusted at this stage (see
     content_extraction.extract_opengraph's docstring for why) — they
     exist on almost every webpage and are not evidence the page actually
     contains what was asked for.
  3. If no JSON-LD match, look for a repeated "listing" pattern (product
     grid, cards, table rows) and run field extraction over each item.
  4. If nothing repeats, fall back to single-record "main content"
     extraction (article/body text + metadata) — no LLM call needed unless
     the caller asked for specific fields.
  5. Only as an absolute last resort — nothing above found anything at
     all — fall back to bare OpenGraph tags, explicitly labeled as generic
     page metadata rather than presented as if it satisfies the request.
  6. Optionally follow pagination up to max_pages, merging + deduping rows.
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

BLOCKED_PAGE_MESSAGE = (
    "The site appears to be protected by bot/CDN detection (e.g. Akamai, Cloudflare) and returned a "
    "challenge or access-denied page instead of real content. Automated extraction was not possible for: {url}"
)
MIN_MAIN_CONTENT_CHARS = 60  # a trivial "Welcome" paragraph shouldn't count as real page content


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
    if ce.looks_like_blocked_page(html):
        raise ce.BlockedPageError(url)

    json_ld = ce.extract_json_ld(html)
    rows = _rows_from_structured(json_ld, fields)
    if rows:
        logger.info("using JSON-LD structured data for %s: %d rows", url, len(rows))
        return rows

    blocks = ce.find_repeated_blocks(html)
    if blocks:
        logger.info("detected a repeated listing pattern on %s: %d items", url, len(blocks))
        target_fields = fields or ["title", "description", "price", "link"]
        item_texts = [b["text"] for b in blocks]
        rows = llm_extractor.extract_fields_from_items(item_texts, target_fields)
        if rows:
            if len(rows) < len(blocks):
                logger.warning(
                    "Only recovered %d/%d listing items on %s (some LLM batches likely failed — see warnings above).",
                    len(rows), len(blocks), url,
                )
            # Recover hrefs the LLM can't see reliably. Rows and blocks are
            # no longer guaranteed to line up 1:1 (a batch can fail and be
            # skipped, or the model can merge/drop an item), so match by
            # position only up to the shorter of the two rather than
            # assuming full alignment.
            for row, block in zip(rows, blocks):
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
    if record and len((record.get("text") or "")) >= MIN_MAIN_CONTENT_CHARS:
        return [record]

    # Absolute last resort: bare OpenGraph tags. These exist on nearly
    # every webpage and are NOT evidence the page contains what was asked
    # for, so they are clearly flagged rather than presented as a match.
    og = ce.extract_opengraph(html)
    if og:
        logger.warning(
            "No product/listing/article data found on %s; falling back to generic page metadata only.", url
        )
        og["_note"] = (
            "Only generic page metadata (OpenGraph/meta tags) was found — no specific product, "
            "listing, or article content was detected on this page."
        )
        return [og]

    return []


async def _fetch_and_extract(url: str, fields: Optional[list[str]]):
    result = await fetchers.fetch(url)
    try:
        rows = _extract_from_page(result.html, url, fields)
        return rows, result
    except ce.BlockedPageError:
        if result.used_browser:
            raise RuntimeError(BLOCKED_PAGE_MESSAGE.format(url=url))
        logger.warning("Page looked blocked on a static fetch; retrying once with a full browser: %s", url)
        result = await fetchers.fetch(url, force_browser=True)
        try:
            rows = _extract_from_page(result.html, url, fields)
            return rows, result
        except ce.BlockedPageError:
            raise RuntimeError(BLOCKED_PAGE_MESSAGE.format(url=url))


async def _run(url: str, fields: Optional[list[str]], max_pages: int) -> list[dict]:
    max_pages = max(1, min(max_pages, config.MAX_PAGES_HARD_LIMIT))
    all_rows: list[dict] = []
    seen_urls: set[str] = set()
    current_url = url

    for page_num in range(1, max_pages + 1):
        if current_url in seen_urls:
            break
        seen_urls.add(current_url)

        rows, result = await _fetch_and_extract(current_url, fields)
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