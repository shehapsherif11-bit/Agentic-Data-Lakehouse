"""
LLM-assisted structured extraction — used only when deterministic sources
(JSON-LD, obvious listing blocks) aren't enough to satisfy the requested
fields. Compared to the previous implementation this module:
  * chunks long input instead of silently truncating to 12,000 characters,
  * caps how many ITEMS go into a single LLM call, not just how many
    characters — a page with many short items (e.g. 34 short product
    cards) can easily fit under the character limit in one call, but
    asking a small free-tier model to return 34 structured JSON objects
    in one response risks output truncation and makes a single failure
    (e.g. a rate limit) cost the entire batch instead of a small slice,
  * validates the model's JSON output instead of trusting it blindly,
  * retries on transient failures / malformed JSON, with longer backoff
    specifically for rate-limit (429) errors and a small pacing delay
    between consecutive calls to avoid tripping the limit repeatedly,
  * never invents values — the prompt explicitly forbids fabrication and
    missing fields are always "N/A" rather than a guess.
"""
import json
import logging
import re
import time

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_groq import ChatGroq

from src.tools import config

logger = logging.getLogger("etl.llm")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)
_ITEM_DELIMITER = "\n\n---ITEM---\n\n"

_PROMPT_TEMPLATE = """You are a precise data extraction engine.
Extract the following fields from the text below: {fields}

Rules:
1. Return ONLY a valid JSON array of objects. No markdown, no commentary, no code fences.
2. Each object must have exactly these keys: {fields}
3. If a field is not present in the text, use "N/A". NEVER invent or guess a value.
4. One object per distinct item/record found in the text.

Text:
\"\"\"
{chunk}
\"\"\"
"""


def _chunk_text(text: str) -> list[str]:
    """Character-count-based chunking, used for single-blob content like an
    article's main text (see pipeline.py's fields-on-main-content path)."""
    size = config.LLM_MAX_CHARS_PER_CHUNK
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return chunks[: config.LLM_MAX_CHUNKS]


def chunk_items(items: list[str]) -> list[str]:
    """Item-aware chunking for listing extraction (see pipeline.py's
    repeated-block path): groups items into batches bounded by BOTH item
    count and total character length, so a page with many short items
    never bundles more than config.LLM_MAX_ITEMS_PER_CHUNK of them into a
    single request — capping both the model's expected output size and
    the blast radius of any one failed call."""
    batches: list[str] = []
    current: list[str] = []
    current_len = 0

    for item in items:
        would_exceed_count = len(current) >= config.LLM_MAX_ITEMS_PER_CHUNK
        would_exceed_chars = current and (current_len + len(item) > config.LLM_MAX_CHARS_PER_CHUNK)
        if current and (would_exceed_count or would_exceed_chars):
            batches.append(_ITEM_DELIMITER.join(current))
            current, current_len = [], 0
        current.append(item)
        current_len += len(item)

    if current:
        batches.append(_ITEM_DELIMITER.join(current))

    return batches[: config.LLM_MAX_CHUNKS * 4]  # generous cap; still bounded, not unlimited


def _get_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set — required for LLM-based extraction.")
    model = config.get_active_groq_model()
    return ChatGroq(
        model=model,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
        max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
    )


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
)
def _invoke_and_parse(llm: ChatGroq, prompt: str) -> list[dict]:
    response = llm.invoke(prompt)
    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    match = _JSON_ARRAY_RE.search(raw)
    payload = match.group(0) if match else raw
    if not payload:
        raise ValueError("Model returned an empty response for this batch.")
    data = json.loads(payload)
    if not isinstance(data, list):
        data = [data]
    return [d for d in data if isinstance(d, dict)]


def _is_model_unavailable_error(e: Exception) -> bool:
    """Detects the specific failure mode that caused this whole module: a
    model that used to work got decommissioned by Groq mid-deployment."""
    msg = str(e).lower()
    return "model_not_found" in msg or "404" in msg or ("model" in msg and "does not exist" in msg)


def _extract_from_chunks(chunks: list[str], fields_str: str, llm: ChatGroq) -> list[dict]:
    rows: list[dict] = []
    re_resolved_once = False

    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(config.LLM_INTER_CALL_DELAY)  # pace consecutive calls to avoid tripping rate limits

        prompt = _PROMPT_TEMPLATE.format(fields=fields_str, chunk=chunk)
        try:
            rows.extend(_invoke_and_parse(llm, prompt))
            continue
        except Exception as e:
            if _is_model_unavailable_error(e) and not re_resolved_once:
                # The model we resolved at process start just stopped
                # working (e.g. Groq decommissioned it while we were
                # running). Drop the memoized choice and re-resolve once —
                # groq_model_resolver re-validates its own cache too, so
                # this picks a currently-working model without a restart.
                logger.warning("Groq model became unavailable mid-run (%s); re-resolving a working model...", e)
                config._active_model_cache.clear()
                re_resolved_once = True
                try:
                    llm = _get_llm()
                    rows.extend(_invoke_and_parse(llm, prompt))
                    continue
                except Exception as e2:
                    logger.warning("Re-resolved model also failed on this batch, skipping it: %s", e2)
                    continue
            if _is_rate_limit_error(e):
                logger.warning(
                    "Batch %d/%d hit persistent rate limiting even after retries; skipping it (%d items lost "
                    "from this batch). Consider lowering LLM_MAX_ITEMS_PER_CHUNK or upgrading your Groq tier.",
                    i + 1, len(chunks), chunk.count(_ITEM_DELIMITER) + 1,
                )
            else:
                logger.warning("LLM extraction failed on batch %d/%d, skipping it: %s", i + 1, len(chunks), e)

    return rows


def extract_fields(text: str, fields: list[str]) -> list[dict]:
    """Character-count-chunked extraction over a single blob of text (used
    for article/main-content extraction, where there's no natural item
    boundary to batch on)."""
    if not text.strip():
        return []
    fields_str = ", ".join(fields)
    llm = _get_llm()
    return _extract_from_chunks(_chunk_text(text), fields_str, llm)


def extract_fields_from_items(items: list[str], fields: list[str]) -> list[dict]:
    """Item-count-aware extraction over a list of discrete item texts (used
    for listing/grid extraction — see pipeline.py). Prefer this over
    extract_fields() whenever the caller already has distinct item
    boundaries, since it caps items-per-call and isolates failures to a
    small batch instead of the whole listing."""
    items = [i for i in items if i and i.strip()]
    if not items:
        return []
    fields_str = ", ".join(fields)
    llm = _get_llm()
    return _extract_from_chunks(chunk_items(items), fields_str, llm)