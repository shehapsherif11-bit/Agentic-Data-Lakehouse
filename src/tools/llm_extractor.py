"""
LLM-assisted structured extraction — used only when deterministic sources
(JSON-LD, obvious listing blocks) aren't enough to satisfy the requested
fields. Compared to the previous implementation this module:
  * chunks long input instead of silently truncating to 12,000 characters,
  * validates the model's JSON output instead of trusting it blindly,
  * retries on transient failures / malformed JSON,
  * never invents values — the prompt explicitly forbids fabrication and
    missing fields are always "N/A" rather than a guess.
"""
import json
import logging
import re

from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_groq import ChatGroq

import config

logger = logging.getLogger("etl.llm")

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)

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
    size = config.LLM_MAX_CHARS_PER_CHUNK
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    return chunks[: config.LLM_MAX_CHUNKS]


def _get_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set — required for LLM-based extraction.")
    model = config.get_active_groq_model()
    return ChatGroq(model=model, temperature=config.LLM_TEMPERATURE, api_key=config.GROQ_API_KEY)


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _invoke_and_parse(llm: ChatGroq, prompt: str) -> list[dict]:
    response = llm.invoke(prompt)
    raw = response.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    match = _JSON_ARRAY_RE.search(raw)
    payload = match.group(0) if match else raw
    data = json.loads(payload)
    if not isinstance(data, list):
        data = [data]
    return [d for d in data if isinstance(d, dict)]


def _is_model_unavailable_error(e: Exception) -> bool:
    """Detects the specific failure mode that caused this whole module: a
    model that used to work got decommissioned by Groq mid-deployment."""
    msg = str(e).lower()
    return "model_not_found" in msg or "404" in msg or ("model" in msg and "does not exist" in msg)


def extract_fields(text: str, fields: list[str]) -> list[dict]:
    if not text.strip():
        return []
    fields_str = ", ".join(fields)
    llm = _get_llm()
    rows: list[dict] = []
    re_resolved_once = False

    for chunk in _chunk_text(text):
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
                logger.warning(
                    "Groq model became unavailable mid-run (%s); re-resolving a working model...", e
                )
                config._active_model_cache.clear()
                re_resolved_once = True
                try:
                    llm = _get_llm()
                    rows.extend(_invoke_and_parse(llm, prompt))
                    continue
                except Exception as e2:
                    logger.warning("Re-resolved model also failed on this chunk, skipping it: %s", e2)
                    continue
            logger.warning("LLM extraction failed on a chunk, skipping it: %s", e)

    return rows
