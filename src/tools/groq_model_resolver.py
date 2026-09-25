"""
groq_model_resolver.py

Self-healing selection of a Groq chat model, so a hardcoded model ID never
becomes a single point of failure again. Groq periodically decommissions
models (see https://console.groq.com/docs/deprecations) — that's exactly
what caused `llama-3.1-8b-instant` (and `llama-3.3-70b-versatile`) to start
returning `404 model_not_found` here; they were both decommissioned in
mid/late-2026.

Instead of pinning one model ID in code, this module:
  1. Fetches Groq's LIVE model catalog once (a single cheap GET request —
     NOT a test-invoke of every model, which is what the old code did on
     every startup).
  2. Scores every *currently listed* text-chat model with a small,
     name-based heuristic — bigger/newer generally scores higher, and
     audio/vision/guard/preview-only models are excluded — instead of
     hardcoding specific IDs that will themselves eventually be retired.
  3. Verifies the top-scoring candidate actually answers a 1-token ping;
     if that specific model errors (access/tier issue, mid-rollout
     deprecation, etc.) it falls through to the next-best candidate,
     trying at most a handful of models, not the whole catalog.
  4. Caches the resolved model id to disk with a short TTL, so normal runs
     don't re-hit the API every time, but the cache expires and
     re-resolves on its own — no code change or redeploy needed the next
     time Groq retires whatever we're currently using.
"""
import json
import logging
import os
import re
import time
from typing import Callable

import requests

logger = logging.getLogger("groq_model_resolver")

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
CACHE_TTL_SECONDS = 6 * 60 * 60  # re-validate every 6 hours
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".groq_model_cache.json")
MAX_CANDIDATES_TO_PROBE = 5  # only probe a handful of top-ranked models, never the whole catalog

# Model families listed in Groq's catalog that aren't a fit for general
# text chat (audio, vision, moderation/guard, agentic tool-wrappers, TTS).
EXCLUDE_SUBSTRINGS = (
    "whisper", "tts", "guard", "vision", "llava", "orpheus",
    "prompt-guard", "safeguard", "compound",
)
# Model families we consider viable general-purpose chat/instruct models.
ALLOWED_FAMILIES = ("gpt-oss", "llama", "qwen", "gemma", "mixtral", "deepseek", "kimi")
# Naming patterns that usually indicate preview/narrower-access tiers —
# still usable as a last resort, but scored down since they're more likely
# to be gated, waitlisted, or pulled with little notice.
PREVIEW_HINTS = ("preview", "scout", "maverick")


def _score(model_id: str) -> float:
    """Higher is better. Negative means disqualified. Deliberately
    name-pattern-based (not a hardcoded ID list) so it keeps working as
    Groq's catalog changes."""
    mid = model_id.lower()
    if any(bad in mid for bad in EXCLUDE_SUBSTRINGS):
        return -1.0
    if not any(fam in mid for fam in ALLOWED_FAMILIES):
        return -1.0

    size_match = re.search(r"(\d+)b", mid)
    size = int(size_match.group(1)) if size_match else 0

    if "gpt-oss" in mid:       # Groq's current flagship production-tier family
        family_bonus = 1000
    elif "llama" in mid:
        family_bonus = 500
    elif "qwen" in mid:
        family_bonus = 400
    else:
        family_bonus = 200

    preview_penalty = 300 if any(h in mid for h in PREVIEW_HINTS) else 0
    return family_bonus + size - preview_penalty


def _fetch_live_models(api_key: str) -> list[str]:
    resp = requests.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    resp.raise_for_status()
    return [m["id"] for m in resp.json().get("data", [])]


def _read_cache() -> str | None:
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if time.time() - data.get("ts", 0) < CACHE_TTL_SECONDS:
            return data.get("model")
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def _write_cache(model: str) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"model": model, "ts": time.time()}, f)
    except OSError:
        pass


def resolve_groq_model(
    api_key: str,
    chat_model_factory: Callable[[str], object],
    fallback: str = "openai/gpt-oss-20b",
) -> str:
    """
    Returns a Groq model id that is verified to work right now.

    `chat_model_factory(model_id)` must return an object with `.invoke(...)`
    (e.g. a `ChatGroq` instance) — injected as a callable so this module has
    no hard dependency on langchain_groq and is trivial to unit test with a
    fake factory instead of hitting the real Groq API.
    """
    cached = _read_cache()
    if cached:
        try:
            chat_model_factory(cached).invoke("hi")
            logger.info("Using cached, verified Groq model: %s", cached)
            return cached
        except Exception as e:
            logger.info("Cached model %s no longer works (%s); re-resolving...", cached, e)

    try:
        live_models = _fetch_live_models(api_key)
    except Exception as e:
        logger.warning("Could not fetch the live Groq model list (%s); using fallback: %s", e, fallback)
        _write_cache(fallback)
        return fallback

    ranked = sorted((m for m in live_models if _score(m) > 0), key=_score, reverse=True)

    if not ranked:
        logger.warning("No suitable chat model found in the live Groq catalog; using fallback: %s", fallback)
        _write_cache(fallback)
        return fallback

    for candidate in ranked[:MAX_CANDIDATES_TO_PROBE]:
        try:
            chat_model_factory(candidate).invoke("hi")
            logger.info("Resolved a working Groq model: %s", candidate)
            _write_cache(candidate)
            return candidate
        except Exception as e:
            logger.info("Candidate model %s failed validation (%s); trying the next one...", candidate, e)

    logger.warning("None of the top %d candidates validated; using fallback: %s", MAX_CANDIDATES_TO_PROBE, fallback)
    _write_cache(fallback)
    return fallback
