"""
llm_cache.py — response cache for deterministic (temperature=0) LLM calls.

The key is a hash of (namespace, model, effort, full prompt text). Because the prompt already
contains the question, chat history, schema context and metric definitions, any change to the
inputs (including a schema/metrics change) produces a different key, so a hit is only ever
served for a byte-identical request. Only values the caller marks valid are stored, so parse
failures are never cached.

Two layers: an in-process LRU (no disk read per hit) backed by atomic JSON files on disk
(survives restarts; TTL-bound). Identical concurrent requests are coalesced: the second caller
waits for the first instead of paying for a duplicate call.
"""
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Tuple

try:
    from . import router_config as cfg
except ImportError:
    import router_config as cfg

logger = logging.getLogger("llm_cache")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llm_cache")

_memory: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
_lock = threading.Lock()
_inflight: dict[str, threading.Event] = {}
stats = {"hits": 0, "misses": 0, "coalesced": 0}


def make_key(namespace: str, model: str, text: str) -> str:
    return hashlib.sha256(f"{namespace}\x00{model}\x00{text}".encode("utf-8")).hexdigest()


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _lookup(key: str):
    """Returns (found, value). Memory first, then disk; expired entries are dropped."""
    ttl = cfg.LLM_CACHE_TTL_SECONDS
    now = time.time()
    with _lock:
        entry = _memory.get(key)
        if entry:
            if now - entry[0] <= ttl:
                _memory.move_to_end(key)
                return True, entry[1]
            del _memory[key]
    try:
        with open(_path(key), "r", encoding="utf-8") as f:
            data = json.load(f)
        if now - data["cached_time"] <= ttl:
            _remember(key, data["cached_time"], data["value"])
            return True, data["value"]
        os.remove(_path(key))
    except (OSError, ValueError, KeyError):
        pass
    return False, None


def _remember(key: str, cached_time: float, value: Any) -> None:
    with _lock:
        _memory[key] = (cached_time, value)
        _memory.move_to_end(key)
        while len(_memory) > cfg.LLM_CACHE_MEMORY_ITEMS:
            _memory.popitem(last=False)


def _store(key: str, value: Any) -> None:
    now = time.time()
    _remember(key, now, value)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = f"{_path(key)}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"cached_time": now, "value": value}, f, ensure_ascii=False)
        os.replace(tmp, _path(key))  # atomic: readers never see a half-written file
    except (OSError, TypeError, ValueError) as e:
        logger.warning("Could not persist LLM cache entry: %s", e)


def get_or_compute(namespace: str, model: str, text: str, compute: Callable[[], Any],
                   is_valid: Callable[[Any], bool] = lambda v: v is not None) -> Tuple[Any, bool]:
    """Returns (value, was_cache_hit). `compute()` must return a JSON-serialisable value."""
    if not cfg.LLM_CACHE_ENABLED:
        return compute(), False

    key = make_key(namespace, model, text)
    found, value = _lookup(key)
    if found:
        stats["hits"] += 1
        logger.info("LLM cache HIT [%s]", namespace)
        return value, True

    with _lock:
        waiter = _inflight.get(key)
        if waiter is None:
            _inflight[key] = threading.Event()
    if waiter is not None:
        # Another thread is already computing this exact request: wait, then read its result.
        waiter.wait(timeout=120)
        stats["coalesced"] += 1
        found, value = _lookup(key)
        if found:
            return value, True
        # Leader failed or result was invalid; compute ourselves (no coalescing on this retry).
        return compute(), False

    try:
        stats["misses"] += 1
        value = compute()
        if is_valid(value):
            _store(key, value)
        return value, False
    finally:
        with _lock:
            event = _inflight.pop(key, None)
        if event:
            event.set()


def clear() -> None:
    """Drop all cached entries (memory + disk). Used by tests and for manual invalidation."""
    with _lock:
        _memory.clear()
    try:
        for name in os.listdir(CACHE_DIR):
            if name.endswith(".json"):
                os.remove(os.path.join(CACHE_DIR, name))
    except OSError:
        pass
