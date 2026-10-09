"""
query_cache.py — result cache for Databricks queries, keyed by the sqlglot-normalized SQL.

Hardening over the first version:
  * atomic writes (temp file + os.replace): readers never see a half-written file
  * in-process LRU in front of the disk files: a hit costs no file read / JSON parse
  * bounded: expired entries are deleted, and the directory is capped by entry count and total bytes
  * thread-safe: one lock guards the LRU and eviction bookkeeping
  * identical concurrent queries are coalesced (lookup_or_claim): one thread executes, the rest wait
  * hit/miss counters (get_stats) so cache effectiveness can be reported
  * invalidated by the Airflow marker file (see src/utils/pipeline_marker.py), plus the TTL
"""
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict

import sqlglot

from src.utils.pipeline_marker import DEFAULT_MARKER_PATH, read_marker

logger = logging.getLogger("query_cache")

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".query_cache")

# Configurable TTL (default 1 hour)
TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "200"))
MAX_BYTES = int(os.getenv("QUERY_CACHE_MAX_BYTES", str(50 * 1024 * 1024)))
MEMORY_ITEMS = int(os.getenv("QUERY_CACHE_MEMORY_ITEMS", "64"))
EVICTION_INTERVAL_SECONDS = 60
INFLIGHT_WAIT_SECONDS = 60

# Written by the DAG after each successful dbt_build_ai run; results cached before it are stale.
AI_RUN_MARKER_PATH = DEFAULT_MARKER_PATH

_lock = threading.Lock()
_memory: "OrderedDict[str, dict]" = OrderedDict()   # sql_hash -> entry dict (as stored on disk)
_inflight: dict = {}                                  # sql_hash -> (owner_thread_id, Event)
_last_eviction = 0.0
_stats = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0, "coalesced": 0}


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_stats() -> dict:
    """Snapshot of hit/miss counters plus the derived hit rate."""
    with _lock:
        snap = dict(_stats)
    total = snap["hits"] + snap["misses"]
    snap["hit_rate"] = round(snap["hits"] / total, 3) if total else 0.0
    return snap


def reset_stats() -> None:
    with _lock:
        for k in _stats:
            _stats[k] = 0


def clear_memory() -> None:
    with _lock:
        _memory.clear()


def _get_invalidating_timestamp(tables=None) -> float:
    """Epoch time of the last successful dbt_build_ai run (0.0 if the marker is missing).
    `tables` is accepted for backward compatibility; the marker covers every table."""
    return read_marker(AI_RUN_MARKER_PATH)


def _normalize_sql(sql: str) -> str:
    """Normalizes the SQL string using sqlglot for consistent hashing."""
    try:
        return sqlglot.transpile(sql, read=None, write="databricks")[0]
    except Exception as e:
        logger.debug(f"sqlglot parsing failed for caching, falling back to basic hash: {e}")
        return " ".join(sql.strip().lower().split())


def _hash_sql(normalized_sql: str) -> str:
    return hashlib.sha256(normalized_sql.encode("utf-8")).hexdigest()


def _key(sql: str) -> str:
    return _hash_sql(_normalize_sql(sql))


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _is_fresh(entry: dict) -> bool:
    cached_time = entry.get("cached_time", 0.0)
    if time.time() - cached_time > TTL_SECONDS:
        return False
    # A pipeline run that finished after we cached this result means the underlying tables changed.
    return _get_invalidating_timestamp() <= cached_time


def _remember(key: str, entry: dict) -> None:
    with _lock:
        _memory[key] = entry
        _memory.move_to_end(key)
        while len(_memory) > MEMORY_ITEMS:
            _memory.popitem(last=False)


def _forget(key: str) -> None:
    with _lock:
        _memory.pop(key, None)
    try:
        os.remove(_path(key))
    except OSError:
        pass


def _lookup(key: str):
    """Returns a fresh entry or None. Memory first, then disk (which warms the memory layer)."""
    with _lock:
        entry = _memory.get(key)
        if entry is not None:
            _memory.move_to_end(key)
    if entry is None:
        try:
            with open(_path(key), "r") as f:
                entry = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.warning(f"Unreadable cache file for {key[:12]}: {e}")
            _forget(key)
            return None
        _remember(key, entry)
    if not _is_fresh(entry):
        _forget(key)
        return None
    return entry


def _as_result(entry: dict) -> dict:
    return {
        "columns": entry.get("columns"),
        "rows": entry.get("rows"),
        "row_count": entry.get("row_count"),
        "executed_at": entry.get("executed_at")
    }


def get_cached_result(sql: str) -> dict:
    """
    Attempts to retrieve a cached query result.
    Returns a dict with 'columns', 'rows', 'row_count', 'executed_at', or None if miss/stale.
    Updates the hit/miss counters.
    """
    entry = _lookup(_key(sql))
    with _lock:
        _stats["hits" if entry else "misses"] += 1
    return _as_result(entry) if entry else None


def lookup_or_claim(sql: str):
    """Cache lookup that also coalesces identical concurrent executions.

    Returns (result, is_leader):
      * (result, False)  -> cache hit; use it.
      * (None, True)     -> miss and THIS caller must execute the query, then call set_cached_result()
                            and ALWAYS release_claim() (use try/finally).
      * (None, False)    -> miss but another thread was executing it and did not produce a result in time;
                            execute without a claim.
    If another thread is already executing the same SQL, this waits for it and returns its result."""
    key = _key(sql)
    me = threading.get_ident()
    waiter = None
    with _lock:
        claim = _inflight.get(key)
        if claim is None:
            entry = None
        elif claim[0] == me:
            claim, entry = None, None  # we already own it (duplicate SQL within one batch): just execute
        else:
            waiter = claim[1]
    if waiter is not None:
        waiter.wait(timeout=INFLIGHT_WAIT_SECONDS)
        with _lock:
            _stats["coalesced"] += 1
        entry = _lookup(key)
        with _lock:
            _stats["hits" if entry else "misses"] += 1
        return (_as_result(entry), False) if entry else (None, False)

    entry = _lookup(key)
    if entry:
        with _lock:
            _stats["hits"] += 1
        return _as_result(entry), False

    with _lock:
        _stats["misses"] += 1
        if key not in _inflight:
            _inflight[key] = (me, threading.Event())
            return None, True
    return None, False


def release_claim(sql: str) -> None:
    """Wake any threads waiting on this SQL. Safe to call when no claim is held."""
    key = _key(sql)
    with _lock:
        claim = _inflight.get(key)
        if claim and claim[0] == threading.get_ident():
            del _inflight[key]
        else:
            claim = None
    if claim:
        claim[1].set()


def set_cached_result(sql: str, columns: list, rows: list, row_count: int, executed_at: str):
    """Caches the result of a query (atomically), then evicts if the directory has grown past its bounds."""
    key = _key(sql)
    entry = {
        "cached_time": time.time(),
        "columns": columns,
        "rows": rows,
        "row_count": row_count,
        "executed_at": executed_at
    }
    _remember(key, entry)
    try:
        _ensure_cache_dir()
        tmp = f"{_path(key)}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f:
            json.dump(entry, f, default=str)
        os.replace(tmp, _path(key))  # atomic: concurrent readers see the old file or the new one, never a partial
        with _lock:
            _stats["stores"] += 1
    except (OSError, TypeError, ValueError) as e:
        logger.warning(f"Error writing cache file for {key[:12]}: {e}")
        try:
            os.remove(tmp)
        except (OSError, UnboundLocalError):
            pass
        return
    _maybe_evict()


def _maybe_evict(force: bool = False) -> None:
    """Delete expired entries, then the oldest ones until the directory is within MAX_ENTRIES / MAX_BYTES.
    Rate-limited so the directory is not listed on every write."""
    global _last_eviction
    now = time.time()
    with _lock:
        if not force and now - _last_eviction < EVICTION_INTERVAL_SECONDS:
            return
        _last_eviction = now
    try:
        files = []
        for name in os.listdir(CACHE_DIR):
            full = os.path.join(CACHE_DIR, name)
            if name.endswith(".tmp"):
                if now - os.path.getmtime(full) > 300:  # orphaned by a crashed writer
                    os.remove(full)
                continue
            if name.endswith(".json"):
                st = os.stat(full)
                files.append((st.st_mtime, st.st_size, name))
    except OSError:
        return

    files.sort()  # oldest first
    marker_ts = _get_invalidating_timestamp()
    keep = []
    for mtime, size, name in files:
        if now - mtime > TTL_SECONDS or mtime < marker_ts:
            _forget(name[:-5])
            with _lock:
                _stats["evictions"] += 1
        else:
            keep.append((mtime, size, name))

    total = sum(size for _, size, _ in keep)
    while keep and (len(keep) > MAX_ENTRIES or total > MAX_BYTES):
        _, size, name = keep.pop(0)
        total -= size
        _forget(name[:-5])
        with _lock:
            _stats["evictions"] += 1
