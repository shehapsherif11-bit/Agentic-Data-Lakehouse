"""
concurrency.py — small helper for running independent blocking calls (LLM / DB) in parallel.

LangGraph nodes here are synchronous (and run fine under both .invoke/.stream and
.ainvoke/.astream, where LangGraph moves them to a worker thread). Inside a node we fan out
independent calls with threads. Each task runs in a *copy* of the caller's contextvars so the
telemetry stage/turn-record context and LangChain callbacks follow the work into the thread.
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Any


def run_parallel(fns: List[Callable[[], Any]], max_workers: int = 4) -> List[Any]:
    """Runs zero-arg callables concurrently; returns results in input order.
    The first exception raised by any callable is re-raised after all have finished."""
    if not fns:
        return []
    if len(fns) == 1:
        return [fns[0]()]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(fns))) as ex:
        futures = [ex.submit(contextvars.copy_context().run, fn) for fn in fns]
        results, first_error = [], None
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                results.append(None)
                first_error = first_error or e
    if first_error:
        raise first_error
    return results
