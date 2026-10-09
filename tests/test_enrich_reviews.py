import asyncio
import importlib.util
import json
import os

import httpx
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "airflow", "dags", "AI", "enrich_reviews.py")
spec = importlib.util.spec_from_file_location("enrich_reviews", _PATH)
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)


def _ok(results):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"results": results})}}]})


_REAL_SLEEP = asyncio.sleep


def _run(handler, comments, monkeypatch, batch_size=25, delays=None):
    monkeypatch.setattr(er, "BATCH_SIZE", batch_size)
    monkeypatch.setattr(er, "CONCURRENCY", 4)
    async def fake_sleep(seconds):                      # no real backoff in tests, but record what was requested
        if delays is not None:
            delays.append(seconds)
        await _REAL_SLEEP(0)
    monkeypatch.setattr(er.asyncio, "sleep", fake_sleep)
    transport = httpx.MockTransport(handler)
    real = httpx.AsyncClient
    monkeypatch.setattr(er.httpx, "AsyncClient", lambda **kw: real(transport=transport, **{k: v for k, v in kw.items() if k != "limits"}))
    return asyncio.run(er.classify_comments(comments))


def test_batches_reviews_into_few_requests(monkeypatch):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        n = len(body["messages"][1]["content"].splitlines())
        return _ok([{"id": i, "sentiment": "positive"} for i in range(n)])

    out = _run(handler, [f"review {i}" for i in range(60)], monkeypatch, batch_size=25)
    assert len(calls) == 3                                  # 60 reviews -> 3 requests, not 60
    assert len(out) == 60 and set(out.values()) == {"Positive"}
    assert calls[0]["response_format"] == {"type": "json_object"} and calls[0]["temperature"] == 0


def test_retries_on_429_honouring_retry_after_then_succeeds(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"retry-after": "1"}, text="rate limited")
        return _ok([{"id": 0, "sentiment": "Negative"}])

    delays = []
    out = _run(handler, ["awful"], monkeypatch, delays=delays)
    assert out == {"awful": "Negative"} and len(attempts) == 3
    assert len(delays) == 2 and all(d >= 1 for d in delays)   # waited at least Groq's retry-after each time


def test_gives_up_after_max_retries_and_returns_nothing(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(429, text="slow down")

    assert _run(handler, ["x"], monkeypatch) == {}
    assert len(attempts) == er.MAX_RETRIES


def test_non_retryable_error_fails_fast(monkeypatch):
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(401, text="bad key")

    assert _run(handler, ["x"], monkeypatch) == {} and len(attempts) == 1


def test_partial_and_invalid_answers_only_score_valid_ids(monkeypatch):
    def handler(request):
        return _ok([{"id": 0, "sentiment": "positive"}, {"id": 1, "sentiment": "angry"}, {"id": 99, "sentiment": "neutral"}])

    out = _run(handler, ["great", "meh", "unscored"], monkeypatch)
    assert out == {"great": "Positive"}   # invalid label and out-of-range id ignored; missing id left for next run
