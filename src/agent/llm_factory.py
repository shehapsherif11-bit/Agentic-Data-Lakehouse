import os
import asyncio
import logging
import threading
import time
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

class CircuitBreakerOpenException(Exception):
    pass


class _BreakerState:
    """Failure counters shared by every CircuitBreakerLLM wrapping the same provider+model.
    Derived runnables (with_structured_output / bind_tools) reuse it, so breaker state
    survives across turns instead of being rebuilt (at zero) on every call."""
    def __init__(self):
        self.failure_count = 0
        self.last_failure_time = 0.0


_breaker_states: Dict[str, _BreakerState] = {}
_breaker_states_lock = threading.Lock()


def _get_breaker_state(key: str) -> _BreakerState:
    with _breaker_states_lock:
        return _breaker_states.setdefault(key, _BreakerState())


def _is_transient(error_str: str) -> bool:
    """Errors worth retrying on the same provider."""
    return "429" in error_str or "rate limit" in error_str or "timeout" in error_str


def _is_provider_failure(error_str: str) -> bool:
    """Errors that mean the provider itself is unhealthy and should count toward the breaker.
    Request-specific failures (400s, schema/validation errors) must not trip it."""
    return _is_transient(error_str) or any(
        s in error_str for s in ("500", "502", "503", "504", "connection", "unavailable", "overloaded")
    )


class CircuitBreakerLLM(Runnable):
    """Wraps an LLM with custom retry, backoff, telemetry, and circuit breaker logic.

    This is the ONLY retry layer for LLM calls (the SDK's own retries are disabled and the
    router no longer stacks tenacity on top). `max_total_seconds` bounds the wall time spent
    retrying one call, so a rate-limited provider fails over to the fallback quickly."""
    def __init__(self, llm: Runnable, max_retries: int = 3, initial_backoff: float = 2.0, max_backoff: float = 10.0,
                 breaker_threshold: int = 5, breaker_cooldown: float = 60.0, max_total_seconds: float = 45.0,
                 state: Optional[_BreakerState] = None):
        self.llm = llm
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown = breaker_cooldown
        self.max_total_seconds = max_total_seconds
        self.state = state or _BreakerState()

    # Back-compat accessors for the shared counters
    @property
    def failure_count(self) -> int:
        return self.state.failure_count

    @property
    def last_failure_time(self) -> float:
        return self.state.last_failure_time

    def _check_breaker(self) -> None:
        if self.state.failure_count >= self.breaker_threshold:
            time_since_failure = time.time() - self.state.last_failure_time
            if time_since_failure < self.breaker_cooldown:
                raise CircuitBreakerOpenException(f"Circuit breaker is OPEN. Cooldown remaining: {self.breaker_cooldown - time_since_failure:.1f}s")
            logger.warning("Circuit breaker half-open: attempting request...")

    def _on_success(self) -> None:
        if self.state.failure_count > 0:
            logger.info("Request succeeded, resetting circuit breaker.")
        self.state.failure_count = 0

    def _on_failure(self, error_str: str) -> None:
        if _is_provider_failure(error_str):
            self.state.failure_count += 1
            self.state.last_failure_time = time.time()
            if self.state.failure_count >= self.breaker_threshold:
                logger.error(f"Circuit breaker TRIPPED! {self.state.failure_count} consecutive failures.")

    def _retry_delay(self, e: Exception, retries: int, backoff: float, started: float) -> Optional[float]:
        """Seconds to wait before the next attempt, or None if the error must propagate now."""
        error_str = str(e).lower()
        if not _is_transient(error_str) or retries >= self.max_retries:
            return None
        if (time.time() - started) + backoff > self.max_total_seconds:
            logger.warning("LLM retry budget (%.0fs) exhausted; failing over.", self.max_total_seconds)
            return None
        logger.warning(f"LLM transient error (attempt {retries+1}/{self.max_retries}). Retrying in {backoff:.1f}s. Error: {e}")
        return backoff

    @staticmethod
    def _with_retry_meta(config: Optional[RunnableConfig], retries: int) -> RunnableConfig:
        # Inject the retry count so the telemetry callback can read it
        config = dict(config) if config else {}
        metadata = dict(config.get("metadata") or {})
        metadata["retry_count"] = retries
        config["metadata"] = metadata
        return config

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        self._check_breaker()
        retries, backoff, started = 0, self.initial_backoff, time.time()
        while True:
            try:
                result = self.llm.invoke(input, config=self._with_retry_meta(config, retries), **kwargs)
                self._on_success()
                return result
            except Exception as e:
                delay = self._retry_delay(e, retries, backoff, started)
                if delay is None:
                    self._on_failure(str(e).lower())
                    raise
                time.sleep(delay)
                retries += 1
                backoff = min(backoff * 2.0, self.max_backoff)

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        """Native async path: awaits the underlying model and sleeps with asyncio, never blocking the loop."""
        self._check_breaker()
        retries, backoff, started = 0, self.initial_backoff, time.time()
        while True:
            try:
                result = await self.llm.ainvoke(input, config=self._with_retry_meta(config, retries), **kwargs)
                self._on_success()
                return result
            except Exception as e:
                delay = self._retry_delay(e, retries, backoff, started)
                if delay is None:
                    self._on_failure(str(e).lower())
                    raise
                await asyncio.sleep(delay)
                retries += 1
                backoff = min(backoff * 2.0, self.max_backoff)

    def _derive(self, llm: Runnable) -> "CircuitBreakerLLM":
        return CircuitBreakerLLM(
            llm,
            max_retries=self.max_retries,
            initial_backoff=self.initial_backoff,
            max_backoff=self.max_backoff,
            breaker_threshold=self.breaker_threshold,
            breaker_cooldown=self.breaker_cooldown,
            max_total_seconds=self.max_total_seconds,
            state=self.state,
        )

    def bind_tools(self, *args, **kwargs):
        return self._derive(self.llm.bind_tools(*args, **kwargs))

    def with_structured_output(self, *args, **kwargs):
        return self._derive(self.llm.with_structured_output(*args, **kwargs))


def get_llm(model_name: str = None, temperature: float = 0.0, max_retries: int = 3, request_timeout: int = 15,
            reasoning_effort: Optional[str] = None, enable_openrouter: bool = True,
            max_total_seconds: float = 45.0):
    """Build the provider chain: Groq (primary) -> OpenRouter (fallback).

    reasoning_effort ("low" | "medium" | "high") is forwarded to Groq reasoning models
    (gpt-oss). Reasoning tokens were ~50-60% of output, so lowering effort on
    classification-style nodes is the cheapest latency win.
    enable_openrouter=False builds a Groq-only runnable (used for cross-model fallbacks so a
    single failure doesn't walk through every provider twice)."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY") if enable_openrouter else None

    primary = None
    fallback = None
    model = model_name if (model_name and not model_name.startswith("nvidia/")) else "openai/gpt-oss-120b"

    # Primary: Groq
    if groq_api_key:
        logger.info(f"Initializing Groq LLM (model: {model}, reasoning_effort: {reasoning_effort})")
        groq_kwargs = {}
        if reasoning_effort and "gpt-oss" in model:  # other models reject the parameter
            groq_kwargs["reasoning_effort"] = reasoning_effort
        # SDK max_retries=0 to disable invisible retries!
        primary = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=groq_api_key,
            max_retries=0,
            request_timeout=request_timeout,
            **groq_kwargs,
        )

    # Secondary: OpenRouter
    if openrouter_api_key:
        fb_model = model_name or os.getenv("GENERAL_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        fallback = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            model=fb_model,
            temperature=temperature,
            max_retries=0,
            request_timeout=35
        )

    def _wrap(llm, provider, retries):
        return CircuitBreakerLLM(
            llm, max_retries=retries, max_total_seconds=max_total_seconds,
            state=_get_breaker_state(f"{provider}:{llm.model_name if hasattr(llm, 'model_name') else model}"),
        )

    if primary and fallback:
        # Breaker throws after max_retries; `with_fallbacks` then tries OpenRouter once.
        safe_primary = _wrap(primary, "groq", max_retries)
        safe_fallback = _wrap(fallback, "openrouter", 1)  # Don't retry fallback too much
        return safe_primary.with_fallbacks([safe_fallback])
    elif primary:
        return _wrap(primary, "groq", max_retries)
    elif fallback:
        return _wrap(fallback, "openrouter", max_retries)
    else:
        raise ValueError("Neither GROQ_API_KEY nor OPENROUTER_API_KEY found in environment.")
