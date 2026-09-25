"""
Fetch layer: tries a fast static HTTP request first, and only pays the cost
of a full headless browser when the page actually needs JS to render content.
This is the single biggest performance win over the old design, which always
launched Selenium/Chrome for every website regardless of whether it needed it.
"""
import asyncio
import hashlib
import os
import time
import logging
from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

logger = logging.getLogger("etl.fetchers")


@dataclass
class FetchResult:
    url: str
    html: str
    status_code: int
    used_browser: bool


def _cache_path(url: str) -> str:
    key = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(config.CACHE_DIR, f"{key}.html")


def _read_cache(url: str) -> str | None:
    path = _cache_path(url)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < config.CACHE_TTL_SECONDS:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _write_cache(url: str, html: str) -> None:
    try:
        with open(_cache_path(url), "w", encoding="utf-8") as f:
            f.write(html)
    except OSError:
        pass


@retry(
    reraise=True,
    stop=stop_after_attempt(config.HTTP_RETRIES),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=6),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
)
async def _fetch_static(client: httpx.AsyncClient, url: str) -> httpx.Response:
    resp = await client.get(url, headers={"User-Agent": config.HTTP_USER_AGENT}, timeout=config.HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp


def _looks_js_rendered(html: str) -> bool:
    """Heuristic: very little visible text relative to markup usually means
    the real content is injected client-side (React/Vue/Angular shells)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(strip=True)
    return len(text) < config.MIN_STATIC_TEXT_CHARS


async def fetch(url: str, use_cache: bool = True, force_browser: bool = False) -> FetchResult:
    if use_cache and not force_browser:
        cached = _read_cache(url)
        if cached is not None:
            logger.info("cache hit: %s", url)
            return FetchResult(url=url, html=cached, status_code=200, used_browser=False)

    if not force_browser:
        try:
            async with httpx.AsyncClient(follow_redirects=True, http2=True) as client:
                resp = await _fetch_static(client, url)
            if not _looks_js_rendered(resp.text):
                _write_cache(url, resp.text)
                return FetchResult(url=url, html=resp.text, status_code=resp.status_code, used_browser=False)
            logger.info("static fetch looks like a JS shell, falling back to browser: %s", url)
        except Exception as e:
            logger.warning("static fetch failed (%s), falling back to browser: %s", e, url)

    html = await _fetch_with_browser(url)
    _write_cache(url, html)
    return FetchResult(url=url, html=html, status_code=200, used_browser=True)


async def _fetch_with_browser(url: str) -> str:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=config.BROWSER_HEADLESS)
        try:
            page = await browser.new_page(user_agent=config.HTTP_USER_AGENT)
            await page.goto(url, timeout=config.BROWSER_NAV_TIMEOUT_MS, wait_until=config.BROWSER_WAIT_UNTIL)
            html = await page.content()
            return html
        finally:
            await browser.close()


async def fetch_many(urls: list[str], use_cache: bool = True) -> list[FetchResult]:
    """Bounded-concurrency fetch for pagination / multi-page crawls."""
    sem = asyncio.Semaphore(config.CONCURRENT_PAGE_FETCHES)

    async def _one(u):
        async with sem:
            try:
                return await fetch(u, use_cache=use_cache)
            except Exception as e:
                logger.error("failed to fetch %s: %s", u, e)
                return None

    results = await asyncio.gather(*[_one(u) for u in urls])
    return [r for r in results if r is not None]