"""
Content understanding layer.

Given raw HTML this module answers three questions:
  1. Does the page already carry machine-readable structured data (JSON-LD)?
     If so, that is the most reliable and cheapest source of truth and
     should be preferred over any LLM guesswork. (OpenGraph/meta tags are
     handled separately — see the note on extract_opengraph() below.)
  2. Does the page look like a *listing* of repeated items (product grid,
     article list, table rows)? If so, isolate each item's block of text so
     downstream extraction can work item-by-item instead of guessing at the
     page as a whole.
  3. Otherwise, what is the single main "article" content of the page, with
     boilerplate (nav, ads, footers, cookie banners) stripped out?
It also finds "next page" links for pagination, and detects bot-protection
/ CDN-challenge pages so they're never mistaken for real content.
"""
import json
import re
import logging
from collections import Counter
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger("etl.content")

BOILERPLATE_TAGS = ["script", "style", "noscript", "header", "footer", "nav", "svg", "form", "iframe"]
BOILERPLATE_HINTS = re.compile(r"(cookie|newsletter|subscribe|advert|breadcrumb|sidebar|^nav$)", re.I)

# Phrases that reliably show up on bot-protection / CDN-challenge / access-
# denied pages (Akamai, Cloudflare, PerimeterX, Imperva, generic WAFs).
# These pages are almost always short and dominated by one of these
# phrases — a page that's long AND happens to mention e.g. "captcha" in
# passing (a security blog) should NOT be flagged, hence the length guard
# in looks_like_blocked_page().
BLOCK_PAGE_SIGNATURES = (
    "access denied", "reference #", "edgesuite.net", "akamai",
    "request blocked", "attention required", "cloudflare",
    "are you a human", "pardon our interruption", "captcha",
    "unusual traffic", "403 forbidden", "just a moment",
    "checking your browser", "bot detection", "security check",
    "verify you are a human", "ray id",
)


class BlockedPageError(Exception):
    """Raised when a fetched page looks like a bot-protection / CDN
    challenge / access-denied page rather than real site content. The
    pipeline must never silently extract one of these as if it were the
    requested data."""


def looks_like_blocked_page(html: str) -> bool:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.extract()
    text = soup.get_text(" ", strip=True).lower()
    if len(text) > 1500:  # real pages are rarely this short; avoid false positives on long legit pages
        return False
    return any(sig in text for sig in BLOCK_PAGE_SIGNATURES)


def extract_json_ld(html: str) -> list[dict]:
    """Pull schema.org JSON-LD blocks only. This is a deliberate structured
    declaration by the site author (Product, Offer, ItemList, Article,
    etc.) — a strong, trustworthy signal of real data, unlike generic
    OpenGraph/meta tags (see extract_opengraph() below)."""
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for c in candidates:
            if isinstance(c, dict) and isinstance(c.get("@graph"), list):
                candidates.extend(c["@graph"])
                continue
            if isinstance(c, dict):
                records.append(c)

    return records


def extract_opengraph(html: str) -> dict:
    """Pull OpenGraph/meta tags (og:title, description, author, ...).

    IMPORTANT: these are generic SEO/social-sharing tags present on almost
    every modern webpage — a blog, a SaaS landing page, and a real product
    listing all have them. They are NOT evidence that the page contains
    the specific data the user asked for, and must never be treated as
    "the answer" the way JSON-LD can be. Callers should only fall back to
    this as an absolute last resort, after listing detection and main-
    content extraction have both failed to find anything more specific —
    and should clearly flag the result as generic page metadata, not the
    requested data."""
    soup = BeautifulSoup(html, "lxml")
    og = {}
    for tag in soup.find_all("meta"):
        prop = tag.get("property") or tag.get("name")
        if prop and (prop.startswith("og:") or prop in ("description", "author")):
            og[prop.replace("og:", "")] = tag.get("content")
    return og


def extract_structured_data(html: str) -> list[dict]:
    """Backward-compatible convenience wrapper: JSON-LD if present,
    otherwise a single OpenGraph record. New code should prefer calling
    extract_json_ld() and extract_opengraph() separately so it can apply
    different trust levels to each (see pipeline.py)."""
    records = extract_json_ld(html)
    if records:
        return records
    og = extract_opengraph(html)
    return [og] if og else []


def extract_main_content(html: str, url: str) -> dict:
    """Readable article/body text + metadata, boilerplate stripped."""
    downloaded = trafilatura.extract(
        html, url=url, output_format="json", with_metadata=True,
        favor_precision=True, include_comments=False, include_tables=True,
    )
    if downloaded:
        try:
            return json.loads(downloaded)
        except json.JSONDecodeError:
            pass
    # Fallback if trafilatura can't parse it at all: brute-force strip + get_text
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(BOILERPLATE_TAGS):
        tag.extract()
    text = "\n".join(l.strip() for l in soup.get_text("\n").splitlines() if l.strip())
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    return {"title": title, "text": text, "url": url}


def find_repeated_blocks(html: str, min_repeats: int = 4) -> list[dict]:
    """
    Heuristic listing detector: find the most common (tag, class) signature
    among elements that hold a meaningful amount of text, and return each
    matching element's cleaned text (+ first link) as one "item". This is how
    we recognize product grids, article cards, search results, etc. without
    writing a per-site scraper.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(BOILERPLATE_TAGS):
        tag.extract()

    signature_counts: Counter = Counter()
    signature_to_elements: dict = {}
    for el in soup.find_all(["div", "li", "article", "tr", "section"]):
        classes = el.get("class") or []
        if not classes:
            continue
        sig = (el.name, tuple(sorted(classes)))
        text_len = len(el.get_text(strip=True))
        if text_len < 8 or text_len > 4000:
            continue
        signature_counts[sig] += 1
        signature_to_elements.setdefault(sig, []).append(el)

    if not signature_counts:
        return []

    best_sig, count = signature_counts.most_common(1)[0]
    if count < min_repeats:
        return []

    blocks = []
    for el in signature_to_elements[best_sig]:
        cls_str = " ".join(el.get("class") or [])
        if BOILERPLATE_HINTS.search(cls_str):
            continue
        text = el.get_text("\n", strip=True)
        link = el.find("a", href=True)
        blocks.append({"text": text, "href": link["href"] if link else None})
    return blocks


def find_next_page(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")

    rel_next = soup.find("link", rel="next") or soup.find("a", rel="next")
    if rel_next and rel_next.get("href"):
        return urljoin(current_url, rel_next["href"])

    for a in soup.find_all("a", href=True):
        label = a.get_text(strip=True).lower()
        if label in ("next", "next page", "»", "load more"):
            return urljoin(current_url, a["href"])

    return None