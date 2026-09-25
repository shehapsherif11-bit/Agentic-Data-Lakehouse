"""
Content understanding layer.

Given raw HTML this module answers three questions:
  1. Does the page already carry machine-readable structured data (JSON-LD /
     OpenGraph / meta tags)? If so, that is the most reliable and cheapest
     source of truth and should be preferred over any LLM guesswork.
  2. Does the page look like a *listing* of repeated items (product grid,
     article list, table rows)? If so, isolate each item's block of text so
     downstream extraction can work item-by-item instead of guessing at the
     page as a whole.
  3. Otherwise, what is the single main "article" content of the page, with
     boilerplate (nav, ads, footers, cookie banners) stripped out?
It also finds "next page" links for pagination.
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


def extract_structured_data(html: str) -> list[dict]:
    """Pull schema.org JSON-LD blocks, with OpenGraph/meta as a fallback."""
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

    if not records:
        og = {}
        for tag in soup.find_all("meta"):
            prop = tag.get("property") or tag.get("name")
            if prop and (prop.startswith("og:") or prop in ("description", "author")):
                og[prop.replace("og:", "")] = tag.get("content")
        if og:
            records.append(og)

    return records


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