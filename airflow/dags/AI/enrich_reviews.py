"""
enrich_reviews.py — sentiment-classify restaurant reviews with Groq, cheaply.

Token / rate-limit friendly by design:
  * incremental: only reviews not yet in ai_enriched_reviews are classified (re-runs cost nothing)
  * capped: ENRICH_MAX_REVIEWS (default 500, 0 = no cap) keeps test runs small
  * deduplicated: identical comments are classified once and fanned back out
  * batched: BATCH_SIZE reviews per request instead of one request per review
  * async + bounded: httpx.AsyncClient behind a semaphore, honouring Groq's 429 `retry-after`
  * failures are NOT written, so the next run simply retries them
"""
import asyncio
import json
import os
import random

import databricks.sql
import httpx
from dotenv import load_dotenv
load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = os.getenv("ENRICH_MODEL", "openai/gpt-oss-20b")      # small + fast model is plenty for 3-way sentiment
BATCH_SIZE = int(os.getenv("ENRICH_BATCH_SIZE", "25"))
CONCURRENCY = int(os.getenv("ENRICH_CONCURRENCY", "4"))       # simultaneous in-flight requests
MAX_REVIEWS = int(os.getenv("ENRICH_MAX_REVIEWS", "500"))     # 0 = classify everything outstanding
MAX_RETRIES = 5
MAX_COMMENT_CHARS = 400                                       # sentiment is clear from the start of a review
INSERT_CHUNK = 2000

SENTIMENTS = {"positive": "Positive", "negative": "Negative", "neutral": "Neutral"}

SYSTEM_PROMPT = (
    "You classify restaurant review sentiment. For every numbered review return exactly one label: "
    "Positive, Negative or Neutral. Respond with ONLY a JSON object of the form "
    '{"results": [{"id": <number>, "sentiment": "<label>"}, ...]} covering every id.'
)


async def _classify_batch(client: httpx.AsyncClient, sem: asyncio.Semaphore, batch: dict[int, str]) -> dict[int, str]:
    """batch: {local_id: comment}. Returns {local_id: 'Positive'|'Negative'|'Neutral'} for the ids the model answered."""
    user = "\n".join(f"{i}. {text[:MAX_COMMENT_CHARS]!r}" for i, text in batch.items())
    payload = {
        "model": MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
    }
    if "gpt-oss" in MODEL:
        payload["reasoning_effort"] = "low"  # classification needs no long chain of thought
    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                resp = await client.post(GROQ_URL, json=payload)
            except httpx.HTTPError:
                resp = None
        if resp is not None and resp.status_code == 200:
            try:
                items = json.loads(resp.json()["choices"][0]["message"]["content"])["results"]
                return {int(it["id"]): SENTIMENTS[str(it["sentiment"]).strip().lower()]
                        for it in items if str(it.get("sentiment", "")).strip().lower() in SENTIMENTS and int(it["id"]) in batch}
            except (KeyError, ValueError, TypeError):
                pass  # malformed answer: retry
        elif resp is not None and resp.status_code not in (429, 500, 502, 503, 504):
            print(f"Groq request failed ({resp.status_code}): {resp.text[:200]}")
            return {}
        # 429 / 5xx / transport error / malformed answer: back off (honour retry-after when Groq sends it)
        wait = float(resp.headers.get("retry-after", 0)) if resp is not None else 0.0
        await asyncio.sleep(max(wait, min(2 ** attempt, 30)) + random.random())
    print(f"Giving up on a batch of {len(batch)} reviews after {MAX_RETRIES} attempts (will be retried next run).")
    return {}


async def classify_comments(comments: list[str]) -> dict[str, str]:
    """Classifies unique comments in batches, concurrently. Returns {comment: sentiment} for the ones that succeeded."""
    batches = [dict(enumerate(comments[i:i + BATCH_SIZE])) for i in range(0, len(comments), BATCH_SIZE)]
    sem = asyncio.Semaphore(CONCURRENCY)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=60, limits=limits) as client:
        results = await asyncio.gather(*(_classify_batch(client, sem, b) for b in batches))
    return {batch[i]: s for batch, res in zip(batches, results) for i, s in res.items()}


def process_reviews():
    connection = databricks.sql.connect(
        server_hostname=DATABRICKS_HOST, http_path=DATABRICKS_HTTP_PATH, access_token=DATABRICKS_TOKEN
    )
    cursor = connection.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace.zomato_gold.ai_enriched_reviews (
                review_id INT, comment STRING, sentiment STRING
            )
        """)

        # Incremental: only reviews that have no sentiment yet
        cursor.execute(f"""
            SELECT r.review_id, r.comment
            FROM workspace.zomato_silver.silver_reviews r
            LEFT ANTI JOIN workspace.zomato_gold.ai_enriched_reviews e ON r.review_id = e.review_id
            {f'LIMIT {MAX_REVIEWS}' if MAX_REVIEWS > 0 else ''}
        """)
        pending = cursor.fetchall()
        if not pending:
            print("No new reviews to process.")
            return

        rows = [(int(rid), (comment or "").strip()) for rid, comment in pending]
        unique = sorted({c for _, c in rows if c})
        print(f"{len(rows)} reviews to score -> {len(unique)} unique comments -> "
              f"{-(-len(unique) // BATCH_SIZE)} requests (model={MODEL}, concurrency={CONCURRENCY}).")

        sentiment_of = asyncio.run(classify_comments(unique)) if unique else {}
        sentiment_of[""] = "Neutral"  # empty comments need no API call

        scored = [(rid, sentiment_of[c]) for rid, c in rows if c in sentiment_of]
        print(f"Scored {len(scored)}/{len(rows)} reviews; {len(rows) - len(scored)} left for the next run.")

        # Bulk write: only (id, sentiment) travel in the SQL (ints + a fixed label set, nothing to escape);
        # the comment text is joined back from silver_reviews inside Databricks.
        for i in range(0, len(scored), INSERT_CHUNK):
            values = ", ".join(f"({rid}, '{s}')" for rid, s in scored[i:i + INSERT_CHUNK])
            cursor.execute(f"""
                INSERT INTO workspace.zomato_gold.ai_enriched_reviews
                SELECT r.review_id, r.comment, v.sentiment
                FROM workspace.zomato_silver.silver_reviews r
                JOIN (VALUES {values}) AS v(review_id, sentiment) ON r.review_id = v.review_id
            """)
        print("Done.")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    process_reviews()
