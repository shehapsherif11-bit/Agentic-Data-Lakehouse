# Live demo

A public, free-to-run version of the app. It runs the **same agent, guard and UI** as the full project, but on a frozen data sample in an in-memory DuckDB instead of the Databricks warehouse. No warehouse, no Databricks credentials, no compute cost, no cold start.

## What the data is

`sample_data/` holds Parquet files for the seven Gold tables and a `manifest.json` that the app shows in its banner.

| Part | How it was drawn |
| :--- | :--- |
| The 10 brands with the most orders | Kept **in full**, so "top N restaurants" gives the real top N |
| Every other brand | A deterministic random 3% (hash of the brand name) |
| Branches, orders, items, menu | All rows of every chosen brand |
| Customers | Those who ordered from a chosen brand. E-mails are replaced with `user_<id>@example.com` |
| Customer segments | Recomputed on the sample with the same K-Means recipe as the Airflow task |

The sample was **not** exported from Databricks. `scripts/build_demo_sample.py` runs the project's own dbt models (`zomato_dbt/models`) on DuckDB from the raw CSV files. To prove that this reproduces the real Gold layer, `--verify` rebuilds the core tables on all 10M orders and prints the platform numbers, which match the Databricks figures in `CURRENT_STATE.md`: 10,000,000 orders, 148,541 branches, 112,818 brands, 8.5% failed orders, average order value 483, delivery time 31.1 min, rating 4.15, discount rate 8.8%, `cost` and `rating_count` 100% NULL.

Brand-level numbers for the 10 biggest brands are exact. City totals, customer rankings and platform averages describe the sample, not the whole platform. The app says so in a banner.

## Where DuckDB differs from Databricks, and what the demo does about it

The agent still writes Databricks SQL. `src/utils/demo_database.py` translates it with `sqlglot`, plus three fixes that `sqlglot` does not do:

- `DAYOFWEEK` counts from 1 = Sunday in Databricks, from 0 = Sunday in DuckDB. The demo adds 1.
- Databricks sorts NULLs first on `ASC` and last on `DESC`. The demo sets the same default.
- Databricks timestamps are UTC. The demo pins the time zone to UTC and stores times without a zone.

Each one is covered by a test in `tests/test_demo_backend.py`.

## Run it locally

```bash
pip install -r demo/requirements.txt
GROQ_API_KEY=... streamlit run demo/streamlit_app.py
```

`demo/streamlit_app.py` sets `DEMO_MODE=1` and runs the normal `app.py`.

## Rebuild the sample

Needs the raw CSV files in `data/` and `scikit-learn`.

```bash
pip install -r demo/requirements.txt scikit-learn
python scripts/build_demo_sample.py --pct 3 --featured 10
```

Each rebuild adds about 37 MB of Parquet to the Git history, so rebuild only when the models change.

## Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and click **Create app**, then **Deploy a public app from GitHub**.
2. Repository: this repo. Branch: the one that holds `demo/`. Main file path: `demo/streamlit_app.py`.
3. **Advanced settings**: choose Python 3.12 or newer, and paste the secrets below.
4. Deploy. Dependencies come from `demo/requirements.txt` (it sits next to the main file).

```toml
GROQ_API_KEY = "your key"
OPENROUTER_API_KEY = "your key"   # optional, used when Groq is rate-limited
DEMO_MODE = "1"
DEMO_MAX_QUESTIONS = "15"         # chat questions per visitor session
SESSION_TOKEN_LIMIT = "40000"     # LLM tokens per visitor session
DAILY_TOKEN_LIMIT = "150000"      # LLM tokens per day for ALL visitors together
```

Visitors share one Groq key, so the daily token limit is what protects it. When the limit is reached the app says so politely. Quick reports keep working because they use no LLM.

## Limits to know about

- Groq's free tier allows 8,000 tokens per minute. Under load the agent falls back to OpenRouter, and a fallback call can take 30 seconds or more. Routes that need no LLM (rankings, week-over-week change, quick reports) are not affected.
- The app is stateless between server restarts: query and LLM caches live on the container's disk.
