"""
build_demo_sample.py - builds the frozen data sample of the public live demo (DEMO_MODE=1) WITHOUT Databricks.

It runs the project's own dbt models (zomato_dbt/models/silver and gold) on DuckDB, starting from the raw CSV
files in data/. The SQL is the same SQL that builds the real Gold layer; only the Jinja (config, source, ref,
incremental blocks) is resolved here and the Spark dialect is translated with sqlglot. Output: one Parquet file
per Gold table plus manifest.json, which the demo UI shows so nobody mistakes the sample for the full dataset.

How the sample is drawn: a deterministic pseudo-random subset of restaurant BRANDS (chosen by hashing the
trimmed brand name, not the order). Every branch, order, order item and menu row of a chosen brand is kept, so
brand-level numbers inside the sample are exact. Customers are the ones who ordered from a chosen brand.
Customer e-mails are replaced with user_<id>@example.com. Customer segments are recomputed on the sample with
the same K-Means recipe as airflow/dags/AI/ml_build_segments.py.

Usage (repo root; needs duckdb, sqlglot, pandas, pyarrow, scikit-learn):
    python scripts/build_demo_sample.py --pct 3     # build the sample
    python scripts/build_demo_sample.py --verify    # rebuild the core tables on ALL data and print the
                                                    # platform numbers, to compare with the Databricks Gold layer
"""
import argparse
import datetime
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.demo_database import to_duckdb  # noqa: E402  (one Databricks -> DuckDB translation, shared with the demo)

RAW = ROOT / "data"
MODELS = ROOT / "zomato_dbt" / "models"
BRONZE, SILVER, GOLD = "workspace.zomato_bronze", "workspace.zomato_silver", "workspace.zomato_gold"

# Dependency order. dim_users needs ai_customer_segments first, so it runs last.
SILVER_MODELS = ["silver_restaurant", "silver_users", "silver_food", "silver_menu", "silver_orders", "silver_order_items"]
GOLD_MODELS = ["dim_resturant", "dim_date", "dim_menu", "fact_orders", "fact_order_items"]
CORE_MODELS = ["silver_restaurant", "silver_orders", "dim_resturant", "fact_orders"]  # what --verify needs
SCHEMA_OF = {**{m: SILVER for m in SILVER_MODELS}, **{m: GOLD for m in GOLD_MODELS + ["dim_users"]}}

CSV_OPTIONS = "header=true, all_varchar=true, quote='\"', escape='\"'"


def csv(name: str) -> str:
    return f"read_csv('{(RAW / name).as_posix()}', {CSV_OPTIONS})"


def underscored(con, name: str) -> str:
    """The CSV with spaces in column names turned into underscores ("Family size" -> Family_size), which is how
    the Databricks bronze load names them."""
    cols = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {csv(name)}").fetchall()]
    return "SELECT " + ", ".join(f'"{c}" AS "{c.replace(" ", "_")}"' for c in cols) + f" FROM {csv(name)}"


def render(model: str) -> str:
    """dbt model -> plain DuckDB SQL. Full refresh: the is_incremental() blocks are dropped."""
    path = next(MODELS.rglob(f"{model}.sql"))
    sql = path.read_text(encoding="utf-8")
    sql = re.sub(r"\{\{\s*config\(.*?\)\s*\}\}", "", sql, flags=re.S)
    sql = re.sub(r"\{%\s*if is_incremental\(\)\s*%\}.*?\{%\s*endif\s*%\}", "", sql, flags=re.S)
    sql = re.sub(r"\{\{\s*source\('zomato_bronze',\s*'(\w+)'\)\s*\}\}", rf"{BRONZE}.\1", sql)
    sql = re.sub(r"\{\{\s*ref\('(\w+)'\)\s*\}\}", lambda m: f"{SCHEMA_OF[m.group(1)]}.{m.group(1)}", sql)
    if "{{" in sql or "{%" in sql:
        raise RuntimeError(f"{model}: unresolved Jinja left in the model")
    # Spark casts the text '29.0' to 29. DuckDB refuses a decimal string for INT. Go through DOUBLE and drop the
    # fraction, as Spark does (every INT column here is a non-negative id, count or number of minutes).
    sql = re.sub(r"TRY_CAST\((\w+) AS INT\)", r"TRY_CAST(FLOOR(TRY_CAST(\1 AS DOUBLE)) AS INT)", sql)
    return to_duckdb(sql)


def count(con, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def run_model(con, model: str):
    started = time.time()
    table = f"{SCHEMA_OF[model]}.{model}"
    con.execute(f"CREATE OR REPLACE TABLE {table} AS {render(model)}")
    print(f"  {model:20} {count(con, table):>12,} rows  ({time.time() - started:.1f}s)", flush=True)


def featured_brands(con, n: int) -> list[str]:
    """The n brands with the most orders in ALL the data, so a "top N restaurants" question gives the real top N."""
    rows = con.execute(f"""
        SELECT TRIM(r.name) AS name
        FROM {csv('restaurant.csv')} r
        JOIN (SELECT r_id, COUNT(*) AS n FROM {csv('orders.csv')} GROUP BY r_id) o ON o.r_id = r.id
        GROUP BY TRIM(r.name) ORDER BY SUM(o.n) DESC, name LIMIT {int(n)}""").fetchall()
    return [r[0] for r in rows]


def load_bronze(con, pct, featured=()):
    """Raw CSVs as all-text bronze tables. With pct, keep a random pct% of brands plus the featured brands."""
    print("Loading raw CSVs...", flush=True)
    for schema in ("zomato_bronze", "zomato_silver", "zomato_gold"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS workspace.{schema}")
    brand_filter = ""
    if pct is not None:
        brand_filter = f"WHERE (hash(TRIM(name)) % 10000) < {round(pct * 100)}"
        if featured:
            con.execute(f"CREATE OR REPLACE TABLE {BRONZE}.featured AS SELECT UNNEST(?) AS name", [list(featured)])
            brand_filter += f" OR TRIM(name) IN (SELECT name FROM {BRONZE}.featured)"
    con.execute(f"CREATE OR REPLACE TABLE {BRONZE}.bronze_restaurant AS SELECT * FROM {csv('restaurant.csv')} {brand_filter}")
    con.execute(f"CREATE OR REPLACE TABLE {BRONZE}.sample_ids AS SELECT DISTINCT id FROM {BRONZE}.bronze_restaurant")
    keep = "" if pct is None else f"WHERE r_id IN (SELECT id FROM {BRONZE}.sample_ids)"
    tables = [("bronze_orders", "orders.csv"), ("bronze_menu", "menu.csv")]
    if pct is not None:  # the full order-items file (23M rows) is only needed for the sample itself
        tables.append(("bronze_order_items", "order_items.csv"))
    for table, file in tables:
        started = time.time()
        con.execute(f"CREATE OR REPLACE TABLE {BRONZE}.{table} AS SELECT * FROM {csv(file)} {keep}")
        print(f"  {table:20} {count(con, f'{BRONZE}.{table}'):>12,} rows  ({time.time() - started:.1f}s)", flush=True)
    user_filter = "" if pct is None else f"WHERE user_id IN (SELECT DISTINCT user_id FROM {BRONZE}.bronze_orders)"
    con.execute(f"CREATE OR REPLACE TABLE {BRONZE}.bronze_users AS SELECT * FROM ({underscored(con, 'users.csv')}) {user_filter}")
    con.execute(f"CREATE OR REPLACE TABLE {BRONZE}.bronze_food AS SELECT * FROM {csv('food.csv')}")


def build_segments(con):
    """Same recipe as airflow/dags/AI/ml_build_segments.py (frequency, AOV, recency -> K-Means with 4 clusters),
    run on the sample. Writes workspace.zomato_gold.ai_customer_segments."""
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    df = con.execute(f"""
        SELECT fo.user_id, COUNT(DISTINCT fo.order_id) AS frequency, AVG(fo.sales_amount) AS aov,
               MAX(dd.full_date) AS last_order_date
        FROM {GOLD}.fact_orders fo JOIN {GOLD}.dim_date dd ON fo.date_id = dd.date_id
        GROUP BY fo.user_id""").df()
    df["last_order_date"] = pd.to_datetime(df["last_order_date"])
    df["recency"] = (df["last_order_date"].max() - df["last_order_date"]).dt.days
    features = df[["frequency", "aov", "recency"]].fillna(0)
    df["cluster"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(features))
    centers = df.groupby("cluster")[["frequency", "aov", "recency"]].mean()
    centers["inv_recency"] = -centers["recency"]
    scaled = MinMaxScaler().fit_transform(centers[["frequency", "aov", "inv_recency"]])
    scores = sorted(
        ((i, s[0] * 0.4 + s[1] * 0.4 + s[2] * 0.2, centers.loc[i, "recency"]) for i, s in zip(centers.index, scaled)),
        key=lambda x: x[1], reverse=True,
    )
    high_value, loyal = scores[0][0], scores[1][0]
    at_risk, occasional = [x[0] for x in sorted(scores[2:], key=lambda x: x[2], reverse=True)]
    df["customer_segment"] = df["cluster"].map(
        {high_value: "High Value", loyal: "Loyal", occasional: "Occasional", at_risk: "At Risk"}
    )
    seg = df[["user_id", "customer_segment"]].astype({"user_id": "int32"})
    con.register("seg_df", seg)
    con.execute(
        f"CREATE OR REPLACE TABLE {GOLD}.ai_customer_segments AS "
        "SELECT user_id::INT AS user_id, customer_segment::VARCHAR AS customer_segment FROM seg_df"
    )
    print(f"  {'ai_customer_segments':20} {len(seg):>12,} rows", flush=True)


def verify():
    """Rebuild the core tables on ALL data and print the platform numbers that CURRENT_STATE.md records from
    the real Databricks Gold layer, so the two can be compared."""
    tmp = Path(tempfile.mkdtemp(prefix="demo_verify_"))
    try:
        con = duckdb.connect(":memory:")
        con.execute("SET TimeZone = 'UTC'")
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")
        con.execute(f"ATTACH '{(tmp / 'verify.duckdb').as_posix()}' AS workspace")
        load_bronze(con, None)
        print("Running the core dbt models on all data...", flush=True)
        for m in CORE_MODELS:
            run_model(con, m)

        def one(sql):
            return con.execute(sql).fetchone()

        failed = "CASE WHEN order_status IN ('Cancelled','Refunded') THEN 1 ELSE 0 END"
        stats = {
            "orders": one(f"SELECT COUNT(*) FROM {GOLD}.fact_orders")[0],
            "branches": one(f"SELECT COUNT(*) FROM {GOLD}.dim_resturant")[0],
            "brands": one(f"SELECT COUNT(DISTINCT restaurant_name) FROM {GOLD}.dim_resturant")[0],
            "date_range": [str(x) for x in one(f"SELECT MIN(order_timestamp)::DATE, MAX(order_timestamp)::DATE FROM {GOLD}.fact_orders")],
            "failed_order_pct": round(one(f"SELECT 100.0 * AVG({failed}) FROM {GOLD}.fact_orders")[0], 2),
            "aov": round(one(f"SELECT SUM(sales_amount) / COUNT(DISTINCT order_id) FROM {GOLD}.fact_orders")[0], 1),
            "avg_delivery_min": round(one(f"SELECT AVG(delivery_time_min) FROM {GOLD}.fact_orders")[0], 2),
            "avg_customer_rating": round(one(f"SELECT AVG(customer_rating) FROM {GOLD}.fact_orders")[0], 2),
            "discount_rate_pct": round(one(f"SELECT 100.0 * SUM(discount) / SUM(subtotal) FROM {GOLD}.fact_orders")[0], 2),
            "unrated_branches_pct": round(one(f"SELECT 100.0 * AVG(CASE WHEN rating IS NULL THEN 1 ELSE 0 END) FROM {GOLD}.dim_resturant")[0], 1),
            "cost_null_pct": round(one(f"SELECT 100.0 * AVG(CASE WHEN cost IS NULL THEN 1 ELSE 0 END) FROM {GOLD}.dim_resturant")[0], 1),
            "rating_count_null_pct": round(one(f"SELECT 100.0 * AVG(CASE WHEN rating_count IS NULL THEN 1 ELSE 0 END) FROM {GOLD}.dim_resturant")[0], 1),
        }
        per_branch = one(
            f"SELECT MEDIAN(c), QUANTILE_CONT(c, 0.1), QUANTILE_CONT(c, 0.9) FROM "
            f"(SELECT COUNT(*) AS c FROM {GOLD}.fact_orders GROUP BY restaurant_id)"
        )
        stats["orders_per_branch_median_p10_p90"] = [round(x, 1) for x in per_branch]
        print(json.dumps(stats, indent=2))
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pct", type=float, default=3.0, help="percent of the other brands to keep at random (default 3)")
    ap.add_argument("--featured", type=int, default=10, help="also keep, in full, the N brands with the most orders (default 10)")
    ap.add_argument("--out", default=str(ROOT / "demo" / "sample_data"))
    ap.add_argument("--verify", action="store_true", help="rebuild the core tables on all data and print platform numbers")
    args = ap.parse_args()
    if args.verify:
        return verify()

    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone = 'UTC'")  # Databricks warehouses default to UTC; a raw timestamp means the same thing here
    con.execute("ATTACH ':memory:' AS workspace")
    featured = featured_brands(con, args.featured)
    print("Featured brands (kept in full):", ", ".join(featured), flush=True)
    load_bronze(con, args.pct, featured)
    print("Running the dbt models on DuckDB...", flush=True)
    for m in SILVER_MODELS + GOLD_MODELS:
        run_model(con, m)
    build_segments(con)
    run_model(con, "dim_users")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.parquet"):
        old.unlink()
    for table in GOLD_MODELS + ["dim_users", "ai_customer_segments"]:
        replace = []
        for column, dtype, *_ in con.execute(f"DESCRIBE {GOLD}.{table}").fetchall():
            if dtype == "TIMESTAMP WITH TIME ZONE":  # stored as UTC wall-clock time, so no reader depends on a time zone
                replace.append(f"CAST({column} AS TIMESTAMP) AS {column}")
        if table == "dim_users":  # no real e-mail addresses in a public repo
            replace.append("'user_' || CAST(user_id AS VARCHAR) || '@example.com' AS email")
        select = f"SELECT * REPLACE ({', '.join(replace)}) FROM {GOLD}.{table}" if replace else f"SELECT * FROM {GOLD}.{table}"
        target = (out / f"{table}.parquet").as_posix()
        con.execute(f"COPY ({select} ORDER BY 1) TO '{target}' (FORMAT parquet, COMPRESSION zstd)")

    def one(sql):
        return con.execute(sql).fetchone()[0]

    manifest = {
        "sample_pct": args.pct,
        "featured_brands": featured,
        "method": "a deterministic hash sample of restaurant brands plus the brands with the most orders; all branches, orders, items and menu rows of a chosen brand are kept; built by running the project's dbt models on DuckDB from the raw CSV files",
        "customer_segments": "recomputed on the sample with the same K-Means recipe as the Airflow task",
        "orders": one(f"SELECT COUNT(*) FROM {GOLD}.fact_orders"),
        "order_items": one(f"SELECT COUNT(*) FROM {GOLD}.fact_order_items"),
        "brands": one(f"SELECT COUNT(DISTINCT restaurant_name) FROM {GOLD}.dim_resturant"),
        "branches": one(f"SELECT COUNT(*) FROM {GOLD}.dim_resturant"),
        "customers": one(f"SELECT COUNT(*) FROM {GOLD}.dim_users"),
        "date_from": str(one(f"SELECT MIN(order_timestamp)::DATE FROM {GOLD}.fact_orders")),
        "date_to": str(one(f"SELECT MAX(order_timestamp)::DATE FROM {GOLD}.fact_orders")),
        "full_orders": one(f"SELECT COUNT(*) FROM {csv('orders.csv')}"),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "emails_masked": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Parquet total: {sum(f.stat().st_size for f in out.glob('*.parquet')) / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
