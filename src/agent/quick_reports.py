"""
quick_reports.py - one-click reports for the UI.

Each report is fixed SQL: no LLM, no routing. It still goes through the SAME safety guard and the SAME query
cache as every question (execute_guarded in analyst_graph.py), so a report is as safe as a chat answer and the
second click is served from the cache.

The SQL uses only constructs that run unchanged on Databricks and, after transpiling, on the demo's DuckDB.
"""
from dataclasses import dataclass

G = "workspace.zomato_gold"


@dataclass(frozen=True)
class QuickReport:
    id: str
    title: str
    description: str
    sql: str
    kind: str   # "line" | "bar" | "barh"
    x: str
    y: str


REPORTS: tuple[QuickReport, ...] = (
    QuickReport(
        id="monthly_revenue",
        title="Revenue by month",
        description="Total revenue and orders for every complete calendar month. Partial months at the edges of the data are left out.",
        sql=f"""
SELECT DATE_TRUNC('month', fo.order_timestamp) AS month,
       ROUND(SUM(fo.sales_amount), 2) AS revenue,
       COUNT(DISTINCT fo.order_id) AS orders
FROM {G}.fact_orders fo
GROUP BY DATE_TRUNC('month', fo.order_timestamp)
HAVING CAST(MIN(fo.order_timestamp) AS DATE) = CAST(DATE_TRUNC('month', MIN(fo.order_timestamp)) AS DATE)
   AND CAST(MAX(fo.order_timestamp) AS DATE) = LAST_DAY(MAX(fo.order_timestamp))
ORDER BY month""",
        kind="line", x="month", y="revenue",
    ),
    QuickReport(
        id="top_brands",
        title="Top 10 restaurants by orders",
        description="Brands ranked by number of orders. Ties are broken alphabetically.",
        sql=f"""
SELECT dr.restaurant_name,
       COUNT(DISTINCT fo.order_id) AS orders,
       ROUND(SUM(fo.sales_amount), 2) AS revenue
FROM {G}.fact_orders fo
JOIN {G}.dim_resturant dr ON fo.restaurant_id = dr.restaurant_id
GROUP BY dr.restaurant_name
ORDER BY orders DESC, dr.restaurant_name
LIMIT 10""",
        kind="barh", x="restaurant_name", y="orders",
    ),
    QuickReport(
        id="city_revenue",
        title="Revenue by city (top 10)",
        description="The ten cities with the highest revenue.",
        sql=f"""
SELECT dr.city,
       ROUND(SUM(fo.sales_amount), 2) AS revenue,
       COUNT(DISTINCT fo.order_id) AS orders
FROM {G}.fact_orders fo
JOIN {G}.dim_resturant dr ON fo.restaurant_id = dr.restaurant_id
GROUP BY dr.city
ORDER BY revenue DESC, dr.city
LIMIT 10""",
        kind="barh", x="city", y="revenue",
    ),
    QuickReport(
        id="failed_rate_by_city",
        title="Failed-order rate by city",
        description="Share of orders that were cancelled or refunded. Only cities with at least 100 orders, so a tiny city cannot top the list by chance.",
        sql=f"""
SELECT dr.city,
       COUNT(*) AS orders,
       ROUND(100.0 * SUM(CASE WHEN fo.order_status IN ('Cancelled', 'Refunded') THEN 1 ELSE 0 END) / COUNT(*), 2) AS failed_order_rate_pct
FROM {G}.fact_orders fo
JOIN {G}.dim_resturant dr ON fo.restaurant_id = dr.restaurant_id
GROUP BY dr.city
HAVING COUNT(*) >= 100
ORDER BY failed_order_rate_pct DESC, dr.city
LIMIT 10""",
        kind="barh", x="city", y="failed_order_rate_pct",
    ),
    QuickReport(
        id="orders_by_weekday",
        title="Orders by day of week",
        description="How many orders land on each weekday.",
        sql=f"""
SELECT dd.day_name,
       COUNT(DISTINCT fo.order_id) AS orders
FROM {G}.fact_orders fo
JOIN {G}.dim_date dd ON fo.date_id = dd.date_id
GROUP BY dd.day_name
ORDER BY orders DESC, dd.day_name""",
        kind="bar", x="day_name", y="orders",
    ),
)

BY_ID = {r.id: r for r in REPORTS}


def run_report(report: QuickReport, execute) -> dict:
    """Run one report. `execute` is analyst_graph.execute_guarded. Returns {success, columns, rows, error, sql}."""
    return execute([{"sql": report.sql.strip(), "purpose": f"quick report: {report.id}"}])[0]
