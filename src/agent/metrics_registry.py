"""
Semantic/Metric Resolver + Data Grain/Join Safety module.
"""

from typing import Dict, List, Optional, Tuple, Any

# ==========================================
# 1. TABLE_GRAINS
# ==========================================
TABLE_GRAINS = {
    "workspace.zomato_gold.fact_orders": {
        "alias": "fo",
        "grain": "one row per order",
        "pk": "order_id",
        "relationships": {
            "dim_date": {"join_key": "date_id", "type": "many-to-one"},
            "dim_resturant": {"join_key": "restaurant_id", "type": "many-to-one"},
            "dim_users": {"join_key": "user_id", "type": "many-to-one"},
            "fact_order_items": {"join_key": "order_id", "type": "one-to-many"}
        }
    },
    "workspace.zomato_gold.fact_order_items": {
        "alias": "fi",
        "grain": "one row per order line item",
        "pk": None,
        "relationships": {
            "fact_orders": {"join_key": "order_id", "type": "many-to-one"},
            "dim_menu": {"join_key": "food_id", "type": "many-to-one"}
        }
    },
    "workspace.zomato_gold.dim_date": {
        "alias": "dd",
        "grain": "one row per date",
        "pk": "date_id",
        "relationships": {}
    },
    "workspace.zomato_gold.dim_menu": {
        "alias": "dm",
        "grain": "one row per menu item",
        "pk": "menu_id",
        "relationships": {
            "dim_resturant": {"join_key": "restaurant_id", "type": "many-to-one"}
        }
    },
    "workspace.zomato_gold.dim_resturant": {
        "alias": "dr",
        "grain": "one row per restaurant branch (pk: restaurant_id). Note: multiple branches share the same restaurant_name (Brand level). To analyze by restaurant/brand, ALWAYS group by restaurant_name.",
        "pk": "restaurant_id",
        "relationships": {}
    },
    "workspace.zomato_gold.dim_users": {
        "alias": "du",
        "grain": "one row per user",
        "pk": "user_id",
        "relationships": {}
    },
    "workspace.zomato_gold.ai_customer_segments": {
        "alias": "cs",
        "grain": "one row per user",
        "pk": "user_id",
        "relationships": {
            "dim_users": {"join_key": "user_id", "type": "one-to-one"}
        }
    }
}

# ==========================================
# 2. COLUMN_CATALOG
# ==========================================
COLUMN_CATALOG = {
    "workspace.zomato_gold.fact_orders": {
        "order_id": "int", "user_id": "int", "restaurant_id": "int", "date_id": "int",
        "order_timestamp": "timestamp", "restaurant_city": "string", "order_status": "string",
        "items_count": "int", "sales_qty": "int", "subtotal": "double", "discount": "double",
        "delivery_fee": "double", "gst": "double", "sales_amount": "double",
        "customer_rating": "double", "delivery_time_min": "int"
    },
    "workspace.zomato_gold.fact_order_items": {
        "order_id": "int", "food_id": "string", "quantity": "int", "price": "double", "line_amount": "double"
    },
    "workspace.zomato_gold.dim_date": {
        "date_id": "int", "full_date": "date", "year": "int", "quarter": "int",
        "quarter_name": "string", "month_number": "int", "month_name": "string",
        "month_short": "string", "day_number": "int", "day_name": "string", "day_type": "string"
    },
    "workspace.zomato_gold.dim_menu": {
        "menu_id": "string", "restaurant_id": "int", "food_id": "string", "item_name": "string",
        "veg_or_non_veg": "string", "cuisine": "string", "price": "double"
    },
    "workspace.zomato_gold.dim_resturant": {
        "restaurant_id": "int", "restaurant_name": "string", "city": "string", "rating": "double",
        "rating_category": "string", "rating_count": "int", "cost": "double", "cuisine": "string",
        "affordability_tier": "string", "cuisine_focus": "string", "popularity_tier": "string"
    },
    "workspace.zomato_gold.dim_users": {
        "user_id": "int", "name": "string", "email": "string", "age": "int", "age_group": "string",
        "gender": "string", "marital_status": "string", "occupation": "string",
        "monthly_income": "string", "educational_qualifications": "string",
        "family_size": "int", "family_segment": "string"
    },
    "workspace.zomato_gold.ai_customer_segments": {
        "user_id": "int", "customer_segment": "string"
    }
}

# ==========================================
# 3. METRICS
# ==========================================
METRICS = {
    # DERIVABLE metrics
    "revenue": {
        "display_name": "Revenue",
        "sql_expression": "SUM(fo.sales_amount)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "sales_amount")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Total sales amount",
        "aggregation_type": "SUM"
    },
    "total_orders": {
        "display_name": "Total Orders",
        "sql_expression": "COUNT(DISTINCT fo.order_id)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "order_id")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Total number of unique orders",
        "aggregation_type": "COUNT"
    },
    "aov": {
        "display_name": "Average Order Value",
        "sql_expression": "SUM(fo.sales_amount) / NULLIF(COUNT(DISTINCT fo.order_id), 0)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "sales_amount"), ("workspace.zomato_gold.fact_orders", "order_id")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Average revenue per order",
        "aggregation_type": "CALCULATION"
    },
    "total_discount": {
        "display_name": "Total Discount",
        "sql_expression": "SUM(fo.discount)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "discount")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Total discount amount",
        "aggregation_type": "SUM"
    },
    "discount_rate": {
        "display_name": "Discount Rate",
        "sql_expression": "SUM(fo.discount) / NULLIF(SUM(fo.subtotal), 0) * 100",
        "required_columns": [("workspace.zomato_gold.fact_orders", "discount"), ("workspace.zomato_gold.fact_orders", "subtotal")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Percentage of discount over subtotal",
        "aggregation_type": "CALCULATION"
    },
    "avg_rating": {
        "display_name": "Average Rating",
        "sql_expression": "AVG(fo.customer_rating)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "customer_rating")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Average customer rating for orders",
        "aggregation_type": "AVG"
    },
    "avg_delivery_time": {
        "display_name": "Average Delivery Time",
        "sql_expression": "AVG(fo.delivery_time_min)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "delivery_time_min")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Average delivery time in minutes",
        "aggregation_type": "AVG"
    },
    "items_per_order": {
        "display_name": "Items per Order",
        "sql_expression": "AVG(fo.items_count)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "items_count")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Average number of items per order",
        "aggregation_type": "AVG"
    },
    "total_customers": {
        "display_name": "Total Customers",
        "sql_expression": "COUNT(DISTINCT fo.user_id)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "user_id")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Total number of unique customers",
        "aggregation_type": "COUNT"
    },
    "avg_order_per_customer": {
        "display_name": "Average Order per Customer",
        "sql_expression": "COUNT(DISTINCT fo.order_id) / NULLIF(COUNT(DISTINCT fo.user_id), 0)",
        "required_columns": [("workspace.zomato_gold.fact_orders", "order_id"), ("workspace.zomato_gold.fact_orders", "user_id")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Average number of orders per customer",
        "aggregation_type": "CALCULATION"
    },
    "total_quantity_sold": {
        "display_name": "Total Quantity Sold",
        "sql_expression": "SUM(fi.quantity)",
        "required_columns": [("workspace.zomato_gold.fact_order_items", "quantity")],
        "source_table": "workspace.zomato_gold.fact_order_items",
        "description": "Total quantity of items sold",
        "aggregation_type": "SUM"
    },
    "avg_item_price": {
        "display_name": "Average Item Price",
        "sql_expression": "AVG(fi.price)",
        "required_columns": [("workspace.zomato_gold.fact_order_items", "price")],
        "source_table": "workspace.zomato_gold.fact_order_items",
        "description": "Average price of items sold",
        "aggregation_type": "AVG"
    },
    
    # MISSING metrics
    "profit": {
        "display_name": "Profit",
        "sql_expression": None,
        "required_columns": [],
        "source_table": None,
        "description": "Profit (Revenue - Cost)",
        "aggregation_type": "CALCULATION",
        "missing_reason": "needs cost/COGS data - missing. Available alternatives: revenue (sales_amount), total_orders, average order value (AOV), and discount_rate."
    },
    "profit_margin": {
        "display_name": "Profit Margin",
        "sql_expression": None,
        "required_columns": [],
        "source_table": None,
        "description": "Profit margin percentage",
        "aggregation_type": "CALCULATION",
        "missing_reason": "needs cost data - missing. Available alternatives: revenue, total_orders, average order value (AOV), and discount_rate."
    },
    "conversion_rate": {
        "display_name": "Conversion Rate",
        "sql_expression": None,
        "required_columns": [],
        "source_table": None,
        "description": "Order conversion rate",
        "aggregation_type": "CALCULATION",
        "missing_reason": "needs visitor/session data - missing"
    },
    "retention_rate": {
        "display_name": "Retention Rate",
        "sql_expression": None,
        "required_columns": [],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Customer retention rate over time",
        "aggregation_type": "COMPLEX",
        "missing_reason": "needs a cohort / repeat-purchase definition that is not in the metric catalog yet"
    },

    # TEMPORAL metric templates
    "revenue_growth_wow": {
        "display_name": "Revenue Growth (WoW)",
        "sql_template": "WITH weekly AS (SELECT dd.year, WEEK(dd.full_date) as week_num, SUM(fo.sales_amount) as rev FROM {schema}.fact_orders fo JOIN {schema}.dim_date dd ON fo.date_id = dd.date_id GROUP BY 1, 2) SELECT rev, LAG(rev) OVER(ORDER BY year, week_num) as prev_rev FROM weekly",
        "required_columns": [("workspace.zomato_gold.fact_orders", "sales_amount"), ("workspace.zomato_gold.dim_date", "date_id")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Week-over-Week revenue growth using LAG",
        "aggregation_type": "TEMPORAL"
    },
    "revenue_growth_mom": {
        "display_name": "Revenue Growth (MoM)",
        "sql_template": "WITH monthly AS (SELECT dd.year, dd.month_number, SUM(fo.sales_amount) as rev FROM {schema}.fact_orders fo JOIN {schema}.dim_date dd ON fo.date_id = dd.date_id GROUP BY 1, 2) SELECT rev, LAG(rev) OVER(ORDER BY year, month_number) as prev_rev FROM monthly",
        "required_columns": [("workspace.zomato_gold.fact_orders", "sales_amount"), ("workspace.zomato_gold.dim_date", "date_id")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Month-over-Month revenue growth using LAG",
        "aggregation_type": "TEMPORAL"
    },
    "order_growth_mom": {
        "display_name": "Order Growth (MoM)",
        "sql_template": "WITH monthly AS (SELECT dd.year, dd.month_number, COUNT(DISTINCT fo.order_id) as ord_cnt FROM {schema}.fact_orders fo JOIN {schema}.dim_date dd ON fo.date_id = dd.date_id GROUP BY 1, 2) SELECT ord_cnt, LAG(ord_cnt) OVER(ORDER BY year, month_number) as prev_ord_cnt FROM monthly",
        "required_columns": [("workspace.zomato_gold.fact_orders", "order_id"), ("workspace.zomato_gold.dim_date", "date_id")],
        "source_table": "workspace.zomato_gold.fact_orders",
        "description": "Month-over-Month order count growth",
        "aggregation_type": "TEMPORAL"
    }
}

# ==========================================
# 4. Functions
# ==========================================
def resolve_metric(metric_name: str) -> dict:
    """Resolve a business metric name to its definition.
    Returns: {name, status: 'available'|'derivable'|'missing'|'complex', 
              definition: dict or None, missing_reason: str or None}
    Uses fuzzy matching on metric names (e.g., 'average order value' -> 'aov')."""
    name_lower = metric_name.lower().strip()
    name_clean = name_lower.replace("_", " ").replace("-", " ")
    
    # Aliases mapping
    aliases = {
        "sales": "revenue",
        "total revenue": "revenue",
        "average order value": "aov",
        "avg order value": "aov",
        "total sales": "revenue",
        "number of orders": "total_orders",
        "customers": "total_customers",
        "average rating": "avg_rating",
        "order count": "total_orders",
        "sales change": "revenue",
        "sales_change": "revenue",
        "sales decline": "revenue",
        "sales_decline": "revenue",
        "sales drop": "revenue",
        "sales_drop": "revenue",
        "sales growth": "revenue",
        "sales_growth": "revenue",
        "revenue change": "revenue",
        "revenue_change": "revenue",
        "revenue decline": "revenue",
        "revenue_decline": "revenue",
        "revenue drop": "revenue",
        "revenue_drop": "revenue",
        "contribution": "revenue",
        "contribution_to_change": "revenue",
        "contribution to change": "revenue",
        "contribution share": "revenue",
        "contribution_share": "revenue",
        "restaurant contribution": "revenue",
        "restaurant_contribution": "revenue",
        "customer retention": "retention_rate",
        "retention": "retention_rate",
        "churn": "retention_rate",
        "churn rate": "retention_rate",
        "order change": "total_orders",
        "order decline": "total_orders",
        "order drop": "total_orders",
        "order growth": "total_orders",
    }
    
    # Try exact, clean, or direct aliases
    resolved_name = aliases.get(name_clean, aliases.get(name_lower, name_lower))
    
    # Strip common analytical suffixes if still unresolved
    if resolved_name not in METRICS:
        import re
        stripped = re.sub(r'_(change|decline|drop|growth|difference|delta|loss|contribution)$', '', name_lower)
        stripped = re.sub(r'^(contribution_to_|restaurant_)', '', stripped)
        if stripped in aliases:
            resolved_name = aliases[stripped]
        elif stripped in METRICS:
            resolved_name = stripped
    
    # Check directly first
    if resolved_name in METRICS:
        metric = METRICS[resolved_name]
        if metric.get("sql_expression") is None and "sql_template" not in metric:
            status = "complex" if metric.get("aggregation_type") == "COMPLEX" else "missing"
            return {
                "name": resolved_name,
                "status": status,
                "definition": metric,
                "missing_reason": metric.get("missing_reason")
            }
        else:
            return {
                "name": resolved_name,
                "status": "derivable",
                "definition": metric,
                "missing_reason": None
            }
            
    # Fuzzy match logic with strict word boundaries
    import re
    search_term = resolved_name.strip()
    if search_term:
        search_pattern = re.compile(rf"\b{re.escape(search_term)}\b", re.IGNORECASE)
        for key, value in METRICS.items():
            disp_lower = value["display_name"].lower()
            if search_pattern.search(disp_lower) or search_pattern.search(key.replace("_", " ")):
                if value.get("sql_expression") is None and "sql_template" not in value:
                    status = "complex" if value.get("aggregation_type") == "COMPLEX" else "missing"
                    return {
                        "name": key,
                        "status": status,
                        "definition": value,
                        "missing_reason": value.get("missing_reason")
                    }
                return {
                    "name": key,
                    "status": "derivable",
                    "definition": value,
                    "missing_reason": None
                }
            
    return {
        "name": metric_name,
        "status": "missing",
        "definition": None,
        "missing_reason": "Unknown metric"
    }

# Metrics whose per-group values add up to the overall total, so "share of total" is meaningful for them.
ADDITIVE_METRICS = ("revenue", "total_orders", "total_discount")

_DERIVED_TOKENS = {"percentage", "percent", "pct", "share", "contribution", "proportion"}
_DERIVED_FILLER = _DERIVED_TOKENS | {"of", "to", "total", "overall", "the", "in", "from"}


def split_derived_metric(metric_name: str) -> tuple:
    """('percentage_of_total_revenue') -> ('revenue', ['share_of_total']).

    The intent LLM names a derived column as if it were a metric; refusing it as "not in the catalog" is wrong
    when the base metric exists. Only touches names the catalog cannot already resolve. Returns (base_or_None,
    derived); (metric_name, []) when the name is not a derived measure."""
    name = str(metric_name or "")
    if resolve_metric(name)["status"] not in ("missing", "complex"):
        return name, []
    tokens = name.lower().replace("_", " ").replace("-", " ").split()
    if not (_DERIVED_TOKENS & set(tokens)):
        return name, []
    base = " ".join(t for t in tokens if t not in _DERIVED_FILLER)
    if base and resolve_metric(base)["status"] in ("missing", "complex"):
        return name, []   # a base we cannot resolve either: keep the original so the user is told what is missing
    return (base or None), ["share_of_total"]


def list_available_metrics() -> list[str]:
    """Display names of every metric the catalog can actually compute (the source of truth for 'what can I ask?')."""
    return [m["display_name"] for m in METRICS.values()
            if m.get("sql_expression") is not None or "sql_template" in m]


_RETENTION_ALTERNATIVES_EN = (
    "Retention itself is not defined, but these related measures are available: customers per period, "
    "average orders per customer, and the K-Means customer segments (e.g. 'At Risk', 'Loyal')."
)
_RETENTION_ALTERNATIVES_AR = (
    "الاحتفاظ بالعملاء (retention) نفسه مش معرّف، لكن فيه مقاييس قريبة متاحة: عدد العملاء، "
    "متوسط الطلبات لكل عميل، وشرائح العملاء (مثل 'At Risk' و 'Loyal')."
)


def alternatives_hint(missing: list[dict], ar: bool = False) -> str:
    """Suggest REAL alternatives for the missing metrics. Never substitutes silently: the caller shows it as a suggestion."""
    names = " ".join(str(m.get("name", "")).lower() for m in missing)
    if any(k in names for k in ("retention", "churn", "loyalty")):
        return _RETENTION_ALTERNATIVES_AR if ar else _RETENTION_ALTERNATIVES_EN
    if any(k in names for k in ("profit", "margin", "cost")):
        return ("البديل المتاح: الإيراد، عدد الطلبات، متوسط قيمة الطلب، ونسبة الخصم." if ar
                else "Available alternatives: revenue, total orders, average order value and discount rate.")
    return ""


def check_data_sufficiency(required_metrics: list[str], required_dimensions: list[str]) -> dict:
    """Check if all required metrics and dimensions can be computed.
    Returns: {sufficient: bool, available: list, missing: list[{name, reason}], warnings: list}"""
    available = []
    missing = []
    warnings = []
    
    all_cols = []
    for table, cols in COLUMN_CATALOG.items():
        all_cols.extend([c.lower() for c in cols.keys()])

    for metric in required_metrics:
        resolved = resolve_metric(metric)
        if resolved["status"] in ("missing", "complex"):
            if metric.lower() in all_cols:
                warnings.append(f"Note: '{metric}' was requested as a metric but it is actually a column/dimension. Proceeding.")
            elif metric.lower() == "years" or metric.lower() == "age": # Hardcode common mistakes
                warnings.append(f"Note: '{metric}' is a dimension, not a metric. Proceeding.")
            else:
                missing.append({"name": metric, "reason": resolved["missing_reason"]})
        else:
            available.append(metric)
            
    # (all_cols was already populated above)
        
    for dim in required_dimensions:
        if dim.lower() not in all_cols:
            warnings.append(f"Dimension '{dim}' may not be directly available.")
            
    return {
        "sufficient": len(missing) == 0,
        "available": available,
        "missing": missing,
        "warnings": warnings
    }

def get_join_safety_context(tables_needed: list[str]) -> str:
    """Generate a textual warning about join cardinality for the SQL generator.
    E.g., warns about one-to-many joins that could cause duplicate counting."""
    warnings = []
    if "workspace.zomato_gold.fact_orders" in tables_needed and "workspace.zomato_gold.fact_order_items" in tables_needed:
        warnings.append("WARNING: Joining fact_orders to fact_order_items creates a 1-to-many fan-out. Do NOT sum order-level metrics (like sales_amount, delivery_fee) directly after this join without handling duplicates. Use order_level subqueries if computing both order metrics and item metrics.")
    
    if len(warnings) > 0:
        return " ".join(warnings)
    return "Join paths appear safe based on requested tables."

def get_table_alias(table_name: str) -> str:
    """Return the standard alias for a table."""
    full_table = table_name if "workspace.zomato_gold" in table_name else f"workspace.zomato_gold.{table_name}"
    return TABLE_GRAINS.get(full_table, {}).get("alias", "")

import functools

@functools.lru_cache(maxsize=1)
def get_schema_context_for_llm() -> str:
    """Build a compact schema reference string for inclusion in LLM prompts.
    Includes table names, columns, types, grain, and relationships."""
    context = "Database Schema Reference (workspace.zomato_gold):\n\n"
    
    for table, grain_info in TABLE_GRAINS.items():
        context += f"Table: {table} (Alias: {grain_info['alias']})\n"
        context += f"Grain: {grain_info['grain']}\n"
        context += f"Primary Key: {grain_info['pk']}\n"
        if grain_info["relationships"]:
            rels = [f"{rel_table} via {r['join_key']} ({r['type']})" for rel_table, r in grain_info["relationships"].items()]
            context += f"Relationships: {', '.join(rels)}\n"
        
        columns = COLUMN_CATALOG.get(table, {})
        cols_str = ", ".join([f"{c}({t})" for c, t in columns.items()])
        context += f"Columns: {cols_str}\n\n"
        
    return context.strip()

@functools.lru_cache(maxsize=32)
def get_lean_schema_context_for_llm(relevant_tables: tuple = None) -> str:
    """Builds a token-efficient schema context including only relevant tables.
    If relevant_tables is omitted, includes core tables (fact_orders, dim_resturant, dim_date)."""
    if not relevant_tables:
        target_tables = [
            "workspace.zomato_gold.fact_orders",
            "workspace.zomato_gold.dim_resturant",
            "workspace.zomato_gold.dim_date"
        ]
    else:
        target_tables = []
        for t in relevant_tables:
            full_t = t if "workspace.zomato_gold" in t else f"workspace.zomato_gold.{t}"
            if full_t in TABLE_GRAINS:
                target_tables.append(full_t)
        if not target_tables:
            target_tables = [
                "workspace.zomato_gold.fact_orders",
                "workspace.zomato_gold.dim_resturant",
                "workspace.zomato_gold.dim_date"
            ]

    context = "Database Schema Reference (workspace.zomato_gold):\n\n"
    for table in target_tables:
        grain_info = TABLE_GRAINS.get(table)
        if not grain_info:
            continue
        context += f"Table: {table} (Alias: {grain_info['alias']})\n"
        context += f"Grain: {grain_info['grain']} | PK: {grain_info['pk']}\n"
        columns = COLUMN_CATALOG.get(table, {})
        cols_str = ", ".join([f"{c}({t})" for c, t in columns.items()])
        context += f"Columns: {cols_str}\n\n"

    return context.strip()
