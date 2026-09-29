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
        "grain": "one row per restaurant",
        "pk": "restaurant_id",
        "relationships": {}
    },
    "workspace.zomato_gold.dim_users": {
        "alias": "du",
        "grain": "one row per user",
        "pk": "user_id",
        "relationships": {}
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
        "missing_reason": "needs cost/COGS data - missing"
    },
    "profit_margin": {
        "display_name": "Profit Margin",
        "sql_expression": None,
        "required_columns": [],
        "source_table": None,
        "description": "Profit margin percentage",
        "aggregation_type": "CALCULATION",
        "missing_reason": "needs cost data - missing"
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
        "missing_reason": "complex"
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
        "order count": "total_orders"
    }
    
    # Try exact, clean, or direct aliases
    resolved_name = aliases.get(name_clean, name_lower)
    
    # also try direct match in aliases for the raw name_lower
    if name_lower in aliases:
        resolved_name = aliases[name_lower]
    
    # Check directly first
    if resolved_name in METRICS:
        metric = METRICS[resolved_name]
        if metric.get("sql_expression") is None and "sql_template" not in metric:
            status = "complex" if metric.get("missing_reason") == "complex" else "missing"
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
                    status = "complex" if value.get("missing_reason") == "complex" else "missing"
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

def check_data_sufficiency(required_metrics: list[str], required_dimensions: list[str]) -> dict:
    """Check if all required metrics and dimensions can be computed.
    Returns: {sufficient: bool, available: list, missing: list[{name, reason}], warnings: list}"""
    available = []
    missing = []
    warnings = []
    
    for metric in required_metrics:
        resolved = resolve_metric(metric)
        if resolved["status"] in ("missing", "complex"):
            missing.append({"name": metric, "reason": resolved["missing_reason"]})
        else:
            available.append(metric)
            
    all_cols = []
    for table, cols in COLUMN_CATALOG.items():
        all_cols.extend(cols.keys())
        
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
