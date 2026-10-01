"""
markdown_utils.py — Centralized Markdown sanitization and table formatting for LLM outputs.

Rules:
1. Escapes literal unescaped '$' so KaTeX does not interpret currency/prices as math mode.
2. Escapes angle brackets before numbers (e.g. <5%, <20%) to avoid broken HTML tag parsing.
3. Neutralizes raw HTML tags (<script>, <iframe>, etc.).
4. Escapes underscores in snake_case identifiers outside code spans (e.g. fact_orders -> fact\\_orders)
   to prevent unintentional italics.
5. Formats sets of 3+ related numeric values into properly aligned Markdown tables:
   - Clear headers
   - Right-aligned numeric columns (---:)
   - Thousands separators (e.g. 1,921,496)
   - Fixed decimal places for percentages
"""

import re
from typing import Any, Dict, List, Optional


def sanitize_for_markdown(text: str) -> str:
    """
    Sanitizes LLM-generated text for safe Markdown / Streamlit rendering.
    Applied in one place: the narrator's final output step before reaching the UI.
    """
    if not text:
        return ""

    # Split into code spans (`...` and ```...```) vs prose so code content remains intact
    parts = re.split(r"(```[\s\S]*?```|`[^`\n]+`)", text)
    sanitized_parts = []

    for i, part in enumerate(parts):
        # Odd indices are code blocks / inline code; keep them untouched
        if i % 2 == 1:
            sanitized_parts.append(part)
            continue

        s = part

        # 1. Escape unescaped dollar signs: (?<!\\)\$ -> \$
        # Handles currency values like $100, $1,921,496, $50.00
        s = re.sub(r"(?<!\\)\$", r"\$", s)

        # 2. Neutralize comparison operators next to numbers (<5%, < 20%, >100)
        s = re.sub(r"<(?=\s*\d)", r"&lt;", s)
        s = re.sub(r"(?<=\d)\s*>", r"&gt;", s)

        # 3. Strip dangerous raw HTML tags
        s = re.sub(r"<\s*(script|style|iframe|object|embed)[^>]*>[\s\S]*?<\s*/\s*\1\s*>", "", s, flags=re.IGNORECASE)

        # 4. Escape underscores in identifier names (e.g., fact_orders -> fact\_orders)
        # Only matches within word characters to avoid altering intentional italic spans
        s = re.sub(r"(?<=[a-zA-Z0-9])_(?=[a-zA-Z0-9])", r"\_", s)

        sanitized_parts.append(s)

    return "".join(sanitized_parts)


def format_markdown_table(
    rows: List[Dict[str, Any]], 
    columns: Optional[List[str]] = None,
    max_rows: int = 15,
    lang: str = "en"
) -> str:
    """
    Renders a list of row dicts as a strictly formatted Markdown table.
    Enforces:
      - Clear headers
      - Right-aligned numeric columns (---:)
      - Left-aligned text columns (:---)
      - Thousands separators for numbers (e.g. 1,921,496)
      - Fixed decimal places for percentages and floating point values
    """
    if not rows or len(rows) < 3:
        return ""

    display_rows = rows[:max_rows]
    cols = columns or list(display_rows[0].keys())

    # Filter out internal/system columns
    ignored = {
        "days_with_data_cur", "days_with_data_prev", 
        "total_sales_cur", "total_sales_prev", "total_delta"
    }
    cols = [c for c in cols if not c.startswith("_") and c not in ignored]

    if not cols:
        return ""

    # Detect numeric columns
    is_numeric = {}
    for c in cols:
        vals = [r.get(c) for r in display_rows if r.get(c) is not None]
        is_numeric[c] = any(isinstance(v, (int, float)) for v in vals)

    # Friendly headers
    header_titles = []
    for c in cols:
        norm = c.lower().replace("_", " ").strip()
        if "pct" in norm or "percent" in norm:
            title = "% Change"
        elif "contribution" in norm:
            title = "% of Net Change"
        elif norm in ("sales cur", "cur sales"):
            title = "Current Sales"
        elif norm in ("sales prev", "prev sales"):
            title = "Prior Sales"
        elif norm == "delta":
            title = "Change"
        else:
            title = c.replace("_", " ").title()
        header_titles.append(title)

    header_line = "| " + " | ".join(header_titles) + " |"
    sep_line = "| " + " | ".join("---:" if is_numeric[c] else ":---" for c in cols) + " |"

    data_lines = []
    for row in display_rows:
        formatted_vals = []
        for c in cols:
            val = row.get(c)
            if val is None:
                formatted_vals.append("-")
            elif isinstance(val, float):
                if "pct" in c.lower() or "percent" in c.lower():
                    formatted_vals.append(f"{val:+.2f}%" if val != 0 else "0.00%")
                elif "contribution" in c.lower():
                    pct_val = val * 100 if abs(val) <= 1.0 else val
                    formatted_vals.append(f"{pct_val:.2f}%")
                elif abs(val) >= 1000 or val.is_integer():
                    formatted_vals.append(f"{val:,.0f}")
                else:
                    formatted_vals.append(f"{val:,.2f}")
            elif isinstance(val, int):
                formatted_vals.append(f"{val:,}")
            else:
                formatted_vals.append(str(val))
        data_lines.append("| " + " | ".join(formatted_vals) + " |")

    return "\n".join([header_line, sep_line] + data_lines)
