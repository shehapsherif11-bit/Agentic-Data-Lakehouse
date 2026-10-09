"""
ranking_text.py — the deterministic answer for ranking results (no LLM narration of tables).

  * answers the question first: "which is THE lowest?" -> names the leader, including ties
  * never mislabels a capped/oversized result as "Top 10000"
  * adds 1–2 observations computed from the rows (gap to #2, spread) — plain arithmetic, nothing invented
  * suggests the natural next step (diagnose / enrich), so the user gets an analyst, not a table dump
"""
from typing import Optional

try:
    from .markdown_utils import format_markdown_table
except ImportError:  # flat-script import
    from markdown_utils import format_markdown_table

MAX_SHOWN = 15


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fmt(v) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 or float(v).is_integer() else f"{v:,.2f}"


def build_ranking_answer(rows: list[dict], cols: list[str], intent: dict, lang: str,
                         proxy_note: Optional[str] = None) -> str:
    ar = lang in ("ar", "mixed")
    asked = intent.get("top_n")
    bottom = str(intent.get("sort_order", "DESC")).upper() == "ASC"
    shown_n = min(len(rows), max(MAX_SHOWN, asked or 0))
    shown = rows[:shown_n]

    metric_cols = [c for c in cols if any(_num(r.get(c)) for r in rows)]
    label_cols = [c for c in cols if c not in metric_cols] or cols[:1]
    # A numeric id next to a name (customers: user_id, name) is part of the label, not the metric.
    if "user_id" in metric_cols and len(metric_cols) > 1:
        metric_cols.remove("user_id")
        label_cols = ["user_id"] + label_cols
    metric = metric_cols[0] if metric_cols else None
    name_col = next((c for c in label_cols if c != "user_id"), label_cols[0])

    parts = []
    if proxy_note:
        parts.append(proxy_note)

    # ---- direct answer for "which is THE lowest / who is THE top customer"
    if intent.get("singular") and metric and shown:
        lead_v = shown[0].get(metric)
        ties = [r for r in shown if r.get(metric) == lead_v]
        names = [str(r.get(name_col)) for r in ties]
        label = metric.replace("_", " ")
        if ar:
            who = " و ".join(f"**{n}**" for n in names)
            word = "الأقل" if bottom else "الأعلى"
            tie = f" (تعادل بين {len(ties)})" if len(ties) > 1 else ""
            parts.append(f"{word} في {label}: {who} — {_fmt(lead_v)}{tie}.")
        else:
            who = ", ".join(f"**{n}**" for n in names)
            word = "Lowest" if bottom else "Highest"
            tie = f" ({len(ties)}-way tie)" if len(ties) > 1 else ""
            parts.append(f"{word} {label}: {who} — {_fmt(lead_v)}{tie}.")

    # ---- heading
    if asked or len(rows) <= MAX_SHOWN:
        n_title = len(shown)
        heading = (f"**{'أقل' if bottom else 'أفضل'} {n_title} نتائج:**" if ar
                   else f"**{'Bottom' if bottom else 'Top'} {n_title} Results:**")
        if intent.get("singular"):
            heading = ("**للسياق، الترتيب كامل:**" if ar else "**For context, the full ranking:**")
    else:
        heading = (f"**عرض أول {shown_n} من {len(rows):,} صف:**" if ar
                   else f"**Showing the first {shown_n} of {len(rows):,} rows:**")
    if asked and len(rows) < asked:
        heading += (f"\n\n_طلبت {asked} لكن المتاح فعلاً {len(rows)} فقط._" if ar
                    else f"\n\n_You asked for {asked}, but only {len(rows)} matching records exist._")
    parts.append(heading)
    parts.append(format_markdown_table(shown, cols, max_rows=shown_n, lang=lang, min_rows=1))

    # ---- observations (arithmetic on the shown rows only)
    obs = []
    vals = [r.get(metric) for r in shown if metric and _num(r.get(metric))]
    if len(vals) >= 2 and vals[1]:
        gap = (vals[0] - vals[1]) / abs(vals[1]) * 100
        if abs(gap) >= 0.05:
            obs.append(f"الفرق بين الأول والتاني {abs(gap):.1f}%." if ar else f"#1 differs from #2 by {abs(gap):.1f}%.")
        else:
            obs.append("الأول والتاني متعادلين." if ar else "#1 and #2 are tied.")
    if len(vals) >= 3 and min(vals) > 0:
        ratio = max(vals) / min(vals)
        obs.append(f"أعلى قيمة في الجدول = {ratio:.1f}× أقل قيمة." if ar else f"The highest value is {ratio:.1f}× the lowest in this table.")
    if obs:
        parts.append(("**ملاحظات:** " if ar else "**Observations:** ") + " ".join(obs))

    # ---- next step
    if name_col == "restaurant_name":
        parts.append("💡 تقدر تسألني: «ليه؟» عشان أشخّص السبب وأقترح حل، أو «زود المدينة والتقييم»."
                     if ar else "💡 Ask “why?” and I will diagnose the cause and suggest fixes, or “add their city and rating”.")
    return "\n\n".join(p for p in parts if p)
