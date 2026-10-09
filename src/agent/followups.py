"""
followups.py — follow-up questions that operate on the restaurants the user is ALREADY looking at.

  * diagnose : "ليه محقق العدد ده بس؟ المشكلة فين؟", "اي السبب؟", "why so low? what should I do?"
               -> root-cause diagnosis + actions (diagnostics.py)
  * enrich   : "زود عليهم المدينة والتقييم", "add the city and rating"
               -> the SAME restaurants, same order, same metric values, plus the requested attributes

Previously both went back through the LLM planner, which lost the previous entity list (it returned 10,000
unrelated rows for "add the city") or narrated "the data does not contain a reason".
"""
from __future__ import annotations

import re
from typing import Callable, Optional

try:
    from . import diagnostics as dx
    from .markdown_utils import format_markdown_table
except ImportError:  # flat-script import
    import diagnostics as dx
    from markdown_utils import format_markdown_table

ENTITY_COL = "restaurant_name"

_A = r"[؀-ۿ]"


def _ar_words(*words: str, prefix: bool = True, whole: bool = True) -> str:
    """Arabic alternatives with real word boundaries (\\b does not work for Arabic): 'ليه' must not match
    inside 'عليهم'. An optional و/ف conjunction prefix is allowed ('وليه')."""
    pre = "(?:و|ف)?" if prefix else ""
    end = f"(?!{_A})" if whole else ""
    return f"(?<!{_A}){pre}(?:{'|'.join(words)}){end}"


_WHY = re.compile(
    r"\bwhy\b|\breason|\broot cause|\bcause[sd]?\b|\bproblem|\bissue|\bwhat'?s wrong|\bfix\b|\bsolve|\bsolution|"
    r"\bimprove|\bwhat should i do|\bhow (can|do) i (fix|improve|increase|grow)|\bdiagnos|"
    r"\bleh\b|\bel sabab\b|\bmoshkela\b|"
    + _ar_words("ليه", "لية", "ليش", "لماذا", "حل", "حلول", "الحل", "الحلول") + "|"
    + _ar_words("السبب", "سبب", "اسباب", "أسباب", "الاسباب", "الأسباب", "المشكل", "مشكل", "تشخيص", "اطور", "أطور", "تطوير",
                whole=False) + "|"
    r"اعمل ا[يى]ه|أعمل ا[يى]ه|اعمل اي(?!" + _A + r")|ازاي ا?زود|إزاي أزود|ازاي اعلي|ازاي احسن|إزاي أحسن"
)
_ADD = re.compile(r"\badd\b|\binclude\b|\bwith (their|its|the)\b|\bshow (their|its|the)\b|\balso\b|"
                  + _ar_words("زود", "زوّد", "ضيف", "ضيّف", "اضف", "أضف", "اضافة", "إضافة", "معاهم", "معاه", "كمان",
                              "اعرض", "هات", "حط", "حطلي", whole=False))
_ATTRS = {
    "city": re.compile(r"\bcit(y|ies)\b|\blocation|مدين|المدينه|المدينة|مكان|منطق|محافظ"),
    "rating": re.compile(r"\brating|\bstars?\b|تقييم|التقييم|ريتنج|نجوم"),
    "cuisine": re.compile(r"\bcuisine|\bfood type|مطبخ|نوع الاكل|نوع الأكل"),
    "branches": re.compile(r"\bbranch|\boutlet|فرع|فروع"),
}
_NEW_TOPIC = re.compile(
    r"\b(retention|churn|segments?|customers?|users?|cit(y|ies)|months?|years?|quarters?|weeks?|declin\w*|drop\w*|"
    r"grow\w*|increas\w*|decreas\w*|trend\w*|20\d\d)\b|20\d\d|"
    # whole Arabic words only: 'سنه' (year) must not match inside 'احسنهم'
    + _ar_words("(?:ال)?(?:عملاء|عميل|شرائح|شريحة|شريحه|شهر|سنة|سنه|ربع|اسبوع|أسبوع|اتجاه)", "نزلت", "نزل", "انخفض",
                "انخفضت", "زادت", "ارتفع", "ارتفعت")
)
_REF = re.compile(r"\b(this|that|these|those|them|it|they)\b|ده|دي|دول|دا|هذا|هذه|عليهم|بتاعهم|بتاعه|بتاعها|العدد|برده|برضه|كده")


def has_entities(evidence: list) -> bool:
    return bool(_entity_table(evidence))


def followup_kind(question: str, evidence: list) -> Optional[str]:
    """'diagnose' | 'enrich' | None — only when the previous result lists restaurants."""
    if not question or not has_entities(evidence):
        return None
    t = question.lower()
    names = entity_names(evidence)
    mentions = any(n.lower() in t for n in names)
    words = re.findall(r"\w+", t)

    attrs = requested_attributes(t)
    if attrs and _ADD.search(t) and not _WHY.search(t):
        return "enrich"
    if _WHY.search(t):
        if mentions:
            return "diagnose"
        # A question that opens a NEW topic ("why did sales decline in 2025?", "why did customer retention drop?")
        # belongs to the analyst, even right after a restaurant ranking.
        if _NEW_TOPIC.search(t):
            return None
        if _REF.search(t) or len(words) <= 6:
            return "diagnose"
    return None


def requested_attributes(text: str) -> list[str]:
    return [k for k, rx in _ATTRS.items() if rx.search(text.lower())]


def _as_rows(ev) -> tuple[list[str], list[dict]]:
    if hasattr(ev, "columns") and hasattr(ev, "rows"):
        cols = list(ev.columns)
        return cols, [dict(zip(cols, r)) for r in ev.rows]
    cols = list(ev.get("columns", []))
    return cols, [r if isinstance(r, dict) else dict(zip(cols, r)) for r in ev.get("rows", [])]


def _entity_table(evidence: list) -> Optional[tuple[list[str], list[dict]]]:
    for ev in evidence or []:
        cols, rows = _as_rows(ev)
        if ENTITY_COL in cols and rows:
            return cols, rows
    return None


_ORDERS_METRIC = re.compile(r"order", re.IGNORECASE)
_NOT_ORDER_COUNT = re.compile(r"avg|average|aov|per_|per |value|growth|rate|ratio|delivery|rating", re.IGNORECASE)


def ranked_metric(evidence: list) -> str:
    """What the previous table ranked the restaurants by: "orders" (a count of orders) or "revenue" (the default).
    A diagnosis of a low ORDER COUNT must not blame basket size, which only moves revenue."""
    table = _entity_table(evidence)
    if not table:
        return "revenue"
    cols, rows = table
    for c in cols:
        if c in (ENTITY_COL, "user_id") or not any(isinstance(r.get(c), (int, float)) for r in rows):
            continue
        return "orders" if _ORDERS_METRIC.search(c) and not _NOT_ORDER_COUNT.search(c) else "revenue"
    return "revenue"


def entity_names(evidence: list) -> list[str]:
    table = _entity_table(evidence)
    if not table:
        return []
    seen, out = set(), []
    for r in table[1]:
        n = r.get(ENTITY_COL)
        if n is not None and n not in seen:
            seen.add(n)
            out.append(str(n))
    return out


def target_names(question: str, evidence: list) -> list[str]:
    """Restaurants named in the question, else all restaurants of the previous result (capped)."""
    names = entity_names(evidence)
    t = (question or "").lower()
    named = [n for n in names if n.lower() in t]
    return (named or names)[:dx.MAX_ENTITIES]


# ------------------------------------------------------------------ enrichment
def enrich_sql(names: list[str], attrs: list[str]) -> str:
    G = dx.G
    return f"""
WITH b AS (
    SELECT restaurant_id, restaurant_name, city, rating, cuisine
    FROM {G}.dim_resturant
    WHERE restaurant_name IN ({", ".join(dx.sql_literal(n) for n in names)})
),
o AS (
    SELECT fo.restaurant_id, COUNT(*) AS orders
    FROM {G}.fact_orders fo
    JOIN b ON fo.restaurant_id = b.restaurant_id
    GROUP BY fo.restaurant_id
)
SELECT b.restaurant_name,
       MAX_BY(b.city, COALESCE(o.orders, 0)) AS top_city,
       COUNT(DISTINCT b.city) AS cities,
       COUNT(DISTINCT b.restaurant_id) AS branches,
       ROUND(AVG(b.rating), 2) AS avg_rating,
       COUNT(b.rating) AS rated_branches,
       MAX_BY(b.cuisine, COALESCE(o.orders, 0)) AS cuisine
FROM b
LEFT JOIN o ON o.restaurant_id = b.restaurant_id
GROUP BY b.restaurant_name
""".strip()


_ATTR_COLS = {
    "city": [("top_city", "المنطقة (أنشط فرع)", "Area (busiest branch)"), ("cities", "عدد المناطق", "Areas")],
    "rating": [("avg_rating", "التقييم", "Rating"), ("rated_branches", "فروع متقيمة", "Rated branches")],
    "cuisine": [("cuisine", "المطبخ", "Cuisine")],
    "branches": [("branches", "الفروع", "Branches")],
}


def render_enrichment(evidence: list, rows: list[dict], attrs: list[str], lang: str) -> str:
    ar = lang in ("ar", "mixed")
    cols, prev_rows = _entity_table(evidence)
    metric_cols = [c for c in cols if c != ENTITY_COL and any(isinstance(r.get(c), (int, float)) for r in prev_rows)]
    by_name = {r.get(ENTITY_COL): r for r in rows}

    want = list(dict.fromkeys(a for a in attrs if a in _ATTR_COLS))
    if "city" in want and "branches" not in want:
        want.append("branches")   # a brand-level "city" is only meaningful next to its branch count
    extra = [c for a in want for c in _ATTR_COLS[a]]

    out_cols = [ENTITY_COL] + metric_cols + [c for c, *_ in extra]
    table_rows, missing = [], []
    for pr in prev_rows:
        name = pr.get(ENTITY_COL)
        e = by_name.get(name)
        if e is None:
            missing.append(str(name))
        row = {ENTITY_COL: name, **{m: pr.get(m) for m in metric_cols}}
        for c, *_ in extra:
            row[c] = (e or {}).get(c)
        table_rows.append(row)

    md = format_markdown_table(table_rows, out_cols, max_rows=len(table_rows), lang=lang, min_rows=1)
    # Friendly headers for the added columns.
    for c, h_ar, h_en in extra:
        md = md.replace(f"| {c.replace('_', ' ').title()} |", f"| {h_ar if ar else h_en} |", 1)

    notes = []
    multi_city = [r[ENTITY_COL] for r in table_rows if (r.get("cities") or 0) > 1]
    if "city" in want and multi_city:
        notes.append(
            f"{len(multi_city)} مطاعم ليها فروع في أكتر من منطقة: المعروض هو منطقة الفرع اللي عليه أكبر عدد طلبات (المنطقة مسجلة بالشكل «الحي، المدينة»)."
            if ar else
            f"{len(multi_city)} restaurant(s) operate in several areas: the area shown is that of their busiest branch (stored as 'locality, city')."
        )
    if "rating" in want:
        unrated = [r[ENTITY_COL] for r in table_rows if r.get("avg_rating") is None]
        notes.append(
            "التقييم = متوسط تقييم الفروع على المنصة (مش تقييم العملاء للطلبات)."
            + (f" مفيش تقييم مسجل لـ: {', '.join(map(str, unrated))}." if unrated else "")
            if ar else
            "Rating = average platform rating of the branches (not customers' order ratings)."
            + (f" No rating recorded for: {', '.join(map(str, unrated))}." if unrated else "")
        )
    if missing:
        notes.append(("مالقيتش في جدول المطاعم: " if ar else "Not found in the restaurant table: ") + ", ".join(missing))

    head = (f"**نفس الـ {len(table_rows)} مطاعم من النتيجة السابقة، بنفس الترتيب، ومعاهم البيانات المطلوبة:**"
            if ar else f"**The same {len(table_rows)} restaurants from the previous result, same order, with the requested details:**")
    tail = ("\n\n💡 عايز تعرف ليه الترتيب كده؟ اسألني «ليه؟» وأنا أشخّص الأسباب."
            if ar else "\n\n💡 Want to know why they rank like this? Ask “why?” and I will diagnose it.")
    return f"{head}\n\n{md}\n\n" + "\n".join(f"- {n}" for n in notes) + tail


# ------------------------------------------------------------------ orchestration
Executor = Callable[[list[dict]], list[dict]]


def run_followup(kind: str, question: str, evidence: list, lang: str, execute: Executor) -> dict:
    """-> {'answer': str, 'evidence': [{'columns','rows','sql'}...] | [] }. `execute` runs guarded read-only SQL
    and returns [{success, columns, rows, error}] in order."""
    names = target_names(question, evidence)
    ar = lang in ("ar", "mixed")
    if not names:
        return {"answer": "مفيش مطاعم في النتيجة السابقة أشتغل عليها." if ar
                else "The previous result has no restaurants to work on.", "evidence": []}

    if kind == "enrich":
        attrs = requested_attributes(question) or ["city", "rating"]
        sql = enrich_sql(names, attrs)
        res = execute([{"sql": sql, "purpose": "enrich previous restaurants"}])[0]
        if not res.get("success"):
            return {"answer": _failed(ar), "evidence": []}
        return {"answer": render_enrichment(evidence, res.get("rows", []), attrs, lang),
                "evidence": [{"columns": res.get("columns", []), "rows": res.get("rows", []), "sql": sql}]}

    # diagnose: targets first (their cities/cuisines decide the benchmark queries)
    t_sql = dx.target_sql(names)
    first = execute([{"sql": t_sql, "purpose": "diagnose targets"},
                     {"sql": dx.PLATFORM_SQL, "purpose": "platform benchmark"},
                     {"sql": dx.QUALITY_SQL, "purpose": "data quality"}])
    targets, platform, quality = first
    if not targets.get("success") or not platform.get("success") or not platform.get("rows"):
        return {"answer": _failed(ar), "evidence": []}
    t_rows = targets.get("rows", [])
    p = dict(platform["rows"][0])
    q = (quality.get("rows") or [None])[0] if quality.get("success") else None
    if q and q.get("branches"):
        p["unrated_share"] = 1 - float(q.get("rated") or 0) / float(q["branches"])
        if q.get("brands"):
            p["branches_per_brand"] = float(q["branches"]) / float(q["brands"])

    cities = sorted({r["city"] for r in t_rows if r.get("city") and int(r.get("cities") or 1) == 1})
    cuisines = sorted({r["cuisine"] for r in t_rows if r.get("cuisine")})
    bench_q = []
    if cities:
        bench_q.append({"sql": dx.group_benchmark_sql("city", cities), "purpose": "city benchmark"})
    if cuisines:
        bench_q.append({"sql": dx.group_benchmark_sql("cuisine", cuisines), "purpose": "cuisine benchmark"})
    bench = execute(bench_q) if bench_q else []
    city_b, cuisine_b = {}, {}
    for spec, res in zip(bench_q, bench):
        if res.get("success"):
            target = city_b if spec["purpose"].startswith("city") else cuisine_b
            target.update({r["grp"]: r for r in res.get("rows", [])})

    order = {n: i for i, n in enumerate(names)}
    t_rows = sorted(t_rows, key=lambda r: order.get(r.get(ENTITY_COL), 999))
    metric = ranked_metric(evidence)
    results = [dx.analyse_entity(r, p, city_b.get(r.get("city")), cuisine_b.get(r.get("cuisine")), metric=metric) for r in t_rows]
    answer = dx.render(results, p, lang, data_gaps=dx.data_gaps(q, lang), metric=metric)
    return {"answer": answer,
            "evidence": [{"columns": targets.get("columns", []), "rows": t_rows, "sql": t_sql}]}


def _failed(ar: bool) -> str:
    return ("معرفتش أجيب البيانات المطلوبة من قاعدة البيانات دلوقتي، ومش هخمّن من غير أرقام. جرّب تاني بعد شوية."
            if ar else "I could not retrieve the required data right now, and I will not guess without numbers. Please try again shortly.")
