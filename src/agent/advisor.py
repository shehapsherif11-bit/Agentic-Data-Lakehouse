"""
advisor.py — the evidence-bound Business Intelligence advisor.

Answers opinion / decision follow-ups ("which restaurant is best? give me one name", "اي احسن مطعم استثمر فيه")
from the rows the analyst ALREADY retrieved. It is deliberately deterministic (no LLM): every number shown is
copied from, or arithmetically derived from, the stored query result, the decision criterion is stated, and the
facts / calculations / limitations / next steps are kept in separate sections.
"""
from typing import Any, Optional

try:
    from . import metrics_registry as metrics
    from .query_intent import LOWER_IS_BETTER, extract_top_n, is_investment_question
except ImportError:  # flat-script import
    import metrics_registry as metrics
    from query_intent import LOWER_IS_BETTER, extract_top_n, is_investment_question


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 or float(v).is_integer() else f"{v:,.2f}"


def _as_table(ev) -> tuple[list[str], list[dict]]:
    """Evidence object or {'columns','rows'} dict -> (columns, list-of-dict rows)."""
    if hasattr(ev, "columns") and hasattr(ev, "rows"):
        cols = list(ev.columns)
        rows = [dict(zip(cols, r)) for r in ev.rows]
    else:
        cols = list(ev.get("columns", []))
        rows = [r if isinstance(r, dict) else dict(zip(cols, r)) for r in ev.get("rows", [])]
    return cols, rows


def _pick_table(evidence: list) -> Optional[tuple[list[str], list[dict]]]:
    """First stored result with >= 2 rows and a numeric column; else the first non-empty one."""
    fallback = None
    for ev in evidence or []:
        cols, rows = _as_table(ev)
        if not rows:
            continue
        fallback = fallback or (cols, rows)
        if len(rows) >= 2 and any(_num(r.get(c)) for r in rows for c in cols):
            return cols, rows
    return fallback


def _metric_label(metric_col: str, intent: Optional[dict]) -> str:
    for m in (intent or {}).get("metrics") or []:
        res = metrics.resolve_metric(m)
        d = res.get("definition") or {}
        if res.get("name") == metric_col or metric_col == "result_val":
            return d.get("display_name") or m
    d = metrics.METRICS.get(metric_col) or {}
    return d.get("display_name") or metric_col.replace("_", " ").title()


def build_advice(question: str, evidence: list, lang: str = "en", previous_intent: Optional[dict] = None,
                 previous_question: Optional[str] = None) -> str:
    ar = lang in ("ar", "mixed")
    table = _pick_table(evidence)
    if not table:
        return _no_data(ar)
    cols, rows = table

    metric_cols = [c for c in cols if any(_num(r.get(c)) for r in rows)]
    label_cols = [c for c in cols if c not in metric_cols]
    if not metric_cols or not label_cols:
        return _no_data(ar)
    label_col, metric_col = label_cols[0], metric_cols[0]
    label = _metric_label(metric_col, previous_intent)

    cand = [r for r in rows if _num(r.get(metric_col))]
    if len(cand) < 2:
        return _single_candidate(ar, cand[0] if cand else None, label_col, metric_col, label)

    lower_better = any(k in metric_col.lower() or k in label.lower().replace(" ", "_") for k in LOWER_IS_BETTER)
    cand.sort(key=lambda r: r[metric_col], reverse=not lower_better)
    n = min(extract_top_n(question) or 1, len(cand))
    picks, best, second = cand[:n], cand[0], cand[1]
    total = sum(r[metric_col] for r in cand)

    # ---- calculated findings (python arithmetic on stored values) ----
    gap_pct = ((best[metric_col] - second[metric_col]) / abs(second[metric_col]) * 100) if second[metric_col] else None
    share_pct = (best[metric_col] / total * 100) if total else None
    direction_en = "lowest" if lower_better else "highest"
    direction_ar = "الأقل" if lower_better else "الأعلى"

    invest = is_investment_question(question)
    names = ", ".join(str(p[label_col]) for p in picks)
    ties = [r for r in cand if r[metric_col] == best[metric_col]]
    tied = n == 1 and len(ties) > 1
    is_revenue = label.lower() in ("revenue", "total revenue", "sales")
    tie_names = " / ".join(str(r[label_col]) for r in ties)
    if tied:
        gap_pct = None

    table_lines = "\n".join(
        f"| {i} | {r[label_col]} | {_fmt(r[metric_col])} |" for i, r in enumerate(cand[:10], 1)
    )
    cmp_en = f"- **{best[label_col]}** is {abs(gap_pct):.1f}% {'below' if lower_better else 'above'} the runner-up ({second[label_col]})." if gap_pct is not None else ""
    cmp_en += f"\n- **{best[label_col]}** is {share_pct:.1f}% of the combined {label.lower()} of the {len(cand)} listed." if share_pct is not None else ""
    cmp_ar = f"- **{best[label_col]}** {'أقل' if lower_better else 'أعلى'} من صاحب المركز التاني ({second[label_col]}) بنسبة {abs(gap_pct):.1f}%." if gap_pct is not None else ""
    cmp_ar += f"\n- **{best[label_col]}** تمثل {share_pct:.1f}% من إجمالي {label} للـ {len(cand)} المعروضين." if share_pct is not None else ""

    if tied:
        cmp_en = f"- {len(ties)}-way tie at {_fmt(best[metric_col])}: {tie_names}." + cmp_en
        cmp_ar = f"- تعادل بين {len(ties)} عند {_fmt(best[metric_col])}: {tie_names}." + cmp_ar
    missing = [metrics.METRICS[k]["display_name"] for k in ("profit", "profit_margin") if k in metrics.METRICS]
    basis_q = previous_question or ""

    if ar:
        head = (f"## تعادل: {tie_names}\n" if tied else f"## اختياري: {names}\n" if n == 1 else f"## اختياراتي: {names}\n")
        crit = (f"**معيار الاختيار:** {direction_ar} في **{label}** بين الـ {len(cand)} المعروضين في النتيجة السابقة — "
                f"ده المؤشر الوحيد المتاح قدامي، مش حكم شامل.")
        if tied:
            crit += (f"\n\nالـ {len(ties)} متعادلين بالظبط في المعيار ده، فمش هختار واحد عشوائياً. "
                     "اسألني «ليه؟» عشان أقارنهم في الإلغاءات والتوصيل والتقييم ومتوسط الطلب.")
        if invest:
            crit += ("\n\n⚠️ ده اختيار مبني على الإيراد فقط، **مش توصية استثمارية**: الإيراد بيقيس الحجم مش الربحية ولا النمو ولا المخاطر."
                     if is_revenue else
                     f"\n\n⚠️ ده اختيار مبني على **{label}** فقط، **مش توصية استثمارية**: المؤشر ده مش بيقيس الربحية ولا النمو ولا المخاطر.")
        return (
            f"{head}\n{crit}\n\n"
            f"### أرقام مقاسة (من الاستعلام)\n| # | الاسم | {label} |\n|---:|:---|---:|\n{table_lines}\n\n"
            f"### حسابات مشتقة\n{cmp_ar}\n\n"
            f"### اللي البيانات دي مبتثبتوش\n"
            f"- مفيش بيانات تكلفة، فمقدرش أحسب **{' / '.join(missing)}**.\n"
            f"- النتيجة دي لقطة واحدة؛ مفيهاش اتجاه زمني (نمو أو انكماش) ولا ثبات العملاء.\n"
            f"- الترتيب بـ {label} مبيثبتش إن المطعم ده الأنسب للاستثمار أو الأحسن جودة.\n\n"
            f"### أنسب تحليلات تكمّل بيها\n"
            f"1. اتجاه الإيراد شهرياً لـ {best[label_col]} والمنافسين (هل بينمو ولا بيقل؟)\n"
            f"2. متوسط قيمة الطلب وعدد الطلبات لكل مطعم (هل الإيراد من حجم ولا من سعر؟)\n"
            f"3. نسبة الخصم والتقييم لكل مطعم (هل الإيراد معتمد على الخصومات؟)\n"
            + (f"\n*مبني على: {basis_q}*" if basis_q else "")
        )

    head = (f"## Tie: {tie_names}\n" if tied else f"## My pick: {names}\n" if n == 1 else f"## My picks: {names}\n")
    crit = (f"**Decision criterion:** the {direction_en} **{label}** among the {len(cand)} restaurants in the previous result — "
            f"the only performance measure I have for them, not an overall judgement.")
    if tied:
        crit += (f"\n\nThese {len(ties)} are exactly tied on this criterion, so I will not pick one at random. "
                 "Ask “why?” to compare them on cancellations, delivery, rating and basket size.")
    if invest:
        crit += ("\n\n⚠️ This is a revenue-based pick, **not an investment recommendation**: revenue measures size, "
                 "not profitability, growth or risk.")
    return (
        f"{head}\n{crit}\n\n"
        f"### Measured (from the query)\n| # | Name | {label} |\n|---:|:---|---:|\n{table_lines}\n\n"
        f"### Calculated\n{cmp_en}\n\n"
        f"### What this data does not prove\n"
        f"- There is no cost data, so I cannot compute **{' / '.join(missing)}**.\n"
        f"- This result is a single snapshot: no time trend (growth or decline) and no customer retention.\n"
        f"- A {label.lower()} ranking does not show that this restaurant is the best investment or the best quality.\n\n"
        f"### Suggested next analyses\n"
        f"1. Monthly revenue trend for {best[label_col]} vs. the others (growing or shrinking?)\n"
        f"2. Average order value and order count per restaurant (is revenue volume-driven or price-driven?)\n"
        f"3. Discount rate and average rating per restaurant (is revenue discount-dependent?)\n"
        + (f"\n*Based on: {basis_q}*" if basis_q else "")
    )


def _single_candidate(ar: bool, row: Optional[dict], label_col: str, metric_col: str, label: str) -> str:
    if not row:
        return _no_data(ar)
    return (
        f"النتيجة السابقة فيها عنصر واحد بس ({row[label_col]}: {_fmt(row[metric_col])} {label})، فمفيش مقارنة أقدر أبني عليها اختيار."
        if ar else
        f"The previous result has only one entity ({row[label_col]}: {_fmt(row[metric_col])} {label}), so there is nothing to compare and no pick to make."
    )


def _no_data(ar: bool) -> str:
    return (
        "مفيش نتيجة سابقة فيها أرقام أقدر أقارن بيها. اسألني سؤال بيانات الأول (مثلاً: أعلى 5 مطاعم من حيث الإيراد) وأنا أرشحلك على أساسه."
        if ar else
        "There is no earlier result with comparable numbers. Ask a data question first (e.g. top 5 restaurants by revenue) and I will base my pick on it."
    )
