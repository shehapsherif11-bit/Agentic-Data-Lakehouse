"""
result_classifier.py — tells apart the outcomes that an LLM tends to blur together:

  * no_rows          the query ran fine but returned zero rows
  * empty_aggregate  a single aggregate row where every value is NULL / 0 (SUM over no rows is NULL while
                     COUNT is 0 — i.e. the filter matched nothing; it is NOT 'revenue = 0')
  * ok               real data

and renders the honest, deterministic explanation (no LLM, so nothing can be invented).
"""
from typing import Any, Iterable, Optional


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def classify_rows(columns: Iterable[str], rows: list[dict]) -> dict:
    """-> {kind, null_columns, zero_columns}. `rows` is a list of dicts."""
    columns = list(columns)
    if not rows:
        return {"kind": "no_rows", "null_columns": [], "zero_columns": []}

    null_cols = [c for c in columns if all(r.get(c) is None for r in rows)]
    zero_cols = [c for c in columns if all(_is_num(r.get(c)) and r.get(c) == 0 for r in rows)]

    kind = "ok"
    if len(rows) == 1:
        numeric_like = [c for c in columns if c in null_cols or c in zero_cols or _is_num(rows[0].get(c))]
        non_empty_numeric = [c for c in numeric_like if _is_num(rows[0].get(c)) and rows[0].get(c) != 0]
        if numeric_like and not non_empty_numeric and len(numeric_like) == len(columns):
            kind = "empty_aggregate"
    return {"kind": kind, "null_columns": null_cols, "zero_columns": zero_cols}


def describe_empty(kind: str, columns: list[str], rows: list[dict], coverage: Optional[dict],
                   lang: str, time_period: Optional[str] = None) -> str:
    """Honest explanation for no_rows / empty_aggregate. `coverage` = {'min_date','max_date'} or None."""
    ar = lang in ("ar", "mixed")
    cov = ""
    if coverage and coverage.get("min_date") and coverage.get("max_date"):
        a, b = str(coverage["min_date"])[:10], str(coverage["max_date"])[:10]
        cov = (f"\n\nالبيانات المتاحة في قاعدة البيانات تغطي الفترة من **{a}** إلى **{b}**."
               if ar else f"\n\nThe orders in the database cover **{a}** to **{b}**.")
    period = f" ({time_period})" if time_period else ""

    if kind == "no_rows":
        return (
            f"الاستعلام اشتغل بنجاح لكنه لم يرجع أي صفوف{period}، يعني مفيش بيانات مطابقة للشرط ده." + cov
            if ar else
            f"The query ran successfully but returned no rows{period}, so there is no matching data." + cov
        ) + (_next_step(ar, bool(cov)))

    shown = ", ".join(f"{c} = {'NULL' if rows[0].get(c) is None else rows[0].get(c)}" for c in columns)
    return (
        f"الاستعلام اشتغل بنجاح لكن مفيش طلبات مطابقة للفترة{period}: النتيجة هي {shown}.\n\n"
        "ده معناه إن **مفيش مبيعات مسجلة في الفترة دي**، مش إن الإيراد صفر فعلاً — "
        "إجمالي الإيراد بيظهر NULL لأنه مجموع على صفوف مفيش منها." + cov
        if ar else
        f"The query ran successfully but no orders match the requested period{period}: the result is {shown}.\n\n"
        "That means **no sales are recorded for that period** — it does not mean revenue was actually zero: "
        "total revenue is NULL because it is a sum over zero matching rows." + cov
    ) + _next_step(ar, bool(cov))


def _next_step(ar: bool, has_coverage: bool) -> str:
    if not has_coverage:
        return ""
    return ("\n\nتحب أشغّل نفس السؤال على آخر شهر فيه بيانات فعلاً؟" if ar
            else "\n\nWant me to run the same question for the latest month that actually has data?")
