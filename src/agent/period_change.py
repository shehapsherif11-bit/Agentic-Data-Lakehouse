"""
period_change.py — "which week/month had the biggest drop (or rise) in revenue/orders, and why?" answered
deterministically, with no LLM in the loop (routing, SQL and narration are all code).

Why this exists: the LLM planner only knew how to compare the latest two months, wrote a different WoW query on
every run, and the catalog's WoW template grouped by (year, WEEK()) and counted the partial last week as a real
week. Here:
  * periods are whole calendar periods inside the data coverage (Monday–Sunday weeks / calendar months), so a
    partial first or last period can never be reported as the "biggest drop";
  * the extreme period is picked by SQL (LAG over the consecutive-period series), the user sees the definition;
  * the explanation is a measured decomposition of the SAME two periods (orders x basket size, who contributed,
    activity, failed orders) — labelled as association, never as proven cause — plus what the data cannot show.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Callable, Optional

FACT = "workspace.zomato_gold.fact_orders"
DATES = "workspace.zomato_gold.dim_date"
RESTAURANTS = "workspace.zomato_gold.dim_resturant"
DEFAULT_CONTRIBUTORS = 5
MAX_CONTRIBUTORS = 20

_AR = r"[؀-ۿ]"
_UNIT_WEEK = re.compile(r"\bweek(s|ly)?\b|\bwow\b|week[- ]over[- ]week|اسبوع|أسبوع|الاسبوع|الأسبوع|اسبوعي|أسبوعي")
_UNIT_MONTH = re.compile(r"\bmonth(s|ly)?\b|\bmom\b|month[- ]over[- ]month|شهر|الشهر|شهري")
_DOWN = re.compile(r"\b(decreas\w*|declin\w*|drop\w*|fall\w*|fell|dip\w*|decrease)\b|نزول|انخفاض|هبوط|نقص|نزل|انخفض|هبط|تراجع|نزلت")
_UP = re.compile(r"\b(increas\w*|growth|grew|rise|rose|jump\w*|surge\w*|spike\w*)\b|زياد|ارتفاع|نمو|طلع|ارتفع")
_SUPERLATIVE = re.compile(
    r"\b(biggest|largest|worst|steepest|sharpest|greatest|maximum|max)\b|أكبر|اكبر|اسوأ|أسوأ|اسوا|أسوا|اعلى|أعلى"
)
_OVER_OVER = re.compile(r"\bwow\b|\bmom\b|week[- ]over[- ]week|month[- ]over[- ]month")
_NAMED_PERIOD = re.compile(
    r"\b20\d\d\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b|"
    r"\b(last|this|previous|past|latest|current)\s+(week|month)\b|يناير|فبراير|مارس|ابريل|أبريل|مايو|يونيو|يوليو|اغسطس|أغسطس|سبتمبر|اكتوبر|أكتوبر|نوفمبر|ديسمبر|"
    r"(?<!" + _AR + r")(الماضي|اللي فات|الحالي)"
)
_ORDERS = re.compile(r"\borders?\b|اوردر|أوردر|طلبات")
_REVENUE = re.compile(r"\brevenue|\bsales\b|ايراد|إيراد|مبيعات")
_N = re.compile(r"\b(?:top\s+)?(\d{1,2})\s+(?:restaurants?|brands?|contributors?|مطاعم)|(\d{1,2})\s*(?:مطاعم|مطعم)")

_DATE_STRING = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:[ T]00:00:00(?:\.0+)?)?$")


def parse_period_change(question: str) -> Optional[dict]:
    """-> {unit, direction, metric, top_n} when the question is 'which week/month had the biggest
    drop/rise (in revenue/orders)'; None otherwise (anything with a named period goes to the planner)."""
    t = (question or "").lower()
    if not t or _NAMED_PERIOD.search(t):
        return None
    unit = "week" if _UNIT_WEEK.search(t) else "month" if _UNIT_MONTH.search(t) else None
    if unit is None:
        return None
    down, up = _DOWN.search(t), _UP.search(t)
    if not (down or up) or not (_SUPERLATIVE.search(t) or _OVER_OVER.search(t)):
        return None
    direction = "down" if (down and (not up or down.start() <= up.start())) else "up"
    metric = "orders" if _ORDERS.search(t) and not _REVENUE.search(t) else "revenue"
    m = _N.search(t)
    n = int(next(g for g in m.groups() if g)) if m else DEFAULT_CONTRIBUTORS
    return {"unit": unit, "direction": direction, "metric": metric, "top_n": max(1, min(n, MAX_CONTRIBUTORS))}


# ------------------------------------------------------------------ SQL
_UNIT_SQL = {
    # (trunc unit, last day of period given its start, start of the next period, consecutive-periods test, baseline length)
    "week": ("week", "DATE_ADD({s}, 6)", "DATE_ADD({s}, 7)", "DATEDIFF(c.p_start, c.prev_start) = 7", 8),
    "month": ("month", "LAST_DAY({s})", "ADD_MONTHS({s}, 1)", "ADD_MONTHS(c.prev_start, 1) = c.p_start", 6),
}


def _agg(metric: str, cond: Optional[str] = None) -> str:
    if metric == "orders":
        return f"COUNT(DISTINCT CASE WHEN {cond} THEN fo.order_id END)" if cond else "COUNT(DISTINCT fo.order_id)"
    return f"SUM(CASE WHEN {cond} THEN fo.sales_amount ELSE 0 END)" if cond else "SUM(fo.sales_amount)"


def extreme_period_sql(unit: str, direction: str, metric: str) -> str:
    trunc, last_day, _next, consecutive, base_n = _UNIT_SQL[unit]
    order = "ASC" if direction == "down" else "DESC"
    return (
        f"WITH bounds AS (SELECT MIN(dd.full_date) AS d0, MAX(dd.full_date) AS d1 "
        f"FROM {FACT} fo JOIN {DATES} dd ON fo.date_id = dd.date_id), "
        f"periods AS (SELECT CAST(DATE_TRUNC('{trunc}', dd.full_date) AS DATE) AS p_start, {_agg(metric)} AS value "
        f"FROM {FACT} fo JOIN {DATES} dd ON fo.date_id = dd.date_id GROUP BY 1), "
        f"full_periods AS (SELECT p.p_start, p.value FROM periods p CROSS JOIN bounds b "
        f"WHERE p.p_start >= b.d0 AND {last_day.format(s='p.p_start')} <= b.d1), "
        f"chg AS (SELECT p_start, value, LAG(p_start) OVER (ORDER BY p_start) AS prev_start, "
        f"LAG(value) OVER (ORDER BY p_start) AS prev_value, "
        f"AVG(value) OVER (ORDER BY p_start ROWS BETWEEN {base_n + 1} PRECEDING AND 2 PRECEDING) AS baseline_avg "
        f"FROM full_periods), "
        f"cands AS (SELECT c.p_start, c.prev_start, c.value, c.prev_value, c.baseline_avg, "
        f"c.value - c.prev_value AS abs_change, "
        f"ROUND(100.0 * (c.value - c.prev_value) / NULLIF(c.prev_value, 0), 2) AS pct_change "
        f"FROM chg c WHERE c.prev_start IS NOT NULL AND {consecutive}) "
        f"SELECT x.p_start AS period_start, x.prev_start AS prev_period_start, x.value AS current_value, "
        f"x.prev_value AS previous_value, x.abs_change, x.pct_change, x.baseline_avg, "
        f"STDDEV_SAMP(x.pct_change) OVER () AS pct_stddev, COUNT(*) OVER () AS periods_compared, b.d0 AS data_start, b.d1 AS data_end "
        f"FROM cands x CROSS JOIN bounds b ORDER BY x.abs_change {order}, x.p_start ASC LIMIT 1"
    )


def _in_period(unit: str, start: str) -> str:
    nxt = _UNIT_SQL[unit][2].format(s=f"DATE'{start}'")
    return f"dd.full_date >= DATE'{start}' AND dd.full_date < {nxt}"


def contributors_sql(unit: str, direction: str, metric: str, cur: str, prev: str, top_n: int) -> str:
    cur_c, prev_c = _in_period(unit, cur), _in_period(unit, prev)
    order = "ASC" if direction == "down" else "DESC"
    return (
        f"WITH per AS (SELECT dr.restaurant_name, {_agg(metric, prev_c)} AS prev_value, {_agg(metric, cur_c)} AS cur_value "
        f"FROM {FACT} fo JOIN {DATES} dd ON fo.date_id = dd.date_id "
        f"JOIN {RESTAURANTS} dr ON fo.restaurant_id = dr.restaurant_id "
        f"WHERE ({prev_c}) OR ({cur_c}) GROUP BY dr.restaurant_name), "
        f"d AS (SELECT restaurant_name, prev_value, cur_value, cur_value - prev_value AS abs_change, "
        f"ROUND(100.0 * (cur_value - prev_value) / NULLIF(prev_value, 0), 2) AS pct_change, "
        f"SUM(cur_value - prev_value) OVER () AS total_change, "
        f"SUM(CASE WHEN cur_value - prev_value < 0 THEN 1 ELSE 0 END) OVER () AS brands_down, COUNT(*) OVER () AS brands "
        f"FROM per) "
        f"SELECT restaurant_name, prev_value, cur_value, abs_change, pct_change, "
        f"ROUND(100.0 * abs_change / NULLIF(total_change, 0), 2) AS share_of_total_change, brands_down, brands "
        f"FROM d ORDER BY abs_change {order}, restaurant_name ASC LIMIT {int(top_n)}"
    )


def drivers_sql(unit: str, cur: str, prev: str) -> str:
    cur_c, prev_c = _in_period(unit, cur), _in_period(unit, prev)
    return (
        f"SELECT CASE WHEN {cur_c} THEN 'current' ELSE 'previous' END AS period, "
        f"SUM(fo.sales_amount) AS revenue, COUNT(DISTINCT fo.order_id) AS orders, "
        f"SUM(fo.sales_amount) / NULLIF(COUNT(DISTINCT fo.order_id), 0) AS aov, "
        f"COUNT(DISTINCT fo.restaurant_id) AS active_branches, COUNT(DISTINCT fo.user_id) AS customers, "
        f"AVG(CASE WHEN fo.order_status IN ('Cancelled', 'Refunded') THEN 1.0 ELSE 0.0 END) AS failed_rate "
        f"FROM {FACT} fo JOIN {DATES} dd ON fo.date_id = dd.date_id "
        f"WHERE ({prev_c}) OR ({cur_c}) GROUP BY 1"
    )


# ------------------------------------------------------------------ rendering
def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _d(v) -> Optional[str]:
    """A strict YYYY-MM-DD from a date/datetime or a date-like string, else None. These values are interpolated
    into SQL, so a string must match IN FULL (no truncating a longer string down to something that looks valid)."""
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()[:10]
    m = _DATE_STRING.match(str(v)) if v is not None else None
    return m.group(1) if m else None


def _num(v: Optional[float], dec: int = 0) -> str:
    return "-" if v is None else f"{v:,.{dec}f}"


def _signed(v: Optional[float], dec: int = 0, suffix: str = "") -> str:
    return "-" if v is None else f"{v:+,.{dec}f}{suffix}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join([":---"] + ["---:"] * (len(headers) - 1)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _period_label(unit: str, start: str, ar: bool) -> str:
    import datetime as dt
    s = dt.date.fromisoformat(start)
    if unit == "week":
        e = s + dt.timedelta(days=6)
        return f"{start} → {e.isoformat()}"
    return s.strftime("%Y-%m")


def render(parsed: dict, extreme: dict, contrib: list[dict], drivers: dict, lang: str) -> str:
    ar = lang in ("ar", "mixed")
    unit, metric, down = parsed["unit"], parsed["metric"], parsed["direction"] == "down"
    cur, prev = _d(extreme["period_start"]), _d(extreme["prev_period_start"])
    c_val, p_val = _f(extreme["current_value"]), _f(extreme["previous_value"])
    chg, pct = _f(extreme["abs_change"]), _f(extreme["pct_change"])
    n_cmp = int(_f(extreme.get("periods_compared")) or 0)
    d0, d1 = _d(extreme.get("data_start")), _d(extreme.get("data_end"))
    base = _f(extreme.get("baseline_avg"))
    sd = _f(extreme.get("pct_stddev"))
    mlabel = (("عدد الطلبات" if metric == "orders" else "الإيراد") if ar else ("orders" if metric == "orders" else "revenue"))
    ulabel = (("أسبوع" if unit == "week" else "شهر") if ar else unit)
    word = ("أكبر انخفاض" if down else "أكبر ارتفاع") if ar else ("largest decrease" if down else "largest increase")
    cur_l, prev_l = _period_label(unit, cur, ar), _period_label(unit, prev, ar)
    dec = 0

    L: list[str] = []
    if ar:
        L.append(f"## {word} في {mlabel} ({ulabel} مقابل {ulabel} السابق)")
        L.append(f"**{ulabel} {cur_l}** مقابل **{prev_l}**: {_num(c_val, dec)} مقابل {_num(p_val, dec)} "
                 f"— التغيير {_signed(chg, dec)} ({_signed(pct, 2, '%')}).")
        L.append(f"_التعريف: التغيير المطلق بين {ulabel}ين كاملين متتاليين ({n_cmp} مقارنة، من {d0} إلى {d1}). "
                 f"الفترات الجزئية في أول وآخر البيانات مستبعدة._")
    else:
        L.append(f"## {word.capitalize()} in {mlabel} ({unit}-over-{unit})")
        L.append(f"**{unit.capitalize()} {cur_l}** vs **{prev_l}**: {_num(c_val, dec)} vs {_num(p_val, dec)} "
                 f"— change {_signed(chg, dec)} ({_signed(pct, 2, '%')}).")
        L.append(f"_Definition: largest absolute change between two consecutive full {unit}s ({n_cmp} comparisons, "
                 f"data {d0} to {d1}). Partial periods at the edges of the data are excluded._")

    # ---- contributors
    if contrib:
        brands, brands_down = int(_f(contrib[0].get("brands")) or 0), int(_f(contrib[0].get("brands_down")) or 0)
        share_hdr = ("% من إجمالي الانخفاض" if down else "% من إجمالي الزيادة") if ar else ("Share of total decline" if down else "Share of total increase")
        hdr = (["المطعم", f"{prev_l}", f"{cur_l}", "التغيير", "% التغيير", share_hdr] if ar
               else ["Restaurant", prev_l, cur_l, "Change", "% change", share_hdr])
        rows = [[str(r["restaurant_name"]), _num(_f(r["prev_value"])), _num(_f(r["cur_value"])), _signed(_f(r["abs_change"])),
                 _signed(_f(r["pct_change"]), 1, "%"), _signed(_f(r["share_of_total_change"]), 1, "%")] for r in contrib]
        top_share = sum(_f(r["share_of_total_change"]) or 0 for r in contrib)
        L.append(("\n### أكبر المساهمين في التغيير (على مستوى البراند)" if ar else "\n### Biggest contributors (brand level)"))
        L.append(_table(hdr, rows))
        if brands:
            L.append((f"أعلى {len(contrib)} براندات = {top_share:.1f}% من إجمالي التغيير؛ {brands_down:,} من {brands:,} براند {'نزلوا' if down else 'زادوا'}."
                      if ar else f"The top {len(contrib)} brands account for {top_share:.1f}% of the total change; "
                                 f"{brands_down:,} of {brands:,} brands {'declined' if down else 'grew'}."))

    # ---- measured decomposition of the same two periods
    cur_d, prev_d = drivers.get("current"), drivers.get("previous")
    if cur_d and prev_d:
        o1, o0, a1, a0 = _f(cur_d["orders"]), _f(prev_d["orders"]), _f(cur_d["aov"]), _f(prev_d["aov"])
        L.append(("\n### التفكيك المقاس للفترتين" if ar else "\n### Measured breakdown of the same two periods"))
        pairs = [
            (("الطلبات" if ar else "Orders"), o0, o1, 0), (("متوسط قيمة الطلب" if ar else "Average order value"), a0, a1, 1),
            (("الفروع النشطة" if ar else "Active branches"), _f(prev_d["active_branches"]), _f(cur_d["active_branches"]), 0),
            (("العملاء" if ar else "Customers"), _f(prev_d["customers"]), _f(cur_d["customers"]), 0),
        ]
        rows = [[name, _num(p, d), _num(c, d), _signed(None if p is None or c is None else c - p, d),
                 _signed(None if not p or c is None else 100 * (c - p) / p, 1, "%")] for name, p, c, d in pairs]
        fr0, fr1 = _f(prev_d["failed_rate"]), _f(cur_d["failed_rate"])
        rows.append([("الطلبات الملغية/المستردة" if ar else "Cancelled/refunded orders"), _num(None if fr0 is None else 100 * fr0, 1) + "%",
                     _num(None if fr1 is None else 100 * fr1, 1) + "%",
                     _signed(None if fr0 is None or fr1 is None else 100 * (fr1 - fr0), 1, " pp"), ""])
        L.append(_table((["المقياس", prev_l, cur_l, "التغيير", "%"] if ar else ["Measure", prev_l, cur_l, "Change", "%"]), rows))
        if None not in (o0, o1, a0, a1) and metric == "revenue":
            # Exact identity: R1 - R0 = (o1 - o0) * a0  +  o1 * (a1 - a0)
            vol, price = (o1 - o0) * a0, o1 * (a1 - a0)
            L.append((f"تحليل حسابي للتغيير في الإيراد: **أثر عدد الطلبات** {_signed(vol)} + **أثر متوسط قيمة الطلب** {_signed(price)} "
                      f"= {_signed(vol + price)}." if ar else
                      f"Arithmetic split of the revenue change: **order-volume effect** {_signed(vol)} + **basket-size effect** "
                      f"{_signed(price)} = {_signed(vol + price)}."))

    # ---- honest limits
    L.append(("\n### حدود التحليل" if ar else "\n### Limits of this analysis"))
    lim = []
    if base and p_val and down and p_val > base * 1.1:
        lim.append((f"الفترة السابقة كانت أعلى من متوسط الفترات قبلها بحوالي {100 * (p_val / base - 1):.0f}%، فجزء من الانخفاض عودة للمستوى الطبيعي مش تراجع حقيقي."
                    if ar else f"The previous {unit} was about {100 * (p_val / base - 1):.0f}% above the average of the {unit}s before it, "
                               f"so part of this drop is a return to the normal level rather than a new decline."))
    elif base and p_val and not down and p_val < base * 0.9:
        lim.append((f"الفترة السابقة كانت أقل من المتوسط بحوالي {100 * (1 - p_val / base):.0f}%، فجزء من الارتفاع تعافي مش نمو."
                    if ar else f"The previous {unit} was about {100 * (1 - p_val / base):.0f}% below its recent average, so part of this rise is a rebound."))
    if sd and pct is not None and n_cmp >= 20:
        z = abs(pct) / sd
        if z < 3.5:
            lim.append((f"التغيير ده ({abs(pct):.2f}%) هو {z:.1f}× التذبذب المعتاد بين {ulabel}ين متتاليين (انحراف معياري {sd:.2f}%). مع {n_cmp} مقارنة، "
                        f"أكبر تغيير بالحجم ده متوقع بالصدفة، فمفيش بالضرورة سبب محدد ورا التغيير." if ar else
                        f"This change ({abs(pct):.2f}%) is {z:.1f}x the typical {unit}-to-{unit} swing (standard deviation {sd:.2f}%). With {n_cmp} comparisons, "
                        f"an extreme this size is expected by chance alone, so there may be no specific cause to find."))
        else:
            lim.append((f"التغيير ده {z:.1f}× التذبذب المعتاد (انحراف معياري {sd:.2f}%)، يعني خارج الطبيعي فعلاً ويستاهل التحقيق." if ar else
                        f"This change is {z:.1f}x the typical swing (standard deviation {sd:.2f}%), so it stands out from ordinary variation and is worth investigating."))
    lim.append(("ده ارتباط مش سببية: الأرقام فوق بتوصف **إيه اللي اتغير**، مش بتثبت ليه. مفيش عندنا بيانات عن الزيارات، التسويق، الإجازات والمواسم، الطقس، أو الأسعار لحظة بلحظة."
                if ar else "This is association, not causation: the figures describe **what changed**, they do not prove why. "
                           "The data has no traffic, marketing, holiday/season, weather or live-pricing information."))
    L.extend(f"- {x}" for x in lim)
    return "\n".join(L)


def _failed(ar: bool, why: str = "") -> str:
    return ("مقدرتش أحسب المقارنة دي من البيانات، ومش هخمّن أرقام." if ar
            else "I could not compute this comparison from the data, and I will not guess numbers.") + (f" ({why})" if why else "")


Executor = Callable[[list], list]


def run_period_change(question: str, lang: str, execute: Executor) -> dict:
    """-> {'answer': str, 'evidence': [{'columns','rows','sql'}]}. `execute` runs guarded read-only SQL."""
    ar = lang in ("ar", "mixed")
    parsed = parse_period_change(question)
    if not parsed:
        return {"answer": _failed(ar), "evidence": []}
    unit, direction, metric = parsed["unit"], parsed["direction"], parsed["metric"]

    sql1 = extreme_period_sql(unit, direction, metric)
    r1 = execute([{"sql": sql1, "purpose": f"{direction} {unit}-over-{unit} extreme"}])[0]
    if not r1.get("success"):
        return {"answer": _failed(ar), "evidence": []}
    rows1 = r1.get("rows") or []
    if not rows1:
        msg = ("مفيش فترتين كاملتين متتاليتين في البيانات أقارن بينهم، فمقدرش أحدد أكبر تغيير." if ar
               else f"The data does not contain two consecutive full {unit}s, so no {unit}-over-{unit} change can be determined.")
        return {"answer": msg, "evidence": [{"columns": r1.get("columns", []), "rows": rows1, "sql": sql1}]}
    extreme = rows1[0]
    cur, prev = _d(extreme.get("period_start")), _d(extreme.get("prev_period_start"))
    if not cur or not prev:   # dates are interpolated into SQL below: only ever a validated YYYY-MM-DD
        return {"answer": _failed(ar), "evidence": [{"columns": r1.get("columns", []), "rows": rows1, "sql": sql1}]}

    sql2 = contributors_sql(unit, direction, metric, cur, prev, parsed["top_n"])
    sql3 = drivers_sql(unit, cur, prev)
    r2, r3 = execute([{"sql": sql2, "purpose": "restaurant contribution to the change"},
                      {"sql": sql3, "purpose": "measured drivers of the two periods"}])
    evidence = [{"columns": r1.get("columns", []), "rows": rows1, "sql": sql1}]
    contrib = r2.get("rows") or [] if r2.get("success") else []
    if r2.get("success"):
        evidence.insert(0, {"columns": r2.get("columns", []), "rows": contrib, "sql": sql2})   # restaurants first: "why?" follow-ups use them
    drivers = {str(r.get("period")): r for r in (r3.get("rows") or [])} if r3.get("success") else {}
    if r3.get("success"):
        evidence.append({"columns": r3.get("columns", []), "rows": r3.get("rows", []), "sql": sql3})
    return {"answer": render(parsed, extreme, contrib, drivers, lang), "evidence": evidence}
