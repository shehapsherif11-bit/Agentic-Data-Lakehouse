"""
diagnostics.py — deterministic root-cause diagnosis for restaurants.

"Why does restaurant X have so few orders? where is the problem? what should I do?"

1. Measure the target restaurants (orders, active span, failed-order rate, delivery time, customer rating,
   listing rating, discount rate, basket size, repeat purchase, recent trend).
2. Benchmark them against the whole platform, their own city and their own cuisine.
3. Flag every KPI that deviates beyond a threshold, rank the deviations as root-cause CANDIDATES (correlation,
   not proven causation), attach a concrete action + target KPI to each, and list what was checked and is normal.

Every number in the output is a query result or plain arithmetic on query results — no LLM is involved.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Optional

G = "workspace.zomato_gold"
MAX_ENTITIES = 15


def sql_literal(value: str) -> str:
    # Databricks/Spark: '' is NOT an escaped quote — 'McDonald''s' is two adjacent literals ('McDonald' 's'),
    # i.e. "McDonalds". The apostrophe must be backslash-escaped.
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _in_list(values) -> str:
    return ", ".join(sql_literal(v) for v in values)


# ------------------------------------------------------------------ SQL
def target_sql(names: list[str]) -> str:
    return f"""
WITH bounds AS (
    SELECT MAX(order_timestamp) AS max_ts FROM {G}.fact_orders
),
t AS (
    SELECT restaurant_id, restaurant_name, city, cuisine, rating
    FROM {G}.dim_resturant
    WHERE restaurant_name IN ({_in_list(names[:MAX_ENTITIES])})
)
SELECT
    t.restaurant_name,
    COUNT(DISTINCT t.restaurant_id) AS branches,
    COUNT(DISTINCT t.city) AS cities,
    MIN(t.city) AS city,
    MIN(t.cuisine) AS cuisine,
    AVG(t.rating) AS listing_rating,
    COUNT(fo.order_id) AS orders,
    COUNT(DISTINCT fo.user_id) AS customers,
    SUM(fo.sales_amount) AS revenue,
    AVG(fo.sales_amount) AS aov,
    AVG(CASE WHEN fo.order_status = 'Cancelled' THEN 1.0 ELSE 0.0 END) AS cancel_rate,
    AVG(CASE WHEN fo.order_status = 'Refunded' THEN 1.0 ELSE 0.0 END) AS refund_rate,
    AVG(fo.delivery_time_min) AS avg_delivery_min,
    AVG(fo.customer_rating) AS avg_customer_rating,
    SUM(fo.discount) / NULLIF(SUM(fo.subtotal), 0) AS discount_rate,
    MIN(fo.order_timestamp) AS first_order,
    MAX(fo.order_timestamp) AS last_order,
    MAX(b.max_ts) AS data_end,
    SUM(CASE WHEN fo.order_timestamp > ADD_MONTHS(b.max_ts, -6) THEN 1 ELSE 0 END) AS orders_last_6m,
    SUM(CASE WHEN fo.order_timestamp <= ADD_MONTHS(b.max_ts, -6)
              AND fo.order_timestamp > ADD_MONTHS(b.max_ts, -12) THEN 1 ELSE 0 END) AS orders_prev_6m
FROM t
LEFT JOIN {G}.fact_orders fo ON fo.restaurant_id = t.restaurant_id
CROSS JOIN bounds b
GROUP BY t.restaurant_name
""".strip()


PLATFORM_SQL = f"""
WITH per AS (
    SELECT restaurant_id, COUNT(*) AS n, COUNT(DISTINCT user_id) AS u
    FROM {G}.fact_orders
    GROUP BY restaurant_id
),
dist AS (
    SELECT PERCENTILE(n, 0.1) AS p10, PERCENTILE(n, 0.25) AS p25, PERCENTILE(n, 0.5) AS median_orders,
           PERCENTILE(n, 0.75) AS p75, PERCENTILE(n, 0.9) AS p90, COUNT(*) AS branches,
           AVG(n / u) AS orders_per_customer
    FROM per
),
rates AS (
    SELECT AVG(sales_amount) AS aov,
           AVG(CASE WHEN order_status = 'Cancelled' THEN 1.0 ELSE 0.0 END) AS cancel_rate,
           AVG(CASE WHEN order_status = 'Refunded' THEN 1.0 ELSE 0.0 END) AS refund_rate,
           AVG(delivery_time_min) AS avg_delivery_min,
           AVG(customer_rating) AS avg_customer_rating,
           SUM(discount) / NULLIF(SUM(subtotal), 0) AS discount_rate
    FROM {G}.fact_orders
)
SELECT d.p10, d.p25, d.median_orders, d.p75, d.p90, d.branches, d.orders_per_customer,
       r.aov, r.cancel_rate, r.refund_rate, r.avg_delivery_min, r.avg_customer_rating, r.discount_rate
FROM dist d CROSS JOIN rates r
""".strip()


QUALITY_SQL = f"""
SELECT COUNT(*) AS branches, COUNT(DISTINCT restaurant_name) AS brands, COUNT(rating) AS rated,
       COUNT(cost) AS with_cost, COUNT(rating_count) AS with_rating_count
FROM {G}.dim_resturant
""".strip()


def data_gaps(quality: Optional[dict], lang: str) -> list[str]:
    """Data-quality limits measured from the restaurant table (never assumed)."""
    if not quality or not quality.get("branches"):
        return []
    ar = lang in ("ar", "mixed")
    total = float(quality["branches"])
    out = []
    for col, label_ar, label_en in (("with_cost", "مستوى الأسعار (cost)", "price level (cost)"),
                                    ("with_rating_count", "عدد التقييمات (rating_count)", "review volume (rating_count)")):
        missing = 1 - float(quality.get(col) or 0) / total
        if missing >= 0.5:
            out.append(f"عمود {label_ar} فاضي في {missing * 100:.0f}% من المطاعم، فمقدرش أستخدمه في التشخيص." if ar
                       else f"The {label_en} column is empty for {missing * 100:.0f}% of restaurants, so it cannot be used.")
    unrated = 1 - float(quality.get("rated") or 0) / total
    if unrated >= 0.2:
        out.append(f"{unrated * 100:.0f}% من المطاعم مالهاش تقييم على المنصة أصلاً." if ar
                   else f"{unrated * 100:.0f}% of restaurants have no platform rating at all.")
    return out


def group_benchmark_sql(column: str, values: list[str]) -> str:
    """Per-city or per-cuisine benchmark (column is 'city' or 'cuisine')."""
    assert column in ("city", "cuisine")
    return f"""
WITH o AS (
    SELECT dr.{column} AS grp, fo.restaurant_id, fo.sales_amount, fo.order_status, fo.delivery_time_min,
           fo.customer_rating, fo.discount, fo.subtotal
    FROM {G}.fact_orders fo
    JOIN {G}.dim_resturant dr ON fo.restaurant_id = dr.restaurant_id
    WHERE dr.{column} IN ({_in_list(values)})
),
per AS (
    SELECT grp, restaurant_id, COUNT(*) AS n FROM o GROUP BY grp, restaurant_id
),
dist AS (
    SELECT grp, COUNT(*) AS branches, PERCENTILE(n, 0.5) AS median_orders FROM per GROUP BY grp
),
rates AS (
    SELECT grp, AVG(sales_amount) AS aov,
           AVG(CASE WHEN order_status = 'Cancelled' THEN 1.0 ELSE 0.0 END) AS cancel_rate,
           AVG(CASE WHEN order_status = 'Refunded' THEN 1.0 ELSE 0.0 END) AS refund_rate,
           AVG(delivery_time_min) AS avg_delivery_min,
           AVG(customer_rating) AS avg_customer_rating,
           SUM(discount) / NULLIF(SUM(subtotal), 0) AS discount_rate
    FROM o GROUP BY grp
)
SELECT d.grp, d.branches, d.median_orders, r.aov, r.cancel_rate, r.refund_rate, r.avg_delivery_min,
       r.avg_customer_rating, r.discount_rate
FROM dist d JOIN rates r ON d.grp = r.grp
""".strip()


# ------------------------------------------------------------------ helpers
def _f(v) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _ts(v) -> Optional[_dt.datetime]:
    if v is None:
        return None
    if isinstance(v, _dt.datetime):
        return v.replace(tzinfo=None)
    if isinstance(v, _dt.date):
        return _dt.datetime(v.year, v.month, v.day)
    try:
        return _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _months(a: Optional[_dt.datetime], b: Optional[_dt.datetime]) -> Optional[float]:
    if not a or not b:
        return None
    return max((b - a).days / 30.44, 1.0)


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v * 100:.1f}%"


def _n(v: Optional[float], d: int = 0) -> str:
    if v is None:
        return "-"
    return f"{v:,.{d}f}"


def wilson_lower(k: float, n: int, z: float = 1.96) -> float:
    """Lower bound of the 95% Wilson interval for a proportion k/n: with a dozen orders, '2 failed' is not evidence of anything."""
    if n <= 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return max(0.0, (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / d)


SMALL_SAMPLE = 30


def _confidence(orders: int) -> str:
    return "high" if orders >= 200 else "medium" if orders >= 30 else "low"


# ------------------------------------------------------------------ analysis
def analyse_entity(t: dict, platform: dict, city: Optional[dict], cuisine: Optional[dict], metric: str = "revenue") -> dict:
    """-> {name, metrics, level, findings[], normal[]} for one restaurant (brand).
    `metric` is what the user ranked by: for "orders" a small basket is NOT a cause (it lowers revenue, not the number of orders)."""
    orders = int(t.get("orders") or 0)
    branches = max(int(t.get("branches") or 1), 1)
    opb = orders / branches
    first, last, end = _ts(t.get("first_order")), _ts(t.get("last_order")), _ts(t.get("data_end"))
    active_months = _months(first, end)
    customers = int(t.get("customers") or 0)
    opc = orders / customers if customers else None
    failed = (_f(t.get("cancel_rate")) or 0) + (_f(t.get("refund_rate")) or 0)

    bench = city if city and int(t.get("cities") or 1) == 1 else platform
    bench_name = "city" if bench is city and city else "platform"
    b_failed = (_f(bench.get("cancel_rate")) or 0) + (_f(bench.get("refund_rate")) or 0)
    conf = _confidence(orders)

    p = {k: _f(platform.get(k)) for k in ("p10", "p25", "median_orders", "p75", "p90", "orders_per_customer")}
    if p["p10"] is not None and opb < p["p10"]:
        level = "very_low"
    elif p["p25"] is not None and opb < p["p25"]:
        level = "low"
    elif p["p90"] is not None and opb > p["p90"]:
        level = "very_high"
    elif p["p75"] is not None and opb > p["p75"]:
        level = "high"
    else:
        level = "typical"
    low = level in ("very_low", "low", "typical")

    findings, normal = [], []

    def add(key, kind, value, benchmark, weight, **extra):
        findings.append({"key": key, "kind": kind, "value": value, "benchmark": benchmark, "weight": weight,
                         "confidence": extra.pop("confidence", conf), "bench_name": bench_name, **extra})

    # 1) failed orders (cancelled + refunded)
    if (failed and b_failed and failed > b_failed * 1.5 and failed - b_failed >= 0.03
            and (orders >= SMALL_SAMPLE or wilson_lower(failed * orders, orders) > b_failed)):
        add("failed_orders", "issue", failed, b_failed, (failed / b_failed))
    elif failed and b_failed and failed < b_failed * 0.6 and not low:
        add("failed_orders", "strength", failed, b_failed, b_failed / max(failed, 1e-9))
    else:
        normal.append(("failed_orders", failed, b_failed))

    # 2) delivery time
    d, bd = _f(t.get("avg_delivery_min")), _f(bench.get("avg_delivery_min"))
    if d is not None and bd and d > bd * 1.2 and d - bd >= 5:
        add("slow_delivery", "issue", d, bd, d / bd)
    elif d is not None and bd and d < bd * 0.8 and not low:
        add("slow_delivery", "strength", d, bd, bd / d)
    elif d is not None:
        normal.append(("slow_delivery", d, bd))

    # 3) customer (order) rating
    r, br = _f(t.get("avg_customer_rating")), _f(bench.get("avg_customer_rating"))
    if r is not None and br and br - r >= 0.3:
        add("low_customer_rating", "issue", r, br, 1 + (br - r))
    elif r is not None and br and r - br >= 0.3 and not low:
        add("low_customer_rating", "strength", r, br, 1 + (r - br))
    elif r is not None:
        normal.append(("low_customer_rating", r, br))
    else:
        add("no_customer_ratings", "issue", None, br, 1.1)

    # 4) listing rating (visibility / social proof)
    lr = _f(t.get("listing_rating"))
    if lr is None:
        # Weak evidence when most of the platform is unrated too (then it does not distinguish this restaurant).
        common = (_f(platform.get("unrated_share")) or 0) > 0.4
        add("unrated_listing", "issue", None, _f(platform.get("unrated_share")), 1.05 if common else 1.3,
            confidence="low" if common else "medium")
    elif lr < 3.5:
        add("weak_listing_rating", "issue", lr, 3.9, 1 + (3.9 - lr))
    else:
        normal.append(("weak_listing_rating", lr, None))

    # 5) discount competitiveness
    dr_, bdr = _f(t.get("discount_rate")), _f(bench.get("discount_rate"))
    if dr_ is not None and bdr and dr_ < bdr * 0.6:
        add("low_discount", "issue", dr_, bdr, bdr / max(dr_, 1e-9) / 2)
    elif dr_ is not None and bdr and dr_ > bdr * 1.6:
        add("discount_dependence", "risk", dr_, bdr, dr_ / bdr / 2)
    elif dr_ is not None:
        normal.append(("low_discount", dr_, bdr))

    # 6) basket size (price barrier vs. small baskets)
    a, ba = _f(t.get("aov")), _f(bench.get("aov"))
    if a is not None and ba and a > ba * 1.3:
        add("high_price", "issue" if low else "strength", a, ba, a / ba)
    elif a is not None and ba and a < ba * 0.7:
        if metric == "orders":
            normal.append(("high_price", a, ba))   # a small basket cannot explain few orders
        else:
            add("small_basket", "issue", a, ba, ba / a)
    elif a is not None:
        normal.append(("high_price", a, ba))

    # 7) repeat purchase
    bopc = p["orders_per_customer"]
    if opc is not None and bopc and opc < bopc * 0.8 and orders >= 30:
        add("weak_repeat", "issue", opc, bopc, bopc / opc)
    elif opc is not None and bopc:
        normal.append(("weak_repeat", opc, bopc))

    # 8) recent trend
    last6, prev6 = int(t.get("orders_last_6m") or 0), int(t.get("orders_prev_6m") or 0)
    if prev6 >= 5 and last6 < prev6 * 0.7:
        add("declining", "issue", last6, prev6, prev6 / max(last6, 1))
    elif prev6 >= 5 and last6 > prev6 * 1.3:
        add("growing", "strength", last6, prev6, last6 / prev6)
    else:
        normal.append(("declining", last6, prev6))

    # 9) life-cycle: a new listing simply had less time
    if first and end and (end - first).days < 183:
        add("new_listing", "context", (end - first).days, None, 1.5, confidence="high")
    if last and end and (end - last).days > 60:
        add("inactive_recently", "issue", (end - last).days, None, 1.6, confidence="high")

    # 10) local market and cuisine demand
    pm = p["median_orders"]
    if city and pm and int(t.get("cities") or 1) == 1:
        cm = _f(city.get("median_orders"))
        if cm is not None and cm < pm * 0.8:
            add("weak_city_market", "context", cm, pm, pm / cm, confidence="high")
        if cm and low and opb < cm * 0.75:
            add("below_local_peers", "symptom", opb, cm, cm / max(opb, 1e-9), confidence="high")
    if cuisine and pm:
        qm = _f(cuisine.get("median_orders"))
        if qm is not None and qm < pm * 0.8:
            add("niche_cuisine", "context", qm, pm, pm / qm, confidence="high")

    findings.sort(key=lambda f: ({"issue": 0, "risk": 1, "context": 2, "symptom": 3, "strength": 4}[f["kind"]], -f["weight"]))

    # Decomposition: revenue = branches x orders-per-branch x average order value, each vs. a typical brand.
    import math
    typical_branches = _f(platform.get("branches_per_brand")) or 1.0
    factors = {
        "branches": branches / typical_branches,
        "orders_per_branch": (opb / pm) if pm else None,
        "aov": (a / ba) if (a and ba) else None,
    }
    valid = {k: v for k, v in factors.items() if v and v > 0 and not (metric == "orders" and k == "aov")}
    driver = max(valid, key=lambda k: abs(math.log(valid[k]))) if valid else None
    return {
        "factors": factors, "driver": driver,
        "name": t.get("restaurant_name"), "level": level, "orders": orders, "branches": branches,
        "orders_per_branch": opb, "active_months": active_months, "first": first, "last": last,
        "city": t.get("city"), "cuisine": t.get("cuisine"), "customers": customers, "opc": opc,
        "failed": failed, "delivery": d, "rating": r, "listing_rating": lr, "discount": dr_, "aov": a,
        "revenue": _f(t.get("revenue")), "findings": findings, "normal": normal, "confidence": conf,
        "bench_name": bench_name,
    }


# ------------------------------------------------------------------ text
_T = {
    "failed_orders": ("ارتفاع الطلبات الفاشلة (إلغاء + استرداد)", "High failed orders (cancelled + refunded)"),
    "slow_delivery": ("التوصيل أبطأ من المنافسين", "Slower delivery than peers"),
    "low_customer_rating": ("تقييم العملاء للطلبات أقل من المنافسين", "Lower customer order ratings than peers"),
    "no_customer_ratings": ("مفيش أي تقييمات من العملاء على الطلبات", "No customer ratings on any order"),
    "unrated_listing": ("المطعم مالوش تقييم على المنصة (ظهور وثقة أقل)", "Listing has no platform rating (weaker visibility / social proof)"),
    "weak_listing_rating": ("تقييم المطعم على المنصة ضعيف", "Weak platform listing rating"),
    "low_discount": ("خصومات أقل بكتير من المنافسين", "Much lower discounts than peers"),
    "discount_dependence": ("اعتماد عالي على الخصومات", "Heavy reliance on discounts"),
    "high_price": ("متوسط قيمة الطلب أعلى من السوق (حاجز سعر محتمل)", "Basket value well above the market (possible price barrier)"),
    "small_basket": ("قيمة الطلب صغيرة مقارنة بالسوق", "Small basket value vs. market"),
    "weak_repeat": ("العملاء مبيرجعوش يطلبوا تاني", "Customers rarely re-order"),
    "declining": ("الطلبات بتقل في آخر 6 شهور", "Orders falling in the last 6 months"),
    "growing": ("الطلبات بتزيد في آخر 6 شهور", "Orders growing in the last 6 months"),
    "new_listing": ("مطعم جديد على المنصة (وقت أقل لتجميع طلبات)", "New listing (less time to accumulate orders)"),
    "inactive_recently": ("مفيش طلبات من فترة (ممكن يكون مقفول أو مش متاح)", "No recent orders (may be closed / unavailable)"),
    "weak_city_market": ("سوق المدينة نفسه ضعيف", "Weak local (city) market"),
    "below_local_peers": ("أقل من مطاعم نفس المدينة", "Below restaurants in the same city"),
    "niche_cuisine": ("نوع المطبخ عليه طلب قليل على مستوى المنصة", "Low platform-wide demand for this cuisine"),
}

_ACTION = {
    "failed_orders": ("راجع أسباب الإلغاء والاسترداد (نفاد أصناف، تأخير قبول الطلب، أخطاء في التحضير) وحدّث توفر المنيو لحظياً. الهدف: نزّل النسبة لـ {b} أو أقل.",
                      "Audit cancellation/refund reasons (out-of-stock items, slow acceptance, preparation errors) and keep menu availability live. Target: {b} or lower."),
    "slow_delivery": ("قلّل وقت التحضير (منيو أبسط للأصناف الأكثر طلباً، تجهيز مسبق في أوقات الذروة) وراجع نطاق التوصيل. الهدف: {b} دقيقة.",
                      "Cut preparation time (simplify best-sellers, pre-prep for peak hours) and review the delivery radius. Target: {b} min."),
    "low_customer_rating": ("حلّل تقييمات وتعليقات العملاء السلبية وعالج أكتر 3 شكاوى (جودة، تغليف، كمية). الهدف: {b}.",
                            "Analyse negative ratings/reviews and fix the top 3 complaints (quality, packaging, portion). Target: {b}."),
    "no_customer_ratings": ("شجّع العملاء يقيّموا (رسالة بعد التوصيل أو حافز بسيط) عشان يبقى فيه مقياس جودة أصلاً.",
                            "Prompt customers to rate (post-delivery message or a small incentive) so quality can be measured at all."),
    "unrated_listing": ("كمّل بيانات المطعم على المنصة (صور، منيو كامل، وصف) واطلب تقييمات من أول العملاء — المطاعم بدون تقييم بتظهر أقل وبتكسب ثقة أقل.",
                        "Complete the listing (photos, full menu, description) and collect first ratings — unrated listings rank lower and convert worse."),
    "weak_listing_rating": ("ركّز على أسباب التقييم الضعيف قبل أي حملة تسويق؛ التسويق لمطعم تقييمه ضعيف بيهدر الميزانية.",
                            "Fix the causes of the weak rating before any marketing push; promoting a poorly-rated listing wastes budget."),
    "low_discount": ("جرّب عرض محدود ومستهدف (مثلاً لأول طلب أو في الأوقات الهادية) وقيس أثره على الطلبات — بس خد بالك إن الربحية مش ظاهرة لأن بيانات التكلفة مش متاحة. متوسط الخصم عند المنافسين {b}.",
                     "Test a limited, targeted offer (first order or off-peak) and measure the order uplift — profitability is unknown because cost data is unavailable. Peer discount rate: {b}."),
    "discount_dependence": ("الخصومات أعلى بكتير من السوق؛ اختبر تقليلها تدريجياً وتابع هل الطلبات بتثبت.",
                            "Discounts are far above market; test reducing them gradually and watch whether orders hold."),
    "high_price": ("ضيف أصناف بسعر دخول أقل أو كومبو صغير عشان تقلل حاجز السعر. متوسط السوق {b}.",
                   "Add entry-price items or small combos to lower the price barrier. Market average basket: {b}."),
    "small_basket": ("زوّد قيمة الطلب بكومبوهات وإضافات (upsell). متوسط السوق {b}.",
                     "Raise basket size with combos and add-ons (upsell). Market average basket: {b}."),
    "weak_repeat": ("اعمل برنامج ولاء أو عرض للطلب التاني خلال أسبوعين. المتوسط على المنصة {b} طلب لكل عميل.",
                    "Launch a loyalty / second-order offer within two weeks. Platform average: {b} orders per customer."),
    "declining": ("راجع اللي اتغير في آخر 6 شهور (أسعار، منيو، تقييمات، منافس جديد قريب) — الطلبات نزلت من {b} لـ {v}.",
                  "Review what changed in the last 6 months (prices, menu, ratings, a new nearby competitor) — orders fell from {b} to {v}."),
    "inactive_recently": ("اتأكد إن المطعم مفتوح ومتاح على التطبيق (ساعات العمل، حالة الحساب).",
                          "Check that the restaurant is open and available in the app (opening hours, account status)."),
    "weak_city_market": ("الطلب في المدينة كلها ضعيف؛ الحل مش تشغيلي بس — قارن بمطاعم نفس المدينة مش بالمنصة كلها، وفكّر في التسويق المحلي.",
                         "Demand is weak city-wide; judge it against local peers, not the whole platform, and consider local marketing."),
    "below_local_peers": ("المطعم أقل من وسيط مدينته ({b} طلب)، فالمشكلة خاصة بيه مش بالسوق — ركّز على الأسباب التشغيلية فوق.",
                          "It is below its own city's median ({b} orders), so the gap is restaurant-specific, not the market — focus on the operational causes above."),
    "niche_cuisine": ("نوع المطبخ عليه طلب قليل؛ ضيف أصناف أوسع جاذبية أو بيعها كباقة مع أصناف شعبية.",
                      "The cuisine has low demand; add broader-appeal items or bundle with popular ones."),
    "new_listing": ("المطعم جديد؛ قارنه بعد ما يكمل 6 شهور، وركّز دلوقتي على أول التقييمات.",
                    "It is new; compare again after 6 months and focus now on first ratings."),
}

_UNIT = {"failed_orders": "pct", "slow_delivery": "min", "low_customer_rating": "r", "weak_listing_rating": "r",
         "low_discount": "pct", "discount_dependence": "pct", "high_price": "money", "small_basket": "money",
         "weak_repeat": "x2", "declining": "int", "growing": "int", "weak_city_market": "int",
         "below_local_peers": "int1", "niche_cuisine": "int", "new_listing": "days", "inactive_recently": "days",
         "no_customer_ratings": "r", "unrated_listing": "r"}


def _fmt_unit(key: str, v) -> str:
    u = _UNIT.get(key)
    if v is None:
        return "-"
    return {"pct": _pct(v), "min": f"{v:.1f}", "r": f"{v:.2f}", "money": _n(v), "x2": f"{v:.2f}",
            "int": _n(v), "int1": _n(v, 1), "days": _n(v)}.get(u, str(v))


def _evidence_line(f: dict, ar: bool) -> str:
    v, b, key = f["value"], f["benchmark"], f["key"]
    bench_word = {"city": ("متوسط مدينته", "its city"), "platform": ("متوسط المنصة", "platform")}[f["bench_name"]]
    if key in ("weak_city_market", "niche_cuisine"):
        return (f"وسيط الطلبات لكل فرع {_fmt_unit(key, v)} مقابل {_fmt_unit(key, b)} على المنصة" if ar
                else f"median orders per branch {_fmt_unit(key, v)} vs {_fmt_unit(key, b)} platform-wide")
    if key == "below_local_peers":
        return (f"{_fmt_unit(key, v)} طلب لكل فرع مقابل وسيط {_fmt_unit(key, b)} في مدينته" if ar
                else f"{_fmt_unit(key, v)} orders per branch vs a city median of {_fmt_unit(key, b)}")
    if key == "declining" or key == "growing":
        return (f"{v} طلب في آخر 6 شهور مقابل {b} في الـ 6 شهور اللي قبلها" if ar
                else f"{v} orders in the last 6 months vs {b} in the 6 months before")
    if key == "new_listing":
        return f"أول طلب من {v} يوم بس" if ar else f"first order only {v} days before the data ends"
    if key == "inactive_recently":
        return f"آخر طلب قبل نهاية البيانات بـ {v} يوم" if ar else f"last order {v} days before the data ends"
    if key == "unrated_listing":
        share = f" (ونفس الحال في {b * 100:.0f}% من كل المطاعم)" if ar and b else (f" (true for {b * 100:.0f}% of all restaurants too)" if b else "")
        return ("rating = NULL في جدول المطاعم" if ar else "rating is NULL in the restaurant table") + share
    if key == "no_customer_ratings":
        return "كل الطلبات customer_rating = NULL" if ar else "every order has customer_rating = NULL"
    return (f"{_fmt_unit(key, v)} مقابل {_fmt_unit(key, b)} ({bench_word[0]})" if ar
            else f"{_fmt_unit(key, v)} vs {_fmt_unit(key, b)} ({bench_word[1]})")


_CONF = {"high": ("عالية", "high"), "medium": ("متوسطة", "medium"), "low": ("منخفضة — عدد الطلبات قليل", "low — few orders")}


def _action(f: dict, ar: bool) -> Optional[str]:
    tmpl = _ACTION.get(f["key"])
    if not tmpl:
        return None
    return tmpl[0 if ar else 1].format(b=_fmt_unit(f["key"], f["benchmark"]), v=_fmt_unit(f["key"], f["value"]))


_T_POS = {
    "failed_orders": ("طلبات فاشلة أقل من المنافسين", "Fewer failed orders than peers"),
    "slow_delivery": ("توصيل أسرع من المنافسين", "Faster delivery than peers"),
    "low_customer_rating": ("تقييم عملاء أعلى من المنافسين", "Higher customer ratings than peers"),
    "high_price": ("متوسط قيمة الطلب أعلى من السوق", "Basket value above the market"),
    "growing": ("الطلبات بتزيد في آخر 6 شهور", "Orders growing in the last 6 months"),
}
_NEUTRAL = {
    "failed_orders": ("نسبة الإلغاء والاسترداد", "cancellation/refund rate"),
    "slow_delivery": ("وقت التوصيل", "delivery time"),
    "low_customer_rating": ("تقييم العملاء", "customer rating"),
    "weak_listing_rating": ("تقييم المنصة", "listing rating"),
    "low_discount": ("نسبة الخصم", "discount rate"),
    "high_price": ("متوسط قيمة الطلب", "basket value"),
    "weak_repeat": ("تكرار الشراء", "repeat purchase"),
    "declining": ("اتجاه آخر 6 شهور", "last-6-months trend"),
}
_FACTOR = {"branches": ("عدد الفروع", "number of branches"),
           "orders_per_branch": ("الطلبات لكل فرع", "orders per branch"),
           "aov": ("متوسط قيمة الطلب", "average order value")}


def _x(v: Optional[float]) -> str:
    return "-" if v is None else (f"×{v:,.1f}" if v >= 0.95 else f"×{v:.2f}")


def _label(f: dict, i: int) -> str:
    if f["kind"] == "strength" and f["key"] in _T_POS:
        return _T_POS[f["key"]][i]
    return _T[f["key"]][i]


def render(results: list[dict], platform: dict, lang: str, metric_label: Optional[str] = None,
           data_gaps: Optional[list[str]] = None, metric: str = "revenue") -> str:
    ar = lang in ("ar", "mixed")
    i = 0 if ar else 1
    if not results:
        return ("مالقيتش المطاعم دي في جدول المطاعم، فمقدرش أشخّصها." if ar
                else "I could not find these restaurants in the restaurant table, so I cannot diagnose them.")

    pm, p10 = _f(platform.get("median_orders")), _f(platform.get("p10"))
    lines: list[str] = []
    multi = len(results) > 1
    low_side = sum(r["level"] in ("very_low", "low", "typical") for r in results) >= sum(r["level"] in ("high", "very_high") for r in results)

    # ---- 1. the problem, measured
    def _span(ms):
        if not ms:
            return ""
        lo, hi = min(ms), max(ms)
        rng = f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}"
        return f" على مدار ~{rng} شهر" if ar else f" over ~{rng} months"

    if multi:
        opbs = [r["orders_per_branch"] for r in results]
        lines.append("## التشخيص" if ar else "## Diagnosis")
        lines.append(
            (f"**{'المشكلة' if low_side else 'الملاحظة'} (مقاسة):** الـ {len(results)} مطاعم عندهم بين {min(opbs):,.0f} و {max(opbs):,.0f} طلب لكل فرع"
             + _span([r["active_months"] for r in results if r["active_months"]])
             + (f"، مقابل وسيط {pm:,.0f} طلب لكل فرع على المنصة (أقل 10% من الفروع تحت {p10:,.0f})." if pm and p10 else "."))
            if ar else
            (f"**The {'problem' if low_side else 'observation'} (measured):** these {len(results)} restaurants have {min(opbs):,.0f}–{max(opbs):,.0f} orders per branch"
             + _span([r["active_months"] for r in results if r["active_months"]])
             + (f", vs a platform median of {pm:,.0f} orders per branch (the bottom 10% of branches are below {p10:,.0f})." if pm and p10 else "."))
        )
    else:
        r = results[0]
        lvl = {"very_low": ("أقل 10% من الفروع", "the bottom 10% of branches"), "low": ("أقل 25%", "the bottom 25%"),
               "typical": ("في النطاق الطبيعي", "within the normal range"), "high": ("أعلى 25%", "the top 25%"),
               "very_high": ("أعلى 10%", "the top 10%")}[r["level"]][i]
        lines.append(f"## التشخيص: {r['name']}" if ar else f"## Diagnosis: {r['name']}")
        lines.append(
            f"**{'المشكلة' if low_side else 'الملاحظة'} (مقاسة):** {r['orders']:,} طلب ({r['orders_per_branch']:,.1f} لكل فرع، {r['branches']} فرع){_span([r['active_months']] if r['active_months'] else [])} — ده في {lvl} على المنصة (الوسيط {_n(pm)})."
            if ar else
            f"**The {'problem' if low_side else 'observation'} (measured):** {r['orders']:,} orders ({r['orders_per_branch']:,.1f} per branch, {r['branches']} branch(es)){_span([r['active_months']] if r['active_months'] else [])} — {lvl} on the platform (median {_n(pm)})."
        )

    summary_at = len(lines)   # the 3-line summary is inserted here once causes and actions are known

    # ---- 2. decomposition: which factor explains the number
    drivers = [r["driver"] for r in results if r.get("driver")]
    if drivers:
        by_orders = metric == "orders"
        lines.append(("\n### تفكيك الرقم: عدد الطلبات = الفروع × الطلبات لكل فرع" if by_orders else
                      "\n### تفكيك الرقم: الإيراد = الفروع × الطلبات لكل فرع × متوسط قيمة الطلب") if ar
                     else ("\n### Breaking the number down: orders = branches × orders per branch" if by_orders else
                           "\n### Breaking the number down: revenue = branches × orders per branch × average order value"))
        lines.append(("كل عامل مقارن بالبراند العادي على المنصة (×1 = زي المتوسط)." if ar
                      else "Each factor is compared with a typical brand on the platform (×1 = average)."))
        hdr = (["المطعم", "الفروع", "طلب/فرع"] + ([] if by_orders else ["متوسط الطلب"]) + ["العامل الأهم"] if ar
               else ["Restaurant", "Branches", "Orders/branch"] + ([] if by_orders else ["Basket"]) + ["Main factor"])
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|:---|" + "---:|" * (len(hdr) - 2) + ":---|")
        for r in results:
            fx = r["factors"]
            drv = _FACTOR[r["driver"]][i] if r.get("driver") else "-"
            basket = "" if by_orders else f"{_n(r['aov'])} ({_x(fx['aov'])}) | "
            lines.append(f"| {r['name']} | {r['branches']:,} ({_x(fx['branches'])}) | {r['orders_per_branch']:,.1f} ({_x(fx['orders_per_branch'])}) | "
                         f"{basket}{drv} |")
        top_driver = max(set(drivers), key=drivers.count)
        cnt = drivers.count(top_driver)
        sample = next(r for r in results if r.get("driver") == top_driver)
        fval = sample["factors"][top_driver]
        direction = ("أعلى" if fval >= 1 else "أقل") if ar else ("above" if fval >= 1 else "below")
        lines.append(
            (f"\n**الاستنتاج:** العامل اللي بيفسّر الرقم أكتر من غيره هو **{_FACTOR[top_driver][0]}**"
             + (f" ({cnt} من {len(results)} مطاعم)" if multi else "")
             + f" — مثلاً {sample['name']} {direction} من المعتاد بـ {_x(fval)}.")
            if ar else
            (f"\n**Conclusion:** the factor that explains the number most is **{_FACTOR[top_driver][1]}**"
             + (f" ({cnt} of {len(results)} restaurants)" if multi else "")
             + f" — e.g. {sample['name']} is {_x(fval)} {direction} typical.")
        )

    # ---- 3. root-cause candidates
    agg: dict[str, dict] = {}
    for r in results:
        for f in r["findings"]:
            if f["kind"] == "strength" and low_side:
                continue
            a = agg.setdefault(f["key"], {"f": f, "who": [], "score": 0.0})
            a["who"].append(r["name"])
            a["score"] += f["weight"]
    order = {"issue": 0, "risk": 1, "context": 2, "symptom": 3, "strength": 4}
    ranked = sorted(agg.values(), key=lambda a: (order[a["f"]["kind"]], -len(a["who"]), -a["score"]))
    causes = [a for a in ranked if a["f"]["kind"] in ("issue", "risk", "context")]
    strengths = [a for a in ranked if a["f"]["kind"] == "strength"]
    symptoms = [a for a in ranked if a["f"]["kind"] == "symptom"]
    if symptoms and low_side:
        # "Below restaurants in the same city" only restates the gap (and rests on the same few orders): show it as measured
        # context, never as a root cause with its own confidence.
        s0 = symptoms[0]
        lines.insert(summary_at, (
            f"- مقارنة بنفس المدينة (وصف للفجوة، مش تفسير لها): {len(s0['who'])} من {len(results)} أقل من وسيط مدينتهم — {_evidence_line(s0['f'], ar)}."
            if ar else
            f"- Versus the same city (describes the gap, does not explain it): {len(s0['who'])} of {len(results)} are below their city's median — {_evidence_line(s0['f'], ar)}."))   # inserted at summary_at: the later 3-line summary lands above it

    if low_side:
        if causes:
            lines.append("\n### الأسباب الجذرية المحتملة (مرتبة حسب قوة الدليل)" if ar else "\n### Likely root causes (ranked by strength of evidence)")
            for n, a in enumerate(causes[:6], 1):
                f = a["f"]
                who = (f" — في {len(a['who'])} من {len(results)} مطاعم" if ar else f" — in {len(a['who'])} of {len(results)} restaurants") if multi else ""
                ev = "" if multi and f["key"] not in ("weak_city_market", "niche_cuisine", "unrated_listing") else f" — {_evidence_line(f, ar)}"
                conf = _CONF[f["confidence"]][i]
                lines.append(f"{n}. **{_label(f, i)}**{who}{ev}. " + (f"الثقة: {conf}." if ar else f"Confidence: {conf}."))
        if not [a for a in causes if a["f"]["kind"] == "issue" and a["f"]["confidence"] != "low"]:
            if causes:
                lines.append("\n⚠️ الإشارات دي كلها ثقتها منخفضة (طلبات قليلة جداً)، فخدها كمؤشرات تتأكد منها مش كأحكام."
                             if ar else "\n⚠️ All of these signals are low-confidence (very few orders): treat them as leads to verify, not verdicts.")
            lines.append(
                "\nمفيش مؤشر تشغيلي (إلغاء، توصيل، تقييم عملاء، خصم، تكرار شراء) بعيد عن المنافسين بشكل **مؤكد**، "
                "فالفرق غالباً في **الطلب والظهور** (ترتيب البحث، التسويق، الموقع) — ودي بيانات مش موجودة عندنا."
                if ar else
                "\nNo operational KPI (cancellations, delivery, customer rating, discount, repeat purchase) deviates from peers with confidence, "
                "so the gap is most likely **demand and visibility** (search ranking, marketing, location) — data we do not have."
            )
    else:
        lines.append("\n### ليه الأداء عالي" if ar else "\n### Why performance is high")
        for a in (strengths + [c for c in causes if c["f"]["kind"] == "context"])[:5]:
            f = a["f"]
            lines.append(f"- **{_label(f, i)}** — {_evidence_line(f, ar)}")
        if not strengths:
            lines.append("- المؤشرات التشغيلية (إلغاء، توصيل، تقييم، خصم) زي متوسط المنصة بالظبط، فالتفوق مش جاي من التشغيل — جاي من الحجم والطلب (شوف التفكيك فوق)."
                         if ar else "- Operational KPIs (cancellations, delivery, rating, discount) match the platform average, so the lead does not come from operations — it comes from scale and demand (see the breakdown above).")
        risks = [c for c in causes if c["f"]["kind"] in ("issue", "risk")]
        if risks:
            lines.append("\n**مخاطر لازم تتابع:** " + "، ".join(_label(c["f"], i) for c in risks[:3]) if ar
                         else "\n**Risks to watch:** " + ", ".join(_label(c["f"], i) for c in risks[:3]))

    # ---- 4. per-entity evidence table
    lines.append("\n### الأرقام (مقاسة)" if ar else "\n### The numbers (measured)")
    hdr = (["المطعم", "الطلبات", "طلب/فرع", "إلغاء+استرداد", "التوصيل (د)", "تقييم العملاء", "تقييم المنصة", "الخصم", "متوسط الطلب"]
           if ar else ["Restaurant", "Orders", "Orders/branch", "Failed %", "Delivery (min)", "Customer rating", "Listing rating", "Discount", "Basket"])
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|:---|" + "---:|" * (len(hdr) - 1))
    for r in results:
        lines.append(f"| {r['name']} | {r['orders']:,} | {r['orders_per_branch']:,.1f} | {_pct(r['failed'])} | "
                     f"{_n(r['delivery'], 1)} | {_n(r['rating'], 2)} | {_n(r['listing_rating'], 2)} | {_pct(r['discount'])} | {_n(r['aov'])} |")
    pf = (_f(platform.get("cancel_rate")) or 0) + (_f(platform.get("refund_rate")) or 0)
    lines.append(f"| **{'متوسط المنصة' if ar else 'Platform'}** | - | {_n(pm)} ({'وسيط' if ar else 'median'}) | {_pct(pf)} | "
                 f"{_n(_f(platform.get('avg_delivery_min')), 1)} | {_n(_f(platform.get('avg_customer_rating')), 2)} | - | "
                 f"{_pct(_f(platform.get('discount_rate')))} | {_n(_f(platform.get('aov')))} |")

    # ---- 5. what was checked and is normal
    family = {"small_basket": "high_price", "discount_dependence": "low_discount", "growing": "declining",
              "unrated_listing": "weak_listing_rating", "no_customer_ratings": "low_customer_rating"}
    flagged = {family.get(k, k) for k in agg}
    checked = {k for r in results for k, *_ in r["normal"]} - flagged
    if metric == "orders":
        checked.discard("high_price")   # basket size is not part of an order-count diagnosis: do not call it "normal"
    if checked:
        names = "، ".join(_NEUTRAL[k][i] for k in sorted(checked) if k in _NEUTRAL) if ar else \
            ", ".join(_NEUTRAL[k][i] for k in sorted(checked) if k in _NEUTRAL)
        lines.append(("\n**اتفحص وطلع طبيعي (مش هو السبب):** " if ar else "\n**Checked and normal (not the cause):** ") + names)

    # ---- 6. actions
    acts = []
    if low_side:
        for a in causes[:4]:
            act = _action(a["f"], ar)
            if act and act not in acts:
                acts.append(act)
        if "orders_per_branch" in drivers:
            acts.append("الطلبات لكل فرع هي نقطة الضعف: حسّن الظهور (صور، منيو كامل، أول تقييمات) واعمل عرض محدود في الأوقات الهادية، وقيس عدد الطلبات الأسبوعي بعد 4 أسابيع مقابل وسيط المنصة."
                        if ar else "Orders per branch is the weak factor: improve visibility (photos, full menu, first ratings) and run a limited off-peak offer; track weekly orders after 4 weeks against the platform median.")
    else:
        if "branches" in drivers:
            acts.append("التفوق جاي من حجم الشبكة: أي توسّع جديد لازم يتقاس بالطلبات لكل فرع عشان الفروع الجديدة متسحبش المتوسط لتحت."
                        if ar else "The lead comes from network size: judge any expansion by orders per branch so new branches do not dilute the average.")
        acts.append("استخدم أعلى المطاعم دي كـ benchmark: قارن الفروع الضعيفة بيهم في الطلبات لكل فرع."
                    if ar else "Use these top restaurants as the benchmark: compare weak branches with them on orders per branch.")
    if not acts:
        acts.append("ابدأ بتحسين الظهور: صور ومنيو كامل وأول تقييمات، وعروض محدودة في الأوقات الهادية، وقيس الأثر بعد 4 أسابيع."
                    if ar else "Start with visibility: photos, full menu, first ratings, and a limited off-peak offer; measure after 4 weeks.")
    lines.append("\n### الحل المقترح" if ar else "\n### Recommended actions")
    lines.extend(f"{n}. {t}" for n, t in enumerate(acts[:5], 1))

    # ---- summary first: the answer in three lines, details below
    summary = []
    if drivers:
        td = max(set(drivers), key=drivers.count)
        fv = next(r for r in results if r.get("driver") == td)["factors"][td]
        summary.append(f"- **العامل الأساسي:** {_FACTOR[td][0]} ({_x(fv)} من المعتاد)." if ar
                       else f"- **Main factor:** {_FACTOR[td][1]} ({_x(fv)} typical).")
    issues = [a for a in causes if a["f"]["kind"] in ("issue", "risk")] if low_side else strengths
    if issues:
        f0 = issues[0]["f"]
        conf = _CONF[f0["confidence"]][i].split(" — ")[0]
        summary.append(f"- **أقوى إشارة:** {_label(f0, i)} (ثقة {conf})." if ar
                       else f"- **Strongest signal:** {_label(f0, i)} ({conf} confidence).")
    elif low_side:
        summary.append("- **أقوى إشارة:** مفيش خلل تشغيلي مؤكد؛ الأرجح ضعف الطلب والظهور." if ar
                       else "- **Strongest signal:** no confirmed operational fault; most likely weak demand / visibility.")
    if acts:
        first = acts[0].split(" — ")[0].split("؛")[0].split(". ")[0].rstrip(".")
        summary.append(f"- **أول خطوة:** {first}." if ar else f"- **First step:** {first}.")
    if summary:
        lines[summary_at:summary_at] = ["\n### الخلاصة" if ar else "\n### In short"] + summary

    # ---- 7. limits
    lim = [
        "ده **ارتباط مش سببية**: الأسباب دي أقوى تفسيرات بتدعمها البيانات، مش إثبات." if ar
        else "This is **correlation, not causation**: these are the explanations the data supports best, not proof.",
    ]
    if any(r["confidence"] == "low" for r in results):
        lim.append("عدد الطلبات قليل جداً (أقل من 30)، فالنسب (إلغاء، تقييم، خصم) حساسة لطلب أو اتنين." if ar
                   else "Order counts are very small (< 30), so rates (cancellations, ratings, discount) swing on one or two orders.")
    lim.append("مفيش بيانات زيارات/ظهور في البحث، ولا تكاليف أو أرباح، ولا تسويق — فالأسباب دي مش ممكن أقيسها." if ar
               else "There is no traffic/search-visibility, cost/profit or marketing data, so those causes cannot be measured.")
    lim.extend(data_gaps or [])
    lines.append("\n### حدود التحليل" if ar else "\n### Limits")
    lines.extend(f"- {t}" for t in lim)

    lines.append(
        "\n💡 **تقدر تسألني بعدها:** «اعرض اتجاه الطلبات الشهري للمطاعم دي» أو «اي احسن مطعم فيهم من وجهة نظرك»."
        if ar else
        "\n💡 **Next you can ask:** “show the monthly order trend for these restaurants” or “which one is best in your opinion?”."
    )
    return "\n".join(lines)
