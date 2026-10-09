"""
query_intent.py — deterministic parsing of the parts of a question that must never depend on an LLM:
how many rows were asked for (top N), which direction a ranking goes, whether the user is asking for an
opinion / decision ("which is best? give me one name"), and which language the user wrote in.

These were previously left to the intent LLM or hardcoded (`LIMIT 5`), which is how "top 6" returned 5 rows.
"""
import re
from typing import Optional

MAX_TOP_N = 100
DEFAULT_TOP_N = 10

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
    "واحد": 1, "واحدة": 1, "اتنين": 2, "اثنين": 2, "اثنان": 2, "تلاتة": 3, "ثلاثة": 3, "ثلاث": 3,
    "اربعة": 4, "أربعة": 4, "اربع": 4, "أربع": 4, "خمسة": 5, "خمس": 5, "ستة": 6, "ست": 6,
    "سبعة": 7, "سبع": 7, "تمانية": 8, "ثمانية": 8, "ثماني": 8, "تسعة": 9, "تسع": 9,
    "عشرة": 10, "عشر": 10, "خمستاشر": 15, "عشرين": 20,
}

_RANK_MARKERS = (
    r"top|bottom|best|worst|highest|lowest|first|leading|biggest|smallest|largest|"
    r"أفضل|افضل|احسن|أحسن|أعلى|اعلى|أكثر|اكثر|اكتر|أقل|اقل|اوطى|أسوأ|اسوأ|اسوا|أضعف|اضعف|اول|أول"
)
_N_TOKEN = r"(\d+|[a-z]+|[؀-ۿ]+)"
_N_AFTER_MARKER = re.compile(rf"(?:{_RANK_MARKERS})\s+{_N_TOKEN}")
_N_BEFORE_MARKER = re.compile(rf"(\d+)\s+(?:{_RANK_MARKERS})")
_N_BEFORE_NOUN = re.compile(
    r"(\d+)\s+(?:restaurants?|brands?|cities|city|customers?|users?|items?|dishes|مطاعم|مطعم|مدن|مدينة|عملاء|عميل|أصناف|اصناف)"
)
_SINGLE = re.compile(r"\b(one|single)\s+(name|restaurant|brand|answer|pick|option)\b|اسم واحد|مطعم واحد|واحد بس|واحد فقط|\bjust one\b|\bonly one\b")

_ASC_MARKERS = re.compile(
    r"\b(bottom|lowest|worst|least|fewest|smallest|weakest|poorest|slowest)\b|"
    r"(?<![؀-ۿ])(ال)?(أقل|اقل|اوطى|أدنى|ادنى|أسوأ|اسوأ|اسوا|أضعف|اضعف)"
)
_DESC_MARKERS = re.compile(
    r"\b(top|best|highest|most|largest|biggest|leading|strongest)\b|"
    r"(أعلى|اعلى|أكثر|اكثر|اكتر|أفضل|افضل|احسن|أحسن)"
)

# Metrics where a SMALLER value is the better one (used when choosing a "best" candidate).
LOWER_IS_BETTER = ("delivery_time", "discount", "cost", "decline", "drop", "loss")


def _to_int(token: str) -> Optional[int]:
    token = token.translate(_DIGITS).strip()
    if token.isdigit():
        return int(token)
    return _WORD_NUMBERS.get(token)


def extract_top_n(question: str) -> Optional[int]:
    """The N the user asked for ('top 6', 'أفضل 6 مطاعم', 'top six', '6 best'), or None when no count is
    stated. A bounded value in [1, MAX_TOP_N]; anything outside that is treated as 'not stated'."""
    if not question:
        return None
    t = question.lower().translate(_DIGITS)
    candidates = []
    for pat in (_N_AFTER_MARKER, _N_BEFORE_MARKER, _N_BEFORE_NOUN):
        for m in pat.finditer(t):
            n = _to_int(m.group(1))
            if n is not None:
                candidates.append(n)
        if candidates:
            break
    n = candidates[0] if candidates else None
    if n is None and _SINGLE.search(t):
        n = 1
    if n is None or not (1 <= n <= MAX_TOP_N):
        return None
    return n


def ranking_direction(question: str) -> str:
    """'ASC' for bottom/lowest/worst style questions, otherwise 'DESC'."""
    t = (question or "").lower()
    if _ASC_MARKERS.search(t) and not _DESC_MARKERS.search(t):
        return "ASC"
    if _ASC_MARKERS.search(t):
        # Both present ("highest and lowest"): the nearest-to-start marker wins; ties go to the explicit bottom.
        asc, desc = _ASC_MARKERS.search(t), _DESC_MARKERS.search(t)
        return "ASC" if asc.start() < desc.start() else "DESC"
    return "DESC"


_ADVISORY = re.compile(
    r"\bin your opinion\b|\byour opinion\b|\byour view\b|\bwhat do you think\b|\bdo you recommend\b|"
    r"\brecommend\b|\bshould i\b|\bworth (it|investing)\b|\binvest(ing|ment)?\b|\badvice\b|\badvise\b|"
    r"\bwhich (one )?(is|would be) (the )?best\b|\bwhich one\b.*\b(pick|choose|best)\b|"
    r"رأيك|رايك|وجه[ةه] نظرك|وجه[ةه] نظر[كي]|انصحني|تنصحني|تنصح|نصيحة|نصيحه|استثمر|استثمار|أستثمر|"
    r"اختار لي|اختارلي|اختر لي|تختار|ترشح|رشح لي|رشحلي"
)
_BEST_WORD = re.compile(r"\b(best|better|top)\b|أفضل|افضل|احسن|أحسن")
_PICK_ONE = re.compile(r"\b(one|single)\b|واحد|اسم")


def is_advisory_question(question: str) -> bool:
    """True when the user wants an opinion / a pick / a decision ('which is best? give me one name',
    'اي احسن مطعم استثمر فيه') rather than a plain ranking ('top 5 restaurants')."""
    if not question:
        return False
    t = question.lower()
    if _ADVISORY.search(t):
        return True
    return bool(_BEST_WORD.search(t) and _PICK_ONE.search(t) and extract_top_n(t) == 1)


_SINGULAR = re.compile(
    r"\b(most|least|highest|lowest|top|best|worst|biggest|smallest)\s+(\w+\s+)?(restaurant|customer|city|user|client|brand|dish|item)\b(?!s)|"
    r"\bwho is the (top|best|biggest|most)|\bwhich (restaurant|customer|city|user|brand)\b(?!s)|"
    r"(?<![؀-ۿ])(?:ال)?(اكتر|أكتر|أكثر|اكثر|اقل|أقل|احسن|أحسن|افضل|أفضل|اعلى|أعلى|اوطى|أسوأ|اسوأ)\s+"
    r"(مطعم|عميل|زبون|مدينة|مدينه|صنف|منطقة|منطقه|براند)(?![؀-ۿ])"
)


def is_singular_superlative(question: str) -> bool:
    """'اي اقل مطعم ...', 'مين اكتر عميل', 'which restaurant has the most orders' — the user wants ONE answer
    (plus ties), not a list."""
    t = (question or "").lower()
    return bool(_SINGULAR.search(t)) and extract_top_n(t) in (None, 1)


def is_investment_question(question: str) -> bool:
    return bool(re.search(r"\binvest|\bworth\b|استثمر|استثمار|أستثمر|\bbuy\b|\bacquire\b|\bpartner", (question or "").lower()))


_DATA_TERMS = re.compile(
    r"\b(restaurants?|customers?|users?|orders?|revenue|sales|profit|margin|rating|cit(y|ies)|cuisine|top|bottom|"
    r"highest|lowest|average|total|how many|how much|trend|month|year)\b|"
    r"مطعم|مطاعم|عميل|عملاء|زبون|طلب|اوردر|أوردر|ايراد|إيراد|مبيعات|ارباح|أرباح|ربح|تقييم|مدين|مطبخ|"
    r"اكتر|أكثر|اقل|أقل|افضل|أفضل|احسن|أحسن|اعلى|أعلى|كام|متوسط|اجمالي|إجمالي|شهر|سنة|سنه"
)


_DIMENSIONS = (
    ("customer", re.compile(r"\b(customers?|clients?|users?|buyers?)\b|عميل|عملاء|العميل|العملاء|زبون|زباين|زبائن|مشتري")),
    ("restaurant_name", re.compile(r"\b(restaurants?|brands?)\b|مطعم|مطاعم|المطعم|المطاعم|براند")),
    ("city", re.compile(r"\b(cit(y|ies))\b|مدينة|مدينه|مدن|المدينة|المدينه")),
    ("cuisine", re.compile(r"\b(cuisines?)\b|مطبخ|مطابخ")),
)


def extract_dimension(question: str) -> Optional[str]:
    """The entity the user ranks/groups by, read from the question itself ('مين اكتر عميل' -> customer).
    Used to correct the intent LLM, which carried 'restaurant' over from context and answered KFC as a customer."""
    t = (question or "").lower()
    hits = [(m.start(), dim) for dim, rx in _DIMENSIONS for m in [rx.search(t)] if m]
    return min(hits)[1] if hits else None


def looks_like_data_question(question: str) -> bool:
    """Deterministic fallback when the router LLM is unavailable: does this need the database?"""
    return bool(_DATA_TERMS.search((question or "").lower()))


def detect_language(text: str) -> str:
    """'ar' / 'en' / 'mixed' from the script of the text (fallback when the router did not classify)."""
    text = text or ""
    arabic = len(re.findall(r"[؀-ۿ]", text))
    latin_words = len(re.findall(r"[A-Za-z]{2,}", text))
    if arabic and latin_words >= 3:
        return "mixed"
    if arabic:
        return "ar"
    return "en"


_SHARE_QUESTION = re.compile(
    r"\bshare\b|\bproportion of\b|percent(age)?\s+(contribution|of\s+(the\s+)?(total|overall|all))|"
    r"%\s*(contribution|of\s+(the\s+)?(total|overall))|contribution\s+(to|of)\s+(the\s+)?(total|overall)|"
    r"حص[ةه]|مساهم[ةه]|نسب[ةه]\s+(مساهم|من\s+(ال)?(اجمالي|إجمالي|كل))"
)


def extract_derived_measures(question: str) -> list[str]:
    """Extra columns the user asked for on top of the ranked metric. Only 'share_of_total' ('percentage
    contribution to overall revenue') is recognised; it is computed in SQL, never by the LLM. Arabic 'نسبة'
    alone is NOT matched: 'نسبة الخصم' is the discount-rate metric."""
    return ["share_of_total"] if _SHARE_QUESTION.search((question or "").lower()) else []
