"""
src/agent/numeric_grounding.py

Numeric and Narrative Grounding Validator (Phase B).
Enforces strict numerical traceability against SQL query evidence:
1. Strips markdown list markers, ordinals, and ranking prefixes (e.g. '1.', '2.', '#1') before extracting numbers.
2. Suffix scaling (K, M, B, million, ألف, مليون, مليار) is accepted ONLY when the token has an explicit scaling suffix.
3. Plain numbers (e.g. 14,000,000) require a raw match against unscaled evidence values within displayed rounding.
4. Percentages are verified only with explicit '%' or percent keywords.
5. Narrative claims ('bulk', 'majority', 'lion's share') must be backed by explicit >50% calculations in the evidence.
"""

import re
import math
import datetime
from typing import List, Dict, Any, Tuple

# Arabic digit mapping
ARABIC_TO_WESTERN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

# Suffix multipliers
SCALING_FACTORS = {
    'K': 1_000,
    'THOUSAND': 1_000,
    'الف': 1_000,
    'ألف': 1_000,
    'M': 1_000_000,
    'MILLION': 1_000_000,
    'مليون': 1_000_000,
    'B': 1_000_000_000,
    'BILLION': 1_000_000_000,
    'مليار': 1_000_000_000,
}

def clean_text_for_grounding(text: str, row_count: int) -> str:
    """Strips markdown list markers, ranking indicators, and currency symbols."""
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        # Strip list markers like "1. ", "15) ", "- 1. " at the start of lines
        cleaned = re.sub(r'^\s*(?:[-*]\s*)?\d{1,3}[.)]\s*', '', line)
        cleaned_lines.append(cleaned)
        
    cleaned_text = "\n".join(cleaned_lines)
    
    # Strip rank prefixes like "#1", "#15", "rank 1", "المركز 1"
    cleaned_text = re.sub(r'(?:#|rank\s*|المركز\s*|المرتبة\s*)\d{1,3}\b', '', cleaned_text, flags=re.IGNORECASE)
    
    # Strip known currency symbols to avoid spurious tokens
    cleaned_text = re.sub(r'[₹$€£¥]|(?:INR|USD|EGP|AED|ج\.م|درهم|جنيه)\b', '', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text


def _parse_numeric_tokens(text: str) -> List[Tuple[str, float, str]]:
    """
    Finds numeric tokens and determines their type:
    Returns list of (raw_token, parsed_value, token_type)
    where token_type is 'percent', 'scaled', or 'raw'.
    """
    tokens = []
    
    # 1. Percentages: e.g. 7.8%, 15 %, 15.6%., 12 percent, 12 بالمئة
    pct_pattern = re.compile(
        r'(\b[0-9٠-٩]+(?:[.,][0-9٠-٩]+)?)\s*(%|\b(?:percent|percentage|بالمئة|في المئة)\b)',
        re.IGNORECASE
    )
    for m in pct_pattern.finditer(text):
        num_str = m.group(1).translate(ARABIC_TO_WESTERN).replace(',', '')
        try:
            val = float(num_str)
            tokens.append((m.group(0), val, 'percent'))
        except ValueError:
            pass

    # 2. Scaled numbers: e.g. 14M, 14.5 million, 42.5M, 100K, 14 مليون
    scaled_pattern = re.compile(
        r'(\b[0-9٠-٩]+(?:[.,][0-9٠-٩]+)?)\s*(k|m|b|thousand|million|billion|ألف|الف|مليون|مليار)\b',
        re.IGNORECASE
    )
    for m in scaled_pattern.finditer(text):
        num_str = m.group(1).translate(ARABIC_TO_WESTERN).replace(',', '')
        suffix = m.group(2).upper()
        # Normalize arabic letters
        if suffix == 'الف': suffix = 'ألف'
        mult = SCALING_FACTORS.get(suffix, 1)
        try:
            base_val = float(num_str)
            tokens.append((m.group(0), base_val * mult, 'scaled'))
        except ValueError:
            pass

    # 3. Raw numbers (with commas or dots): e.g. 42,466,314 or 61731
    # Match numbers not immediately followed by scale suffix or %
    raw_pattern = re.compile(
        r'\b[0-9٠-٩]{1,3}(?:,[0-9٠-٩]{3})+(?:\.[0-9٠-٩]+)?\b|\b[0-9٠-٩]+(?:\.[0-9٠-٩]+)?\b'
    )
    
    # We collect text with percentages and scaled numbers masked out to avoid duplicate capture
    masked_text = pct_pattern.sub(' ', text)
    masked_text = scaled_pattern.sub(' ', masked_text)
    
    for m in raw_pattern.finditer(masked_text):
        raw_str = m.group(0)
        num_str = raw_str.translate(ARABIC_TO_WESTERN).replace(',', '')
        try:
            val = float(num_str)
            tokens.append((raw_str, val, 'raw'))
        except ValueError:
            pass

    return tokens


def ungrounded_numbers(text: str, evidence: List[Dict[str, Any]]) -> List[str]:
    """
    Finds numeric values in text that do not correspond to any number in evidence or derived metrics.
    Follows Phase B grounding rules:
    - Strips list markers and rankings.
    - Scaled tokens (e.g. 14M) match evidence (e.g. 13,994,523) within rounding tolerance.
    - Raw tokens (e.g. 14,000,000) must match evidence directly (does not match 13,994,523).
    - Percentages must match percent columns, ratios, or calculated shares.
    """
    row_count = len(evidence)
    cleaned = clean_text_for_grounding(text, row_count)
    tokens = _parse_numeric_tokens(cleaned)
    
    if not tokens:
        return []

    # Extract all numbers from evidence
    evidence_raw_numbers = set()
    evidence_percentages = set()
    
    for row in evidence:
        for k, val in row.items():
            k_lower = k.lower()
            if isinstance(val, (int, float)):
                fval = float(val)
                evidence_raw_numbers.add(fval)
                evidence_raw_numbers.add(abs(fval))
                if any(p in k_lower for p in ['pct', 'percent', 'share', 'ratio', 'contribution']):
                    evidence_percentages.add(abs(fval))
                    evidence_percentages.add(abs(fval) * 100) # support both 0.078 and 7.8%
            elif isinstance(val, (str, datetime.date, datetime.datetime)):
                for num_str in re.findall(r'\d+(?:\.\d+)?', str(val)):
                    try:
                        n = float(num_str)
                        evidence_raw_numbers.add(n)
                        evidence_raw_numbers.add(abs(n))
                    except ValueError:
                        pass

    # Allow row count and ranking positions 1..row_count
    for rank in range(1, row_count + 1):
        evidence_raw_numbers.add(float(rank))
    evidence_raw_numbers.add(float(row_count))
    
    # Allow common calendar years in this dataset (2023-2027)
    for yr in [2023.0, 2024.0, 2025.0, 2026.0, 2027.0]:
        evidence_raw_numbers.add(yr)

    # Derived shares: ratio of entity delta or metric to sum / total
    ev_list = [abs(x) for x in evidence_raw_numbers if x > 0]
    derived_ratios = set()
    for i in range(len(ev_list)):
        for j in range(len(ev_list)):
            if i != j and ev_list[j] > 0:
                r = ev_list[i] / ev_list[j]
                if r <= 1.0:
                    derived_ratios.add(r)
                    derived_ratios.add(r * 100)

    ungrounded = []

    for raw_token, val, token_type in tokens:
        # Check against small integers (1..10 or days/months)
        if val.is_integer() and (0 <= val <= max(12, row_count)):
            continue

        matched = False

        if token_type == 'raw':
            # Raw number match: unscaled numbers must match an exact evidence number (within 1.0 for int/float rounding)
            for ev in evidence_raw_numbers:
                if math.isclose(val, ev, rel_tol=0.0001, abs_tol=1.0):
                    matched = True
                    break

        elif token_type == 'scaled':
            # Scaled match: e.g. 1.2M matching 1,234,567 (2.8%) or 14M matching 13,994,523 (0.04%)
            for ev in evidence_raw_numbers:
                if math.isclose(val, ev, rel_tol=0.035, abs_tol=100.0):
                    matched = True
                    break

        elif token_type == 'percent':
            # Percent match: e.g. 7.8% or 7.22%
            for p in evidence_percentages:
                if math.isclose(val, p, rel_tol=0.05, abs_tol=0.2):
                    matched = True
                    break
            if not matched:
                for r in derived_ratios:
                    if math.isclose(val, r, rel_tol=0.05, abs_tol=0.2):
                        matched = True
                        break

        if not matched:
            ungrounded.append(raw_token.strip())

    return ungrounded


def unbacked_narrative_claims(text: str, evidence: List[Dict[str, Any]]) -> List[str]:
    """Flags qualitative proportion claims (e.g. 'represent the bulk') when not backed by math."""
    violations = []
    lower_text = text.lower()
    
    bulk_keywords = [
        "represent the bulk", "represents the bulk", "constitute the bulk", "constitutes the bulk",
        "bulk of the decline", "bulk of the drop", "bulk of the loss", "bulk of the overall",
        "majority of the decline", "majority of the drop", "lion's share",
        "accounted for the bulk", "accounts for the bulk",
        "معظم الانخفاض", "أغلبية الانخفاض", "معظم التراجع", "أغلبية التراجع", "الجزء الأكبر من الانخفاض"
    ]
    
    if any(kw in lower_text for kw in bulk_keywords):
        # Look for explicit percentage or share column >= 50%
        has_share_proof = any(
            isinstance(v, (int, float)) and v >= 50
            for row in evidence
            for k, v in row.items()
            if any(term in k.lower() for term in ["share", "pct", "percent", "ratio", "proportion", "contribution"])
        )
        if not has_share_proof:
            violations.append(
                "Proportion hallucination: Claimed entities represent the 'bulk' or 'majority' "
                "without explicit >50% share calculation in data."
            )
            
    return violations
