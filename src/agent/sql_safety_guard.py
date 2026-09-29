"""
SQL Safety Guard — Deterministic security layer for all SQL before Databricks execution.

This module is SEPARATE from analytical SQL validation.
It enforces read-only access regardless of what the LLM generates.

Pipeline position:
    LLM generates SQL → sql_validator (structural) → sql_safety_guard → Databricks execution
"""

import re
import logging
from typing import NamedTuple

logger = logging.getLogger("sql_safety_guard")


class SafetyVerdict(NamedTuple):
    """Immutable result of a safety check."""
    safe: bool
    reason: str  # human-readable explanation (empty string when safe)


# ──────────────────────────────────────────────
# 1.  Blocked keyword patterns
# ──────────────────────────────────────────────
#   Word-boundary matching so column names like "updated_at" don't trigger.
_WRITE_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "MERGE",
    "UPSERT", "REPLACE",
]
_DDL_KEYWORDS = [
    "CREATE", "DROP", "ALTER", "TRUNCATE", "RENAME",
]
_DCL_KEYWORDS = [
    "GRANT", "REVOKE", "DENY",
]
_DANGEROUS_KEYWORDS = [
    "EXEC", "EXECUTE", "CALL",           # stored-proc execution
    "COPY", "UNLOAD", "LOAD",            # bulk data movement
    "SET",                                # session variable mutation
    "OPTIMIZE", "VACUUM", "ANALYZE",     # DDL-like maintenance
    "MSCK",                               # Hive metastore repair
]

_ALL_BLOCKED: list[re.Pattern] = []
for kw in _WRITE_KEYWORDS + _DDL_KEYWORDS + _DCL_KEYWORDS + _DANGEROUS_KEYWORDS:
    _ALL_BLOCKED.append(re.compile(r"\b" + kw + r"\b", re.IGNORECASE))


# ──────────────────────────────────────────────
# 2.  Suspicious patterns (regex)
# ──────────────────────────────────────────────
_SUSPICIOUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Multiple statements (semicolons followed by non-whitespace/non-end)
    (re.compile(r";\s*\S"), "Multiple SQL statements detected (possible injection)."),
    # SQL comments that could hide payloads
    (re.compile(r"--.*\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|EXEC)\b", re.IGNORECASE),
     "Suspicious comment containing a blocked keyword."),
    # UNION-based injection attempt
    (re.compile(r"\bUNION\s+ALL\s+SELECT\b.*\bFROM\s+information_schema\b", re.IGNORECASE),
     "Possible UNION-based injection targeting information_schema."),
    # INTO OUTFILE / DUMPFILE (MySQL-style exfiltration — unlikely on Databricks but safe to block)
    (re.compile(r"\bINTO\s+(OUTFILE|DUMPFILE)\b", re.IGNORECASE),
     "INTO OUTFILE/DUMPFILE is not allowed."),
    # xp_cmdshell, sp_ (SQL Server stored procs — defense in depth)
    (re.compile(r"\b(xp_cmdshell|sp_executesql|sp_oacreate)\b", re.IGNORECASE),
     "Dangerous stored procedure reference detected."),
    # DBFS / cloud paths in SQL (could be used for data exfiltration)
    (re.compile(r"dbfs:/|s3://|gs://|abfss://|wasbs://", re.IGNORECASE),
     "Direct cloud storage path reference is not allowed."),
]


# ──────────────────────────────────────────────
# 3.  Allowed catalogs/schemas (whitelist)
# ──────────────────────────────────────────────
_ALLOWED_SCHEMAS = {
    "workspace.zomato_gold",
    "workspace.zomato_silver",
    "workspace.zomato_bronze",
}

_TABLE_REF_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_.]+)", re.IGNORECASE
)


# ──────────────────────────────────────────────
# 4.  Public API
# ──────────────────────────────────────────────
def check_sql_safety(sql: str) -> SafetyVerdict:
    """
    Run ALL deterministic safety checks on a SQL string.

    Returns SafetyVerdict(safe=True, reason="") when the query passes,
    or SafetyVerdict(safe=False, reason="...") on the first failure.

    This function is intentionally fail-closed: if it cannot verify
    the SQL, it rejects it.
    """
    if not sql or not sql.strip():
        return SafetyVerdict(False, "Empty SQL query.")

    cleaned = _strip_sql_comments(sql)

    # ── Check 1: blocked keywords ──
    for pattern in _ALL_BLOCKED:
        if pattern.search(cleaned):
            kw = pattern.pattern.replace(r"\b", "")
            return SafetyVerdict(False, f"Blocked keyword detected: {kw}")

    # ── Check 2: must start with SELECT or WITH (CTE) ──
    first_keyword = _first_keyword(cleaned)
    if first_keyword not in ("SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"):
        return SafetyVerdict(
            False,
            f"Query must start with SELECT/WITH/SHOW/DESCRIBE/EXPLAIN, "
            f"found: {first_keyword or '(nothing)'}",
        )

    # ── Check 3: reject multiple statements ──
    if _has_multiple_statements(cleaned):
        return SafetyVerdict(False, "Multiple SQL statements are not allowed.")

    # ── Check 4: suspicious patterns ──
    for pattern, reason in _SUSPICIOUS_PATTERNS:
        if pattern.search(cleaned):
            return SafetyVerdict(False, reason)

    # ── Check 5: table reference whitelist ──
    for match in _TABLE_REF_PATTERN.finditer(cleaned):
        table_ref = match.group(1).lower()
        
        # Extract catalog and schema if provided
        parts = table_ref.split('.')
        
        if len(parts) >= 3:
            full_schema = f"{parts[0]}.{parts[1]}"
        elif len(parts) == 2:
            full_schema = f"workspace.{parts[0]}" # default catalog
        else:
            full_schema = "workspace.zomato_gold" # default catalog and schema
            
        if full_schema not in {s.lower() for s in _ALLOWED_SCHEMAS}:
            # Exceptions for built-in or allowed non-schema tables
            if table_ref not in ['information_schema.tables', 'information_schema.columns']:
                return SafetyVerdict(
                    False,
                    f"Table reference '{table_ref}' is outside allowed schemas: "
                    f"{', '.join(sorted(_ALLOWED_SCHEMAS))}.",
                )

    # ── Check 6: length sanity ──
    if len(sql) > 10_000:
        return SafetyVerdict(False, "SQL query exceeds maximum allowed length (10 000 chars).")

    return SafetyVerdict(True, "")


def check_multiple_queries(queries: list[dict]) -> list[dict]:
    """
    Validate a batch of query dicts [{sql, purpose, ...}].

    Returns the same list with added fields:
        - safety_passed: bool
        - safety_reason: str  (empty when passed)

    Queries that fail safety will NOT be executed.
    """
    results = []
    for q in queries:
        verdict = check_sql_safety(q.get("sql", ""))
        updated = {**q, "safety_passed": verdict.safe, "safety_reason": verdict.reason}
        if not verdict.safe:
            logger.warning(
                "SQL BLOCKED by safety guard | purpose='%s' | reason='%s' | sql='%.200s'",
                q.get("purpose", ""), verdict.reason, q.get("sql", ""),
            )
        results.append(updated)
    return results


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────
def _strip_sql_comments(sql: str) -> str:
    """Remove single-line (--) and multi-line (/* */) comments."""
    # Multi-line
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Single-line
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql.strip()


def _first_keyword(sql: str) -> str | None:
    """Return the first SQL keyword (uppercased) or None."""
    match = re.match(r"\s*(\w+)", sql)
    return match.group(1).upper() if match else None


def _has_multiple_statements(sql: str) -> bool:
    """
    Detect multiple statements separated by semicolons.
    Ignores trailing semicolons and semicolons inside string literals.
    """
    # Remove string literals to avoid false positives
    no_strings = re.sub(r"'[^']*'", "''", sql)
    # Split on semicolons and check for non-empty segments
    parts = [p.strip() for p in no_strings.split(";") if p.strip()]
    return len(parts) > 1
