"""
src/agent/token_budget.py

Session and Global Daily Token Budget Manager (Phase B-1).
Prevents hitting Groq 429 daily rate limits (TPD 200,000) by tracking usage
and providing early circuit-breaking with user-friendly bilingual messages.
"""

import os
import json
import time
import datetime
from threading import Lock

# Tunable thresholds
DAILY_GLOBAL_TOKEN_LIMIT = int(os.getenv("DAILY_TOKEN_LIMIT", "180000"))
SESSION_TOKEN_LIMIT = int(os.getenv("SESSION_TOKEN_LIMIT", "60000"))

_BUDGET_LOCK = Lock()
_DAILY_USAGE = {
    "date": "",
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "question_count": 0,
}
_SESSION_USAGE = {}


def _get_today_str() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def record_tokens(session_id: str, prompt_tokens: int, completion_tokens: int):
    """Records prompt and completion tokens for the current day and session."""
    total = prompt_tokens + completion_tokens
    today = _get_today_str()

    with _BUDGET_LOCK:
        if _DAILY_USAGE["date"] != today:
            _DAILY_USAGE["date"] = today
            _DAILY_USAGE["total_prompt_tokens"] = 0
            _DAILY_USAGE["total_completion_tokens"] = 0
            _DAILY_USAGE["total_tokens"] = 0
            _DAILY_USAGE["question_count"] = 0
            _SESSION_USAGE.clear()

        _DAILY_USAGE["total_prompt_tokens"] += prompt_tokens
        _DAILY_USAGE["total_completion_tokens"] += completion_tokens
        _DAILY_USAGE["total_tokens"] += total
        _DAILY_USAGE["question_count"] += 1

        sess = _SESSION_USAGE.setdefault(session_id, {"prompt": 0, "completion": 0, "total": 0, "count": 0})
        sess["prompt"] += prompt_tokens
        sess["completion"] += completion_tokens
        sess["total"] += total
        sess["count"] += 1


def check_budget_available(session_id: str, lang: str = "en") -> tuple[bool, str]:
    """
    Checks if session or global daily limit has been exceeded.
    Returns (is_available, error_message).
    """
    today = _get_today_str()
    with _BUDGET_LOCK:
        if _DAILY_USAGE["date"] == today:
            daily_total = _DAILY_USAGE["total_tokens"]
            if daily_total >= DAILY_GLOBAL_TOKEN_LIMIT:
                if lang == "ar":
                    msg = (
                        f"⚠️ عذراً، تم الوصول إلى الحد الأقصى للاستهلاك اليومي "
                        f"({daily_total:,} / {DAILY_GLOBAL_TOKEN_LIMIT:,} رمز). يرجى المحاولة غداً."
                    )
                else:
                    msg = (
                        f"⚠️ The daily token limit has been reached "
                        f"({daily_total:,} / {DAILY_GLOBAL_TOKEN_LIMIT:,} tokens). Please try again tomorrow."
                    )
                return False, msg

        sess = _SESSION_USAGE.get(session_id, {})
        sess_total = sess.get("total", 0)
        if sess_total >= SESSION_TOKEN_LIMIT:
            if lang == "ar":
                msg = (
                    f"⚠️ تم الوصول إلى الحد الأقصى لاستهلاك الجلسة الحالية "
                    f"({sess_total:,} / {SESSION_TOKEN_LIMIT:,} رمز). يرجى بدء محادثة جديدة."
                )
            else:
                msg = (
                    f"⚠️ Your session token limit has been reached "
                    f"({sess_total:,} / {SESSION_TOKEN_LIMIT:,} tokens). Please start a new session."
                )
            return False, msg

    return True, ""


def get_budget_status() -> dict:
    """Returns current usage snapshot for reporting and UI."""
    with _BUDGET_LOCK:
        return {
            "daily_tokens_used": _DAILY_USAGE["total_tokens"],
            "daily_tokens_limit": DAILY_GLOBAL_TOKEN_LIMIT,
            "daily_questions_count": _DAILY_USAGE["question_count"],
            "remaining_daily_tokens": max(0, DAILY_GLOBAL_TOKEN_LIMIT - _DAILY_USAGE["total_tokens"]),
        }
