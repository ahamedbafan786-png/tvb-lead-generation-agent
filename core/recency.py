"""
core/recency.py
Date-based 18-month recency validation for TVB qualification pipeline.
Separates latest round date from financial evidence date.
"""

import re
from datetime import datetime, date
from typing import Optional, Tuple

RECENCY_THRESHOLD_DAYS = 548  # 18 months ~ 548 days

MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12
}

def parse_financial_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str or not isinstance(date_str, str):
        return None

    clean = date_str.strip().lower()
    if not clean:
        return None

    # 1. Full date: YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", clean)
    if m_iso:
        try:
            return date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))
        except ValueError:
            return None

    # 2. Year-Month: YYYY-MM or YYYY/MM
    m_ym = re.search(r"\b((?:19|20)\d{2})[-/](\d{1,2})\b", clean)
    if m_ym:
        try:
            return date(int(m_ym.group(1)), int(m_ym.group(2)), 1)
        except ValueError:
            return None

    # 3. Month Name + Year: e.g. "August 2025", "aug 2025"
    m_my = re.search(r"\b([a-z]{3,9})\s+((?:19|20)\d{2})\b", clean)
    if m_my:
        month_name = m_my.group(1)
        year = int(m_my.group(2))
        if month_name in MONTH_MAP:
            try:
                return date(year, MONTH_MAP[month_name], 1)
            except ValueError:
                return None

    # If the string contains a hyphenated/slashed year (e.g. 2025-99-99 or 2025-13),
    # it was an invalid date format attempt and should not fall back to year-only.
    if re.search(r"\b(?:19|20)\d{2}[-/]", clean):
        return None

    # 4. Standalone Year: YYYY (e.g. "2025")
    m_year = re.search(r"(?<![-/])\b((?:19|20)\d{2})\b(?![-/])", clean)
    if m_year:
        try:
            year = int(m_year.group(1))
            return date(year, 12, 31)
        except ValueError:
            return None

    return None

def is_older_than_18_months(date_str: Optional[str], reference_date: Optional[date] = None) -> bool:
    if not date_str:
        return True
    ref = reference_date or datetime.now().date()
    parsed = parse_financial_date(date_str)
    if not parsed:
        return True
    # Future dates cannot qualify as valid recent past evidence
    if parsed > ref:
        return True
    return (ref - parsed).days > RECENCY_THRESHOLD_DAYS

def evaluate_recency_basis(
    financial_type: str,
    evidence_date: Optional[str],
    round_date: Optional[str],
    revenue_date: Optional[str] = None,
    reference_date: Optional[date] = None
) -> Tuple[bool, str, Optional[str]]:
    ref = reference_date or datetime.now().date()

    if financial_type in ("ARR", "ANNUAL_REVENUE", "TTM_REVENUE"):
        candidates = [
            (revenue_date, f"{financial_type} evidence date"),
            (evidence_date, f"{financial_type} confirmation evidence"),
        ]
    elif financial_type == "CUMULATIVE_FUNDING":
        candidates = [
            (evidence_date, f"Cumulative confirmation evidence ({evidence_date})" if evidence_date else "Cumulative confirmation evidence"),
            (round_date, f"Latest round date ({round_date})" if round_date else "Latest round date"),
        ]
    else:
        # SINGLE_ROUND_FUNDING or other
        candidates = [
            (round_date, f"Financing round date ({round_date})" if round_date else "Financing round date"),
            (evidence_date, f"Financing confirmation evidence ({evidence_date})" if evidence_date else "Financing confirmation evidence"),
        ]

    def _check_date(d_str: Optional[str]) -> Tuple[str, Optional[date], Optional[int]]:
        if not d_str or not isinstance(d_str, str) or not d_str.strip():
            return "missing", None, None
        parsed = parse_financial_date(d_str)
        if not parsed:
            return "malformed", None, None
        delta = (ref - parsed).days
        if delta < 0:
            return "future", parsed, delta
        elif delta == 0:
            return "today", parsed, 0
        else:
            return "valid_past", parsed, delta

    # Pass 1: Select the highest-priority candidate that is a valid historical date (today or valid_past).
    # This guarantees future dates or malformed dates cannot override valid historical evidence.
    for d_str, label in candidates:
        status, parsed, delta = _check_date(d_str)
        if status in ("today", "valid_past"):
            approx_months = round(delta / 30.4, 1)
            if delta > RECENCY_THRESHOLD_DAYS:
                return False, f"Dated: {label} is ~{approx_months}m old (exceeds 18m limit)", d_str
            return True, f"Recent: {label} is ~{approx_months}m old (within 18m window)", d_str

    # Pass 2: No valid historical date found. Report future or malformed on highest priority present candidate.
    for d_str, label in candidates:
        status, parsed, delta = _check_date(d_str)
        if status == "future":
            return False, f"Future date: {label} is in the future ({d_str} > {ref}); recency unverified", d_str
        elif status == "malformed":
            return False, f"Date '{d_str}' unparseable; recency unverified", d_str

    # All candidate dates were missing
    return False, "Financial recency unverified (no verifiable date published)", None
