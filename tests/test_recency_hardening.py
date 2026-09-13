"""
Deterministic & Adversarial Test Suite: Recency & Date Validation Hardening
=============================================================================
Tests all requirements for preventing future financial dates from qualifying,
handling malformed calendar dates safely without exceptions, preserving 18-month
boundary arithmetic, and ensuring future dates cannot override valid historical evidence.

Section 8: Recency Decision Tests:
1. valid recent date -> ACCEPT
2. date exactly at accepted boundary (548 days) -> ACCEPT
3. date just outside boundary (549 days) -> REJECT
4. future date -> REJECT
5. malformed date -> REJECT / UNPARSEABLE safely
6. impossible calendar date (e.g. 2026-02-31, 2025-02-29) -> safe invalid state
7. missing date -> REJECT when recency required
8. valid current financial evidence + old round -> preserve existing valid behavior (ACCEPT)
9. future evidence + qualifying amount -> REJECT
10. future latest round + valid historical evidence -> future date cannot override valid evidence

Section 9: Pipeline Crash-Safety Tests:
- candidate with 2026-02-31 does not crash pipeline
- candidate with 2025-13-01 does not crash pipeline
- candidate with 2099-01-01 does not crash pipeline
"""

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from core.extractor import evaluate_financial_qualification
from core.models import CandidateCompany, ContactPathResult
from core.pipeline import run_lead_pipeline
from core.recency import (
    RECENCY_THRESHOLD_DAYS,
    evaluate_recency_basis,
    is_older_than_18_months,
    parse_financial_date,
)

FIXED_REF = date(2026, 9, 13)


# ==============================================================================
# 1. PARSE_FINANCIAL_DATE TESTS: CALENDAR SAFETY & MALFORMED INPUTS
# ==============================================================================

def test_parse_date_valid_formats():
    assert parse_financial_date("2026-05-15") == date(2026, 5, 15)
    assert parse_financial_date("2026/05/15") == date(2026, 5, 15)
    assert parse_financial_date("2026-05") == date(2026, 5, 1)
    assert parse_financial_date("2026/05") == date(2026, 5, 1)
    assert parse_financial_date("May 2026") == date(2026, 5, 1)
    assert parse_financial_date("august 2025") == date(2025, 8, 1)
    assert parse_financial_date("2025") == date(2025, 12, 31)
    assert parse_financial_date("Raised Seed in 2025") == date(2025, 12, 31)


def test_parse_date_leap_years():
    # 2024 is a leap year -> valid
    assert parse_financial_date("2024-02-29") == date(2024, 2, 29)
    # 2025 is not a leap year -> invalid, must safely return None
    assert parse_financial_date("2025-02-29") is None


def test_parse_date_impossible_calendar_dates_safely_return_none():
    assert parse_financial_date("2026-02-31") is None
    assert parse_financial_date("2025-04-31") is None  # April has 30 days
    assert parse_financial_date("2025-13-01") is None  # Invalid month 13
    assert parse_financial_date("2025-00-10") is None  # Invalid month 00
    assert parse_financial_date("2025-12-00") is None  # Invalid day 00
    assert parse_financial_date("2025-12-32") is None  # Invalid day 32
    assert parse_financial_date("2025-13") is None     # Invalid YYYY-MM
    assert parse_financial_date("2025-00") is None     # Invalid YYYY-MM


def test_parse_date_malformed_and_empty_safely_return_none():
    assert parse_financial_date("") is None
    assert parse_financial_date("   ") is None
    assert parse_financial_date(None) is None
    assert parse_financial_date("not-a-date") is None
    assert parse_financial_date("sometime in the spring") is None
    assert parse_financial_date("2025-xyz") is None


# ==============================================================================
# 2. SECTION 8 RECENCY DECISION TESTS
# ==============================================================================

def test_1_valid_recent_date_accepted():
    """1. valid recent date -> ACCEPT"""
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date="2026-05-01",
        round_date=None,
        reference_date=FIXED_REF,
    )
    assert is_rec is True
    assert "Recent" in note
    assert eff == "2026-05-01"


def test_2_exact_boundary_accepted():
    """2. date exactly at accepted boundary (548 days) -> ACCEPT"""
    boundary_date = FIXED_REF - timedelta(days=RECENCY_THRESHOLD_DAYS)  # Exactly 548 days ago
    boundary_str = boundary_date.isoformat()
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date=boundary_str,
        round_date=None,
        reference_date=FIXED_REF,
    )
    assert is_rec is True
    assert "Recent" in note


def test_3_just_outside_boundary_rejected():
    """3. date just outside boundary (549 days) -> REJECT"""
    outside_date = FIXED_REF - timedelta(days=RECENCY_THRESHOLD_DAYS + 1)  # 549 days ago
    outside_str = outside_date.isoformat()
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date=outside_str,
        round_date=None,
        reference_date=FIXED_REF,
    )
    assert is_rec is False
    assert "Dated" in note
    assert "exceeds 18m limit" in note


def test_4_future_date_rejected():
    """4. future date -> REJECT (both with fixed ref and today's actual system date)."""
    # Fixed ref
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date="2099-01-01",
        round_date=None,
        reference_date=FIXED_REF,
    )
    assert is_rec is False
    assert "Future date" in note
    assert "in the future" in note

    # Real system date (without reference_date)
    is_rec_live, note_live, eff_live = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date="2099-01-01",
        round_date=None,
    )
    assert is_rec_live is False
    assert "Future date" in note_live

    # is_older_than_18_months helper must also reject future date
    assert is_older_than_18_months("2099-01-01", reference_date=FIXED_REF) is True
    assert is_older_than_18_months("2099-01-01") is True


def test_5_malformed_date_rejected_safely():
    """5. malformed date -> REJECT / UNPARSEABLE safely."""
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date="not-a-date",
        round_date=None,
        reference_date=FIXED_REF,
    )
    assert is_rec is False
    assert "unparseable" in note
    assert eff == "not-a-date"


def test_6_impossible_calendar_date_safe_invalid_state():
    """6. impossible calendar date -> safe invalid state (no ValueError raised)."""
    for bad_date in ["2026-02-31", "2025-02-29", "2025-13-01", "2025-00-10"]:
        is_rec, note, eff = evaluate_recency_basis(
            financial_type="CUMULATIVE_FUNDING",
            evidence_date=bad_date,
            round_date=None,
            reference_date=FIXED_REF,
        )
        assert is_rec is False
        assert "unparseable" in note
        assert eff == bad_date


def test_7_missing_date_rejected():
    """7. missing date -> REJECT when recency is required."""
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date=None,
        round_date=None,
        reference_date=FIXED_REF,
    )
    assert is_rec is False
    assert "no verifiable date published" in note
    assert eff is None


def test_8_valid_current_evidence_plus_old_round_accepted():
    """8. valid current financial evidence + old round -> preserve existing valid behavior (ACCEPT)."""
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date="2026-05-01",  # Recent confirmation
        round_date="2022-01-01",     # Old round
        reference_date=FIXED_REF,
    )
    assert is_rec is True
    assert "Recent" in note
    assert "Cumulative confirmation evidence (2026-05-01)" in note
    assert eff == "2026-05-01"


def test_9_future_evidence_plus_qualifying_amount_rejected():
    """9. future evidence + qualifying amount -> REJECT."""
    cand = CandidateCompany(
        name="FutureProof AI",
        description="Autonomous agents",
        industry="AI",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=3_000_000,
        financial_evidence_date="2099-01-01",
        latest_round_date=None,
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "RECENCY"
    assert "Future date" in audit.rejection_reason or "in the future" in audit.rejection_reason


def test_10_future_latest_round_cannot_override_valid_historical_evidence():
    """10. future latest round + valid historical evidence -> future date cannot override valid evidence."""
    # Under SINGLE_ROUND_FUNDING: round_date is future 2099-01-01, but evidence_date is recent 2026-05-01
    is_rec, note, eff = evaluate_recency_basis(
        financial_type="SINGLE_ROUND_FUNDING",
        evidence_date="2026-05-01",
        round_date="2099-01-01",
        reference_date=FIXED_REF,
    )
    assert is_rec is True
    assert "Recent" in note
    assert eff == "2026-05-01"
    assert "Financing confirmation evidence (2026-05-01)" in note

    # Under CUMULATIVE_FUNDING: evidence_date is 2026-05-01, round_date is 2099-01-01
    is_rec_cum, note_cum, eff_cum = evaluate_recency_basis(
        financial_type="CUMULATIVE_FUNDING",
        evidence_date="2026-05-01",
        round_date="2099-01-01",
        reference_date=FIXED_REF,
    )
    assert is_rec_cum is True
    assert eff_cum == "2026-05-01"

    # In qualification pipeline: candidate with valid evidence date passes even if round_date is future
    cand = CandidateCompany(
        name="ValidEvidenceCo",
        description="Cloud SaaS",
        industry="SaaS",
        hq_country="UK",
        is_tech_platform=True,
        total_cumulative_funding_usd=3_000_000,
        financial_evidence_date="2026-01-01",
        latest_round_date="2099-01-01",
        financial_source_url="https://techcrunch.com/validevidence",
        financial_source_quote="ValidEvidenceCo raised $3M total funding.",
        trusted_source_urls=["https://techcrunch.com/validevidence"],
        trusted_source_text="ValidEvidenceCo raised $3M total funding.",
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"


# ==============================================================================
# 3. POLICY QUALIFICATION EVALUATION ACROSS ALL POLICIES (A, B, C, D)
# ==============================================================================

def test_policy_a_future_date_rejected():
    cand = CandidateCompany(
        name="PolicyACo",
        description="SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_500_000,
        financial_evidence_date="2099-01-01",
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "RECENCY"


def test_policy_a_malformed_date_rejected_safely():
    cand = CandidateCompany(
        name="PolicyACo",
        description="SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_500_000,
        financial_evidence_date="2026-02-31",
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "RECENCY"
    assert "unparseable" in audit.rejection_reason


def test_policy_b_future_date_rejected():
    cand = CandidateCompany(
        name="PolicyBCo",
        description="SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        latest_round_amount_usd=2_500_000,
        latest_round_date="2099-01-01",
        financial_evidence_date="2099-01-01",
        financial_followup_complete=True,
        subsequent_rounds_discovered=False,
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "RECENCY"


def test_policy_b_malformed_date_rejected_safely():
    cand = CandidateCompany(
        name="PolicyBCo",
        description="SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        latest_round_amount_usd=2_500_000,
        latest_round_date="2025-13-01",
        financial_followup_complete=True,
        subsequent_rounds_discovered=False,
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "RECENCY"
    assert "unparseable" in audit.rejection_reason


def test_policy_c_future_date_rejected():
    cand = CandidateCompany(
        name="PolicyCCo",
        description="SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="Generated $3M in ARR.",
        revenue_date="2099-01-01",
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "RECENCY"


def test_policy_c_malformed_date_rejected_safely():
    cand = CandidateCompany(
        name="PolicyCCo",
        description="SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="Generated $3M in ARR.",
        revenue_date="2026-02-31",
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "RECENCY"
    assert "unparseable" in audit.rejection_reason


def test_policy_d_scale_conflict_unaltered():
    """Policy D (excessive funding disqualifying ARR) remains strictly unaltered."""
    cand = CandidateCompany(
        name="PolicyDCo",
        description="SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="Generated $3M in ARR.",
        financial_source_url="https://saas-news.com/policyd",
        trusted_source_urls=["https://saas-news.com/policyd"],
        trusted_source_text="Generated $3M in ARR.",
        total_cumulative_funding_usd=25_000_000,
        revenue_date="2026-01-01",
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "Policy D" in audit.rejection_reason


# ==============================================================================
# 4. SECTION 9 PIPELINE CRASH-SAFETY TESTS
# ==============================================================================

def test_pipeline_crash_safety_malformed_feb31():
    """Candidate with 2026-02-31 does NOT crash the pipeline; rejected gracefully."""
    cand = CandidateCompany(
        name="Feb31Startup",
        description="Data platform",
        industry="Data",
        hq_country="Germany",
        hq_city="Munich",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_500_000.0,
        financial_evidence_date="2026-02-31",
        founder_name="Karl Schmidt",
        founder_title="CEO",
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Karl Schmidt",
                    founder_title_found="CEO",
                    contact_path_found=True,
                    pages_checked=["https://feb31.eu/about"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kw: c):
                        summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert summary.funnel.passed_financial == 0
    assert len(summary.leads) == 0


def test_pipeline_crash_safety_malformed_month13():
    """Candidate with 2025-13-01 does NOT crash the pipeline; rejected gracefully."""
    cand = CandidateCompany(
        name="Month13Startup",
        description="Cloud orchestration",
        industry="Cloud",
        hq_country="France",
        hq_city="Paris",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_000_000.0,
        financial_evidence_date="2025-13-01",
        founder_name="Pierre Simon",
        founder_title="CEO",
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Pierre Simon",
                    founder_title_found="CEO",
                    contact_path_found=True,
                    pages_checked=["https://m13.eu/about"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kw: c):
                        summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert summary.funnel.passed_financial == 0
    assert len(summary.leads) == 0


def test_pipeline_crash_safety_future_year2099():
    """Candidate with 2099-01-01 does NOT crash and is rejected at financial/recency gate."""
    cand = CandidateCompany(
        name="Year2099Startup",
        description="Quantum SaaS",
        industry="Quantum",
        hq_country="Estonia",
        hq_city="Tallinn",
        is_tech_platform=True,
        total_cumulative_funding_usd=3_000_000.0,
        financial_evidence_date="2099-01-01",
        founder_name="Jaan Tamm",
        founder_title="CEO",
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Jaan Tamm",
                    founder_title_found="CEO",
                    contact_path_found=True,
                    pages_checked=["https://y2099.ee/about"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kw: c):
                        summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert summary.funnel.passed_financial == 0
    assert len(summary.leads) == 0
