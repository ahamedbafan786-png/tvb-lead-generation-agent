"""
Adversarial Verification Suite: Policy B Follow-Up Verification Safety
======================================================================
Tests all requirements for gating Policy B (Single Round Fallback) on
strictly verified completion of the targeted financial follow-up.

Test Cases Covered:
A. $3M single round + successful follow-up + no subsequent rounds -> Policy B may qualify
B. $3M single round + successful follow-up + subsequent round found -> REJECT
C. $3M single round + timeout -> NOT QUALIFIED under Policy B
D. $3M single round + HTTP 429 -> NOT QUALIFIED under Policy B
E. $3M single round + HTTP 403/quota error -> NOT QUALIFIED under Policy B
F. $3M single round + malformed response -> NOT QUALIFIED under Policy B
G. $3M single round + missing follow-up result -> NOT QUALIFIED under Policy B
H. $3M single round + unknown subsequent-round state -> NOT QUALIFIED under Policy B
I. $3M single round + valid cumulative funding independently confirms $3M -> Policy A may qualify
J. ARR $3M with independent valid evidence -> Policy C may qualify even if unrelated follow-up fails
K. TavilyBudgetExhaustedError must be re-raised (not swallowed)
L. Pipeline integration: Policy B candidate rejected when follow-up search fails
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.discovery import (
    TavilyAuthError,
    TavilyBudgetExhaustedError,
    TavilyQuotaError,
    TavilyRateLimitError,
    TavilyTimeoutError,
)
from core.extractor import (
    evaluate_financial_qualification,
    execute_targeted_funding_follow_up,
)
from core.models import CandidateCompany


def _make_single_round_candidate(
    name: str = "TestTarget AI",
    amount: float = 3_000_000.0,
    round_name: str = "Seed",
    round_date: str = "2026-01-01",
    evidence_date: str = "2026-01-01",
) -> CandidateCompany:
    """Helper to create a standard candidate with single-round funding only."""
    return CandidateCompany(
        name=name,
        description="Enterprise automation platform",
        industry="Enterprise Software",
        hq_country="Germany",
        hq_city="Berlin",
        is_tech_platform=True,
        latest_round_amount_usd=amount,
        latest_round_name=round_name,
        latest_round_date=round_date,
        financial_evidence_date=evidence_date,
        total_cumulative_funding_usd=None,
        revenue_amount_usd=None,
        founder_name="Max Mustermann",
        founder_title="CEO",
    )


# ==============================================================================
# CASE A: $3M single round + successful follow-up + no subsequent rounds -> QUALIFIES
# ==============================================================================

def test_case_a_single_round_successful_followup_no_subsequent_rounds_qualifies():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "subsequent_rounds_discovered": False,
        "acquisition_or_closure_discovered": False,
        "latest_round_amount_usd": 3000000.0,
        "latest_round_date": "2026-01-01",
        "financial_evidence_date": "2026-01-01",
        "evidence_snippet": "Raised $3M Seed",
        "financial_source_url": "https://news.eu/seed",
    })
    mock_client.models.generate_content.return_value = mock_resp

    with patch("core.discovery.search_tavily", return_value=[{"title": "Seed round", "url": "https://news.eu/seed", "content": "Raised $3M Seed"}]):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is True
    assert updated.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "SINGLE_ROUND_FUNDING"
    assert audit.confidence_rating == "MEDIUM"
    assert "Policy B" in audit.audit_summary


# ==============================================================================
# CASE B: $3M single round + successful follow-up + subsequent round found -> REJECT
# ==============================================================================

def test_case_b_single_round_successful_followup_subsequent_round_found_rejects():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "subsequent_rounds_discovered": True,
        "acquisition_or_closure_discovered": False,
        "latest_round_name": "Series B",
        "evidence_snippet": "Company later raised Series A and Series B totaling $25M.",
    })
    mock_client.models.generate_content.return_value = mock_resp

    with patch("core.discovery.search_tavily", return_value=[{"title": "Series B", "url": "https://news.eu/b", "content": "Later raised Series B"}]):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is True
    assert updated.subsequent_rounds_discovered is True

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "subsequent rounds" in audit.rejection_reason


# ==============================================================================
# CASE C: $3M single round + timeout -> NOT QUALIFIED under Policy B
# ==============================================================================

def test_case_c_single_round_timeout_not_qualified():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()

    with patch("core.discovery.search_tavily", side_effect=TavilyTimeoutError("Request timed out after 10s")):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is False
    assert updated.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason
    assert "Cannot confirm absence of subsequent rounds" in audit.rejection_reason


# ==============================================================================
# CASE D: $3M single round + HTTP 429 -> NOT QUALIFIED under Policy B
# ==============================================================================

def test_case_d_single_round_http_429_not_qualified():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()

    with patch("core.discovery.search_tavily", side_effect=TavilyRateLimitError("HTTP 429: Rate limit exceeded")):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is False
    assert updated.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason


# ==============================================================================
# CASE E: $3M single round + HTTP 403 / plan quota error -> NOT QUALIFIED under Policy B
# ==============================================================================

def test_case_e1_single_round_http_403_not_qualified():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()

    with patch("core.discovery.search_tavily", side_effect=TavilyAuthError("HTTP 403: Forbidden")):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is False
    assert updated.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason


def test_case_e2_single_round_plan_quota_exhausted_not_qualified():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()

    with patch("core.discovery.search_tavily", side_effect=TavilyQuotaError("HTTP 432: Monthly plan quota exceeded")):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is False
    assert updated.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason


# ==============================================================================
# CASE F: $3M single round + malformed response -> NOT QUALIFIED under Policy B
# ==============================================================================

def test_case_f1_single_round_malformed_json_not_qualified():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "NOT VALID JSON {broken: 123"
    mock_client.models.generate_content.return_value = mock_resp

    with patch("core.discovery.search_tavily", return_value=[{"title": "News", "url": "https://news.eu", "content": "Some text"}]):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is False
    assert updated.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason


def test_case_f2_single_round_non_dict_json_not_qualified():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps("A string instead of an object")
    mock_client.models.generate_content.return_value = mock_resp

    with patch("core.discovery.search_tavily", return_value=[{"title": "News", "url": "https://news.eu", "content": "Some text"}]):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is False
    assert updated.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason


# ==============================================================================
# CASE G: $3M single round + missing follow-up result -> NOT QUALIFIED under Policy B
# ==============================================================================

def test_case_g_single_round_missing_followup_result_not_qualified():
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()

    # Tavily returns empty results (0 search hits)
    with patch("core.discovery.search_tavily", return_value=[]):
        updated = execute_targeted_funding_follow_up(
            client=mock_client,
            candidate=candidate,
            tavily_api_key="mock_key"
        )

    assert updated.financial_followup_complete is False
    assert updated.subsequent_rounds_discovered is False
    # LLM should not even have been called because sources_text was empty
    mock_client.models.generate_content.assert_not_called()

    qualified, audit = evaluate_financial_qualification(updated)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason


# ==============================================================================
# CASE H: $3M single round + unknown subsequent-round state -> NOT QUALIFIED under Policy B
# ==============================================================================

def test_case_h_single_round_unknown_state_direct_evaluation_not_qualified():
    """Candidate created fresh with default financial_followup_complete=False must NOT qualify under Policy B."""
    candidate = _make_single_round_candidate()
    assert candidate.financial_followup_complete is False
    assert candidate.subsequent_rounds_discovered is False

    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "financial follow-up verification is incomplete" in audit.rejection_reason


# ==============================================================================
# CASE I: $3M single round + valid cumulative funding -> QUALIFIES under Policy A
# ==============================================================================

def test_case_i_single_round_with_valid_cumulative_funding_qualifies_policy_a():
    """Candidate with valid cumulative funding qualifies under Policy A even if follow-up was never completed."""
    candidate = _make_single_round_candidate()
    candidate.total_cumulative_funding_usd = 3_000_000.0
    candidate.financial_evidence_date = "2026-01-01"
    candidate.financial_source_url = "https://techcrunch.com/seed-raise"
    candidate.financial_source_quote = "Company secured $3M in total funding."
    candidate.trusted_source_urls = ["https://techcrunch.com/seed-raise"]
    candidate.trusted_source_text = "Company secured $3M in total funding."
    assert candidate.financial_followup_complete is False

    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "CUMULATIVE_FUNDING"
    assert audit.confidence_rating == "HIGH"
    assert "Policy A" in audit.audit_summary


# ==============================================================================
# CASE J: ARR $3M with independent valid evidence -> QUALIFIES under Policy C
# ==============================================================================

def test_case_j_arr_3m_with_independent_evidence_qualifies_policy_c():
    """Candidate with valid ARR qualifies under Policy C even if follow-up failed/incomplete."""
    candidate = _make_single_round_candidate()
    candidate.revenue_amount_usd = 3_000_000.0
    candidate.revenue_period = "ARR"
    candidate.financial_source_url = "https://saas-news.com/revenue"
    candidate.financial_source_quote = "Generating $3M ARR as of January 2026."
    candidate.revenue_date = "2026-01-01"
    candidate.trusted_source_urls = ["https://saas-news.com/revenue"]
    candidate.trusted_source_text = "Generating $3M ARR as of January 2026."
    assert candidate.financial_followup_complete is False

    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "ARR"
    assert audit.confidence_rating == "HIGH"
    assert "Policy C" in audit.audit_summary


# ==============================================================================
# CASE K: TavilyBudgetExhaustedError must be re-raised (not swallowed)
# ==============================================================================

def test_case_k_tavily_budget_exhausted_error_reraised():
    """TavilyBudgetExhaustedError must propagate to the caller so pipeline can terminate gracefully."""
    candidate = _make_single_round_candidate()
    mock_client = MagicMock()

    with patch("core.discovery.search_tavily", side_effect=TavilyBudgetExhaustedError("Hard budget 60 reached")):
        with pytest.raises(TavilyBudgetExhaustedError):
            execute_targeted_funding_follow_up(
                client=mock_client,
                candidate=candidate,
                tavily_api_key="mock_key"
            )


# ==============================================================================
# CASE L: Pipeline Integration: Failed follow-up on Policy B candidate does not qualify
# ==============================================================================

def test_case_l_pipeline_rejects_candidate_when_followup_fails_on_policy_b():
    """Integration check: in the pipeline, if follow-up search fails on a Policy B candidate, the candidate is rejected."""
    from core.pipeline import run_lead_pipeline
    from core.models import ContactPathResult

    cand = _make_single_round_candidate(name="UnconfirmedStartup")

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Max Mustermann",
                    founder_title_found="CEO",
                    contact_path_found=True,
                    pages_checked=["https://unconfirmed.eu/about"]
                )):
                    # Tavily follow-up times out
                    with patch("core.discovery.search_tavily", side_effect=TavilyTimeoutError("Timeout")):
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=("max@unconfirmed.eu", "https://unconfirmed.eu/about")):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    # Candidate was NOT qualified due to incomplete follow-up
    assert summary.funnel.passed_financial == 0
    assert len(summary.leads) == 0
