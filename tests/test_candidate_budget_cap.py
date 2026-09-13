"""
tests/test_candidate_budget_cap.py
Comprehensive test suite verifying deterministic enforcement of MAX_CANDIDATES_PER_RUN.

Verifies:
- Deduplication precedes candidate cap
- Candidate cap limits unique candidates entering downstream processing
- Beyond-cap candidates are audited with rejection_stage="CANDIDATE_BUDGET"
- Zero Tavily credits consumed for skipped candidates
- Multi-query discovery respects global cap and terminates early
- target_count does not override candidate cap
- Edge cases: MAX=0, MAX=1, MAX=None, negative raises ValueError
- End-to-end scenario: 20 raw -> 15 unique -> 10 processed + 5 skipped
"""

from datetime import date, timedelta
import pytest
from unittest.mock import patch, MagicMock
from core.models import (
    CandidateCompany,
    ContactPathResult,
    PipelineSummary,
    FunnelMetrics,
    LeadRecord,
    CandidateAuditRecord
)
from core.pipeline import run_lead_pipeline

RECENT_DATE = (date.today() - timedelta(days=60)).strftime("%Y-%m-%d")


def _create_mock_candidate(name: str, hq_country: str = "Germany", hq_city: str = "Berlin", funding: float = 2_000_000.0) -> CandidateCompany:
    return CandidateCompany(
        name=name,
        website_url=f"https://{name.lower().replace(' ', '')}.com",
        description="B2B SaaS platform",
        industry="Software",
        hq_country=hq_country,
        hq_city=hq_city,
        is_tech_platform=True,
        total_cumulative_funding_usd=funding,
        founder_name=f"Founder {name}",
        founder_title="CEO",
        revenue_amount_usd=2_000_000.0,
        revenue_type="ARR",
        revenue_period="2026",
        financial_evidence_date=RECENT_DATE,
        latest_round_date=RECENT_DATE,
        financial_followup_complete=True,
        financial_source_url="https://techcrunch.com/seed",
        financial_source_quote="Raised $2M in seed funding.",
        trusted_source_urls=["https://techcrunch.com/seed"],
        trusted_source_text="Raised $2M in seed funding."
    )


# ==============================================================================
# Case A: 100 raw candidates with MAX=50 -> at most 50 unique candidates processed
# ==============================================================================

def test_case_a_100_raw_candidates_max_50():
    candidates = [_create_mock_candidate(f"Company {i}") for i in range(100)]

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):
        
        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=50
        )

    assert summary.funnel.raw_candidates_found == 100
    assert summary.funnel.unique_candidates_found == 100
    assert summary.funnel.candidates_processed == 50
    assert summary.candidates_processed == 50
    assert summary.funnel.candidates_skipped_candidate_budget == 50
    assert summary.candidates_skipped_candidate_budget == 50
    assert summary.exit_reason == "MAX_CANDIDATES_PER_RUN reached (50)"

    budget_skipped = [rec for rec in summary.audit_log if rec.rejection_stage == "CANDIDATE_BUDGET"]
    assert len(budget_skipped) == 50


# ==============================================================================
# Case B: Deduplication precedes cap
# ==============================================================================

def test_case_b_deduplication_precedes_cap():
    # 50 unique candidates, each duplicated once = 100 raw candidates
    unique_candidates = [_create_mock_candidate(f"UniqueCompany {i}") for i in range(50)]
    raw_candidates = unique_candidates + unique_candidates

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=raw_candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=50
        )

    # All 50 unique candidates fit within cap of 50
    assert summary.funnel.raw_candidates_found == 100
    assert summary.funnel.unique_candidates_found == 50
    assert summary.funnel.candidates_processed == 50
    assert summary.funnel.candidates_skipped_candidate_budget == 0


def test_case_b_deduplication_with_excess_unique():
    # 40 unique candidates duplicated = 80 raw candidates. Cap = 30.
    unique_candidates = [_create_mock_candidate(f"Company {i}") for i in range(40)]
    raw_candidates = unique_candidates + unique_candidates

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=raw_candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=30
        )

    assert summary.funnel.raw_candidates_found == 80
    assert summary.funnel.unique_candidates_found == 40
    assert summary.funnel.candidates_processed == 30
    assert summary.funnel.candidates_skipped_candidate_budget == 10


# ==============================================================================
# Case C: Exactly N unique candidates -> all N processed
# ==============================================================================

def test_case_c_exactly_n_unique_candidates():
    candidates = [_create_mock_candidate(f"Company {i}") for i in range(10)]

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=10
        )

    assert summary.funnel.candidates_processed == 10
    assert summary.funnel.candidates_skipped_candidate_budget == 0
    budget_skips = [r for r in summary.audit_log if r.rejection_stage == "CANDIDATE_BUDGET"]
    assert len(budget_skips) == 0


# ==============================================================================
# Case D: N+1 unique candidates -> final one skipped by candidate budget
# ==============================================================================

def test_case_d_n_plus_one_unique_candidates():
    candidates = [_create_mock_candidate(f"Company {i}") for i in range(11)]

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=10
        )

    assert summary.funnel.unique_candidates_found == 11
    assert summary.funnel.candidates_processed == 10
    assert summary.funnel.candidates_skipped_candidate_budget == 1

    budget_skips = [r for r in summary.audit_log if r.rejection_stage == "CANDIDATE_BUDGET"]
    assert len(budget_skips) == 1
    assert budget_skips[0].company_name == "Company 10"


# ==============================================================================
# Case E: Audit record format for candidate budget rejection
# ==============================================================================

def test_case_e_candidate_budget_audit_record_format():
    candidates = [_create_mock_candidate(f"Corp {i}") for i in range(3)]

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=2
        )

    budget_skips = [r for r in summary.audit_log if r.rejection_stage == "CANDIDATE_BUDGET"]
    assert len(budget_skips) == 1
    rec = budget_skips[0]

    assert rec.company_name == "Corp 2"
    assert rec.qualification_status == "REJECTED"
    assert rec.rejection_stage == "CANDIDATE_BUDGET"
    assert "MAX_CANDIDATES_PER_RUN (2) was reached" in rec.rejection_reason
    assert "MAX_CANDIDATES_PER_RUN was reached (2)" in rec.audit_summary


# ==============================================================================
# Case F: Candidate budget skip makes 0 additional Tavily calls
# ==============================================================================

def test_case_f_skipped_candidates_consume_zero_tavily_calls():
    candidates = [_create_mock_candidate(f"Firm {i}") for i in range(5)]

    mock_followup = MagicMock(side_effect=lambda cl, c, **kw: c)

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)) as mock_contact, \
         patch("core.pipeline.execute_targeted_funding_follow_up", mock_followup), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=2
        )

    # Candidates 0 and 1 entered downstream processing
    assert mock_contact.call_count <= 2
    assert summary.funnel.candidates_processed == 2
    assert summary.funnel.candidates_skipped_candidate_budget == 3
    # Skipped candidates Firm 2, Firm 3, Firm 4 never had follow-up called
    called_names = [call.args[1].name for call in mock_followup.call_args_list]
    for skipped in ["Firm 2", "Firm 3", "Firm 4"]:
        assert skipped not in called_names


# ==============================================================================
# Case G: Candidate count across multiple discovery queries respects global run cap
# ==============================================================================

def test_case_g_candidate_cap_across_multiple_queries():
    query1_cands = [_create_mock_candidate(f"Q1 Company {i}") for i in range(6)]
    query2_cands = [_create_mock_candidate(f"Q2 Company {i}") for i in range(8)]
    query3_cands = [_create_mock_candidate(f"Q3 Company {i}") for i in range(10)]

    mock_extract = MagicMock(side_effect=[query1_cands, query2_cands, query3_cands])

    with patch("core.pipeline.generate_queries", return_value=["query 1", "query 2", "query 3"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", mock_extract), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=3,
            max_candidates=10
        )

    # Query 1 gave 6 candidates (processed: 6)
    # Query 2 gave 8 candidates (processed: 4 to reach 10; skipped: 4)
    # Total processed reached 10 -> pipeline terminated before Query 3
    assert summary.total_queries_run == 2
    assert mock_extract.call_count == 2
    assert summary.funnel.candidates_processed == 10
    assert summary.funnel.candidates_skipped_candidate_budget == 4
    assert summary.exit_reason == "MAX_CANDIDATES_PER_RUN reached (10)"


# ==============================================================================
# Case H: target_count does not override candidate cap
# ==============================================================================

def test_case_h_target_count_does_not_override_candidate_cap():
    cands = [_create_mock_candidate(f"TargetTest {i}") for i in range(10)]

    with patch("core.pipeline.generate_queries", return_value=["query 1", "query 2"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=cands), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        # target_count=15 requested, but max_candidates=5
        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=15,
            max_queries=2,
            max_candidates=5
        )

    assert summary.funnel.candidates_processed == 5
    assert summary.funnel.candidates_skipped_candidate_budget == 5
    assert summary.exit_reason == "MAX_CANDIDATES_PER_RUN reached (5)"


# ==============================================================================
# Case I: Financial qualification semantics unaffected
# ==============================================================================

def test_case_i_financial_qualification_semantics_unaffected():
    # Low funding candidate within cap must still fail financial evaluation
    low_funding_cand = _create_mock_candidate("LowFundingCo", funding=100_000.0)
    low_funding_cand.revenue_amount_usd = 100_000.0

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=[low_funding_cand]), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=5,
            max_queries=1,
            max_candidates=10
        )

    assert summary.funnel.candidates_processed == 1
    assert summary.funnel.candidates_skipped_candidate_budget == 0
    assert len(summary.leads) == 0
    assert summary.audit_log[0].rejection_stage == "FINANCIAL"


# ==============================================================================
# Case J: Email verification semantics unaffected
# ==============================================================================

def test_case_j_email_verification_semantics_unaffected():
    # Candidate passes financial, tech, non-US; verified email produces LeadRecord
    cand = _create_mock_candidate("SuccessCo")

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=[cand]), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
             contact_path_found=True,
             founder_name_found="Founder SuccessCo",
             founder_title_found="CEO",
             exact_email_found="alex@successco.com",
             email_source_url="https://successco.com/team"
         )), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=("alex@successco.com", "https://successco.com/team")):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=5,
            max_queries=1,
            max_candidates=10
        )

    assert summary.funnel.candidates_processed == 1
    assert len(summary.leads) == 1
    assert summary.leads[0].company_name == "SuccessCo"
    assert summary.leads[0].verified_email == "alex@successco.com"


# ==============================================================================
# Edge Cases: MAX=0, MAX=1, MAX=None, Negative Value
# ==============================================================================

def test_edge_case_max_0():
    candidates = [_create_mock_candidate(f"ZeroCap {i}") for i in range(5)]

    with patch("core.pipeline.generate_queries", return_value=["query 1", "query 2"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=5,
            max_queries=2,
            max_candidates=0
        )

    assert summary.total_queries_run == 1
    assert summary.funnel.unique_candidates_found == 5
    assert summary.funnel.candidates_processed == 0
    assert summary.funnel.candidates_skipped_candidate_budget == 5
    assert summary.exit_reason == "MAX_CANDIDATES_PER_RUN reached (0)"

    budget_skips = [r for r in summary.audit_log if r.rejection_stage == "CANDIDATE_BUDGET"]
    assert len(budget_skips) == 5


def test_edge_case_max_1():
    candidates = [_create_mock_candidate(f"SingleCap {i}") for i in range(5)]

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=5,
            max_queries=1,
            max_candidates=1
        )

    assert summary.funnel.unique_candidates_found == 5
    assert summary.funnel.candidates_processed == 1
    assert summary.funnel.candidates_skipped_candidate_budget == 4
    assert summary.exit_reason == "MAX_CANDIDATES_PER_RUN reached (1)"


def test_edge_case_max_none_disabled():
    candidates = [_create_mock_candidate(f"NoneCap {i}") for i in range(10)]

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=candidates), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)), \
         patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda cl, c, **kw: c), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=100,
            max_queries=1,
            max_candidates=None
        )

    # When None, MAX_CANDIDATES_PER_RUN defaults to config (150). With 10 candidates, all 10 are processed.
    assert summary.funnel.candidates_processed == 10
    assert summary.funnel.candidates_skipped_candidate_budget == 0


def test_edge_case_negative_max_raises_value_error():
    with pytest.raises(ValueError, match="Invalid MAX_CANDIDATES_PER_RUN: -1"):
        run_lead_pipeline(api_key="mock_key", max_candidates=-1)

    with pytest.raises(ValueError, match="Invalid MAX_CANDIDATES_PER_RUN: -100"):
        run_lead_pipeline(api_key="mock_key", max_candidates=-100)


# ==============================================================================
# Test 10: End-to-End Scenario (20 raw, 15 unique, MAX=10)
# ==============================================================================

def test_10_end_to_end_scenario():
    """
    Mocked scenario:
    - 20 raw candidates discovered
    - 5 are duplicates (15 unique)
    - MAX_CANDIDATES_PER_RUN = 10
    - Exactly 10 unique candidates enter downstream processing
    - Exactly 5 unique candidates skipped by candidate budget
    - 0 extra Tavily calls made for skipped candidates
    """
    unique_cands = [_create_mock_candidate(f"Scenario Company {i}") for i in range(15)]
    # Duplicate first 5 to create 20 raw candidates
    raw_cands = unique_cands + unique_cands[:5]
    assert len(raw_cands) == 20

    mock_followup = MagicMock(side_effect=lambda cl, c, **kw: c)

    with patch("core.pipeline.generate_queries", return_value=["query 1"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])), \
         patch("core.pipeline.extract_structured_candidates", return_value=raw_cands), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=False)) as mock_contact, \
         patch("core.pipeline.execute_targeted_funding_follow_up", mock_followup), \
         patch("core.pipeline.verify_founder_email_on_site", return_value=False):

        summary = run_lead_pipeline(
            api_key="mock_key",
            target_count=20,
            max_queries=1,
            max_candidates=10
        )

    # Telemetry assertions
    assert summary.funnel.raw_candidates_found == 20
    assert summary.funnel.unique_candidates_found == 15
    assert summary.funnel.candidates_processed == 10
    assert summary.candidates_processed == 10
    assert summary.funnel.candidates_skipped_candidate_budget == 5
    assert summary.candidates_skipped_candidate_budget == 5

    # Exit reason
    assert summary.exit_reason == "MAX_CANDIDATES_PER_RUN reached (10)"

    # Audit log verification
    budget_skips = [rec for rec in summary.audit_log if rec.rejection_stage == "CANDIDATE_BUDGET"]
    assert len(budget_skips) == 5
    for skip in budget_skips:
        assert skip.qualification_status == "REJECTED"
        assert "MAX_CANDIDATES_PER_RUN (10) was reached" in skip.rejection_reason

    # Zero Tavily follow-up calls for skipped candidates
    assert mock_contact.call_count <= 10
    called_names = [call.args[1].name for call in mock_followup.call_args_list]
    for i in range(10, 15):
        assert f"Scenario Company {i}" not in called_names
