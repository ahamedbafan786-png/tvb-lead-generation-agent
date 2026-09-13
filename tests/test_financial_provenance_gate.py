"""
tests/test_financial_provenance_gate.py
Comprehensive test suite verifying the financial provenance gate:
- 15 adversarial tests ensuring that source-free or hallucinated financial claims fail-closed
- 4 end-to-end pipeline integration scenarios verifying provenance enforcement across discovery and follow-up
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from core.models import (
    CandidateCompany,
    EvidenceItem,
    ContactPathResult,
)
from core.extractor import (
    evaluate_financial_qualification,
    verify_financial_provenance,
    extract_financial_amounts_from_text,
    is_amount_corroborated_by_text,
    extract_structured_candidates,
    execute_targeted_funding_follow_up,
)
from core.pipeline import run_lead_pipeline


def _create_base_candidate(name: str = "TestCorp") -> CandidateCompany:
    """Helper to create a standard valid EU tech candidate."""
    return CandidateCompany(
        name=name,
        website_url="https://testcorp.eu",
        description="B2B AI workflow automation platform",
        industry="Enterprise Software",
        hq_country="Germany",
        hq_city="Berlin",
        is_tech_platform=True,
        founder_name="Elena Weber",
        founder_title="CEO",
    )


# ==============================================================================
# PART 1: 15 ADVERSARIAL PROVENANCE UNIT TESTS
# ==============================================================================

def test_01_source_free_cumulative_funding_rejected():
    """1. Source-free $3M cumulative funding must fail-closed at FINANCIAL stage."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = None
    cand.financial_source_quote = None

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"
    assert "provenance" in audit.rejection_reason.lower() or "source" in audit.rejection_reason.lower()


def test_02_source_free_arr_rejected():
    """2. Source-free $3M ARR must fail-closed at FINANCIAL stage."""
    cand = _create_base_candidate()
    cand.revenue_amount_usd = 3_000_000.0
    cand.revenue_period = "ARR"
    cand.revenue_date = "2026-01-01"
    cand.financial_source_url = None
    cand.financial_source_quote = None

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"


def test_03_fake_source_url_not_in_trusted_sources_rejected():
    """3. Candidate with source URL not present in trusted Tavily sources must be rejected."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://fabricated-financial-news.org/funding"
    cand.financial_source_quote = "TestCorp raised $3M in total funding."
    cand.trusted_source_urls = ["https://techcrunch.com/realsource", "https://reuters.com/realsource2"]
    cand.trusted_source_text = "TechCrunch: TestCorp raised funding."

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"
    assert "trusted" in audit.rejection_reason.lower() or "authenticated" in audit.rejection_reason.lower()


def test_04_real_source_url_missing_quote_rejected():
    """4. Source URL in trusted sources but missing source quote must fail-closed."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://techcrunch.com/2026/testcorp-funding"
    cand.financial_source_quote = None
    cand.trusted_source_urls = ["https://techcrunch.com/2026/testcorp-funding"]
    cand.trusted_source_text = "TestCorp raised $3M in total funding."

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"
    assert "quote" in audit.rejection_reason.lower()


def test_05_real_source_url_matching_quote_qualifies():
    """5. Authenticated source URL and matching quote in trusted text must qualify."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://techcrunch.com/2026/testcorp-funding"
    cand.financial_source_quote = "TestCorp raised $3M in total funding."
    cand.trusted_source_urls = ["https://techcrunch.com/2026/testcorp-funding"]
    cand.trusted_source_text = "TestCorp raised $3M in total funding."

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "CUMULATIVE_FUNDING"
    assert audit.confidence_rating == "HIGH"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "VERIFIED"


def test_06_model_url_not_in_trusted_search_results_rejected():
    """6. Model-invented URL absent from discovery search results must be rejected."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 4_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://nonexistent-tavily-result.com/article"
    cand.financial_source_quote = "Raised $4M"
    cand.trusted_source_urls = ["https://actual-source-1.com", "https://actual-source-2.com"]

    is_valid, reason, _, _ = verify_financial_provenance(cand, 4_000_000.0, "CUMULATIVE_FUNDING")
    assert is_valid is False
    assert "trusted search results" in reason.lower()


def test_07_model_amount_contradicts_source_amount_rejected():
    """7. Contradiction: Source quote says $1.5M, but candidate claims $3M -> REJECT."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://techcrunch.com/2026/testcorp"
    cand.financial_source_quote = "TestCorp announced a $1.5M pre-seed round yesterday."
    cand.trusted_source_urls = ["https://techcrunch.com/2026/testcorp"]
    cand.trusted_source_text = "TestCorp announced a $1.5M pre-seed round yesterday."

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"
    assert "contradiction" in audit.rejection_reason.lower() or "corroborate" in audit.rejection_reason.lower()


def test_08_source_backed_cumulative_funding_policy_a_qualifies():
    """8. Source-backed cumulative funding in [$2M, $5M] qualifies under Policy A."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 2_500_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://techcrunch.com/seed-round"
    cand.financial_source_quote = "Company secured $2.5M cumulative financing."
    cand.trusted_source_urls = ["https://techcrunch.com/seed-round"]
    cand.trusted_source_text = "Company secured $2.5M cumulative financing."

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "CUMULATIVE_FUNDING"
    assert "Policy A" in audit.audit_summary


def test_09_source_backed_arr_policy_c_qualifies():
    """9. Source-backed ARR in [$2M, $5M] qualifies under Policy C."""
    cand = _create_base_candidate()
    cand.revenue_amount_usd = 2_500_000.0
    cand.revenue_period = "ARR"
    cand.revenue_date = "2026-01-01"
    cand.financial_source_url = "https://saas-metrics.eu/testcorp"
    cand.financial_source_quote = "TestCorp exceeded $2.5M in ARR."
    cand.trusted_source_urls = ["https://saas-metrics.eu/testcorp"]
    cand.trusted_source_text = "TestCorp exceeded $2.5M in ARR."

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "ARR"
    assert "Policy C" in audit.audit_summary


def test_10_source_backed_policy_b_qualifies():
    """10. Source-backed single-round funding with successful follow-up qualifies under Policy B."""
    cand = _create_base_candidate()
    cand.latest_round_amount_usd = 3_000_000.0
    cand.latest_round_name = "Seed"
    cand.latest_round_date = "2026-01-01"
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://news.eu/seed-report"
    cand.financial_source_quote = "Raised $3M Seed round."
    cand.trusted_source_urls = ["https://news.eu/seed-report"]
    cand.trusted_source_text = "Raised $3M Seed round."
    cand.financial_followup_complete = True
    cand.subsequent_rounds_discovered = False

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "SINGLE_ROUND_FUNDING"
    assert "Policy B" in audit.audit_summary


def test_11_unverified_financial_evidence_cannot_trigger_policy_d():
    """11. Unverified ARR cannot trigger Policy D scale conflict."""
    cand = _create_base_candidate()
    cand.revenue_amount_usd = 3_000_000.0
    cand.revenue_period = "ARR"
    cand.total_cumulative_funding_usd = 25_000_000.0
    cand.revenue_date = "2026-01-01"
    # No source URL or quote
    cand.financial_source_url = None
    cand.financial_source_quote = None

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    # Unverified ARR must not trigger Policy D scale conflict rejection
    assert "Policy D" not in audit.rejection_reason


def test_12_malformed_source_data_safe_rejection():
    """12. Malformed source URL (e.g. localhost/private IP) is safely rejected without crash."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "http://127.0.0.1:8080/exploit"
    cand.financial_source_quote = "TestCorp raised $3M."
    cand.trusted_source_urls = ["http://127.0.0.1:8080/exploit"]
    cand.trusted_source_text = "TestCorp raised $3M."

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"


def test_13_missing_financial_evidence_status_fails_closed():
    """13. Candidate with missing financial fields entirely fails closed."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = None
    cand.latest_round_amount_usd = None
    cand.revenue_amount_usd = None

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"


def test_14_evidence_marked_verified_without_trusted_source_forced_unverified():
    """14. If evidence is superficially marked VERIFIED without trusted source, audit forces UNVERIFIED."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://untrusted-source.org/claim"
    cand.financial_source_quote = "Raised $3M."
    cand.trusted_source_urls = ["https://some-other-source.com"]
    cand.supporting_evidence = [
        EvidenceItem(
            evidence_type="FINANCIAL_SOURCE",
            source_url="https://untrusted-source.org/claim",
            source_quote="Raised $3M.",
            verification_status="VERIFIED",
        )
    ]

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"


def test_15_fake_citation_combined_with_valid_amount_rejected():
    """15. Candidate with valid range amount but fabricated source citation fails closed."""
    cand = _create_base_candidate()
    cand.total_cumulative_funding_usd = 4_500_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = "https://hallucinated-citation.net/press-release"
    cand.financial_source_quote = "Secured 4.5M USD in financing."
    cand.trusted_source_urls = []
    cand.trusted_source_text = None

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.evidence is not None
    assert audit.evidence.financial.verification_status == "UNVERIFIED"


# ==============================================================================
# PART 2: 4 END-TO-END PIPELINE INTEGRATION SCENARIOS
# ==============================================================================

def test_e2e_scenario_a_grounded_discovery_surfaces_source_qualifies():
    """Scenario A: Grounded discovery surfaces source with $3M cumulative funding -> Qualifies E2E."""
    source_url = "https://techcrunch.com/2026/alphatech-seed"
    source_quote = "AlphaTech raised $3M in cumulative funding to date."
    trusted_sources = [
        {
            "title": "AlphaTech funding",
            "url": source_url,
            "content": source_quote,
        }
    ]

    cand = _create_base_candidate(name="AlphaTech")
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = source_url
    cand.financial_source_quote = source_quote
    cand.trusted_source_urls = [source_url]
    cand.trusted_source_text = f"Title: AlphaTech funding\nURL: {source_url}\nContent: {source_quote}"

    with patch("core.pipeline.genai.Client"):
        with patch("core.pipeline.generate_queries", return_value=["test query"]):
            with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", trusted_sources)):
                with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                    with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                        founder_name_found="Elena Weber",
                        founder_title_found="CEO",
                        contact_path_found=True,
                        pages_checked=["https://testcorp.eu/about"]
                    )):
                        with patch("core.pipeline.execute_targeted_funding_follow_up", return_value=cand):
                            with patch("core.pipeline.verify_founder_email_on_site", return_value=("elena@testcorp.eu", "https://testcorp.eu/about")):
                                summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert summary.funnel.passed_financial == 1
    assert len(summary.leads) == 1
    lead = summary.leads[0]
    assert lead.company_name == "AlphaTech"
    assert lead.evidence is not None
    assert lead.evidence.financial.verification_status == "VERIFIED"
    assert lead.financial_source_url == source_url


def test_e2e_scenario_b_source_free_candidate_rejected_in_pipeline():
    """Scenario B: Grounded discovery has no funding source for candidate; model invents $3M -> REJECTED."""
    cand = _create_base_candidate(name="NoSourceTech")
    cand.total_cumulative_funding_usd = 3_000_000.0
    cand.financial_evidence_date = "2026-01-01"
    cand.financial_source_url = None
    cand.financial_source_quote = None
    cand.trusted_source_urls = []
    cand.trusted_source_text = None

    with patch("core.pipeline.genai.Client"):
        with patch("core.pipeline.generate_queries", return_value=["test query"]):
            with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
                with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                    with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                        founder_name_found="Elena Weber",
                        founder_title_found="CEO",
                        contact_path_found=True,
                        pages_checked=["https://testcorp.eu/about"]
                    )):
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=("elena@testcorp.eu", "https://testcorp.eu/about")):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    # Must be rejected at financial stage
    assert summary.funnel.passed_financial == 0
    assert len(summary.leads) == 0


def test_e2e_scenario_c_followup_surfaces_real_seed_source_policy_b_qualifies():
    """Scenario C: Follow-up surfaces real $3M seed source -> Policy B qualifies E2E."""
    cand = _create_base_candidate(name="SeedTarget")
    cand.latest_round_amount_usd = 3_000_000.0
    cand.latest_round_name = "Seed"
    cand.latest_round_date = "2026-01-01"
    cand.financial_evidence_date = "2026-01-01"

    followup_results = [
        {
            "title": "SeedTarget raises $3M Seed",
            "url": "https://techfunding.eu/seedtarget",
            "content": "SeedTarget announced it raised $3M in seed funding."
        }
    ]

    mock_resp_json = json.dumps({
        "subsequent_rounds_discovered": False,
        "acquisition_or_closure_discovered": False,
        "latest_round_amount_usd": 3000000.0,
        "latest_round_date": "2026-01-01",
        "financial_evidence_date": "2026-01-01",
        "evidence_snippet": "SeedTarget announced it raised $3M in seed funding.",
        "financial_source_url": "https://techfunding.eu/seedtarget",
    })

    with patch("core.pipeline.genai.Client"):
        with patch("core.pipeline.generate_queries", return_value=["test query"]):
            with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
                with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                    with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                        founder_name_found="Elena Weber",
                        founder_title_found="CEO",
                        contact_path_found=True,
                        pages_checked=["https://testcorp.eu/about"]
                    )):
                        with patch("core.discovery.search_tavily", return_value=followup_results):
                            with patch("core.extractor.generate_with_fallback", return_value=mock_resp_json):
                                with patch("core.pipeline.verify_founder_email_on_site", return_value=("elena@testcorp.eu", "https://testcorp.eu/about")):
                                    summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert summary.funnel.passed_financial == 1
    assert len(summary.leads) == 1
    lead = summary.leads[0]
    assert lead.company_name == "SeedTarget"
    assert lead.evidence is not None
    assert lead.evidence.financial.financial_type == "SINGLE_ROUND_FUNDING"
    assert lead.evidence.financial.verification_status == "VERIFIED"
    assert lead.financial_source_url == "https://techfunding.eu/seedtarget"


def test_e2e_scenario_d_followup_source_contradiction_rejected():
    """Scenario D: Follow-up surfaces source saying $800K, model claims $3M -> Contradiction rejected."""
    cand = _create_base_candidate(name="ContradictCorp")
    cand.latest_round_amount_usd = 3_000_000.0
    cand.latest_round_name = "Seed"
    cand.latest_round_date = "2026-01-01"
    cand.financial_evidence_date = "2026-01-01"

    followup_results = [
        {
            "title": "ContradictCorp pre-seed",
            "url": "https://techfunding.eu/contradict",
            "content": "ContradictCorp raised $800k in pre-seed funding."
        }
    ]

    mock_resp_json = json.dumps({
        "subsequent_rounds_discovered": False,
        "acquisition_or_closure_discovered": False,
        "latest_round_amount_usd": 3000000.0,
        "latest_round_date": "2026-01-01",
        "financial_evidence_date": "2026-01-01",
        "evidence_snippet": "ContradictCorp raised $800k in pre-seed funding.",
        "financial_source_url": "https://techfunding.eu/contradict",
    })

    with patch("core.pipeline.genai.Client"):
        with patch("core.pipeline.generate_queries", return_value=["test query"]):
            with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
                with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                    with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                        founder_name_found="Elena Weber",
                        founder_title_found="CEO",
                        contact_path_found=True,
                        pages_checked=["https://testcorp.eu/about"]
                    )):
                        with patch("core.discovery.search_tavily", return_value=followup_results):
                            with patch("core.extractor.generate_with_fallback", return_value=mock_resp_json):
                                with patch("core.pipeline.verify_founder_email_on_site", return_value=("elena@testcorp.eu", "https://testcorp.eu/about")):
                                    summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert summary.funnel.passed_financial == 0
    assert len(summary.leads) == 0
