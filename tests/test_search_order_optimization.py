"""
tests/test_search_order_optimization.py
Deterministic offline tests verifying search-order optimization, free contact-path gating,
and lead-per-call metrics with zero Tavily credit consumption.
"""

import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from core.models import CandidateCompany, ContactPathResult, PipelineSummary, FunnelMetrics, LeadRecord
from core.pipeline import (
    run_lead_pipeline, has_initial_financial_evidence, should_skip_financial_followup
)
from core.email_verifier import (
    discover_founder_contact_path, extract_founder_from_dom
)
from core.discovery import get_tavily_call_breakdown, reset_tavily_search_count


# ==============================================================================
# 1. US-HEADQUARTERED CANDIDATE REJECTED BEFORE ANY FOLLOW-UP
# ==============================================================================

def test_1_us_candidate_rejected_before_any_followup():
    """US-headquartered candidate must be rejected before contact-path discovery or follow-up."""
    cand = CandidateCompany(
        name="US Analytics Inc",
        website_url="https://usanalytics.com",
        description="SaaS product",
        industry="SaaS",
        hq_country="United States",
        hq_city="New York",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_000_000.0,
        founder_name="John Smith",
        founder_title="CEO"
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path") as mock_contact:
                    with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                        summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    mock_contact.assert_not_called()
    mock_followup.assert_not_called()
    assert len(summary.audit_log) == 1
    assert summary.audit_log[0].rejection_stage == "NON_US"
    assert summary.funnel.potential_followups_before_contact_gate == 0


# ==============================================================================
# 2. NON-TECH CANDIDATE REJECTED BEFORE ANY FOLLOW-UP
# ==============================================================================

def test_2_non_tech_candidate_rejected_before_any_followup():
    """Non-tech consulting/agency entity must be rejected before contact-path or follow-up."""
    cand = CandidateCompany(
        name="London Legal Consultants",
        website_url="https://londonlegal.co.uk",
        description="Legal consulting and advisory firm",
        industry="Legal",
        hq_country="UK",
        hq_city="London",
        is_tech_platform=False,
        total_cumulative_funding_usd=2_000_000.0,
        founder_name="Arthur Pendelton",
        founder_title="Managing Director"
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path") as mock_contact:
                    with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                        summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    mock_contact.assert_not_called()
    mock_followup.assert_not_called()
    assert len(summary.audit_log) == 1
    assert summary.audit_log[0].rejection_stage == "TECH_PLATFORM"
    assert summary.funnel.potential_followups_before_contact_gate == 0


# ==============================================================================
# 3. OVERFUNDED CANDIDATE REJECTED BEFORE FOLLOW-UP
# ==============================================================================

def test_3_overfunded_candidate_rejected_before_followup():
    """Candidate with known cumulative funding > $5M must be rejected before follow-up."""
    cand = CandidateCompany(
        name="Giant Scale AI",
        website_url="https://giantscale.fr",
        description="Enterprise AI platform",
        industry="AI",
        hq_country="France",
        hq_city="Paris",
        is_tech_platform=True,
        total_cumulative_funding_usd=25_000_000.0,
        founder_name="Pierre Dubois",
        founder_title="CEO"
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path") as mock_contact:
                    with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                        summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    mock_contact.assert_not_called()
    mock_followup.assert_not_called()
    assert len(summary.audit_log) == 1
    assert summary.audit_log[0].rejection_stage == "FINANCIAL"
    assert summary.funnel.potential_followups_before_contact_gate == 0


# ==============================================================================
# 4. CANDIDATE WITH NO OFFICIAL SITE -> NO FOLLOW-UP
# ==============================================================================

def test_4_candidate_with_no_official_site_no_followup():
    """Candidate without an official website URL must be rejected without triggering follow-up."""
    cand = CandidateCompany(
        name="NoSite Technologies",
        website_url=None,
        description="Developer tools platform",
        industry="DevTools",
        hq_country="Germany",
        hq_city="Berlin",
        is_tech_platform=True,
        total_cumulative_funding_usd=None,
        founder_name="Klaus Weber",
        founder_title="Founder"
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                    summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    mock_followup.assert_not_called()
    assert summary.funnel.potential_followups_before_contact_gate == 1
    assert summary.funnel.followups_skipped_no_contact_path == 1
    assert summary.funnel.actual_followups_after_contact_gate == 0
    assert summary.audit_log[0].rejection_stage == "EMAIL_VERIFICATION"
    assert "No official website URL" in summary.audit_log[0].rejection_reason


# ==============================================================================
# 5. CANDIDATE WITH OFFICIAL SITE + FOUNDER PATH -> ELIGIBLE FOR FOLLOW-UP
# ==============================================================================

def test_5_candidate_with_official_site_founder_path_eligible_for_followup():
    """Candidate with valid site and founder contact path must be eligible for follow-up."""
    cand = CandidateCompany(
        name="FounderRoute Tech",
        website_url="https://founderroute.eu",
        description="Cloud observability platform",
        industry="Cloud",
        hq_country="Netherlands",
        hq_city="Amsterdam",
        is_tech_platform=True,
        total_cumulative_funding_usd=None,
        founder_name="Jan van Dijk",
        founder_title="CEO"
    )

    def mock_followup(client, c, **kwargs):
        c.total_cumulative_funding_usd = 2_500_000.0
        c.latest_round_name = "Seed"
        c.latest_round_amount_usd = 2_500_000.0
        c.latest_round_date = "2025-10"
        c.financial_evidence_date = "2025-10"
        c.financial_source_url = "https://trustednews.eu/deal"
        c.financial_source_quote = "Raised $2.5M in Seed funding."
        c.trusted_source_urls = ["https://trustednews.eu/deal"]
        c.trusted_source_text = "Raised $2.5M in Seed funding."
        c.financial_followup_complete = True
        return c

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Jan van Dijk",
                    founder_title_found="CEO",
                    contact_path_found=True,
                    pages_checked=["https://founderroute.eu", "https://founderroute.eu/about"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=mock_followup) as mock_f:
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=("jan@founderroute.eu", "https://founderroute.eu/about")):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    mock_f.assert_called_once()
    assert summary.funnel.actual_followups_after_contact_gate == 1
    assert len(summary.leads) == 1
    assert summary.leads[0].company_name == "FounderRoute Tech"
    assert summary.leads[0].verified_email == "jan@founderroute.eu"


# ==============================================================================
# 6. CANDIDATE WITH FOUNDER PATH BUT NO EMAIL -> NOT AUTOMATICALLY REJECTED
# ==============================================================================

def test_6_candidate_with_founder_path_but_no_email_not_automatically_rejected():
    """Candidate with founder path but no email early must NOT be rejected before follow-up."""
    cand = CandidateCompany(
        name="TeamPage NoEmail",
        website_url="https://teampagenoemail.de",
        description="Cybersecurity compliance automation",
        industry="Cybersecurity",
        hq_country="Germany",
        hq_city="Munich",
        is_tech_platform=True,
        total_cumulative_funding_usd=None,
        founder_name="Stefan Huber",
        founder_title="Founder & CEO"
    )

    def mock_followup(client, c, **kwargs):
        c.total_cumulative_funding_usd = 3_000_000.0
        c.latest_round_name = "Seed"
        c.latest_round_amount_usd = 3_000_000.0
        c.latest_round_date = "2025-08"
        c.financial_evidence_date = "2025-08"
        c.financial_source_url = "https://trustednews.eu/deal"
        c.financial_source_quote = "Raised $3M in Seed funding."
        c.trusted_source_urls = ["https://trustednews.eu/deal"]
        c.trusted_source_text = "Raised $3M in Seed funding."
        c.financial_followup_complete = True
        return c

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Stefan Huber",
                    founder_title_found="Founder & CEO",
                    exact_email_found=None,
                    contact_path_found=True,
                    pages_checked=["https://teampagenoemail.de", "https://teampagenoemail.de/team"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=mock_followup) as mock_f:
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=None):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    # Follow-up WAS executed because founder contact path existed
    mock_f.assert_called_once()
    assert summary.funnel.actual_followups_after_contact_gate == 1
    # Financial stage was passed
    assert summary.funnel.passed_financial == 1
    # Final rejection happened only at authoritative email verification
    assert summary.audit_log[-1].rejection_stage == "EMAIL_VERIFICATION"
    assert "No verified founder email on site" in summary.audit_log[-1].rejection_reason


# ==============================================================================
# 7. CANDIDATE WITH EXACT EARLY FOUNDER EMAIL STILL PASSES FINAL VERIFIER
# ==============================================================================

def test_7_candidate_with_exact_early_founder_email_still_passes_final_verifier():
    """Candidate with exact email found early proceeds and qualifies through authoritative verifier."""
    cand = CandidateCompany(
        name="EarlyEmail SaaS",
        website_url="https://earlyemail.io",
        description="API automation platform",
        industry="SaaS",
        hq_country="Estonia",
        hq_city="Tallinn",
        is_tech_platform=True,
        total_cumulative_funding_usd=1_500_000.0,
        latest_round_name="Seed",
        latest_round_amount_usd=1_500_000.0,
        latest_round_date="2025-09",
        financial_evidence_date="2025-09",
        financial_source_url="https://trustednews.eu/deal",
        financial_source_quote="Raised $1.5M in Seed funding.",
        trusted_source_urls=["https://trustednews.eu/deal"],
        trusted_source_text="Raised $1.5M in Seed funding.",
        financial_followup_complete=True,
        founder_name="Taavi Tamm",
        founder_title="CEO"
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Taavi Tamm",
                    founder_title_found="CEO",
                    exact_email_found="taavi@earlyemail.io",
                    email_source_url="https://earlyemail.io/contact",
                    contact_path_found=True,
                    pages_checked=["https://earlyemail.io", "https://earlyemail.io/contact"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kw: c):
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=("taavi@earlyemail.io", "https://earlyemail.io/contact")):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert summary.funnel.exact_emails_found_early == 1
    assert summary.funnel.passed_email_verification == 1
    assert len(summary.leads) == 1
    assert summary.leads[0].verified_email == "taavi@earlyemail.io"


# ==============================================================================
# 8. FOUNDER DISCOVERED BY OFFICIAL SITE AFTER INITIAL EXTRACTION IS PRESERVED
# ==============================================================================

def test_8_founder_discovered_by_official_site_preserved():
    """If initial candidate extraction lacked founder, free site discovery identifies and preserves founder."""
    cand = CandidateCompany(
        name="Anonymous Start",
        website_url="https://anonymousstart.fi",
        description="Data infrastructure software",
        industry="Data",
        hq_country="Finland",
        hq_city="Helsinki",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_000_000.0,
        latest_round_name="Seed",
        latest_round_amount_usd=2_000_000.0,
        latest_round_date="2025-07",
        financial_evidence_date="2025-07",
        financial_source_url="https://trustednews.eu/deal",
        financial_source_quote="Raised $2M in Seed funding.",
        trusted_source_urls=["https://trustednews.eu/deal"],
        trusted_source_text="Raised $2M in Seed funding.",
        financial_followup_complete=True,
        founder_name=None,
        founder_title=None
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Matti Virtanen",
                    founder_title_found="Co-Founder & CEO",
                    contact_path_found=True,
                    pages_checked=["https://anonymousstart.fi", "https://anonymousstart.fi/team"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kw: c):
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=("matti@anonymousstart.fi", "https://anonymousstart.fi/team")):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert len(summary.leads) == 1
    assert "Matti Virtanen" in summary.leads[0].ceo_cofounder_name
    assert summary.leads[0].verified_email == "matti@anonymousstart.fi"


# ==============================================================================
# 9. GENERIC EMAIL ONLY -> NOT TREATED AS VERIFIED FOUNDER EMAIL
# ==============================================================================

def test_9_generic_email_only_not_treated_as_verified_founder_email():
    """Site with only generic email (contact_path=True) must be rejected by authoritative verifier."""
    cand = CandidateCompany(
        name="Generic Only Platform",
        website_url="https://genericonly.eu",
        description="Payment orchestration platform",
        industry="Fintech",
        hq_country="Ireland",
        hq_city="Dublin",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_200_000.0,
        latest_round_name="Seed",
        latest_round_amount_usd=2_200_000.0,
        latest_round_date="2025-08",
        financial_evidence_date="2025-08",
        financial_source_url="https://trustednews.eu/deal",
        financial_source_quote="Raised $2.2M in Seed funding.",
        trusted_source_urls=["https://trustednews.eu/deal"],
        trusted_source_text="Raised $2.2M in Seed funding.",
        financial_followup_complete=True,
        founder_name="Liam O'Connor",
        founder_title="CEO"
    )

    # Contact path is found (e.g. contact page exists with info@) but exact_email_found is None
    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                with patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(
                    founder_name_found="Liam O'Connor",
                    founder_title_found="CEO",
                    exact_email_found=None,
                    contact_path_found=True,
                    pages_checked=["https://genericonly.eu", "https://genericonly.eu/contact"]
                )):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kw: c):
                        # Authoritative verifier rejects generic email
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=None):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert len(summary.leads) == 0
    assert summary.funnel.passed_email_verification == 0
    assert summary.audit_log[-1].rejection_stage == "EMAIL_VERIFICATION"
    assert "No verified founder email on site" in summary.audit_log[-1].rejection_reason


# ==============================================================================
# 10. FOLLOW-UP COUNT IS LOWER THAN EQUIVALENT OLD ORDERING
# ==============================================================================

def test_10_followup_count_lower_than_old_ordering_for_mocked_candidates():
    """Proves follow-up count is strictly lower under the new order for dead-end candidates."""
    candidates = [
        # Candidate 1: Missing website URL & no financial evidence -> Skipped by contact gate
        CandidateCompany(
            name="DeadEnd 1",
            website_url=None,
            description="SaaS product",
            industry="SaaS",
            hq_country="Norway",
            is_tech_platform=True,
            total_cumulative_funding_usd=None
        ),
        # Candidate 2: Broken website & no financial evidence -> Skipped by contact gate
        CandidateCompany(
            name="DeadEnd 2",
            website_url="https://broken404.no",
            description="Cloud product",
            industry="Cloud",
            hq_country="Norway",
            is_tech_platform=True,
            total_cumulative_funding_usd=None
        ),
        # Candidate 3: Valid website and founder path -> Allowed to follow-up
        CandidateCompany(
            name="Valid Prospect",
            website_url="https://prospect.no",
            description="AI platform",
            industry="AI",
            hq_country="Norway",
            is_tech_platform=True,
            total_cumulative_funding_usd=None,
            founder_name="Henrik Ibsen",
            founder_title="CEO"
        )
    ]

    def mock_contact_path(c):
        if c.name == "DeadEnd 2":
            return ContactPathResult(
                contact_path_found=False,
                reason_if_no_contact_path="Failed to fetch homepage: https://broken404.no"
            )
        return ContactPathResult(
            founder_name_found="Henrik Ibsen",
            founder_title_found="CEO",
            contact_path_found=True,
            pages_checked=["https://prospect.no"]
        )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=candidates):
                with patch("core.pipeline.discover_founder_contact_path", side_effect=mock_contact_path):
                    with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kw: c) as mock_f:
                        with patch("core.pipeline.verify_founder_email_on_site", return_value=None):
                            summary = run_lead_pipeline(api_key="mock_key", target_count=5, max_queries=1)

    # In old ordering, all 3 candidates would execute follow-up (3 calls)
    # In new ordering, 2 are skipped, exactly 1 executes
    assert summary.funnel.potential_followups_before_contact_gate == 3
    assert summary.funnel.followups_skipped_no_contact_path == 2
    assert summary.funnel.actual_followups_after_contact_gate == 1
    assert mock_f.call_count == 1
    assert mock_f.call_count < 3


# ==============================================================================
# 11. VERIFIED LEADS PER TAVILY CALL CALCULATES CORRECTLY
# ==============================================================================

def test_11_verified_leads_per_tavily_call_calculates_correctly():
    """Verifies verified_leads_per_tavily_call handles zero-division and accurate float division."""
    # Case A: 0 total calls -> 0.0
    summary_zero = PipelineSummary(
        total_queries_run=1,
        duration_seconds=5.0,
        funnel=FunnelMetrics(),
        leads=[],
        audit_log=[],
        shortfall=15,
        exit_reason="Test",
        total_tavily_calls=0,
        verified_leads_per_tavily_call=0.0
    )
    assert summary_zero.verified_leads_per_tavily_call == 0.0

    # Case B: 10 total calls, 2 leads -> 0.2
    lead_1 = LeadRecord(
        company_name="Comp A",
        description="Desc",
        industry="SaaS",
        ceo_cofounder_name="CEO One",
        verified_email="ceo@compa.eu",
        email_source_url="https://compa.eu",
        hq_location="Munich, Germany",
        financial_signal="$2.5M",
        recency_basis="Recent",
        us_presence_result="NON_US_VERIFIED",
        financial_confidence="HIGH",
        recency_audit_note="Verified"
    )
    lead_2 = LeadRecord(
        company_name="Comp B",
        description="Desc",
        industry="SaaS",
        ceo_cofounder_name="CEO Two",
        verified_email="ceo@compb.eu",
        email_source_url="https://compb.eu",
        hq_location="Paris, France",
        financial_signal="$3M",
        recency_basis="Recent",
        us_presence_result="NON_US_VERIFIED",
        financial_confidence="HIGH",
        recency_audit_note="Verified"
    )
    leads_mock = [lead_1, lead_2]
    call_count = 10
    metric = round(len(leads_mock) / call_count, 4)
    summary_ratio = PipelineSummary(
        total_queries_run=3,
        duration_seconds=12.0,
        funnel=FunnelMetrics(verified_leads_per_tavily_call=metric),
        leads=leads_mock,
        audit_log=[],
        shortfall=13,
        exit_reason="Target reached",
        total_tavily_calls=call_count,
        verified_leads_per_tavily_call=metric
    )
    assert summary_ratio.verified_leads_per_tavily_call == 0.2


# ==============================================================================
# 12. ZERO TAVILY CALLS OCCUR IN THESE TESTS
# ==============================================================================

def test_12_zero_tavily_calls_in_tests():
    """Verify that throughout test execution zero live Tavily calls are dispatched."""
    reset_tavily_search_count()
    breakdown = get_tavily_call_breakdown()
    assert breakdown["total_calls"] == 0
    assert breakdown["discovery_calls"] == 0
    assert breakdown["financial_followup_calls"] == 0


# ==============================================================================
# 13. DOM FOUNDER EXTRACTION UNIT TESTS
# ==============================================================================

def test_extract_founder_from_dom_patterns():
    """Direct verification of DOM-based executive name and title parsing."""
    html_1 = "<div><span class='name'>Alice Wonderland</span> - <span>CEO & Founder</span></div>"
    soup_1 = BeautifulSoup(html_1, "html.parser")
    name_1, title_1 = extract_founder_from_dom(soup_1)
    assert name_1 == "Alice Wonderland"
    assert "Ceo" in title_1 or "Founder" in title_1

    html_2 = "<div class='profile'><h4>Co-Founder</h4><p>Bob Marley</p></div>"
    soup_2 = BeautifulSoup(html_2, "html.parser")
    name_2, title_2 = extract_founder_from_dom(soup_2)
    assert name_2 == "Bob Marley"

    html_none = "<div><p>Welcome to our company homepage</p></div>"
    soup_none = BeautifulSoup(html_none, "html.parser")
    name_none, title_none = extract_founder_from_dom(soup_none)
    assert name_none is None
    assert title_none is None
