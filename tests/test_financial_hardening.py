"""
tests/test_financial_hardening.py
Automated regression tests covering:
- Date parsing and recency basis evaluation
- Tests A-G requested in targeted correction pass
- Original failure mode cases (Deskbird, HiveMQ, etc.)
- 10 adversarial financial scenarios
- Strict email-guessing rejection invariant
"""

from datetime import date
from core.recency import parse_financial_date, is_older_than_18_months, evaluate_recency_basis
from core.models import CandidateCompany
from core.extractor import evaluate_financial_qualification, passes_non_us_criterion

REF_DATE = date(2026, 9, 12)

def _attach_mock_provenance(cand: CandidateCompany) -> CandidateCompany:
    amt = cand.total_cumulative_funding_usd or cand.revenue_amount_usd or cand.latest_round_amount_usd or 2_000_000.0
    url = cand.financial_source_url or "https://trustednews.com/funding-article"
    cand.financial_source_url = url
    if not cand.financial_source_quote:
        if cand.revenue_amount_usd and cand.revenue_period == "ARR":
            cand.financial_source_quote = f"Company generated ${amt:,.0f} in ARR."
        else:
            cand.financial_source_quote = f"Company raised ${amt:,.0f} in funding."
    cand.trusted_source_urls = [url]
    cand.trusted_source_text = cand.financial_source_quote
    cand.financial_followup_complete = True
    return cand

def test_A_old_round_recent_cumulative_confirmation():
    candidate = _attach_mock_provenance(CandidateCompany(
        name="ReliableSaaS",
        description="B2B platform",
        industry="DevTools",
        hq_country="UK",
        is_tech_platform=True,
        total_cumulative_funding_usd=4_200_000,
        latest_round_amount_usd=2_000_000,
        latest_round_date="2024-02-15",
        financial_evidence_date="2026-07-01",
        founder_name="Alice Smith"
    ))
    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified
    assert audit.is_recent is True
    assert "2026-07-01" in audit.recency_basis

def test_B_old_round_no_recent_confirmation():
    candidate = CandidateCompany(
        name="StaleRound Co",
        description="Platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=3_000_000,
        latest_round_date="2024-02-15",
        financial_evidence_date=None,
        founder_name="Bob Jones"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert audit.rejection_stage == "RECENCY"

def test_C_qualifying_arr_no_funding():
    candidate = _attach_mock_provenance(CandidateCompany(
        name="OrganicGrowth",
        description="Workflow automation",
        industry="B2B SaaS",
        hq_country="France",
        is_tech_platform=True,
        revenue_amount_usd=2_400_000,
        revenue_period="ARR",
        financial_source_quote="OrganicGrowth achieved $2.4M in ARR.",
        revenue_date="2026-01-10",
        total_cumulative_funding_usd=None,
        founder_name="Claire Martin"
    ))
    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified
    assert audit.financial_type == "ARR"
    assert "$2,400,000 ARR" in audit.financial_figure_used

def test_D_qualifying_arr_excessive_cumulative_funding():
    candidate = _attach_mock_provenance(CandidateCompany(
        name="ScaledUnicorn",
        description="Enterprise platform",
        industry="Enterprise SaaS",
        hq_country="Sweden",
        is_tech_platform=True,
        revenue_amount_usd=2_400_000,
        revenue_period="ARR",
        financial_source_quote="ScaledUnicorn reported $2.4M ARR.",
        revenue_date="2026-03-01",
        total_cumulative_funding_usd=20_000_000,
        founder_name="David Lindqvist"
    ))
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert "Policy D" in audit.rejection_reason

def test_E_non_us_hq_minimal_us_signal():
    candidate = CandidateCompany(
        name="BerlinCloud",
        description="Cloud orchestration",
        industry="DevTools",
        hq_country="Germany",
        hq_city="Berlin",
        is_tech_platform=True,
        us_presence_evidence="Incorporated Delaware holding for funding; all operations and team in Berlin.",
        founder_name="Stefan Meyer"
    )
    is_non_us, result, evidence = passes_non_us_criterion(candidate)
    assert is_non_us is True
    assert result == "PASSED_MINIMAL_US_SIGNAL"

def test_F_non_us_hq_strong_us_presence():
    candidate = CandidateCompany(
        name="NominalLondon Co",
        description="Analytics",
        industry="SaaS",
        hq_country="UK",
        hq_city="London",
        is_tech_platform=True,
        us_presence_evidence="Executive team and primary operations in the US (San Francisco office).",
        founder_name="Edward King"
    )
    is_non_us, result, evidence = passes_non_us_criterion(candidate)
    assert is_non_us is False
    assert result == "REJECTED_US_OPERATIONS"

def test_G_non_us_hq_merely_us_customers():
    candidate = CandidateCompany(
        name="ParisSec",
        description="Security software",
        industry="Cybersecurity",
        hq_country="France",
        hq_city="Paris",
        is_tech_platform=True,
        us_presence_evidence="80% of revenue from US customers; founding team and operations based in Paris.",
        founder_name="Gilles Moreau"
    )
    is_non_us, result, evidence = passes_non_us_criterion(candidate)
    assert is_non_us is True
    assert "PASSED" in result

def test_no_guessed_or_synthesized_emails_accepted():
    from core.email_verifier import verify_founder_email_on_site
    candidate = CandidateCompany(
        name="StealthCompany",
        description="Software tool",
        industry="SaaS",
        hq_country="UK",
        website_url="https://non-existent-domain-404.io",
        is_tech_platform=True,
        founder_name="John Doe"
    )
    res = verify_founder_email_on_site(candidate)
    assert res is None

def test_adv_scenario_2_boundary_funding_5m():
    candidate = _attach_mock_provenance(CandidateCompany(
        name="BoundarySeed",
        description="Data tools",
        industry="DevTools",
        hq_country="UK",
        is_tech_platform=True,
        total_cumulative_funding_usd=5_000_000,
        latest_round_amount_usd=5_000_000,
        latest_round_date="2026-01-01",
        founder_name="Jane Boundary"
    ))
    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified
    assert audit.confidence_rating == "HIGH"

def test_adv_scenario_8_acquired_company_rejected():
    candidate = CandidateCompany(
        name="AcquiredCorp",
        description="Acquired startup",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_000_000,
        acquisition_or_closure_discovered=True,
        founder_name="Hans Schmidt"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert "acquired or defunct" in audit.rejection_reason

def test_adv_scenario_9_subsequent_unquantified_rejected():
    candidate = CandidateCompany(
        name="UnquantifiedGrowth",
        description="Platform",
        industry="SaaS",
        hq_country="France",
        is_tech_platform=True,
        latest_round_amount_usd=2_000_000,
        subsequent_rounds_discovered=True,
        founder_name="Pierre Growth"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert "subsequent rounds or growth financing detected" in audit.rejection_reason

def test_recency_recent_2026():
    assert not is_older_than_18_months("2026-05-10", reference_date=REF_DATE)
    assert not is_older_than_18_months("May 2026", reference_date=REF_DATE)

def test_recency_borderline_date():
    assert not is_older_than_18_months("2025-04-01", reference_date=REF_DATE)
    assert not is_older_than_18_months("April 2025", reference_date=REF_DATE)
    assert is_older_than_18_months("2025-02-15", reference_date=REF_DATE)

def test_recency_old_2024():
    assert is_older_than_18_months("2024-11-20", reference_date=REF_DATE)
    assert is_older_than_18_months("2024", reference_date=REF_DATE)

def test_recency_old_2022():
    assert is_older_than_18_months("2022-06-15", reference_date=REF_DATE)
    assert is_older_than_18_months("2022", reference_date=REF_DATE)

def test_recency_malformed_and_unknown():
    assert is_older_than_18_months(None, reference_date=REF_DATE)
    assert is_older_than_18_months("", reference_date=REF_DATE)
    assert is_older_than_18_months("sometime recently", reference_date=REF_DATE)

def test_case_1_deskbird_stale_round():
    candidate = CandidateCompany(
        name="Deskbird",
        description="Hybrid workplace platform",
        industry="Enterprise SaaS",
        hq_country="Switzerland",
        hq_city="St. Gallen",
        is_tech_platform=True,
        latest_round_amount_usd=5_000_000,
        latest_round_name="Pre-Series A",
        latest_round_date="2022-08-01",
        total_cumulative_funding_usd=44_000_000,
        subsequent_rounds_discovered=True,
        founder_name="Ivan Cossu",
        founder_title="CEO"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert "exceeds $5M" in audit.rejection_reason

def test_case_2_large_cumulative_funding():
    candidate = CandidateCompany(
        name="ScaleUp Tech",
        description="Fintech infrastructure",
        industry="Fintech",
        hq_country="UK",
        is_tech_platform=True,
        latest_round_amount_usd=2_000_000,
        latest_round_date="2026-01-15",
        total_cumulative_funding_usd=15_000_000,
        founder_name="Jane Doe"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert "exceeds $5M" in audit.rejection_reason

def test_case_3_valid_small_company():
    candidate = _attach_mock_provenance(CandidateCompany(
        name="MicroSaaS",
        description="Developer workflow tools",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_500_000,
        latest_round_name="Seed",
        latest_round_date="2025-08-01",
        subsequent_rounds_discovered=False,
        founder_name="Max Mustermann"
    ))
    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified
    assert audit.confidence_rating == "HIGH"
    assert audit.financial_type == "CUMULATIVE_FUNDING"

def test_case_4_revenue_company():
    candidate = _attach_mock_provenance(CandidateCompany(
        name="BootstrappedHQ",
        description="Compliance monitoring platform",
        industry="B2B SaaS",
        hq_country="Netherlands",
        is_tech_platform=True,
        revenue_amount_usd=2_400_000,
        revenue_period="ARR",
        financial_source_quote="BootstrappedHQ reached $2.4M ARR.",
        revenue_date="2025-10-01",
        founder_name="Lars Jansen"
    ))
    qualified, audit = evaluate_financial_qualification(candidate)
    assert qualified
    assert audit.financial_type == "ARR"
    assert "$2,400,000 ARR" in audit.financial_figure_used

def test_case_5_high_revenue():
    candidate = CandidateCompany(
        name="BigRevenue Co",
        description="Logistics scheduling platform",
        industry="Logistics",
        hq_country="France",
        is_tech_platform=True,
        revenue_amount_usd=7_000_000,
        revenue_period="ARR",
        financial_source_quote="BigRevenue Co generates $7M in ARR.",
        revenue_date="2026-02-01",
        founder_name="Pierre Dubois"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert "exceeds $5M" in audit.rejection_reason

def test_case_6_unknown_financials():
    candidate = CandidateCompany(
        name="Vague Platform",
        description="Social sharing tool",
        industry="MarTech",
        hq_country="Sweden",
        is_tech_platform=True,
        total_cumulative_funding_usd=None,
        revenue_amount_usd=None,
        financial_evidence_snippet="The company is experiencing rapid growth with seven-figure traction.",
        founder_name="Anna Lind"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert audit.financial_type == "INSUFFICIENT"

def test_case_7_old_financial_signal():
    candidate = CandidateCompany(
        name="Dated Startup",
        description="Recruitment SaaS",
        industry="HR Tech",
        hq_country="Estonia",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_000_000,
        latest_round_date="2022-04-10",
        founder_name="Taavi Kallas"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert audit.rejection_stage == "RECENCY"
    assert "exceeds 18m" in audit.rejection_reason

def test_case_8_smaller_round_larger_cumulative():
    candidate = CandidateCompany(
        name="MultiRound Systems",
        description="Security API platform",
        industry="Cybersecurity",
        hq_country="Canada",
        is_tech_platform=True,
        latest_round_amount_usd=3_000_000,
        latest_round_name="Series A extension",
        latest_round_date="2026-01-10",
        total_cumulative_funding_usd=12_000_000,
        founder_name="Sarah Smith"
    )
    qualified, audit = evaluate_financial_qualification(candidate)
    assert not qualified
    assert "exceeds $5M" in audit.rejection_reason

def test_fallback_model_and_quota_error_configuration():
    from core.config import FALLBACK_MODEL, GEMINI_MODEL
    from core.discovery import is_search_quota_exhausted

    # 1. Fallback model must not be the deprecated/invalid gemini-3-flash
    assert FALLBACK_MODEL != "gemini-3-flash", "FALLBACK_MODEL cannot be invalid 'gemini-3-flash'"

    # 2. Fallback model must be a non-empty, valid string identifier
    assert isinstance(FALLBACK_MODEL, str) and len(FALLBACK_MODEL) > 0
    assert "gemini" in FALLBACK_MODEL.lower()
    assert FALLBACK_MODEL == "gemini-3.5-flash-lite" or "flash" in FALLBACK_MODEL

    # 3. Quota error classifier correctly identifies 429 and RESOURCE_EXHAUSTED without API calls
    class MockError(Exception):
        pass
    quota_err = MockError("429 RESOURCE_EXHAUSTED. You exceeded your current quota.")
    transient_err = MockError("503 UNAVAILABLE. Service temporarily overloaded.")
    assert is_search_quota_exhausted(quota_err) is True
    assert is_search_quota_exhausted(transient_err) is False

def test_parse_financial_amount_all_formats():
    from core.extractor import parse_financial_amount

    # None and empty
    assert parse_financial_amount(None) is None
    assert parse_financial_amount("") is None
    assert parse_financial_amount("null") is None
    assert parse_financial_amount("N/A") is None

    # Numeric direct
    assert parse_financial_amount(3000000) == 3000000.0
    assert parse_financial_amount(2.5) == 2.5

    # Commas
    assert parse_financial_amount("$7,800,000") == 7800000.0
    assert parse_financial_amount("1,500,000 USD") == 1500000.0

    # Millions
    assert parse_financial_amount("$289M") == 289000000.0
    assert parse_financial_amount("2.5 million") == 2500000.0
    assert parse_financial_amount("3M") == 3000000.0

    # Billions
    assert parse_financial_amount("$3.38B") == 3380000000.0

    # Ranges (takes higher end for conservative cumulative/round ceiling)
    assert parse_financial_amount("10M-25M") == 25000000.0
    assert parse_financial_amount("$1M to $5M") == 5000000.0


# ==============================================================================
# Borderline Cases 1 through 12 (Mandatory TVB Verification Suite)
# ==============================================================================

def test_borderline_case_1_funding_999999_rejected():
    cand = CandidateCompany(
        name="Case1Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=999_999,
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert "below $1M floor" in audit.rejection_reason

def test_borderline_case_2_funding_1000000_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="Case2Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=1_000_000,
        financial_evidence_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "CUMULATIVE_FUNDING"

def test_borderline_case_3_funding_5000000_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="Case3Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=5_000_000,
        financial_evidence_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified
    assert audit.qualification_status == "ACCEPTED"

def test_borderline_case_4_funding_5000001_rejected():
    cand = CandidateCompany(
        name="Case4Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=5_000_001,
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert "exceeds $5M ceiling" in audit.rejection_reason

def test_borderline_case_5_arr_999999_rejected():
    cand = CandidateCompany(
        name="Case5Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=999_999,
        revenue_period="ARR",
        financial_source_quote="ARR of $999,999.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert "below $1M floor" in audit.rejection_reason

def test_borderline_case_6_arr_1000000_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="Case6Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=1_000_000,
        revenue_period="ARR",
        financial_source_quote="ARR of $1,000,000.",
        revenue_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "ARR"

def test_borderline_case_7_arr_5000000_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="Case7Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=5_000_000,
        revenue_period="ARR",
        financial_source_quote="ARR of $5,000,000.",
        revenue_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified
    assert audit.qualification_status == "ACCEPTED"

def test_borderline_case_8_arr_5000001_rejected():
    cand = CandidateCompany(
        name="Case8Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=5_000_001,
        revenue_period="ARR",
        financial_source_quote="ARR of $5,000,001.",
        revenue_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert "exceeds $5M ceiling" in audit.rejection_reason

def test_borderline_case_9_funding_20m_arr_3m_rejected():
    cand = _attach_mock_provenance(CandidateCompany(
        name="Case9Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=20_000_000,
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="ARR of $3,000,000.",
        revenue_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert not qualified
    assert audit.rejection_stage == "FINANCIAL"
    assert "Policy D" in audit.rejection_reason

def test_borderline_case_10_funding_none_arr_3m_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="Case10Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=None,
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="ARR of $3,000,000.",
        revenue_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "ARR"

def test_borderline_case_11_old_round_current_cumulative_confirmation_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="Case11Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=3_000_000,
        latest_round_date="2023-05-15",
        financial_evidence_date="2026-05-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified
    assert audit.qualification_status == "ACCEPTED"
    assert audit.is_recent is True

def test_borderline_case_12_old_round_no_current_confirmation_rejected():
    cand = CandidateCompany(
        name="Case12Co",
        description="SaaS",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        total_cumulative_funding_usd=3_000_000,
        latest_round_date="2023-05-15",
        financial_evidence_date=None
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert not qualified
    assert audit.rejection_stage == "RECENCY"


