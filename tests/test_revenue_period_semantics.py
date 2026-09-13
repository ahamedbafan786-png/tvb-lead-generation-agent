"""
Comprehensive test suite for revenue period distinction and ARR qualification semantics.

Verifies:
1. Revenue periods: ARR, ANNUAL, TTM, MONTHLY, QUARTERLY, LIFETIME, UNKNOWN.
2. Policy C strictly requires verified ARR ($1M-$5M).
3. Non-recurring annual revenue and TTM revenue are rejected as ANNUAL_REVENUE and TTM_REVENUE.
4. Monthly, quarterly, and lifetime revenue are rejected.
5. Generic/unspecified revenue without recurring proof cannot qualify.
6. SaaS classification alone cannot convert generic revenue to ARR.
7. Search queries mentioning "ARR" cannot inject ARR attribution into unevidenced snippets.
8. Model claims of ARR contradicted by source snippet (monthly/lifetime) are overridden to safe rejection.
9. Scale conflict (Policy D) correctly disqualifies qualifying ARR when cumulative funding exceeds $5M.
10. Revenue period and financial type are preserved across all audit structures, LeadRecord, and FinancialEvidence.
"""

import pytest
from core.models import (
    CandidateCompany,
    CandidateAuditRecord,
    FinancialEvidence,
    LeadRecord,
    RevenuePeriod,
)
from core.extractor import (
    evaluate_financial_qualification,
    validate_and_classify_revenue,
)


def _attach_mock_provenance(cand: CandidateCompany) -> CandidateCompany:
    amt = cand.revenue_amount_usd or cand.total_cumulative_funding_usd or 2_000_000.0
    url = cand.financial_source_url or "https://trustednews.com/revenue-article"
    cand.financial_source_url = url
    if not cand.financial_source_quote:
        cand.financial_source_quote = f"Company reported ${amt:,.0f} in ARR."
    cand.trusted_source_urls = [url]
    cand.trusted_source_text = cand.financial_source_quote
    cand.financial_followup_complete = True
    return cand

# ==============================================================================
# SECTION 1: REVENUE PERIOD VALIDATION AND CLASSIFICATION UNIT TESTS
# ==============================================================================

def test_validate_arr_explicit_keyword():
    """Snippet explicitly mentioning ARR validates as ARR."""
    cand = CandidateCompany(
        name="CloudMetric",
        description="DevOps platform",
        revenue_amount_usd=2_500_000,
        revenue_period="ARR",
        financial_source_quote="CloudMetric has reached $2.5M in ARR as of Q1 2026."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.ARR
    assert "Corroborated ARR" in reason


def test_validate_annual_recurring_revenue_phrase():
    """Snippet with 'annual recurring revenue' validates as ARR."""
    cand = CandidateCompany(
        name="DataPulse",
        description="Analytics SaaS",
        revenue_amount_usd=3_000_000,
        financial_source_quote="DataPulse reported annual recurring revenue of $3 million."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.ARR


def test_validate_generic_annual_revenue_non_recurring():
    """Annual revenue without recurring indicators classifies as ANNUAL."""
    cand = CandidateCompany(
        name="TradConsulting",
        description="Professional IT services",
        revenue_amount_usd=4_000_000,
        financial_source_quote="Generated $4M in annual revenue for the fiscal year 2025."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.ANNUAL
    assert "non-recurring" in reason.lower() or "accounting" in reason.lower() or "annual revenue" in reason.lower()


def test_validate_trailing_twelve_months_ttm():
    """TTM revenue classifies as TTM."""
    cand = CandidateCompany(
        name="LogiTrack",
        description="Logistics platform",
        revenue_amount_usd=3_500_000,
        financial_source_quote="Generated $3.5M in TTM revenue through December 2025."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.TTM


def test_validate_monthly_revenue():
    """Monthly revenue / MRR classifies as MONTHLY."""
    cand = CandidateCompany(
        name="SubScript",
        description="Subscription manager",
        revenue_amount_usd=250_000,
        financial_source_quote="Current monthly recurring revenue stands at $250k MRR."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.MONTHLY


def test_validate_quarterly_revenue():
    """Quarterly revenue classifies as QUARTERLY."""
    cand = CandidateCompany(
        name="FinScale",
        description="Fintech platform",
        revenue_amount_usd=1_200_000,
        financial_source_quote="Achieved Q4 quarterly revenue of $1.2M."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.QUARTERLY


def test_validate_lifetime_revenue():
    """Cumulative lifetime sales/revenue classifies as LIFETIME."""
    cand = CandidateCompany(
        name="CommerceGrid",
        description="E-commerce infra",
        revenue_amount_usd=4_500_000,
        financial_source_quote="Surpassed $4.5M in lifetime sales and cumulative GMV since launch."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.LIFETIME


def test_validate_generic_revenue_no_period():
    """Generic revenue snippet without period defaults to UNKNOWN."""
    cand = CandidateCompany(
        name="GenTech",
        description="Software development",
        revenue_amount_usd=2_000_000,
        financial_source_quote="The company generated $2M in revenue last year."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.UNKNOWN


# ==============================================================================
# SECTION 2: ANTI-HALLUCINATION & ANTI-UPGRADE VERIFICATION
# ==============================================================================

def test_anti_upgrade_model_claims_arr_snippet_is_generic():
    """If model outputs revenue_period='ARR' but snippet has no recurring proof, downgrade to UNKNOWN."""
    cand = CandidateCompany(
        name="OptimisticAI",
        description="AI assistant",
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",  # Model claimed ARR
        financial_source_quote="OptimisticAI generated $3M in total revenue."  # No recurring proof
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.UNKNOWN
    assert "Uncorroborated" in reason or "downgraded" in reason.lower()


def test_anti_upgrade_saas_classification_does_not_infer_arr():
    """Being a SaaS company does not convert generic revenue into ARR."""
    cand = CandidateCompany(
        name="PureSaaS",
        description="B2B Cloud SaaS platform for enterprises",
        industry="SaaS",
        revenue_amount_usd=2_000_000,
        financial_source_quote="PureSaaS brought in $2M in sales in 2025."
    )
    period, _ = validate_and_classify_revenue(cand)
    assert period != RevenuePeriod.ARR
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert "Policy C requires verified ARR" in audit.rejection_reason


def test_anti_upgrade_search_query_cannot_inject_arr():
    """Query having ARR cannot qualify snippet lacking ARR evidence."""
    cand = CandidateCompany(
        name="SearchBiased",
        description="Software tool",
        revenue_amount_usd=3_000_000,
        financial_source_quote="SearchBiased reported $3,000,000 in revenue."
    )
    # The snippet only says revenue, even if a search query asked for 'ARR'
    period, _ = validate_and_classify_revenue(cand)
    assert period != RevenuePeriod.ARR


def test_contradiction_model_claims_arr_snippet_says_monthly():
    """Model claims ARR but source snippet proves monthly revenue -> overrides to MONTHLY."""
    cand = CandidateCompany(
        name="ContradictCo",
        description="App",
        revenue_amount_usd=300_000,
        revenue_period="ARR",
        financial_source_quote="Reached monthly revenue of $300,000."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.MONTHLY
    assert "Contradiction" in reason

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.financial_type == "MONTHLY_REVENUE"
    assert "monthly revenue" in audit.rejection_reason.lower()


def test_contradiction_model_claims_arr_snippet_says_lifetime():
    """Model claims ARR but source snippet proves lifetime revenue -> overrides to LIFETIME."""
    cand = CandidateCompany(
        name="LifetimeCo",
        description="Marketplace",
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="Crossed $3M in total lifetime sales since inception."
    )
    period, reason = validate_and_classify_revenue(cand)
    assert period == RevenuePeriod.LIFETIME
    assert "Contradiction" in reason

    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.financial_type == "LIFETIME_REVENUE"


# ==============================================================================
# SECTION 3: POLICY C FINANCIAL QUALIFICATION BY REVENUE PERIOD
# ==============================================================================

def test_policy_c_case_a_arr_qualifies():
    """Case A: $3M ARR with corroborating snippet QUALIFIES under Policy C."""
    cand = _attach_mock_provenance(CandidateCompany(
        name="ValidARRCo",
        description="Cloud infrastructure",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="Company hit $3M ARR in January 2026.",
        financial_evidence_date="2026-01-15",
        founder_name="Greta Weber"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "ARR"
    assert audit.revenue_period == "ARR"
    assert "$3,000,000 ARR" in audit.financial_figure_used
    assert "Policy C" in audit.audit_summary


def test_policy_c_case_b_annual_recurring_revenue_qualifies():
    """Case B: Annual recurring revenue wording QUALIFIES under Policy C."""
    cand = _attach_mock_provenance(CandidateCompany(
        name="RecurCo",
        description="Billing engine",
        industry="Fintech",
        hq_country="France",
        is_tech_platform=True,
        revenue_amount_usd=2_500_000,
        revenue_period="ARR",
        financial_source_quote="Recorded $2.5M in annual recurring revenue.",
        financial_evidence_date="2026-02-01",
        founder_name="Luc Besson"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"
    assert audit.financial_type == "ARR"
    assert audit.revenue_period == "ARR"


def test_policy_c_case_c_annual_revenue_rejected():
    """Case C: Non-recurring annual revenue ($3M) is REJECTED as ANNUAL_REVENUE."""
    cand = CandidateCompany(
        name="AnnualAccountingCo",
        description="Accounting platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        financial_source_quote="Annual revenue reached $3M for the financial year.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.financial_type == "ANNUAL_REVENUE"
    assert audit.revenue_period == "ANNUAL"
    assert "is not verified as recurring ARR" in audit.rejection_reason


def test_policy_c_case_d_generic_revenue_rejected():
    """Case D: Generic revenue ($3M) without period is REJECTED."""
    cand = CandidateCompany(
        name="GenericRevCo",
        description="Data analytics",
        industry="SaaS",
        hq_country="Netherlands",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        financial_source_quote="The startup pulled in $3M revenue last year.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.financial_type == "UNKNOWN"
    assert audit.revenue_period == "UNKNOWN"
    assert "Policy C requires verified ARR" in audit.rejection_reason


def test_policy_c_case_e_monthly_revenue_rejected():
    """Case E: Monthly revenue ($250k) is REJECTED as MONTHLY_REVENUE."""
    cand = CandidateCompany(
        name="MonthlyCo",
        description="Subscription app",
        industry="SaaS",
        hq_country="Spain",
        is_tech_platform=True,
        revenue_amount_usd=250_000,
        financial_source_quote="Generates $250,000 monthly revenue.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.financial_type == "MONTHLY_REVENUE"
    assert audit.revenue_period == "MONTHLY"
    assert "monthly revenue" in audit.rejection_reason.lower()


def test_policy_c_case_f_quarterly_revenue_rejected():
    """Case F: Quarterly revenue ($1.5M) is REJECTED as QUARTERLY_REVENUE."""
    cand = CandidateCompany(
        name="QuarterlyCo",
        description="Fintech app",
        industry="Fintech",
        hq_country="Sweden",
        is_tech_platform=True,
        revenue_amount_usd=1_500_000,
        financial_source_quote="Reported quarterly revenue of $1.5M in Q3.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.financial_type == "QUARTERLY_REVENUE"
    assert audit.revenue_period == "QUARTERLY"
    assert "quarterly revenue" in audit.rejection_reason.lower()


def test_policy_c_case_g_lifetime_revenue_rejected():
    """Case G: Lifetime revenue ($4M) is REJECTED as LIFETIME_REVENUE."""
    cand = CandidateCompany(
        name="LifetimeSalesCo",
        description="E-commerce SaaS",
        industry="Commerce",
        hq_country="UK",
        is_tech_platform=True,
        revenue_amount_usd=4_000_000,
        financial_source_quote="Generated over $4M in cumulative lifetime revenue.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.financial_type == "LIFETIME_REVENUE"
    assert audit.revenue_period == "LIFETIME"
    assert "lifetime" in audit.rejection_reason.lower()


def test_policy_c_case_h_ttm_revenue_rejected():
    """Case H: TTM revenue ($3M) is REJECTED as TTM_REVENUE."""
    cand = CandidateCompany(
        name="TTMCo",
        description="Supply chain platform",
        industry="Logistics",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        financial_source_quote="Trailing twelve months (TTM) revenue was $3M.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.qualification_status == "REJECTED"
    assert audit.financial_type == "TTM_REVENUE"
    assert audit.revenue_period == "TTM"
    assert "is not verified as recurring ARR" in audit.rejection_reason


# ==============================================================================
# SECTION 4: ARR BOUNDARY TESTING ($1M - $5M)
# ==============================================================================

def test_arr_boundary_999999_rejected():
    cand = CandidateCompany(
        name="FloorMinusOne",
        description="Platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=999_999,
        revenue_period="ARR",
        financial_source_quote="Crossed $999,999 in ARR.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert "below $1M floor" in audit.rejection_reason


def test_arr_boundary_1000000_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="FloorExact",
        description="Platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=1_000_000,
        revenue_period="ARR",
        financial_source_quote="Crossed exactly $1,000,000 in ARR.",
        financial_evidence_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"


def test_arr_boundary_5000000_accepted():
    cand = _attach_mock_provenance(CandidateCompany(
        name="CeilingExact",
        description="Platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=5_000_000,
        revenue_period="ARR",
        financial_source_quote="Reached $5,000,000 in ARR.",
        financial_evidence_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is True
    assert audit.qualification_status == "ACCEPTED"


def test_arr_boundary_5000001_rejected():
    cand = CandidateCompany(
        name="CeilingPlusOne",
        description="Platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=5_000_001,
        revenue_period="ARR",
        financial_source_quote="Reached $5,000,001 in ARR.",
        financial_evidence_date="2026-01-01"
    )
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert "exceeds $5M ceiling" in audit.rejection_reason


# ==============================================================================
# SECTION 5: POLICY D SCALE CONFLICT WITH VALID ARR
# ==============================================================================

def test_policy_d_scale_conflict_with_arr():
    """Qualifying ARR ($3M) but excessive cumulative funding ($15M) triggers Policy D."""
    cand = _attach_mock_provenance(CandidateCompany(
        name="ScaleConflictCo",
        description="Enterprise SaaS",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000,
        revenue_period="ARR",
        financial_source_quote="ARR reached $3,000,000.",
        total_cumulative_funding_usd=15_000_000,
        revenue_date="2026-01-01"
    ))
    qualified, audit = evaluate_financial_qualification(cand)
    assert qualified is False
    assert audit.rejection_stage == "FINANCIAL"
    assert "Policy D" in audit.rejection_reason
    assert audit.revenue_period == "ARR"
    assert audit.evidence.financial.revenue_period == "ARR"


# ==============================================================================
# SECTION 6: DATA MODEL SERIALIZATION & TRACEABILITY
# ==============================================================================

def test_model_serialization_with_revenue_period():
    """CandidateCompany, CandidateAuditRecord, FinancialEvidence, LeadRecord serialize revenue_period."""
    cand = CandidateCompany(
        name="ModelTestCo",
        description="Test",
        revenue_amount_usd=2_500_000,
        revenue_period=RevenuePeriod.ARR,
        financial_source_quote="Has $2.5M in ARR."
    )
    cand_dict = cand.model_dump()
    assert cand_dict["revenue_period"] == "ARR"

    fin_ev = FinancialEvidence(
        amount_usd=2_500_000,
        financial_type="ARR",
        revenue_period="ARR",
        source_url="https://example.com/financials",
        source_quote="Has $2.5M in ARR."
    )
    fin_dict = fin_ev.model_dump()
    assert fin_dict["revenue_period"] == "ARR"

    lead = LeadRecord(
        company_name="ModelTestCo",
        description="Test",
        industry="SaaS",
        ceo_cofounder_name="Jane Doe (CEO)",
        verified_email="jane@modeltest.co",
        email_source_url="https://modeltest.co/contact",
        hq_location="Berlin, Germany",
        financial_signal="$2,500,000 ARR",
        revenue_period="ARR",
        recency_basis="Recent",
        us_presence_result="PASSED_NON_US_HQ",
        financial_confidence="HIGH",
        recency_audit_note="Verified"
    )
    lead_dict = lead.model_dump()
    assert lead_dict["revenue_period"] == "ARR"
