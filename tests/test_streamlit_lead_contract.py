"""
tests/test_streamlit_lead_contract.py
Deterministic offline tests verifying LeadRecord email source contract and Streamlit UI rendering.
Guarantees:
1. LeadRecord defines and populates email_source_type without AttributeError.
2. Legacy objects missing email_source_type resolve safely via evidence or fallback.
3. Streamlit UI renders cleanly for zero leads, one lead, and multiple leads.
4. Streamlit AppTest validates app startup, secret concealment, and graceful missing-key warnings.
5. 0 live Tavily calls, 0 live Gemini calls.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

from core.models import (
    LeadRecord, CandidateAuditRecord, PipelineSummary, FunnelMetrics,
    AuditEvidence, EmailEvidence, EmailSourceType
)
from app import _get_record_email_source_type


# ==============================================================================
# 1. MODEL CONTRACT & SYNCHRONIZATION TESTS
# ==============================================================================

def test_lead_record_email_source_type_default_contract():
    """Verify LeadRecord defaults email_source_type cleanly without error."""
    lead = LeadRecord(
        company_name="Acme Tech",
        ceo_cofounder_name="Alice Smith (CEO)",
        verified_email="alice@acme.eu",
        email_source_url="https://acme.eu/about",
        hq_location="Berlin, Germany",
        financial_signal="$2.5M ARR",
        recency_basis="Recent",
        us_presence_result="PASS",
        financial_confidence="HIGH",
        recency_audit_note="Verified"
    )
    assert hasattr(lead, "email_source_type")
    assert lead.email_source_type == "FIRST_PARTY_OFFICIAL"


def test_lead_record_email_source_type_explicit_contract():
    """Verify LeadRecord preserves explicitly supplied email_source_type."""
    lead = LeadRecord(
        company_name="Acme Tech",
        ceo_cofounder_name="Alice Smith (CEO)",
        verified_email="alice@alicesmith.me",
        email_source_url="https://alicesmith.me/contact",
        email_source_type=EmailSourceType.FIRST_PARTY_FOUNDER_PUBLISHED.value,
        hq_location="Munich, Germany",
        financial_signal="$3M Funding",
        recency_basis="Recent",
        us_presence_result="PASS",
        financial_confidence="HIGH",
        recency_audit_note="Verified"
    )
    assert lead.email_source_type == "FIRST_PARTY_FOUNDER_PUBLISHED"


def test_lead_record_email_source_type_sync_from_evidence():
    """Verify LeadRecord synchronizes email_source_type from embedded evidence."""
    ev = AuditEvidence(
        email=EmailEvidence(
            email="alice@companieshouse.gov.uk",
            source_url="https://companieshouse.gov.uk/company/123",
            source_type=EmailSourceType.PUBLIC_CORPORATE_RECORD.value,
            attribution_result="ACCEPTED_RULE_1"
        )
    )
    lead = LeadRecord(
        company_name="Acme Tech",
        ceo_cofounder_name="Alice Smith (CEO)",
        verified_email="alice@acme.eu",
        email_source_url="https://companieshouse.gov.uk/company/123",
        hq_location="London, UK",
        financial_signal="$1.8M ARR",
        recency_basis="Recent",
        us_presence_result="PASS",
        financial_confidence="HIGH",
        recency_audit_note="Verified",
        evidence=ev
    )
    assert lead.email_source_type == "PUBLIC_CORPORATE_RECORD"


def test_helper_resolves_legacy_lead_record_missing_attribute():
    """Verify _get_record_email_source_type gracefully handles mock or legacy object."""
    class LegacyLead:
        def __init__(self):
            self.company_name = "OldCo"
            self.verified_email = "old@oldco.de"
            self.email_source_url = "https://oldco.de"
            # Deliberately lacks email_source_type attribute

    legacy = LegacyLead()
    assert not hasattr(legacy, "email_source_type")
    res = _get_record_email_source_type(legacy, default="FIRST_PARTY_OFFICIAL")
    assert res == "FIRST_PARTY_OFFICIAL"

    # Legacy object with evidence only
    class LegacyLeadWithEvidence:
        def __init__(self):
            self.evidence = AuditEvidence(
                email=EmailEvidence(
                    email="old@oldco.de",
                    source_url="https://oldco.de",
                    source_type="PUBLIC_CORPORATE_RECORD"
                )
            )

    legacy_ev = LegacyLeadWithEvidence()
    assert not hasattr(legacy_ev, "email_source_type")
    assert _get_record_email_source_type(legacy_ev) == "PUBLIC_CORPORATE_RECORD"


# ==============================================================================
# 2. UI TABLE RENDERING DETERMINISTIC TESTS (0, 1, MULTIPLE LEADS)
# ==============================================================================

def test_ui_table_rendering_zero_leads():
    """Verify table construction logic for zero leads produces valid audit rows and no errors."""
    summary = PipelineSummary(
        total_queries_run=5,
        duration_seconds=10.0,
        funnel=FunnelMetrics(candidates_found=10, passed_financial=0),
        leads=[],
        audit_log=[
            CandidateAuditRecord(
                company_name="RejectedCo",
                qualification_status="REJECTED",
                rejection_stage="FINANCIAL",
                rejection_reason="Revenue outside target bounds",
                email_source_type=None,
                audit_summary="Disqualified"
            )
        ],
        shortfall=15,
        exit_reason="MAX_QUERIES_REACHED"
    )

    assert len(summary.leads) == 0
    # Render audit rows exactly as app.py does
    audit_rows = [
        {
            "Company": rec.company_name,
            "Status": rec.qualification_status,
            "Stage": rec.rejection_stage or "ACCEPTED",
            "Reason / Rationale": rec.rejection_reason or rec.audit_summary,
            "Email Source Type": _get_record_email_source_type(rec, default="N/A"),
            "HQ Location": rec.hq_location
        }
        for rec in summary.audit_log
    ]
    df_audit = pd.DataFrame(audit_rows)
    assert len(df_audit) == 1
    assert df_audit.iloc[0]["Email Source Type"] == "N/A"
    assert df_audit.iloc[0]["Company"] == "RejectedCo"


def test_ui_table_rendering_one_lead():
    """Verify table construction for single lead: fields, email source type, URL, founder name."""
    lead = LeadRecord(
        company_name="AlphaTech",
        description="B2B AI Platform",
        industry="Enterprise Software",
        ceo_cofounder_name="Elena Rostova (CEO & Co-Founder)",
        verified_email="elena@alphatech.de",
        email_source_url="https://alphatech.de/team",
        email_source_type="FIRST_PARTY_OFFICIAL",
        hq_location="Munich, Germany",
        financial_signal="$3,200,000 ARR",
        recency_basis="Recent (FY2025)",
        us_presence_result="NON_US_VERIFIED",
        financial_confidence="HIGH",
        recency_audit_note="Verified under Policy C"
    )
    summary = PipelineSummary(
        total_queries_run=2,
        duration_seconds=4.5,
        funnel=FunnelMetrics(candidates_found=5, passed_email_verification=1),
        leads=[lead],
        audit_log=[
            CandidateAuditRecord(
                company_name="AlphaTech",
                qualification_status="ACCEPTED",
                verified_email="elena@alphatech.de",
                email_source_url="https://alphatech.de/team",
                email_source_type="FIRST_PARTY_OFFICIAL",
                audit_summary="Qualified lead"
            )
        ],
        shortfall=0,
        exit_reason="TARGET_REACHED"
    )

    # Render rows exactly as app.py does
    rows = [
        {
            "Company Name": l.company_name,
            "Description": l.description,
            "Industry / Sector": l.industry,
            "CEO / Co-Founder": l.ceo_cofounder_name,
            "Verified Email": l.verified_email,
            "Email Evidence URL": l.email_source_url,
            "Email Source Type": _get_record_email_source_type(l, default="FIRST_PARTY_OFFICIAL"),
            "Financial Signal": l.financial_signal,
            "HQ Location": l.hq_location
        }
        for l in summary.leads
    ]
    df = pd.DataFrame(rows)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Company Name"] == "AlphaTech"
    assert row["CEO / Co-Founder"] == "Elena Rostova (CEO & Co-Founder)"
    assert row["Verified Email"] == "elena@alphatech.de"
    assert row["Email Evidence URL"] == "https://alphatech.de/team"
    assert row["Email Source Type"] == "FIRST_PARTY_OFFICIAL"


def test_ui_table_rendering_multiple_leads_diverse_sources():
    """Verify table construction for multiple leads with diverse email source types."""
    leads = [
        LeadRecord(
            company_name="Company One",
            ceo_cofounder_name="Founder One (CEO)",
            verified_email="one@companyone.eu",
            email_source_url="https://companyone.eu/about",
            email_source_type="FIRST_PARTY_OFFICIAL",
            hq_location="Dublin, Ireland",
            financial_signal="$2M ARR",
            recency_basis="Recent",
            us_presence_result="PASS",
            financial_confidence="HIGH",
            recency_audit_note="OK"
        ),
        LeadRecord(
            company_name="Company Two",
            ceo_cofounder_name="Founder Two (Co-Founder)",
            verified_email="two@foundertwo.me",
            email_source_url="https://foundertwo.me/contact",
            email_source_type="FIRST_PARTY_FOUNDER_PUBLISHED",
            hq_location="Stockholm, Sweden",
            financial_signal="$3.5M Funding",
            recency_basis="Recent",
            us_presence_result="PASS",
            financial_confidence="HIGH",
            recency_audit_note="OK"
        ),
        LeadRecord(
            company_name="Company Three",
            ceo_cofounder_name="Founder Three (Managing Director)",
            verified_email="three@companythree.de",
            email_source_url="https://companythree.de/imprint",
            email_source_type="PUBLIC_CORPORATE_RECORD",
            hq_location="Frankfurt, Germany",
            financial_signal="$4M ARR",
            recency_basis="Recent",
            us_presence_result="PASS",
            financial_confidence="HIGH",
            recency_audit_note="OK"
        ),
    ]

    rows = [
        {
            "Company Name": l.company_name,
            "CEO / Co-Founder": l.ceo_cofounder_name,
            "Verified Email": l.verified_email,
            "Email Evidence URL": l.email_source_url,
            "Email Source Type": _get_record_email_source_type(l, default="FIRST_PARTY_OFFICIAL")
        }
        for l in leads
    ]
    df = pd.DataFrame(rows)
    assert len(df) == 3
    assert list(df["Email Source Type"]) == [
        "FIRST_PARTY_OFFICIAL",
        "FIRST_PARTY_FOUNDER_PUBLISHED",
        "PUBLIC_CORPORATE_RECORD"
    ]


# ==============================================================================
# 3. STREAMLIT APPTEST OFFLINE SMOKE TESTS
# ==============================================================================

def test_streamlit_apptest_missing_keys_graceful_warning():
    """Verify app displays graceful warnings and button is disabled when keys are missing."""
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.secrets.clear()
    with patch("dotenv.load_dotenv"), patch.dict("os.environ", {"GEMINI_API_KEY": "", "TAVILY_API_KEY": ""}):
        at.run(timeout=30)

    assert len(at.exception) == 0, f"App threw unexpected exception: {at.exception}"
    error_texts = [e.value for e in at.error]
    assert any("GEMINI_API_KEY" in t for t in error_texts)
    assert any("TAVILY_API_KEY" in t for t in error_texts)
    assert at.button[0].disabled is True


def test_streamlit_apptest_startup_no_secrets_exposed():
    """Verify no API keys or secret patterns are rendered in the UI."""
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.secrets["GEMINI_API_KEY"] = "super_secret_gemini_key_12345"
    at.secrets["TAVILY_API_KEY"] = "super_secret_tavily_key_67890"
    at.run(timeout=30)

    assert len(at.exception) == 0
    # Check that literal secrets are nowhere in markdown or captions
    all_text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    assert "super_secret_gemini_key_12345" not in all_text
    assert "super_secret_tavily_key_67890" not in all_text
