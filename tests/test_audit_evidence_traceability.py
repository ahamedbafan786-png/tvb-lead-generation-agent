"""
tests/test_audit_evidence_traceability.py
Comprehensive test suite verifying financial and qualification source evidence
traceability, anti-hallucination verification, bounded sources, and structured audit outputs.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from core.models import (
    CandidateCompany, LeadRecord, CandidateAuditRecord,
    AuditEvidence, EvidenceItem, FinancialEvidence, RecencyEvidence,
    HQEvidence, USPresenceEvidence, TechPlatformEvidence, FounderEvidence,
    EmailEvidence, PipelineSummary, FunnelMetrics, ContactPathResult
)
from core.extractor import (
    execute_targeted_funding_follow_up, evaluate_financial_qualification,
    passes_non_us_criterion, passes_tech_platform_criterion
)
from core.pipeline import run_lead_pipeline


def test_financial_source_url_and_quote_survive_follow_up():
    """Verify that trusted source URL and quote survive follow-up and populate on candidate."""
    cand = CandidateCompany(
        name="SourceCorp",
        website_url="https://sourcecorp.eu",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Alice Smith",
        founder_title="CEO"
    )

    fake_tavily_results = [
        {
            "title": "SourceCorp raises €2.5M Seed round",
            "url": "https://techcrunch.com/2025/11/sourcecorp-seed",
            "content": "Berlin-based SourceCorp secured $2.7M in seed funding in November 2025."
        }
    ]

    mock_gemini_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "total_cumulative_funding_usd": 2700000,
        "latest_round_amount_usd": 2700000,
        "latest_round_name": "Seed",
        "latest_round_date": "2025-11-15",
        "financial_evidence_date": "2025-11-15",
        "financial_evidence_snippet": "Berlin-based SourceCorp secured $2.7M in seed funding in November 2025.",
        "financial_source_url": "https://techcrunch.com/2025/11/sourcecorp-seed",
        "financial_source_quote": "Berlin-based SourceCorp secured $2.7M in seed funding in November 2025.",
        "revenue_amount_usd": None,
        "revenue_date": None,
        "subsequent_rounds_discovered": False,
        "financial_conflict_detected": False,
        "acquisition_or_closure_discovered": False,
        "us_presence_detected": False,
        "us_presence_category": None,
        "us_locations": [],
        "us_presence_evidence": None,
        "financial_followup_complete": True
    })

    with patch("core.extractor.generate_with_fallback", return_value=mock_response.text):
        with patch("core.discovery.search_tavily", return_value=fake_tavily_results):
            res_cand = execute_targeted_funding_follow_up(mock_gemini_client, cand, tavily_api_key="mock_key")

    assert res_cand.financial_source_url == "https://techcrunch.com/2025/11/sourcecorp-seed"
    assert "secured $2.7M" in res_cand.financial_source_quote
    assert len(res_cand.supporting_evidence) >= 1
    assert res_cand.supporting_evidence[0].source_url == "https://techcrunch.com/2025/11/sourcecorp-seed"


def test_anti_hallucination_rejects_invented_url():
    """If Gemini invents a URL not in Tavily search results, it must be rejected."""
    cand = CandidateCompany(
        name="HallucinateCorp",
        website_url="https://hallucinatecorp.de",
        hq_country="Germany",
        is_tech_platform=True
    )

    fake_tavily_results = [
        {
            "title": "HallucinateCorp Funding",
            "url": "https://actual-news.de/funding/hallucinatecorp",
            "content": "HallucinateCorp closed $2.5M in Series Seed in November 2025."
        }
    ]

    mock_gemini_client = MagicMock()
    # Model invents fake url
    fake_json = json.dumps({
        "total_cumulative_funding_usd": 2500000,
        "latest_round_amount_usd": 2500000,
        "latest_round_name": "Seed",
        "latest_round_date": "2025-11-01",
        "financial_evidence_date": "2025-11-01",
        "financial_evidence_snippet": "HallucinateCorp closed $2.5M in Series Seed in November 2025.",
        "financial_source_url": "https://fake-hallucinated-site.com/fake-article",
        "financial_source_quote": "HallucinateCorp closed $2.5M in Series Seed in November 2025.",
        "revenue_amount_usd": None,
        "revenue_date": None,
        "subsequent_rounds_discovered": False,
        "financial_conflict_detected": False,
        "acquisition_or_closure_discovered": False,
        "us_presence_detected": False,
        "us_presence_category": None,
        "us_locations": [],
        "us_presence_evidence": None,
        "financial_followup_complete": True
    })

    with patch("core.extractor.generate_with_fallback", return_value=fake_json):
        with patch("core.discovery.search_tavily", return_value=fake_tavily_results):
            res_cand = execute_targeted_funding_follow_up(mock_gemini_client, cand, tavily_api_key="mock_key")

    # The invented URL must NOT be accepted! It should fallback to the real URL matching the snippet, or None
    assert res_cand.financial_source_url != "https://fake-hallucinated-site.com/fake-article"
    # In this case, snippet matches actual-news.de so it safely mapped to the verified Tavily URL
    assert res_cand.financial_source_url == "https://actual-news.de/funding/hallucinatecorp"


def test_anti_hallucination_rejects_unmatched_invented_url():
    """If Gemini invents a URL and the snippet doesn't match any result, financial_source_url is None."""
    cand = CandidateCompany(
        name="PhantomCorp",
        website_url="https://phantomcorp.de",
        hq_country="Germany",
        is_tech_platform=True
    )

    fake_tavily_results = [
        {
            "title": "Random unrelated article",
            "url": "https://actual-news.de/other",
            "content": "Nothing about funding here."
        }
    ]

    mock_gemini_client = MagicMock()
    fake_json = json.dumps({
        "total_cumulative_funding_usd": 2500000,
        "latest_round_amount_usd": 2500000,
        "latest_round_name": "Seed",
        "latest_round_date": "2025-11-01",
        "financial_evidence_date": "2025-11-01",
        "financial_evidence_snippet": "PhantomCorp got $2.5M completely fabricated snippet.",
        "financial_source_url": "https://fake-hallucinated-site.com/phantom",
        "financial_source_quote": "PhantomCorp got $2.5M completely fabricated snippet.",
        "financial_followup_complete": True
    })

    with patch("core.extractor.generate_with_fallback", return_value=fake_json):
        with patch("core.discovery.search_tavily", return_value=fake_tavily_results):
            res_cand = execute_targeted_funding_follow_up(mock_gemini_client, cand, tavily_api_key="mock_key")

    assert res_cand.financial_source_url is None


def test_supporting_sources_bounded_max_5():
    """Verify that supporting evidence list is strictly bounded to at most 5 items."""
    cand = CandidateCompany(
        name="BoundedCorp",
        website_url="https://boundedcorp.de",
        hq_country="Germany"
    )

    fake_tavily_results = [
        {
            "title": f"Article {i}",
            "url": f"https://news-{i}.com/article",
            "content": f"BoundedCorp funding article number {i} with some details."
        }
        for i in range(10)
    ]

    mock_gemini_client = MagicMock()
    fake_json = json.dumps({
        "total_cumulative_funding_usd": 3000000,
        "latest_round_amount_usd": 3000000,
        "latest_round_name": "Seed",
        "latest_round_date": "2025-11-01",
        "financial_evidence_date": "2025-11-01",
        "financial_evidence_snippet": "BoundedCorp funding article number 0",
        "financial_source_url": "https://news-0.com/article",
        "financial_source_quote": "BoundedCorp funding article number 0",
        "financial_followup_complete": True
    })

    with patch("core.extractor.generate_with_fallback", return_value=fake_json):
        with patch("core.discovery.search_tavily", return_value=fake_tavily_results):
            res_cand = execute_targeted_funding_follow_up(mock_gemini_client, cand, tavily_api_key="mock_key")

    assert len(res_cand.supporting_evidence) <= 5
    for item in res_cand.supporting_evidence:
        assert item.source_url.startswith("https://news-")
        assert len(item.source_quote) <= 250


def test_evaluate_financial_qualification_populates_structured_audit_evidence():
    """Verify evaluate_financial_qualification creates complete structured AuditEvidence."""
    cand = CandidateCompany(
        name="AuditedFinCorp",
        website_url="https://auditedfin.de",
        hq_country="Germany",
        hq_city="Berlin",
        industry="SaaS",
        is_tech_platform=True,
        founder_name="Bob Miller",
        founder_title="Founder & CEO",
        total_cumulative_funding_usd=2500000,
        financial_evidence_date="2025-11-01",
        latest_round_date="2025-11-01",
        financial_source_url="https://trustednews.com/auditedfin",
        financial_source_quote="AuditedFin raised $2.5M in November 2025.",
        trusted_source_urls=["https://trustednews.com/auditedfin"],
        trusted_source_text="AuditedFin raised $2.5M in November 2025.",
        financial_followup_complete=True
    )

    is_ok, audit_record = evaluate_financial_qualification(cand)

    assert is_ok is True
    assert audit_record.qualification_status == "ACCEPTED"
    assert audit_record.financial_source_url == "https://trustednews.com/auditedfin"
    assert audit_record.financial_source_quote == "AuditedFin raised $2.5M in November 2025."
    assert audit_record.evidence is not None

    evidence = audit_record.evidence
    assert isinstance(evidence, AuditEvidence)
    assert evidence.financial is not None
    assert evidence.financial.amount_usd == 2500000
    assert evidence.financial.financial_type == "CUMULATIVE_FUNDING"
    assert evidence.financial.source_url == "https://trustednews.com/auditedfin"
    assert evidence.financial.verification_status == "VERIFIED"

    assert evidence.recency is not None
    assert evidence.recency.evidence_date == "2025-11-01"
    assert evidence.recency.is_recent is True
    assert evidence.recency.verification_status == "VERIFIED"

    assert evidence.hq is not None
    assert evidence.hq.country == "Germany"
    assert evidence.hq.city == "Berlin"
    assert evidence.hq.verification_status == "VERIFIED"

    assert evidence.tech_platform is not None
    assert evidence.tech_platform.is_tech_platform is True
    assert evidence.tech_platform.verification_status == "VERIFIED"

    assert evidence.founder is not None
    assert evidence.founder.founder_name == "Bob Miller"
    assert evidence.founder.verification_status == "VERIFIED"


def test_unverified_status_when_financial_source_url_missing():
    """If financial_source_url is missing, candidate must be REJECTED and financial evidence verification_status must be UNVERIFIED."""
    cand = CandidateCompany(
        name="NoUrlCorp",
        website_url="https://nourlcorp.de",
        hq_country="Germany",
        total_cumulative_funding_usd=2000000,
        financial_evidence_date="2025-11-01",
        financial_source_url=None,
        financial_followup_complete=True
    )

    is_ok, audit_record = evaluate_financial_qualification(cand)
    assert is_ok is False
    assert audit_record.qualification_status == "REJECTED"
    assert audit_record.rejection_stage == "FINANCIAL"
    assert audit_record.evidence is not None
    assert audit_record.evidence.financial.verification_status == "UNVERIFIED"


def test_us_presence_evidence_preserves_rejection():
    """Verify US presence evidence preserves category, locations, quote, and REJECTED status."""
    cand = CandidateCompany(
        name="UsBranchCorp",
        website_url="https://usbranch.eu",
        hq_country="France",
        is_tech_platform=True,
        total_cumulative_funding_usd=2000000,
        financial_evidence_date="2025-11-01",
        financial_source_url="https://news.com/usbranch",
        financial_followup_complete=True,
        us_presence_detected=True,
        us_presence_category="US_OFFICE",
        us_locations=["San Francisco, CA"],
        us_presence_evidence="Maintains a US office in San Francisco",
        us_presence_source_url="https://usbranch.eu/contact",
        us_presence_source_quote="Maintains a US office in San Francisco"
    )

    non_us_ok, result_label, result_evidence = passes_non_us_criterion(cand)
    assert non_us_ok is False

    is_ok, audit_record = evaluate_financial_qualification(cand)
    assert audit_record.evidence.us_presence is not None
    assert audit_record.evidence.us_presence.category in {"US_OFFICE", "US_OPERATIONAL_OFFICE"}
    assert "San Francisco, CA" in audit_record.evidence.us_presence.locations
    assert audit_record.evidence.us_presence.verification_status == "REJECTED"


def test_lead_record_and_candidate_audit_record_json_serialization():
    """Verify LeadRecord and CandidateAuditRecord with AuditEvidence serialize cleanly to JSON and dict."""
    ev = AuditEvidence(
        financial=FinancialEvidence(
            amount_usd=2500000,
            financial_type="CUMULATIVE_FUNDING",
            source_url="https://news.com/series-a",
            source_quote="Raised $2.5M in Series A",
            evidence_date="2025-11-10",
            verification_status="VERIFIED"
        ),
        recency=RecencyEvidence(
            evidence_date="2025-11-10",
            recency_basis="Recent (round on 2025-11-10)",
            source_url="https://news.com/series-a",
            is_recent=True,
            verification_status="VERIFIED"
        ),
        hq=HQEvidence(
            country="Germany",
            city="Munich",
            source_url="https://munich-tech.de",
            source_quote="Munich, Germany",
            verification_status="VERIFIED"
        ),
        us_presence=USPresenceEvidence(
            category="NONE",
            locations=[],
            source_url="https://munich-tech.de/about",
            verification_status="VERIFIED"
        ),
        tech_platform=TechPlatformEvidence(
            is_tech_platform=True,
            platform_type="Enterprise SaaS",
            source_url="https://munich-tech.de",
            verification_status="VERIFIED"
        ),
        founder=FounderEvidence(
            founder_name="Stefan Mueller",
            founder_title="CEO",
            source_url="https://munich-tech.de/team",
            verification_status="VERIFIED"
        ),
        email=EmailEvidence(
            email="stefan@munich-tech.de",
            source_url="https://munich-tech.de/impressum",
            attribution_result="ACCEPTED_RULE_1",
            verification_status="VERIFIED"
        ),
        supporting_sources=[
            EvidenceItem(
                evidence_type="FINANCIAL_SOURCE",
                title="Funding News",
                source_url="https://news.com/series-a",
                source_quote="Raised $2.5M in Series A",
                extracted_value="Funding News",
                verification_status="VERIFIED"
            )
        ]
    )

    lead = LeadRecord(
        company_name="MunichTech",
        description="Cloud enterprise solutions",
        industry="Enterprise SaaS",
        ceo_cofounder_name="Stefan Mueller (CEO)",
        verified_email="stefan@munich-tech.de",
        email_source_url="https://munich-tech.de/impressum",
        hq_location="Munich, Germany",
        financial_signal="$2,500,000 USD (Cumulative)",
        financial_evidence_date="2025-11-10",
        latest_round_date="2025-11-10",
        recency_basis="Recent (round on 2025-11-10)",
        us_presence_result="NON_US_VERIFIED",
        us_presence_evidence=None,
        financial_confidence="HIGH",
        recency_audit_note="Qualified under Policy A",
        financial_source_url="https://news.com/series-a",
        financial_source_quote="Raised $2.5M in Series A",
        founder_source_url="https://munich-tech.de/team",
        founder_source_quote="Stefan Mueller (CEO)",
        tech_source_url="https://munich-tech.de",
        tech_source_quote="Cloud enterprise solutions",
        us_presence_source_url="https://munich-tech.de/about",
        evidence=ev
    )

    lead_dict = lead.model_dump()
    lead_json = lead.model_dump_json()
    assert isinstance(lead_dict, dict)
    assert isinstance(lead_json, str)
    parsed = json.loads(lead_json)
    assert parsed["company_name"] == "MunichTech"
    assert parsed["financial_source_url"] == "https://news.com/series-a"
    assert parsed["evidence"]["financial"]["amount_usd"] == 2500000
    assert parsed["evidence"]["email"]["email"] == "stefan@munich-tech.de"
    assert parsed["evidence"]["email"]["attribution_result"] == "ACCEPTED_RULE_1"


def test_end_to_end_mocked_pipeline_preserves_evidence():
    """Run pipeline end-to-end with mocks and verify LeadRecord and AuditLog retain all evidence."""
    cand = CandidateCompany(
        name="PipelineVerifiedCorp",
        website_url="https://pipelineverified.de",
        description="AI-powered compliance engine",
        industry="B2B SaaS",
        hq_country="Germany",
        hq_city="Berlin",
        is_tech_platform=True,
        founder_name="Greta Thun",
        founder_title="CEO",
        total_cumulative_funding_usd=2200000,
        financial_evidence_date="2025-11-10",
        latest_round_date="2025-11-10",
        financial_followup_complete=True,
        financial_source_url="https://tech-portal.de/pipelineverified",
        financial_source_quote="PipelineVerifiedCorp closed $2.2M Seed in November 2025.",
        trusted_source_urls=["https://tech-portal.de/pipelineverified"],
        trusted_source_text="PipelineVerifiedCorp closed $2.2M Seed in November 2025."
    )

    mock_contact_path = ContactPathResult(
        contact_path_found=True,
        exact_email_found="greta@pipelineverified.de",
        email_source_url="https://pipelineverified.de/contact",
        founder_name_found="Greta Thun",
        founder_title_found="CEO",
        founder_source_url="https://pipelineverified.de/about",
        attribution_context="RULE_1_EXACT_NAME_MATCH"
    )

    with patch("core.pipeline.genai.Client"):
        with patch("core.pipeline.generate_queries", return_value=["test query"]):
            with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
                with patch("core.pipeline.extract_structured_candidates", return_value=[cand]):
                    with patch("core.pipeline.discover_founder_contact_path", return_value=mock_contact_path):
                        with patch("core.pipeline.execute_targeted_funding_follow_up", return_value=cand):
                            with patch("core.pipeline.verify_founder_email_on_site", return_value=("greta@pipelineverified.de", "https://pipelineverified.de/contact")):
                                summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    assert len(summary.leads) == 1
    lead = summary.leads[0]
    assert lead.company_name == "PipelineVerifiedCorp"
    assert lead.verified_email == "greta@pipelineverified.de"
    assert lead.email_source_url == "https://pipelineverified.de/contact"
    assert lead.financial_source_url == "https://tech-portal.de/pipelineverified"
    assert lead.founder_source_url == "https://pipelineverified.de/about"

    assert lead.evidence is not None
    assert lead.evidence.financial.amount_usd == 2200000
    assert lead.evidence.financial.source_url == "https://tech-portal.de/pipelineverified"
    assert lead.evidence.email.email == "greta@pipelineverified.de"
    assert lead.evidence.founder.founder_name == "Greta Thun"
    assert lead.evidence.founder.source_url == "https://pipelineverified.de/about"

    accepted_audit = [a for a in summary.audit_log if a.qualification_status == "ACCEPTED"][0]
    assert accepted_audit.financial_source_url == "https://tech-portal.de/pipelineverified"
    assert accepted_audit.founder_source_url == "https://pipelineverified.de/about"
    assert accepted_audit.evidence is not None
