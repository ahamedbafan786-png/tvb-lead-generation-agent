"""
tests/test_tavily_budget_control.py
Deterministic unit tests for Tavily budget limits, call breakdown telemetry,
cache safety (zero counter burn on hits), and pipeline search-order pre-filtering.
"""

import io
import json
from unittest.mock import MagicMock, patch
import pytest

from core.discovery import (
    search_tavily, get_total_tavily_calls, get_tavily_call_breakdown,
    reset_tavily_search_count, clear_tavily_cache, TavilyBudgetExhaustedError
)
from core.config import TAVILY_MAX_CALLS_PER_RUN, MAX_FINANCIAL_USD
from core.models import CandidateCompany
from core.pipeline import run_lead_pipeline


@pytest.fixture(autouse=True)
def reset_state():
    clear_tavily_cache()
    reset_tavily_search_count()
    yield
    clear_tavily_cache()
    reset_tavily_search_count()


def make_mock_tavily_response(results=None):
    if results is None:
        results = [{"title": "Test Title", "url": "https://example.com", "content": "Sample snippet", "score": 0.9}]
    mock_resp = io.BytesIO(json.dumps({"results": results}).encode("utf-8"))
    mock_resp.status = 200
    return mock_resp


def test_hard_budget_limit_exhaustion(monkeypatch):
    """Verify that reaching the per-run budget raises TavilyBudgetExhaustedError."""
    monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 3)

    with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: make_mock_tavily_response()):
        # 3 calls should succeed
        res1 = search_tavily("query 1", api_key="mock_key")
        res2 = search_tavily("query 2", api_key="mock_key")
        res3 = search_tavily("query 3", api_key="mock_key")
        assert len(res1) == 1
        assert len(res2) == 1
        assert len(res3) == 1
        assert get_total_tavily_calls() == 3

        # 4th call must raise TavilyBudgetExhaustedError
        with pytest.raises(TavilyBudgetExhaustedError) as exc_info:
            search_tavily("query 4", api_key="mock_key")
        assert "Per-run search budget reached" in str(exc_info.value)
        assert "TAVILY_BUDGET_EXHAUSTED" in str(exc_info.value)
        # Counter should remain 3
        assert get_total_tavily_calls() == 3


def test_independent_category_counters():
    """Verify discovery vs financial_followup vs other calls are tracked separately."""
    with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: make_mock_tavily_response()):
        search_tavily("discovery query 1", api_key="mock_key", category="discovery")
        search_tavily("discovery query 2", api_key="mock_key", category="discovery")
        search_tavily("followup query 1", api_key="mock_key", category="financial_followup")
        search_tavily("followup query 2", api_key="mock_key", category="financial_followup")
        search_tavily("followup query 3", api_key="mock_key", category="financial_followup")
        search_tavily("other query 1", api_key="mock_key", category="other")

    breakdown = get_tavily_call_breakdown()
    assert breakdown["discovery_calls"] == 2
    assert breakdown["financial_followup_calls"] == 3
    assert breakdown["other_calls"] == 1
    assert breakdown["total_calls"] == 6
    assert get_total_tavily_calls() == 6


def test_cache_hit_zero_counter_burn():
    """Verify that cached queries do not increment any counter or invoke the network."""
    with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: make_mock_tavily_response()) as mock_net:
        # 1st call: network hit
        res1 = search_tavily("B2B SaaS Seed Series A", api_key="mock_key", category="discovery")
        assert len(res1) == 1
        assert mock_net.call_count == 1
        assert get_total_tavily_calls() == 1

        # 2nd call: exact query cache hit
        res2 = search_tavily("B2B SaaS Seed Series A", api_key="mock_key", category="discovery")
        assert len(res2) == 1
        assert mock_net.call_count == 1
        assert get_total_tavily_calls() == 1

        # 3rd call: case-insensitive & trimmed whitespace cache hit
        res3 = search_tavily("   b2b saas seed series a   ", api_key="mock_key", category="discovery")
        assert len(res3) == 1
        assert mock_net.call_count == 1
        assert get_total_tavily_calls() == 1

        breakdown = get_tavily_call_breakdown()
        assert breakdown["discovery_calls"] == 1
        assert breakdown["total_calls"] == 1


def test_semantically_different_query_misses_cache():
    """Verify different queries trigger separate network calls."""
    with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: make_mock_tavily_response()) as mock_net:
        search_tavily("Fintech Seed London", api_key="mock_key", category="discovery")
        search_tavily("Healthtech Series A Berlin", api_key="mock_key", category="discovery")

    assert mock_net.call_count == 2
    assert get_total_tavily_calls() == 2


def test_pipeline_skips_followup_for_us_headquarters():
    """Verify pipeline pre-filters US candidates before triggering targeted follow-up."""
    us_cand = CandidateCompany(
        name="US FinTech Inc",
        description="US payment processor",
        industry="Fintech",
        hq_country="United States",
        hq_city="San Francisco",
        is_tech_platform=True,
        total_cumulative_funding_usd=2_000_000.0
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[us_cand]):
                with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                    summary = run_lead_pipeline(
                        api_key="mock_gemini",
                        target_count=1,
                        max_queries=1,
                        tavily_api_key="mock_tavily"
                    )

    # Follow-up should NOT have been called
    mock_followup.assert_not_called()
    assert len(summary.audit_log) == 1
    rec = summary.audit_log[0]
    assert rec.qualification_status == "REJECTED"
    assert rec.rejection_stage == "NON_US"
    assert "Disqualified before follow-up" in rec.audit_summary


def test_pipeline_skips_followup_for_non_tech_platform():
    """Verify pipeline pre-filters non-tech entities (e.g. consulting, VC, recruiting)."""
    agency_cand = CandidateCompany(
        name="Munich Talent Agency",
        description="Recruitment and staffing agency for tech companies",
        industry="Staffing",
        hq_country="Germany",
        hq_city="Munich",
        is_tech_platform=False,
        total_cumulative_funding_usd=1_500_000.0
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[agency_cand]):
                with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                    summary = run_lead_pipeline(
                        api_key="mock_gemini",
                        target_count=1,
                        max_queries=1,
                        tavily_api_key="mock_tavily"
                    )

    mock_followup.assert_not_called()
    assert len(summary.audit_log) == 1
    rec = summary.audit_log[0]
    assert rec.qualification_status == "REJECTED"
    assert rec.rejection_stage == "TECH_PLATFORM"
    assert "Disqualified before follow-up" in rec.audit_summary


def test_pipeline_skips_followup_for_acquired_or_defunct():
    """Verify pipeline pre-filters acquired or defunct companies."""
    acquired_cand = CandidateCompany(
        name="OldCloud Ltd",
        description="Cloud software acquired by mega corp",
        industry="Cloud",
        hq_country="UK",
        hq_city="London",
        is_tech_platform=True,
        acquisition_or_closure_discovered=True,
        total_cumulative_funding_usd=2_000_000.0
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[acquired_cand]):
                with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                    summary = run_lead_pipeline(
                        api_key="mock_gemini",
                        target_count=1,
                        max_queries=1,
                        tavily_api_key="mock_tavily"
                    )

    mock_followup.assert_not_called()
    assert len(summary.audit_log) == 1
    rec = summary.audit_log[0]
    assert rec.qualification_status == "REJECTED"
    assert rec.rejection_stage == "FINANCIAL"
    assert "Acquired or closed" in rec.audit_summary


def test_pipeline_skips_followup_for_overfunded_company():
    """Verify pipeline pre-filters companies with cumulative funding > $5M."""
    overfunded_cand = CandidateCompany(
        name="MegaScale AI",
        description="Massive AI infrastructure",
        industry="AI",
        hq_country="France",
        hq_city="Paris",
        is_tech_platform=True,
        total_cumulative_funding_usd=25_000_000.0
    )

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=[overfunded_cand]):
                with patch("core.pipeline.execute_targeted_funding_follow_up") as mock_followup:
                    summary = run_lead_pipeline(
                        api_key="mock_gemini",
                        target_count=1,
                        max_queries=1,
                        tavily_api_key="mock_tavily"
                    )

    mock_followup.assert_not_called()
    assert len(summary.audit_log) == 1
    rec = summary.audit_log[0]
    assert rec.qualification_status == "REJECTED"
    assert rec.rejection_stage == "FINANCIAL"
    assert "> $5,000,000 cap" in rec.audit_summary


def test_pipeline_exit_reason_on_tavily_budget_exhausted():
    """Verify pipeline gracefully halts with TAVILY_BUDGET_EXHAUSTED exit reason."""
    with patch("core.pipeline.generate_queries", return_value=["query 1", "query 2"]):
        with patch("core.pipeline.search_grounded_candidates", side_effect=TavilyBudgetExhaustedError("Budget cap reached")):
            summary = run_lead_pipeline(
                api_key="mock_gemini",
                target_count=5,
                max_queries=2,
                tavily_api_key="mock_tavily"
            )

    assert summary.exit_reason == "TAVILY_BUDGET_EXHAUSTED"
    assert summary.tavily_budget_remaining == TAVILY_MAX_CALLS_PER_RUN
