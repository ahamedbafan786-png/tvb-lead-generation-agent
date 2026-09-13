"""
tests/test_tavily_cache_lifecycle.py
Deterministic unit and integration test suite verifying the Tavily in-memory
cache lifecycle, cross-run isolation, within-run deduplication, budget preservation,
and zero-credit consumption.

ZERO live Tavily calls, ZERO live Gemini calls.
"""

import json
import socket
import urllib.request
import urllib.error
import pytest
from unittest.mock import patch, MagicMock

from core.discovery import (
    search_tavily,
    clear_tavily_cache,
    reset_tavily_search_count,
    get_total_tavily_calls,
    get_tavily_call_breakdown,
    get_tavily_cache_size,
    _TAVILY_CACHE,
    TavilyBudgetExhaustedError,
    TavilyTimeoutError,
    TavilyRateLimitError
)
from core.pipeline import run_lead_pipeline
from core.models import CandidateCompany, CandidateAuditRecord, AuditEvidence, ContactPathResult


def _make_mock_urlopen(results=None, status=200):
    mock_resp = MagicMock()
    data = {"results": results or []}
    mock_resp.read.return_value = json.dumps(data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    return mock_resp


@pytest.fixture(autouse=True)
def clean_tavily_state():
    """Ensure clean Tavily counters and cache before and after every test."""
    clear_tavily_cache()
    reset_tavily_search_count()
    yield
    clear_tavily_cache()
    reset_tavily_search_count()


# ==============================================================================
# A. Cache cleared at pipeline start
# ==============================================================================
def test_case_a_cache_cleared_at_pipeline_start():
    """Pre-existing cache entries must be purged at run_lead_pipeline initialization."""
    # Poison the cache with a stale search entry
    _TAVILY_CACHE["stale query:5"] = [{"title": "Stale", "url": "https://stale.com"}]
    assert get_tavily_cache_size() == 1

    with patch("core.pipeline.genai.Client"), \
         patch("core.pipeline.generate_queries", return_value=["test query"]), \
         patch("core.pipeline.search_grounded_candidates", return_value=("", [])):
        summary = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)

    # Cache should have been wiped clean when run_lead_pipeline started
    assert "stale query:5" not in _TAVILY_CACHE


# ==============================================================================
# B. Identical query in same run -> cache hit (zero requests)
# ==============================================================================
def test_case_b_identical_query_in_same_run_cache_hit():
    """Identical queries in the same run must hit cache and trigger zero outbound requests."""
    res = [{"title": "Hit Corp", "url": "https://hitcorp.eu", "content": "SaaS"}]

    with patch("urllib.request.urlopen", return_value=_make_mock_urlopen(res)) as mock_url:
        # First call: cache miss -> network dispatch
        r1 = search_tavily("European SaaS ARR", api_key="mock_key", max_results=5)
        assert r1 == res
        assert mock_url.call_count == 1
        assert get_total_tavily_calls() == 1
        assert get_tavily_cache_size() == 1

        # Second call: cache hit -> zero network dispatch
        r2 = search_tavily("European SaaS ARR", api_key="mock_key", max_results=5)
        assert r2 == res
        assert mock_url.call_count == 1  # Still 1!
        assert get_total_tavily_calls() == 1  # Zero budget slots consumed!

        bd = get_tavily_call_breakdown()
        assert bd["tavily_cache_hits"] == 1
        assert bd["tavily_requests_attempted"] == 1
        assert bd["tavily_requests_succeeded"] == 1


# ==============================================================================
# C. Same query across runs -> fresh request
# ==============================================================================
def test_case_c_same_query_across_runs_fresh_request():
    """Across distinct pipeline runs, the cache must not leak and a fresh request is dispatched."""
    res_run1 = [{"title": "Old News", "url": "https://old.eu", "content": "Round 1"}]
    res_run2 = [{"title": "Fresh News", "url": "https://fresh.eu", "content": "Round 2"}]

    # Run 1
    with patch("urllib.request.urlopen", return_value=_make_mock_urlopen(res_run1)) as mock_url1:
        out1 = search_tavily("fintech series a berlin", api_key="mock_key", max_results=5)
        assert out1 == res_run1
        assert mock_url1.call_count == 1

    # End of Run 1 / Start of Run 2 (pipeline boundary)
    clear_tavily_cache()
    reset_tavily_search_count()

    # Run 2 with updated web results
    with patch("urllib.request.urlopen", return_value=_make_mock_urlopen(res_run2)) as mock_url2:
        out2 = search_tavily("fintech series a berlin", api_key="mock_key", max_results=5)
        assert out2 == res_run2
        assert mock_url2.call_count == 1  # Dispatched fresh request!


# ==============================================================================
# D. Different queries don't collide, whitespace normalizes safely
# ==============================================================================
def test_case_d_different_queries_do_not_collide():
    """Distinct semantic queries must never collide; whitespace variations must normalize safely."""
    res_funding = [{"title": "Funding", "url": "https://funding.com"}]
    res_founder = [{"title": "Founder", "url": "https://founder.com"}]

    with patch("urllib.request.urlopen", side_effect=[
        _make_mock_urlopen(res_funding),
        _make_mock_urlopen(res_founder)
    ]) as mock_url:
        r_fund = search_tavily("Berlin SaaS funding", api_key="mock_key")
        r_fndr = search_tavily("Berlin SaaS founder email", api_key="mock_key")

        assert r_fund == res_funding
        assert r_fndr == res_founder
        assert mock_url.call_count == 2
        assert get_tavily_cache_size() == 2

        # Safe normalization: internal/leading/trailing whitespace variation hits existing cache
        r_fund_ws = search_tavily("   Berlin   SaaS    funding   ", api_key="mock_key")
        assert r_fund_ws == res_funding
        assert mock_url.call_count == 2  # Cache hit!


def test_case_d_different_max_results_do_not_collide():
    """Same query string with different max_results must not collide in cache."""
    res_5 = [{"title": "Five", "url": "https://five.com"}]
    res_10 = [{"title": "Ten", "url": "https://ten.com"}]

    with patch("urllib.request.urlopen", side_effect=[
        _make_mock_urlopen(res_5),
        _make_mock_urlopen(res_10)
    ]) as mock_url:
        r5 = search_tavily("ai robotics stockholm", max_results=5, api_key="mock_key")
        r10 = search_tavily("ai robotics stockholm", max_results=10, api_key="mock_key")

        assert r5 == res_5
        assert r10 == res_10
        assert mock_url.call_count == 2
        assert get_tavily_cache_size() == 2


# ==============================================================================
# E. Cache hit consumes zero Tavily budget
# ==============================================================================
def test_case_e_cache_hit_consumes_zero_budget():
    """Verify that multiple cache hits consume exactly 0 budget attempt slots."""
    res = [{"title": "Cached", "url": "https://cached.com"}]

    with patch("urllib.request.urlopen", return_value=_make_mock_urlopen(res)):
        search_tavily("repeat query", api_key="mock_key")
        assert get_total_tavily_calls() == 1

        for _ in range(10):
            search_tavily("repeat query", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_cache_hits"] == 10
        assert bd["total_tavily_calls"] == 1


# ==============================================================================
# F. Cache clearing does not reset mid-run budget unexpectedly
# ==============================================================================
def test_case_f_cache_clearing_does_not_reset_budget_counters():
    """clear_tavily_cache must only clear the cache dictionary, NOT the budget counters."""
    res = [{"title": "A", "url": "https://a.com"}]

    with patch("urllib.request.urlopen", return_value=_make_mock_urlopen(res)):
        search_tavily("query 1", api_key="mock_key")
        search_tavily("query 2", api_key="mock_key")
        search_tavily("query 3", api_key="mock_key")
        assert get_total_tavily_calls() == 3
        assert get_tavily_cache_size() == 3

        # Clear cache mid-run
        clear_tavily_cache()

        # Cache is empty, but budget count is strictly preserved!
        assert get_tavily_cache_size() == 0
        assert get_total_tavily_calls() == 3


# ==============================================================================
# G. Sequential pipeline / Streamlit runs do not reuse old results
# ==============================================================================
def test_case_g_sequential_pipeline_runs_do_not_reuse_old_cache():
    """Simulates clicking Run twice in Streamlit: Run 2 must not see Run 1 cache."""
    cand1 = CandidateCompany(
        name="StartupOne",
        website_url="https://one.de",
        description="B2B platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=2_000_000.0,
        revenue_period="ARR",
        financial_evidence_date="2024-05-01",
        founder_name="Alice",
        founder_title="CEO"
    )
    cand2 = CandidateCompany(
        name="StartupTwo",
        website_url="https://two.de",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        revenue_amount_usd=3_000_000.0,
        revenue_period="ARR",
        financial_evidence_date="2024-05-01",
        founder_name="Bob",
        founder_title="CEO"
    )

    audit1 = CandidateAuditRecord(
        company_name="StartupOne",
        qualification_status="ACCEPTED",
        financial_figure_used="$2,000,000 ARR",
        revenue_period="ARR",
        recency_basis="Financial Date",
        confidence_rating="HIGH",
        audit_summary="Qualified",
        evidence=AuditEvidence()
    )
    audit2 = CandidateAuditRecord(
        company_name="StartupTwo",
        qualification_status="ACCEPTED",
        financial_figure_used="$3,000,000 ARR",
        revenue_period="ARR",
        recency_basis="Financial Date",
        confidence_rating="HIGH",
        audit_summary="Qualified",
        evidence=AuditEvidence()
    )

    with patch("core.pipeline.genai.Client"), \
         patch("core.pipeline.generate_queries", return_value=["search query 1"]), \
         patch("core.pipeline.search_grounded_candidates", side_effect=[
             ("RAW TEXT 1", ["https://one.de"]),
             ("RAW TEXT 2", ["https://two.de"])
         ]), \
         patch("core.pipeline.extract_structured_candidates", side_effect=[[cand1], [cand2]]), \
         patch("core.pipeline.discover_founder_contact_path", return_value=ContactPathResult(contact_path_found=True)), \
         patch("core.pipeline.verify_founder_email_on_site", side_effect=[
             ("alice@one.de", "https://one.de"),
             ("bob@two.de", "https://two.de")
         ]), \
         patch("core.pipeline.evaluate_financial_qualification", side_effect=[
             (True, audit1),
             (True, audit2)
         ]):

        # Run 1
        summary1 = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)
        assert summary1.funnel.candidates_found == 1
        assert len(summary1.leads) == 1
        assert summary1.leads[0].company_name == "StartupOne"

        # Run 2 (new user click in same process)
        summary2 = run_lead_pipeline(api_key="mock_key", target_count=1, max_queries=1)
        assert summary2.funnel.candidates_found == 1
        assert len(summary2.leads) == 1
        assert summary2.leads[0].company_name == "StartupTwo"


# ==============================================================================
# H. Cache + failed request behavior remains correct
# ==============================================================================
def test_case_h_failed_requests_are_not_cached():
    """Failed requests consume a budget slot but must NOT be placed into the cache."""
    mock_error = urllib.error.HTTPError("https://api.tavily.com", 429, "Rate Limited", {}, None)

    with patch("urllib.request.urlopen", side_effect=mock_error):
        with pytest.raises(TavilyRateLimitError):
            search_tavily("failing query", api_key="mock_key")

        # 1 slot consumed
        assert get_total_tavily_calls() == 1
        # Zero items cached!
        assert get_tavily_cache_size() == 0
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1
        assert bd["tavily_requests_succeeded"] == 0

    # If network recovers, next request is dispatched fresh (not a cached error)
    res_recovered = [{"title": "Recovered", "url": "https://rec.com"}]
    with patch("urllib.request.urlopen", return_value=_make_mock_urlopen(res_recovered)):
        out = search_tavily("failing query", api_key="mock_key")
        assert out == res_recovered
        assert get_total_tavily_calls() == 2
        assert get_tavily_cache_size() == 1


# ==============================================================================
# I. Cache + hard budget behavior remains correct
# ==============================================================================
def test_case_i_cached_queries_succeed_even_when_budget_full():
    """A query already cached in this run can still be read even if the budget is full."""
    res = [{"title": "Cached Win", "url": "https://win.eu"}]

    with patch("urllib.request.urlopen", return_value=_make_mock_urlopen(res)):
        # Call 1 with max_budget=1
        r1 = search_tavily("budget query", api_key="mock_key", max_budget=1)
        assert r1 == res
        assert get_total_tavily_calls() == 1

        # Budget is now 1/1 (exhausted for NEW outbound requests)
        # But reading the cached query succeeds with 0 budget cost!
        r1_cached = search_tavily("budget query", api_key="mock_key", max_budget=1)
        assert r1_cached == res

        # A new uncached query must fail with TavilyBudgetExhaustedError
        with pytest.raises(TavilyBudgetExhaustedError):
            search_tavily("uncached query", api_key="mock_key", max_budget=1)

        # Budget attempts strictly remain at 1
        assert get_total_tavily_calls() == 1
