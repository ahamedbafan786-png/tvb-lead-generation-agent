"""
tests/test_tavily_discovery.py
Deterministic unit tests for Tavily discovery layer:
- Result parsing and caching
- Empty results handling
- Error handling: Auth, Quota, Rate limit, Timeout, Malformed JSON
- Source URL preservation and extraction prompt formatting
- Zero Google Search Grounding invocation
"""

import io
import json
import socket
import urllib.request
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from core.discovery import (
    search_tavily, format_tavily_sources, search_grounded_candidates,
    TavilyAuthError, TavilyQuotaError, TavilyRateLimitError, TavilyTimeoutError,
    TavilySearchError, clear_tavily_cache, reset_tavily_search_count, get_tavily_search_count
)
from core.models import CandidateCompany
from core.extractor import execute_targeted_funding_follow_up


@pytest.fixture(autouse=True)
def clean_cache_and_counter():
    clear_tavily_cache()
    reset_tavily_search_count()
    yield
    clear_tavily_cache()
    reset_tavily_search_count()


def test_tavily_result_parsing():
    mock_response_data = {
        "results": [
            {
                "title": "Acme Cloud Secures $3M Seed",
                "url": "https://acmecloud.eu/press/seed",
                "content": "Acme Cloud, a Munich-based B2B developer platform, raised $3M in Seed funding.",
                "score": 0.98
            },
            {
                "title": "Nordic SaaS Index 2026",
                "url": "https://nordicsaas.com/index",
                "content": "Stockholm software company DataPulse reaches $2.5M ARR.",
                "score": 0.89
            }
        ]
    }
    mock_resp = io.BytesIO(json.dumps(mock_response_data).encode("utf-8"))
    mock_resp.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp):
        results = search_tavily("B2B SaaS seed Munich", api_key="test_key")

    assert len(results) == 2
    assert results[0]["title"] == "Acme Cloud Secures $3M Seed"
    assert results[0]["url"] == "https://acmecloud.eu/press/seed"
    assert "Munich-based" in results[0]["content"]
    assert get_tavily_search_count() == 1

    # Verify cache prevents repeated network requests
    with patch("urllib.request.urlopen") as mock_url:
        cached_results = search_tavily("B2B SaaS seed Munich", api_key="test_key")
        assert len(cached_results) == 2
        mock_url.assert_not_called()
        assert get_tavily_search_count() == 1


def test_empty_tavily_result():
    mock_resp = io.BytesIO(json.dumps({"results": []}).encode("utf-8"))
    mock_resp.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp):
        raw_text, sources = search_grounded_candidates(client=None, query="Nonexistent Query", tavily_api_key="test_key")

    assert raw_text == ""
    assert sources == []


def test_tavily_auth_error_handling():
    with pytest.raises(TavilyAuthError) as exc_info:
        search_tavily("some query", api_key="")
    assert "TAVILY_AUTH_ERROR" in str(exc_info.value)

    http_error = urllib.error.HTTPError(
        url="https://api.tavily.com/search",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=io.BytesIO(b'{"detail": "Invalid API key"}')
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(TavilyAuthError) as exc_info:
            search_tavily("some query", api_key="invalid_key")
        assert "401" in str(exc_info.value)


def test_tavily_quota_error_handling():
    http_error = urllib.error.HTTPError(
        url="https://api.tavily.com/search",
        code=432,
        msg="Payment Required",
        hdrs={},
        fp=io.BytesIO(b'{"detail": "Monthly search credit limit reached"}')
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(TavilyQuotaError) as exc_info:
            search_tavily("some query", api_key="test_key")
        assert "TAVILY_QUOTA_ERROR" in str(exc_info.value)


def test_tavily_rate_limit_error_handling():
    http_error = urllib.error.HTTPError(
        url="https://api.tavily.com/search",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=io.BytesIO(b'{"detail": "Rate limit exceeded"}')
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(TavilyRateLimitError) as exc_info:
            search_tavily("some query", api_key="test_key")
        assert "TAVILY_RATE_LIMIT" in str(exc_info.value)


def test_tavily_timeout_error_handling():
    with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
        with pytest.raises(TavilyTimeoutError) as exc_info:
            search_tavily("some query", api_key="test_key")
        assert "TAVILY_TIMEOUT" in str(exc_info.value)


def test_source_formatting_and_preservation():
    mock_results = [
        {
            "title": "Company One Series A",
            "url": "https://companyone.com/news",
            "content": "Company One raised $2.5M in Series A funding in Berlin."
        },
        {
            "title": "Company Two ARR Milestone",
            "url": "https://companytwo.io/blog",
            "content": "Company Two hit $1.8M ARR with its developer localization platform."
        }
    ]
    formatted = format_tavily_sources(mock_results)
    assert "SOURCE 1" in formatted
    assert "Title: Company One Series A" in formatted
    assert "URL: https://companyone.com/news" in formatted
    assert "SOURCE 2" in formatted
    assert "URL: https://companytwo.io/blog" in formatted

    mock_resp = io.BytesIO(json.dumps({"results": mock_results}).encode("utf-8"))
    mock_resp.status = 200
    with patch("urllib.request.urlopen", return_value=mock_resp):
        raw_text, sources = search_grounded_candidates(client=None, query="test", tavily_api_key="test_key")

    assert len(sources) == 2
    assert "https://companyone.com/news" in sources
    assert "https://companytwo.io/blog" in sources
    assert "SOURCE 1" in raw_text


def test_no_google_search_grounding_invoked():
    mock_client = MagicMock()
    mock_results = [{
        "title": "Example Corp",
        "url": "https://example.com",
        "content": "Example software platform Paris."
    }]
    mock_resp = io.BytesIO(json.dumps({"results": mock_results}).encode("utf-8"))
    mock_resp.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp):
        raw_text, sources = search_grounded_candidates(client=mock_client, query="test query", tavily_api_key="test_key")

    mock_client.models.generate_content.assert_not_called()
    assert "https://example.com" in sources


def test_tavily_follow_up_financial_search_mock():
    candidate = CandidateCompany(
        name="TargetSaaS",
        description="Workflow tooling",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        latest_round_amount_usd=2_000_000,
        founder_name="Hans Bauer"
    )

    tavily_results = [{
        "title": "TargetSaaS Total Capital Raised",
        "url": "https://eu-startups.com/targetsaas",
        "content": "TargetSaaS total cumulative funding reached $3.2M across Seed and Series A rounds."
    }]
    mock_tavily_resp = io.BytesIO(json.dumps({"results": tavily_results}).encode("utf-8"))
    mock_tavily_resp.status = 200

    gemini_json_resp = {
        "total_cumulative_funding_usd": 3200000.0,
        "latest_round_name": "Series A",
        "latest_round_amount_usd": 2000000.0,
        "latest_round_date": "2025-11-15",
        "financial_evidence_date": "2026-02-20",
        "subsequent_rounds_discovered": False,
        "acquisition_or_closure_discovered": False,
        "us_presence_detected": False,
        "evidence_snippet": "Total cumulative funding reached $3.2M."
    }

    mock_gemini_client = MagicMock()
    mock_model_response = MagicMock()
    mock_model_response.text = json.dumps(gemini_json_resp)
    mock_gemini_client.models.generate_content.return_value = mock_model_response

    with patch("urllib.request.urlopen", return_value=mock_tavily_resp):
        updated_cand = execute_targeted_funding_follow_up(
            client=mock_gemini_client,
            candidate=candidate,
            tavily_api_key="test_key"
        )

    assert updated_cand.total_cumulative_funding_usd == 3_200_000.0
    assert updated_cand.latest_round_name == "Series A"
    assert updated_cand.financial_evidence_date == "2026-02-20"
    call_kwargs = mock_gemini_client.models.generate_content.call_args[1]
    config = call_kwargs.get("config", {})
    assert "tools" not in config
