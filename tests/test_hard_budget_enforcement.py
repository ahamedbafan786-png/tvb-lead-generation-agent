"""
tests/test_hard_budget_enforcement.py

Adversarial tests for the Codex-identified defect:
The Tavily 60-call budget must be TRULY HARD — every outbound request attempt
must consume a budget slot BEFORE dispatch, regardless of outcome.

Tests A-L cover all required adversarial scenarios.
"""

import io
import json
import socket
import urllib.error
import urllib.request

import pytest
from unittest.mock import patch, MagicMock

from core.discovery import (
    search_tavily, get_total_tavily_calls, get_tavily_call_breakdown,
    reset_tavily_search_count, clear_tavily_cache,
    TavilyBudgetExhaustedError, TavilyTimeoutError, TavilyRateLimitError,
    TavilySearchError, TavilyAuthError,
)


@pytest.fixture(autouse=True)
def clean_state():
    clear_tavily_cache()
    reset_tavily_search_count()
    yield
    clear_tavily_cache()
    reset_tavily_search_count()


def _mock_response(results=None):
    if results is None:
        results = [{"title": "T", "url": "https://e.com", "content": "c", "score": 0.9}]
    resp = io.BytesIO(json.dumps({"results": results}).encode("utf-8"))
    resp.status = 200
    return resp


def _http_error(code, body=""):
    resp = io.BytesIO(body.encode("utf-8"))
    return urllib.error.HTTPError(
        url="https://api.tavily.com/search",
        code=code,
        msg=f"HTTP {code}",
        hdrs={},
        fp=resp,
    )


# ---------------------------------------------------------------------------
# A. 60 successful requests → #61 blocked
# ---------------------------------------------------------------------------

class TestHardBudget60Successes:
    def test_case_a_sixty_successes_then_blocked(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 60)

        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _mock_response()):
            for i in range(60):
                search_tavily(f"unique query {i}", api_key="mock_key")
            assert get_total_tavily_calls() == 60

            # 61st must be blocked BEFORE urlopen
            with pytest.raises(TavilyBudgetExhaustedError):
                search_tavily("query 61", api_key="mock_key")
            assert get_total_tavily_calls() == 60

        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_succeeded"] == 60
        assert bd["tavily_requests_failed"] == 0


# ---------------------------------------------------------------------------
# B. 60 failed requests → #61 blocked
# ---------------------------------------------------------------------------

class TestHardBudget60Failures:
    def test_case_b_sixty_failures_then_blocked(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 60)

        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            for i in range(60):
                with pytest.raises(TavilyTimeoutError):
                    search_tavily(f"fail query {i}", api_key="mock_key")
            assert get_total_tavily_calls() == 60

            # 61st must be blocked BEFORE urlopen
            with pytest.raises(TavilyBudgetExhaustedError):
                search_tavily("fail query 61", api_key="mock_key")
            assert get_total_tavily_calls() == 60

        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_succeeded"] == 0
        assert bd["tavily_requests_failed"] == 60


# ---------------------------------------------------------------------------
# C. 30 successes + 30 failures → #61 blocked
# ---------------------------------------------------------------------------

class TestHardBudgetMixed:
    def test_case_c_mixed_then_blocked(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 60)

        call_count = [0]

        def mixed_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 30:
                return _mock_response()
            else:
                raise socket.timeout("timed out")

        with patch("urllib.request.urlopen", side_effect=mixed_side_effect):
            for i in range(30):
                search_tavily(f"success query {i}", api_key="mock_key")
            for i in range(30):
                with pytest.raises(TavilyTimeoutError):
                    search_tavily(f"timeout query {i}", api_key="mock_key")
            assert get_total_tavily_calls() == 60

            with pytest.raises(TavilyBudgetExhaustedError):
                search_tavily("query 61", api_key="mock_key")

        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_succeeded"] == 30
        assert bd["tavily_requests_failed"] == 30


# ---------------------------------------------------------------------------
# D. 59 requests + cache hits → next real request allowed
# ---------------------------------------------------------------------------

class TestCacheDoesNotConsumeAttemptBudget:
    def test_case_d_cache_hits_dont_block(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 60)

        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _mock_response()):
            # 59 unique queries
            for i in range(59):
                search_tavily(f"unique query {i}", api_key="mock_key")
            assert get_total_tavily_calls() == 59

            # 100 cache hits (re-query existing queries)
            for i in range(100):
                search_tavily(f"unique query {i % 59}", api_key="mock_key")
            # Still only 59 outbound attempts
            assert get_total_tavily_calls() == 59

            # 60th unique query should be allowed
            search_tavily("unique query 59", api_key="mock_key")
            assert get_total_tavily_calls() == 60

        bd = get_tavily_call_breakdown()
        assert bd["tavily_cache_hits"] == 100
        assert bd["tavily_requests_succeeded"] == 60


# ---------------------------------------------------------------------------
# E. cache hits consume zero attempt budget (explicit)
# ---------------------------------------------------------------------------

class TestCacheHitsZeroBudget:
    def test_case_e_cache_hits_zero_budget(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 3)

        with patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _mock_response()):
            search_tavily("query A", api_key="mock_key")
            assert get_total_tavily_calls() == 1

        # Now call the same query 50 times — all cache hits
        for _ in range(50):
            search_tavily("query A", api_key="mock_key")
        assert get_total_tavily_calls() == 1

        bd = get_tavily_call_breakdown()
        assert bd["tavily_cache_hits"] == 50


# ---------------------------------------------------------------------------
# F. timeout still consumes an attempt slot
# ---------------------------------------------------------------------------

class TestTimeoutConsumesSlot:
    def test_case_f_timeout_consumes_slot(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with pytest.raises(TavilyTimeoutError):
                search_tavily("timeout test", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1
        assert bd["tavily_requests_succeeded"] == 0


# ---------------------------------------------------------------------------
# G. HTTP 429 still consumes an attempt slot
# ---------------------------------------------------------------------------

class TestHTTP429ConsumesSlot:
    def test_case_g_429_consumes_slot(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        with patch("urllib.request.urlopen", side_effect=_http_error(429)):
            with pytest.raises(TavilyRateLimitError):
                search_tavily("rate limited query", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1


# ---------------------------------------------------------------------------
# H. HTTP 403 still consumes an attempt slot
# ---------------------------------------------------------------------------

class TestHTTP403ConsumesSlot:
    def test_case_h_403_consumes_slot(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        with patch("urllib.request.urlopen", side_effect=_http_error(403)):
            with pytest.raises(TavilyAuthError):
                search_tavily("forbidden query", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1


# ---------------------------------------------------------------------------
# I. HTTP 500 still consumes an attempt slot
# ---------------------------------------------------------------------------

class TestHTTP500ConsumesSlot:
    def test_case_i_500_consumes_slot(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        with patch("urllib.request.urlopen", side_effect=_http_error(500)):
            with pytest.raises(TavilySearchError):
                search_tavily("server error query", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1


# ---------------------------------------------------------------------------
# J. Repeated budget-exhaustion attempts never dispatch network calls
# ---------------------------------------------------------------------------

class TestRepeatedExhaustionNeverDispatches:
    def test_case_j_no_dispatch_after_exhaustion(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 2)

        urlopen_call_count = [0]
        original_side_effect = lambda *a, **kw: _mock_response()

        def counting_urlopen(*args, **kwargs):
            urlopen_call_count[0] += 1
            return _mock_response()

        with patch("urllib.request.urlopen", side_effect=counting_urlopen):
            search_tavily("q1", api_key="mock_key")
            search_tavily("q2", api_key="mock_key")
            assert urlopen_call_count[0] == 2

            # Now try 10 more — ALL must be blocked without calling urlopen
            for i in range(10):
                with pytest.raises(TavilyBudgetExhaustedError):
                    search_tavily(f"blocked q{i+3}", api_key="mock_key")

            # urlopen was never called again
            assert urlopen_call_count[0] == 2
            assert get_total_tavily_calls() == 2


# ---------------------------------------------------------------------------
# K. Counter/category telemetry remains consistent
# ---------------------------------------------------------------------------

class TestTelemetryConsistency:
    def test_case_k_telemetry_consistent(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 10)

        call_idx = [0]

        def alternating(*args, **kwargs):
            call_idx[0] += 1
            if call_idx[0] % 3 == 0:
                raise socket.timeout("timed out")
            return _mock_response()

        with patch("urllib.request.urlopen", side_effect=alternating):
            for i in range(6):
                try:
                    cat = "discovery" if i < 3 else "financial_followup"
                    search_tavily(f"telemetry q{i}", api_key="mock_key", category=cat)
                except TavilyTimeoutError:
                    pass

        assert get_total_tavily_calls() == 6
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_attempted"] == 6
        assert bd["tavily_requests_succeeded"] + bd["tavily_requests_failed"] == 6
        assert bd["discovery_calls"] == 3
        assert bd["financial_followup_calls"] == 3


# ---------------------------------------------------------------------------
# L. No successful response is required for budget reservation
# ---------------------------------------------------------------------------

class TestNoSuccessRequiredForReservation:
    def test_case_l_all_failures_still_reserve(self, monkeypatch):
        """Budget is consumed even when EVERY request fails."""
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        failures = [
            _http_error(500),
            _http_error(429),
            socket.timeout("timed out"),
            urllib.error.URLError("Connection refused"),
            _http_error(500),
        ]

        with patch("urllib.request.urlopen", side_effect=failures):
            for i in range(5):
                try:
                    search_tavily(f"all fail q{i}", api_key="mock_key")
                except (TavilySearchError, TavilyRateLimitError, TavilyTimeoutError):
                    pass

        assert get_total_tavily_calls() == 5
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_succeeded"] == 0
        assert bd["tavily_requests_failed"] == 5

        # 6th must be blocked
        with pytest.raises(TavilyBudgetExhaustedError):
            search_tavily("blocked after failures", api_key="mock_key")
        assert get_total_tavily_calls() == 5


# ---------------------------------------------------------------------------
# Additional: Connection error consumes a slot
# ---------------------------------------------------------------------------

class TestConnectionErrorConsumesSlot:
    def test_connection_error_consumes_slot(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(TavilySearchError):
                search_tavily("connection error query", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1


# ---------------------------------------------------------------------------
# Additional: Malformed JSON consumes a slot
# ---------------------------------------------------------------------------

class TestMalformedJSONConsumesSlot:
    def test_malformed_json_consumes_slot(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        bad_resp = io.BytesIO(b"not json at all")
        bad_resp.status = 200
        with patch("urllib.request.urlopen", return_value=bad_resp):
            with pytest.raises(TavilySearchError):
                search_tavily("malformed json query", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1


# ---------------------------------------------------------------------------
# Additional: HTTP 401 consumes a slot
# ---------------------------------------------------------------------------

class TestHTTP401ConsumesSlot:
    def test_401_consumes_slot(self, monkeypatch):
        monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 5)

        with patch("urllib.request.urlopen", side_effect=_http_error(401)):
            with pytest.raises(TavilyAuthError):
                search_tavily("auth error query", api_key="mock_key")

        assert get_total_tavily_calls() == 1
        bd = get_tavily_call_breakdown()
        assert bd["tavily_requests_failed"] == 1
