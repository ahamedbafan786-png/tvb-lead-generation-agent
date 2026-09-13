"""
tests/test_gemini_client.py
Unit tests for core.gemini_client rate-limit pacing and model fallback rotation.
All tests use mocking — zero live API calls.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from core.gemini_client import (
    generate_with_fallback,
    AllModelsExhaustedError,
    reset_model_state,
    all_models_exhausted,
    get_exhausted_models,
    _is_daily_quota_error,
    _parse_retry_delay,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_state():
    """Reset module-level state before every test."""
    reset_model_state()
    yield
    reset_model_state()


def _make_client():
    """Create a mock genai.Client."""
    return MagicMock()


def _make_response(text: str):
    """Create a mock response with .text attribute."""
    resp = MagicMock()
    resp.text = text
    return resp


def _daily_rpd_error(model: str = "gemini-3.5-flash"):
    """Simulate a 429 daily RPD quota error."""
    return Exception(
        f"429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
        f"generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        f"limit: 20, model: {model} "
        f"quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier "
        f"retryDelay: '50s'"
    )


def _rpm_rate_limit_error():
    """Simulate a 429 RPM (per-minute) rate limit error."""
    return Exception(
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_requests_per_minute, "
        "quotaId: GenerateRequestsPerMinutePerProjectPerModel "
        "retryDelay: '5s'"
    )


# ── Helper function tests ────────────────────────────────────────────

def test_is_daily_quota_error_true():
    assert _is_daily_quota_error("FreeTier something PerDay blah") is True
    assert _is_daily_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier") is True


def test_is_daily_quota_error_false():
    assert _is_daily_quota_error("PerMinute something") is False
    assert _is_daily_quota_error("random error message") is False


def test_parse_retry_delay():
    assert _parse_retry_delay("retryDelay: '50s'") == 50.0
    assert _parse_retry_delay("retryDelay: '5.5s'") == 5.5
    assert _parse_retry_delay("no delay info here") is None


# ── generate_with_fallback tests ─────────────────────────────────────

@patch("core.gemini_client.MIN_INTER_CALL_DELAY_SECONDS", 0.0)
def test_primary_model_succeeds():
    """When the primary model works, return its response directly."""
    client = _make_client()
    client.models.generate_content.return_value = _make_response('{"result": "ok"}')

    result = generate_with_fallback(client, "test prompt", {"temperature": 0.0})
    assert result == '{"result": "ok"}'
    assert not all_models_exhausted()


@patch("core.gemini_client.MIN_INTER_CALL_DELAY_SECONDS", 0.0)
def test_fallback_on_daily_rpd_exhaustion():
    """When primary model hits daily RPD, falls through to fallback model."""
    client = _make_client()

    # First call (primary) → daily RPD error; second call (fallback) → success
    client.models.generate_content.side_effect = [
        _daily_rpd_error("gemini-3.5-flash"),
        _make_response('{"fallback": true}'),
    ]

    result = generate_with_fallback(
        client, "test", {"temperature": 0.0},
        primary_model="gemini-3.5-flash",
        fallback_model="gemini-3.5-flash-lite",
    )
    assert result == '{"fallback": true}'
    assert "gemini-3.5-flash" in get_exhausted_models()
    assert "gemini-3.5-flash-lite" not in get_exhausted_models()


@patch("core.gemini_client.MIN_INTER_CALL_DELAY_SECONDS", 0.0)
def test_all_models_exhausted_raises():
    """When both models hit daily RPD, raise AllModelsExhaustedError."""
    client = _make_client()

    client.models.generate_content.side_effect = [
        _daily_rpd_error("gemini-3.5-flash"),
        _daily_rpd_error("gemini-3.5-flash-lite"),
    ]

    with pytest.raises(AllModelsExhaustedError):
        generate_with_fallback(
            client, "test", {"temperature": 0.0},
            primary_model="gemini-3.5-flash",
            fallback_model="gemini-3.5-flash-lite",
        )

    assert all_models_exhausted()


@patch("core.gemini_client.MIN_INTER_CALL_DELAY_SECONDS", 0.0)
@patch("core.gemini_client.time.sleep")  # Don't actually sleep in tests
def test_rpm_limit_retries_and_succeeds(mock_sleep):
    """On RPM (per-minute) limit, sleep and retry — succeed on retry."""
    client = _make_client()

    client.models.generate_content.side_effect = [
        _rpm_rate_limit_error(),
        _make_response('{"retry_ok": true}'),
    ]

    result = generate_with_fallback(
        client, "test", {"temperature": 0.0},
        primary_model="gemini-3.5-flash",
        fallback_model="gemini-3.5-flash-lite",
    )
    assert result == '{"retry_ok": true}'
    mock_sleep.assert_called()  # Should have slept for retry delay
    assert not all_models_exhausted()  # Model should NOT be marked exhausted


@patch("core.gemini_client.MIN_INTER_CALL_DELAY_SECONDS", 0.0)
def test_non_rate_limit_error_propagates():
    """Non-429 errors are raised immediately without fallback."""
    client = _make_client()
    client.models.generate_content.side_effect = ValueError("Bad input")

    with pytest.raises(ValueError, match="Bad input"):
        generate_with_fallback(client, "test", {"temperature": 0.0})


@patch("core.gemini_client.MIN_INTER_CALL_DELAY_SECONDS", 0.0)
def test_reset_clears_exhaustion():
    """reset_model_state() clears all exhausted models."""
    client = _make_client()
    client.models.generate_content.side_effect = [
        _daily_rpd_error("gemini-3.5-flash"),
        _daily_rpd_error("gemini-3.5-flash-lite"),
    ]

    with pytest.raises(AllModelsExhaustedError):
        generate_with_fallback(
            client, "test", {"temperature": 0.0},
            primary_model="gemini-3.5-flash",
            fallback_model="gemini-3.5-flash-lite",
        )

    assert all_models_exhausted()
    reset_model_state()
    assert not all_models_exhausted()
    assert len(get_exhausted_models()) == 0


@patch("core.gemini_client.MIN_INTER_CALL_DELAY_SECONDS", 0.0)
def test_pre_exhausted_model_skipped():
    """If primary is already exhausted, skip directly to fallback."""
    client = _make_client()

    # Pre-exhaust primary
    client.models.generate_content.side_effect = [
        _daily_rpd_error("gemini-3.5-flash"),
        _make_response('{"first_run": true}'),
    ]
    generate_with_fallback(
        client, "first", {"temperature": 0.0},
        primary_model="gemini-3.5-flash",
        fallback_model="gemini-3.5-flash-lite",
    )

    # Second call: primary should be skipped, fallback used directly
    client.models.generate_content.side_effect = [
        _make_response('{"second_run": true}'),
    ]
    result = generate_with_fallback(
        client, "second", {"temperature": 0.0},
        primary_model="gemini-3.5-flash",
        fallback_model="gemini-3.5-flash-lite",
    )
    assert result == '{"second_run": true}'
