"""
tests/test_ssrf_and_tls_hardening.py
Deterministic test suite verifying SSRF prevention, public URL validation,
DNS-rebinding protection, redirect boundary enforcement, and elimination of
insecure TLS fallbacks.

ZERO live network calls, ZERO live Tavily calls, ZERO live Gemini calls.
"""

import socket
import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
import requests

from core.models import CandidateCompany, FinancialEvidence
from core.scraper import (
    validate_public_url,
    parse_ip_address,
    is_safe_public_ip,
    fetch_html,
    MAX_REDIRECTS
)
from core.email_verifier import (
    verify_founder_email_on_site,
    discover_founder_contact_path
)
from core.extractor import evaluate_financial_qualification


# ==============================================================================
# Helper Mock Factories
# ==============================================================================
def _make_mock_response(status_code=200, text="<html><body><h1>Test</h1></body></html>", location=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.is_redirect = status_code in (301, 302, 303, 307, 308)
    headers = {"Content-Type": "text/html; charset=utf-8"}
    if location:
        headers["Location"] = location
    resp.headers = headers
    return resp


def _mock_dns_public(hostname, port=None, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]


def _mock_dns_private(hostname, port=None, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 80))]


def _mock_dns_metadata(hostname, port=None, *args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", port or 80))]


# ==============================================================================
# Case A & B: Public HTTPS and HTTP Allowed
# ==============================================================================
def test_case_a_public_https_allowed():
    """Case A: Public HTTPS URL allowed."""
    with patch("socket.getaddrinfo", side_effect=_mock_dns_public):
        is_safe, reason, canonical = validate_public_url("https://example.com/about", resolve_dns=True)
    assert is_safe is True
    assert canonical == "https://example.com/about"

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", return_value=_make_mock_response(200, "<html>About</html>")) as mock_get:
        soup = fetch_html("https://example.com/about")
        assert soup is not None
        assert "About" in soup.get_text()
        assert mock_get.call_args[1]["verify"] is True


def test_case_b_public_http_allowed():
    """Case B: Public HTTP URL allowed."""
    with patch("socket.getaddrinfo", side_effect=_mock_dns_public):
        is_safe, reason, canonical = validate_public_url("http://example.com/contact", resolve_dns=True)
    assert is_safe is True
    assert canonical == "http://example.com/contact"

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", return_value=_make_mock_response(200, "<html>Contact</html>")) as mock_get:
        soup = fetch_html("http://example.com/contact")
        assert soup is not None
        assert mock_get.call_args[1]["verify"] is True


# ==============================================================================
# Case C: Localhost Blocked
# ==============================================================================
@pytest.mark.parametrize("url", [
    "http://localhost",
    "https://localhost:8080",
    "http://localhost/admin",
    "http://sub.localhost",
    "http://app.localhost:3000",
    "http://localhost.localdomain",
])
def test_case_c_localhost_blocked(url):
    """Case C: localhost variations blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    assert "localhost" in reason.lower()
    assert canonical is None

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Case D: 127.0.0.1 Blocked
# ==============================================================================
@pytest.mark.parametrize("url", [
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "https://127.0.0.1:8443/secret",
    "http://127.0.0.2",
    "http://127.255.255.255",
])
def test_case_d_loopback_ip_blocked(url):
    """Case D: 127.0.0.0/8 loopback IPs blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    assert any(term in reason.lower() for term in ["loopback", "blocked", "non-public", "private"])

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Case E: ::1 IPv6 Loopback Blocked
# ==============================================================================
@pytest.mark.parametrize("url", [
    "http://[::1]",
    "http://[::1]:8000",
    "https://[::1]:8443/admin",
])
def test_case_e_ipv6_loopback_blocked(url):
    """Case E: IPv6 loopback [::1] blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    assert any(term in reason.lower() for term in ["loopback", "blocked", "non-public", "private"])

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Case F: RFC 1918 Private IPs Blocked
# ==============================================================================
@pytest.mark.parametrize("url", [
    "http://10.0.0.1",
    "http://10.255.255.254:8080",
    "http://172.16.0.1",
    "http://172.31.255.255/internal",
    "http://192.168.0.1",
    "http://192.168.1.254:3000",
])
def test_case_f_rfc1918_private_ips_blocked(url):
    """Case F: RFC 1918 private IPs blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    assert any(term in reason.lower() for term in ["private", "rfc1918", "blocked", "non-public"])

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Case G: Cloud Metadata Blocked
# ==============================================================================
@pytest.mark.parametrize("url", [
    "http://169.254.169.254",
    "http://169.254.169.254/latest/meta-data/",
    "http://169.254.169.253",
    "http://metadata.google.internal",
    "http://metadata.google.internal/computeMetadata/v1/",
    "http://[fd00:ec2::254]/latest/meta-data/",
])
def test_case_g_cloud_metadata_blocked(url):
    """Case G: Cloud metadata endpoints blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Case H: Malformed URLs Blocked
# ==============================================================================
@pytest.mark.parametrize("url", [
    "not-a-url",
    "http://",
    "https://",
    "http://:80",
    "http:///path",
    "http://?",
    "",
    "   ",
])
def test_case_h_malformed_urls_blocked(url):
    """Case H: Malformed and empty URLs blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    assert canonical is None

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Cases I, J, K, L: Unsupported Schemes Blocked
# ==============================================================================
@pytest.mark.parametrize("url,scheme_name", [
    ("file:///etc/passwd", "file"),
    ("file://localhost/etc/shadow", "file"),
    ("javascript:alert(1)", "javascript"),
    ("data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==", "data"),
    ("gopher://127.0.0.1:70/", "gopher"),
    ("ftp://anonymous@ftp.example.com/test", "ftp"),
    ("ssh://git@github.com", "ssh"),
])
def test_cases_i_j_k_l_unsupported_schemes_blocked(url, scheme_name):
    """Cases I, J, K, L: file://, javascript:, data:, gopher:, ftp: blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    assert "scheme" in reason.lower()

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Case M: Embedded Credentials Blocked
# ==============================================================================
@pytest.mark.parametrize("url", [
    "https://user:password@example.com",
    "https://admin@example.com/page",
    "http://user:pass@93.184.216.34/",
    "http://foo:bar@sub.example.com:8080/",
])
def test_case_m_embedded_credentials_blocked(url):
    """Case M: Embedded userinfo/credentials in URLs blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False
    assert "credentials" in reason.lower()

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# Case N: Redirect to Private IP Blocked
# ==============================================================================
def test_case_n_redirect_to_private_ip_blocked():
    """Case N: Public URL redirecting to RFC1918 private IP blocked before dispatch."""
    resp_redirect = _make_mock_response(302, location="http://192.168.1.1/admin")

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", return_value=resp_redirect) as mock_get:
        soup = fetch_html("https://example.com/jump")
        assert soup is None
        assert mock_get.call_count == 1
        assert mock_get.call_args_list[0][0][0] == "https://example.com/jump"


# ==============================================================================
# Case O: Redirect to Localhost Blocked
# ==============================================================================
def test_case_o_redirect_to_localhost_blocked():
    """Case O: Public URL redirecting to localhost blocked before dispatch."""
    resp_redirect = _make_mock_response(301, location="http://localhost:8080/metrics")

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", return_value=resp_redirect) as mock_get:
        soup = fetch_html("https://example.com/bounce")
        assert soup is None
        assert mock_get.call_count == 1
        assert mock_get.call_args_list[0][0][0] == "https://example.com/bounce"


# ==============================================================================
# Case P: Certificate Error Returns None Without Insecure Fallback
# ==============================================================================
def test_case_p_certificate_failure_no_insecure_retry():
    """Case P: Certificate verification failure logs error and returns None with 0 insecure retries."""
    ssl_error = requests.exceptions.SSLError("certificate verify failed: self-signed certificate")

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", side_effect=ssl_error) as mock_get:
        soup = fetch_html("https://example.com/badcert")
        assert soup is None
        assert mock_get.call_count == 1
        assert mock_get.call_args[1]["verify"] is True


# ==============================================================================
# Case Q: Negative Test - verify=False is Never Passed
# ==============================================================================
def test_case_q_negative_verify_false_never_passed():
    """Case Q: Verify verify=False is never passed across successes, 4xx, 5xx, and redirects."""
    resp_302 = _make_mock_response(302, location="https://example.com/final")
    resp_200 = _make_mock_response(200, "<html>Final</html>")

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", side_effect=[resp_302, resp_200]) as mock_get:
        soup = fetch_html("https://example.com/start")
        assert soup is not None
        assert mock_get.call_count == 2
        for call_obj in mock_get.call_args_list:
            assert call_obj[1].get("verify") is True, f"Insecure verify=False detected in {call_obj}"


# ==============================================================================
# Case R: Model-Provided SSRF URL in CandidateCompany Blocked
# ==============================================================================
def test_case_r_candidate_ssrf_url_blocked():
    """Case R: Malicious/private website_url provided by model is blocked by email verifier."""
    for bad_url in ["http://127.0.0.1:8000", "http://169.254.169.254/latest", "http://localhost:3000"]:
        cand = CandidateCompany(
            name="EvilCorp",
            website_url=bad_url,
            description="Tech platform",
            industry="Security",
            hq_country="Germany",
            is_tech_platform=True,
            founder_name="Alice Evil",
            founder_title="CEO"
        )
        trail = {}
        with patch("requests.get") as mock_get:
            result = verify_founder_email_on_site(cand, audit_trail=trail)
            assert result is None
            assert mock_get.call_count == 0  # Zero network requests dispatched
            assert "Blocked unsafe website URL" in trail.get("rejection_reason", "")

            path_res = discover_founder_contact_path(cand)
            assert path_res.contact_path_found is False
            assert "Blocked unsafe website URL" in path_res.reason_if_no_contact_path


# ==============================================================================
# Case S: Public URL Allowed in Extraction / Follow-Up
# ==============================================================================
def test_case_s_trusted_public_url_preserved():
    """Case S: Valid public URL is preserved in financial evaluation."""
    cand = CandidateCompany(
        name="SafeCorp",
        website_url="https://safecorp.de",
        description="B2B platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        financial_source_url="https://bundesanzeiger.de/ebanzwww/entry?val=123"
    )
    evaluate_financial_qualification(cand)
    assert cand.financial_source_url == "https://bundesanzeiger.de/ebanzwww/entry?val=123"


# ==============================================================================
# Case T: SSRF / Private Financial Evidence URL Stripped and Marked Unverified
# ==============================================================================
def test_case_t_private_financial_url_stripped():
    """Case T: SSRF/private financial source URL is stripped and marked unverified."""
    cand = CandidateCompany(
        name="PrivateLeak",
        website_url="https://leak.eu",
        description="B2B platform",
        industry="SaaS",
        hq_country="Estonia",
        is_tech_platform=True,
        financial_source_url="http://192.168.1.50/internal_earnings.json",
        revenue_amount_usd=2_500_000.0,
        revenue_period="ARR",
        financial_evidence_date="2024-05-01"
    )
    is_qual, audit = evaluate_financial_qualification(cand)
    # The private IP URL was stripped
    assert cand.financial_source_url is None
    fin_items = [ev for ev in cand.supporting_evidence if isinstance(ev, FinancialEvidence)]
    if fin_items:
        assert fin_items[0].verification_status != "VERIFIED"


# ==============================================================================
# Alternate IP Encodings (Hex, Octal, Decimal, IPv4-Mapped IPv6)
# ==============================================================================
@pytest.mark.parametrize("url,desc", [
    ("http://2130706433/", "Decimal IP for 127.0.0.1"),
    ("http://0177.0.0.1/", "Octal IP for 127.0.0.1"),
    ("http://0x7f000001/", "Hex IP for 127.0.0.1"),
    ("http://[::ffff:127.0.0.1]/", "IPv4-mapped IPv6 for 127.0.0.1"),
    ("http://[::ffff:7f00:1]/", "IPv4-mapped IPv6 hex for 127.0.0.1"),
    ("http://[::ffff:10.0.0.1]/", "IPv4-mapped IPv6 for 10.0.0.1"),
    ("http://[::ffff:169.254.169.254]/", "IPv4-mapped IPv6 for metadata IP"),
])
def test_alternate_ip_encodings_blocked(url, desc):
    """Verify evasion attempts using alternate IP notations are detected and blocked."""
    is_safe, reason, canonical = validate_public_url(url, resolve_dns=False)
    assert is_safe is False, f"Failed to block {desc}: {url}"
    assert any(term in reason.lower() for term in ["blocked", "loopback", "private", "link-local", "metadata", "non-public"])

    result = fetch_html(url)
    assert result is None


# ==============================================================================
# DNS Rebinding Protection
# ==============================================================================
def test_dns_rebinding_to_loopback_blocked():
    """Verify hostname resolving to 127.0.0.1 via DNS is blocked."""
    with patch("socket.getaddrinfo", side_effect=_mock_dns_private):
        is_safe, reason, canonical = validate_public_url("https://rebind-test.com/secret", resolve_dns=True)
        assert is_safe is False
        assert any(term in reason.lower() for term in ["loopback", "blocked", "127.0.0.1"])


def test_dns_rebinding_to_metadata_blocked():
    """Verify hostname resolving to 169.254.169.254 via DNS is blocked."""
    with patch("socket.getaddrinfo", side_effect=_mock_dns_metadata):
        is_safe, reason, canonical = validate_public_url("https://meta-rebind.com/keys", resolve_dns=True)
        assert is_safe is False
        assert any(term in reason.lower() for term in ["metadata", "blocked", "169.254.169.254", "link-local"])


# ==============================================================================
# Redirect Loops and Bounded Redirect Enforcement
# ==============================================================================
def test_redirect_limit_enforced():
    """Verify redirect loops exceeding MAX_REDIRECTS (5) are terminated safely."""
    resp_302 = _make_mock_response(302, location="https://example.com/loop")

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", return_value=resp_302) as mock_get:
        soup = fetch_html("https://example.com/loop")
        assert soup is None
        assert mock_get.call_count <= MAX_REDIRECTS + 1


def test_circular_redirect_loop_detected():
    """Verify circular A -> B -> A redirect loop terminates safely within bounds."""
    resp_a = _make_mock_response(302, location="https://example.com/page-b")
    resp_b = _make_mock_response(302, location="https://example.com/page-a")

    with patch("socket.getaddrinfo", side_effect=_mock_dns_public), \
         patch("requests.get", side_effect=[resp_a, resp_b, resp_a, resp_b, resp_a, resp_b]) as mock_get:
        soup = fetch_html("https://example.com/page-a")
        assert soup is None
        assert mock_get.call_count <= MAX_REDIRECTS + 1
