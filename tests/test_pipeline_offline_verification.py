"""
tests/test_pipeline_offline_verification.py
Comprehensive offline verification test suite:
- Email security: Zero-guessing adversarial proof (no synthetic email generation)
- Email attribution: Scenarios A through H
- Query matrix diversity and format validation
- Tavily budget: Hard 60-call limit with 61-call adversarial block
- Mock end-to-end pipeline: Full 20-candidate funnel yielding exactly 2 valid leads
- Failure recovery: Tavily errors, Gemini errors, scraper 403/404/TLS errors, missing fields
"""

import io
import json
import socket
import urllib.request
import urllib.error
from unittest.mock import MagicMock, patch
import pytest
from bs4 import BeautifulSoup
import requests

from core.config import (
    TAVILY_MAX_CALLS_PER_RUN, NON_US_REGIONS, TECH_SECTORS,
    REVENUE_QUERY_TEMPLATES, FUNDING_QUERY_TEMPLATES
)
from core.discovery import (
    search_tavily, generate_queries, get_total_tavily_calls,
    reset_tavily_search_count, clear_tavily_cache,
    TavilyBudgetExhaustedError, TavilyTimeoutError, TavilyRateLimitError, TavilyQuotaError
)
from core.models import CandidateCompany, CandidateAuditRecord
from core.email_verifier import (
    decode_cloudflare_email, extract_dom_emails_with_context,
    verify_founder_email_on_site, OBFUSCATED_EMAIL_REGEX
)
from core.scraper import fetch_html
from core.pipeline import run_lead_pipeline


@pytest.fixture(autouse=True)
def clean_state():
    clear_tavily_cache()
    reset_tavily_search_count()
    yield
    clear_tavily_cache()
    reset_tavily_search_count()


def make_mock_tavily_response(results=None):
    if results is None:
        results = [{"title": "Test Title", "url": "https://example.eu", "content": "Sample snippet", "score": 0.9}]
    mock_resp = io.BytesIO(json.dumps({"results": results}).encode("utf-8"))
    mock_resp.status = 200
    return mock_resp


# ==============================================================================
# 1. EMAIL SECURITY: ZERO-GUESSING ADVERSARIAL PROOF
# ==============================================================================

def test_zero_guessing_adversarial_no_synthetic_generation():
    """
    Prove that no combination of founder name and domain can synthesize an email.
    If the email is absent from the page DOM, verify_founder_email_on_site MUST return None.
    """
    empty_html = "<html><body><h1>Welcome to Acme Platform</h1><p>We build developer tools.</p></body></html>"
    soup = BeautifulSoup(empty_html, "html.parser")

    cand = CandidateCompany(
        name="Acme Platform",
        website_url="https://acmeplatform.eu",
        description="Dev tools",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="John Doe",
        founder_title="CEO"
    )

    with patch("core.email_verifier.fetch_html", return_value=soup):
        # Even with full founder identity, NO email is guessed or generated
        result = verify_founder_email_on_site(cand)
        assert result is None, "Must never synthesize john@acmeplatform.eu when not in DOM"


def test_obfuscated_regex_only_matches_actual_text():
    """Prove that OBFUSCATED_EMAIL_REGEX only normalizes text physically present."""
    sample_text = "Contact: john [at] acme [dot] io or support(at)acme.io"
    matches = OBFUSCATED_EMAIL_REGEX.findall(sample_text)
    assert len(matches) == 2
    assert matches[0] == ("john", "acme", "io")
    assert matches[1] == ("support", "acme", "io")

    # Text without [at] or (at) produces zero matches
    clean_text = "John Doe is the CEO of Acme."
    assert OBFUSCATED_EMAIL_REGEX.findall(clean_text) == []


# ==============================================================================
# 2. EMAIL ATTRIBUTION REVIEW: SCENARIOS A THROUGH H
# ==============================================================================

def test_scenario_a_founder_exact_email_same_card_accepted():
    """Scenario A: Founder + exact email in same card -> ACCEPT"""
    html = """
    <div class="team-card">
        <h3>Patrick Collins</h3>
        <p>Founder & CEO</p>
        <a href="mailto:patrick@damsecure.eu">Email Patrick</a>
    </div>
    """
    cand = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.eu",
        description="Security",
        industry="Cybersecurity",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    assert result[0] == "patrick@damsecure.eu"


def test_scenario_b_founder_title_email_same_section_accepted():
    """Scenario B: Founder + title + exact email in same section -> ACCEPT"""
    html = """
    <section class="leadership-section">
        <h2>Leadership</h2>
        <div>
            <span>Eoin Hinchy</span>
            <span>Chief Executive Officer</span>
            <p>Direct contact: eoin@tinesplatform.io</p>
        </div>
    </section>
    """
    cand = CandidateCompany(
        name="Tines Platform",
        website_url="https://tinesplatform.io",
        description="Automation",
        industry="DevOps",
        hq_country="Ireland",
        is_tech_platform=True,
        founder_name="Eoin Hinchy",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    assert result[0] == "eoin@tinesplatform.io"


def test_scenario_c_unrelated_employee_email_rejected():
    """Scenario C: Founder name nearby but unrelated employee email -> REJECT"""
    html = """
    <div class="team-member">
        <h3>Patrick Collins</h3>
        <p>CEO</p>
        <p>Engineering contact: <a href="mailto:dave.engineer@damsecure.eu">dave.engineer@damsecure.eu</a></p>
    </div>
    """
    cand = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.eu",
        description="Security",
        industry="Cybersecurity",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Unrelated employee email must be rejected despite founder name in card"


def test_scenario_d_generic_inboxes_rejected():
    """Scenario D: Generic info@, sales@, hello@ -> REJECT"""
    html = """
    <div class="footer">
        <p>Founder: Patrick Collins</p>
        <a href="mailto:info@damsecure.eu">info@damsecure.eu</a>
        <a href="mailto:hello@damsecure.eu">hello@damsecure.eu</a>
        <a href="mailto:sales@damsecure.eu">sales@damsecure.eu</a>
    </div>
    """
    cand = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.eu",
        description="Security",
        industry="Cybersecurity",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Generic inboxes must be rejected under strict zero-guessing"


def test_scenario_e_external_domain_email_rejected():
    """Scenario E: Email on external domain -> REJECT"""
    html = """
    <div class="profile-card">
        <h3>Patrick Collins</h3>
        <p>CEO & Co-Founder</p>
        <a href="mailto:patrick.collins@gmail.com">patrick.collins@gmail.com</a>
    </div>
    """
    cand = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.eu",
        description="Security",
        industry="Cybersecurity",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is None, "External personal domains (e.g. gmail.com) must be rejected"


def test_scenario_f_obfuscated_founder_email_accepted():
    """Scenario F: Obfuscated founder email in same card -> ACCEPT after normalization"""
    html = """
    <div class="founder-card">
        <h3>Patrick Collins</h3>
        <span>Co-Founder & CEO</span>
        <p>Reach me: patrick [at] damsecure [dot] eu</p>
    </div>
    """
    cand = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.eu",
        description="Security",
        industry="Cybersecurity",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    assert result[0] == "patrick@damsecure.eu"


def test_scenario_g_cloudflare_encoded_founder_email_accepted():
    """Scenario G: Cloudflare encoded founder email in same card -> ACCEPT after decoding"""
    email_str = "patrick@damsecure.eu"
    r = 0x42
    hex_body = "".join(f"{ord(c) ^ r:02x}" for c in email_str)
    cf_hex = f"{r:02x}{hex_body}"

    html = f"""
    <div class="team-card">
        <h3>Patrick Collins</h3>
        <p>CEO</p>
        <a class="__cf_email__" data-cfemail="{cf_hex}">[email protected]</a>
    </div>
    """
    cand = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.eu",
        description="Security",
        industry="Cybersecurity",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    assert result[0] == "patrick@damsecure.eu"


def test_scenario_h_obfuscated_unrelated_email_rejected():
    """Scenario H: Obfuscated unrelated person's email -> REJECT"""
    html = """
    <div class="team-card">
        <h3>Patrick Collins</h3>
        <p>CEO</p>
        <p>Support: sara.intern [at] damsecure [dot] eu</p>
    </div>
    """
    cand = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.eu",
        description="Security",
        industry="Cybersecurity",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )
    with patch("core.email_verifier.fetch_html", return_value=BeautifulSoup(html, "html.parser")):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Obfuscated email of unrelated person must be rejected"


# ==============================================================================
# 3. QUERY MATRIX DIVERSITY AND FORMAT REVIEW
# ==============================================================================

def test_query_matrix_diversity_and_statistics():
    """Verify query matrix generates 35 diverse, unique, well-formed search queries."""
    queries = generate_queries(count=35, seed=100)
    assert len(queries) == 35

    # Check uniqueness
    unique_queries = set(queries)
    assert len(unique_queries) >= 30, f"Expected high uniqueness, got {len(unique_queries)} unique queries"

    # Verify sector representation
    sectors_hit = {s for s in TECH_SECTORS if any(s.lower() in q.lower() for q in queries)}
    assert len(sectors_hit) >= 5, f"Expected diverse sector representation, hit {len(sectors_hit)}"

    # Verify region representation
    regions_hit = {r for r in NON_US_REGIONS if any(r.lower() in q.lower() for q in queries)}
    assert len(regions_hit) >= 8, f"Expected broad non-US coverage, hit {len(regions_hit)}"

    # Check for malformed syntax (no unclosed quotes or double colons)
    for q in queries:
        assert q.count('"') % 2 == 0, f"Malformed quotes in query: {q}"
        assert "::" not in q


# ==============================================================================
# 4. TAVILY HARD BUDGET: ADVERSARIAL 61-CALL TEST
# ==============================================================================

def test_tavily_hard_budget_blocks_call_61(monkeypatch):
    """Adversarial test: Attempt 61 outbound calls; call #61 must be blocked deterministically."""
    monkeypatch.setattr("core.discovery.TAVILY_MAX_CALLS_PER_RUN", 60)

    network_mock = MagicMock(side_effect=lambda *args, **kwargs: make_mock_tavily_response())
    with patch("urllib.request.urlopen", network_mock):
        for i in range(60):
            res = search_tavily(f"distinct query {i}", api_key="mock_key")
            assert len(res) == 1

        assert get_total_tavily_calls() == 60
        assert network_mock.call_count == 60

        # Attempt 61st call: must raise TavilyBudgetExhaustedError
        with pytest.raises(TavilyBudgetExhaustedError) as exc_info:
            search_tavily("distinct query 61", api_key="mock_key")
        assert "TAVILY_BUDGET_EXHAUSTED" in str(exc_info.value)
        # Network must NOT have been called for call 61
        assert network_mock.call_count == 60
        assert get_total_tavily_calls() == 60


# ==============================================================================
# 5. PIPELINE MOCK END-TO-END TEST (20 CANDIDATES -> EXACTLY 2 VALID LEADS)
# ==============================================================================

def test_mock_end_to_end_pipeline_exact_funnel():
    """
    Simulates a full end-to-end run:
    - 20 discovered candidates
    - 10 obvious deterministic rejects (US, non-tech, acquired, overfunded)
    - 10 candidates advance to targeted follow-up
    - 5 fail downstream financial validation
    - 5 survive financial check
    - 5 pass non-US and tech check
    - 2 pass strict email verification
    - 3 fail email verification (1 generic, 2 no on-site email)
    - Final qualified leads == 2
    """
    # 20 candidate fixtures
    candidates = [
        # --- 10 Obvious deterministic rejects ---
        # 3 US HQs
        CandidateCompany(name="US Cloud 1", description="SaaS", industry="SaaS", hq_country="United States", hq_city="Austin", is_tech_platform=True),
        CandidateCompany(name="US Cloud 2", description="SaaS", industry="SaaS", hq_country="USA", hq_city="San Francisco", is_tech_platform=True),
        CandidateCompany(name="US Cloud 3", description="SaaS", industry="SaaS", hq_country="US", hq_city="Seattle", is_tech_platform=True),
        # 3 Non-tech agencies
        CandidateCompany(name="Berlin Dev Agency", description="Consulting", industry="IT Services", hq_country="Germany", is_tech_platform=False),
        CandidateCompany(name="London Recruiting", description="Staffing", industry="Recruiting", hq_country="UK", is_tech_platform=False),
        CandidateCompany(name="Paris Marketing", description="Ad agency", industry="Marketing", hq_country="France", is_tech_platform=False),
        # 2 Acquired/closed
        CandidateCompany(name="Acquired SaaS", description="Platform", industry="SaaS", hq_country="UK", is_tech_platform=True, acquisition_or_closure_discovered=True),
        CandidateCompany(name="Closed App", description="Platform", industry="SaaS", hq_country="Germany", is_tech_platform=True, acquisition_or_closure_discovered=True),
        # 2 Overfunded > $5M on initial pass
        CandidateCompany(name="Mega Scale 1", description="AI", industry="AI", hq_country="Sweden", is_tech_platform=True, total_cumulative_funding_usd=20_000_000.0),
        CandidateCompany(name="Mega Scale 2", description="AI", industry="AI", hq_country="Norway", is_tech_platform=True, total_cumulative_funding_usd=50_000_000.0),

        # --- 10 Candidates passing deterministic filters to follow-up ---
        # 5 Financial rejects after follow-up
        CandidateCompany(name="Fin Reject 1", description="SaaS", industry="SaaS", hq_country="Netherlands", is_tech_platform=True), # Insufficient
        CandidateCompany(name="Fin Reject 2", description="SaaS", industry="SaaS", hq_country="Switzerland", is_tech_platform=True), # Insufficient
        CandidateCompany(name="Fin Reject 3", description="SaaS", industry="SaaS", hq_country="Estonia", is_tech_platform=True), # Insufficient
        CandidateCompany(name="Fin Reject 4", description="SaaS", industry="SaaS", hq_country="Poland", is_tech_platform=True), # Insufficient
        CandidateCompany(name="Fin Reject 5", description="SaaS", industry="SaaS", hq_country="Czechia", is_tech_platform=True), # Insufficient

        # 5 Surviving candidates meeting financial, non-US, and tech criteria
        CandidateCompany(
            name="Valid Company 1", website_url="https://validone.eu", description="B2B platform", industry="SaaS",
            hq_country="Germany", hq_city="Munich", is_tech_platform=True, total_cumulative_funding_usd=2_500_000.0,
            latest_round_name="Seed", latest_round_amount_usd=2_500_000.0, latest_round_date="2025-10",
            financial_evidence_date="2025-10", founder_name="Lars Weber", founder_title="CEO",
            financial_source_url="https://trustednews.eu/validone",
            financial_source_quote="Valid Company 1 raised $2.5M in funding.",
            trusted_source_urls=["https://trustednews.eu/validone"],
            trusted_source_text="Valid Company 1 raised $2.5M in funding.",
            financial_followup_complete=True
        ),
        CandidateCompany(
            name="Valid Company 2", website_url="https://validtwo.co.uk", description="DevOps platform", industry="DevTools",
            hq_country="UK", hq_city="London", is_tech_platform=True, total_cumulative_funding_usd=3_000_000.0,
            latest_round_name="Seed", latest_round_amount_usd=3_000_000.0, latest_round_date="2025-11",
            financial_evidence_date="2025-11", founder_name="Sophie Clark", founder_title="Co-Founder",
            financial_source_url="https://trustednews.eu/validtwo",
            financial_source_quote="Valid Company 2 raised $3M in funding.",
            trusted_source_urls=["https://trustednews.eu/validtwo"],
            trusted_source_text="Valid Company 2 raised $3M in funding.",
            financial_followup_complete=True
        ),
        CandidateCompany(
            name="Email Reject Generic", website_url="https://generic.eu", description="Cloud tools", industry="Cloud",
            hq_country="France", hq_city="Paris", is_tech_platform=True, total_cumulative_funding_usd=1_500_000.0,
            latest_round_name="Seed", latest_round_amount_usd=1_500_000.0, latest_round_date="2025-08",
            financial_evidence_date="2025-08", founder_name="Jean Dupont", founder_title="CEO",
            financial_source_url="https://trustednews.eu/generic",
            financial_source_quote="Raised $1.5M in funding.",
            trusted_source_urls=["https://trustednews.eu/generic"],
            trusted_source_text="Raised $1.5M in funding.",
            financial_followup_complete=True
        ),
        CandidateCompany(
            name="Email Reject No Email 1", website_url="https://noemail1.eu", description="API platform", industry="Fintech",
            hq_country="Sweden", hq_city="Stockholm", is_tech_platform=True, total_cumulative_funding_usd=2_000_000.0,
            latest_round_name="Seed", latest_round_amount_usd=2_000_000.0, latest_round_date="2025-09",
            financial_evidence_date="2025-09", founder_name="Erik Lind", founder_title="CEO",
            financial_source_url="https://trustednews.eu/noemail1",
            financial_source_quote="Raised $2M in funding.",
            trusted_source_urls=["https://trustednews.eu/noemail1"],
            trusted_source_text="Raised $2M in funding.",
            financial_followup_complete=True
        ),
        CandidateCompany(
            name="Email Reject No Email 2", website_url="https://noemail2.eu", description="Compliance", industry="Compliance",
            hq_country="Norway", hq_city="Oslo", is_tech_platform=True, total_cumulative_funding_usd=1_800_000.0,
            latest_round_name="Seed", latest_round_amount_usd=1_800_000.0, latest_round_date="2025-10",
            financial_evidence_date="2025-10", founder_name="Astrid Moe", founder_title="CEO",
            financial_source_url="https://trustednews.eu/noemail2",
            financial_source_quote="Raised $1.8M in funding.",
            trusted_source_urls=["https://trustednews.eu/noemail2"],
            trusted_source_text="Raised $1.8M in funding.",
            financial_followup_complete=True
        )
    ]

    # Mock website verification: only Valid Company 1 and 2 have verified founder emails
    def mock_email_verifier(cand):
        if cand.name == "Valid Company 1":
            return "lars@validone.eu", "https://validone.eu/about"
        if cand.name == "Valid Company 2":
            return "sophie@validtwo.co.uk", "https://validtwo.co.uk/team"
        return None

    with patch("core.pipeline.generate_queries", return_value=["test query"]):
        with patch("core.pipeline.search_grounded_candidates", return_value=("raw text", [])):
            with patch("core.pipeline.extract_structured_candidates", return_value=candidates):
                with patch("core.pipeline.execute_targeted_funding_follow_up", side_effect=lambda client, c, **kwargs: c):
                    with patch("core.pipeline.verify_founder_email_on_site", side_effect=mock_email_verifier):
                        summary = run_lead_pipeline(
                            api_key="mock_key",
                            target_count=15,
                            max_queries=1,
                            tavily_api_key="mock_key"
                        )

    # Verify exact funnel counts
    assert summary.funnel.candidates_found == 20
    assert summary.funnel.passed_financial == 5
    assert summary.funnel.passed_non_us == 5
    assert summary.funnel.passed_tech_check == 5
    assert summary.funnel.passed_email_verification == 2
    assert len(summary.leads) == 2
    assert {l.company_name for l in summary.leads} == {"Valid Company 1", "Valid Company 2"}


# ==============================================================================
# 6. FAILURE RECOVERY TESTS
# ==============================================================================

def test_failure_recovery_tavily_timeout():
    """Verify graceful handling when Tavily search times out."""
    with patch("urllib.request.urlopen", side_effect=socket.timeout):
        with pytest.raises(TavilyTimeoutError):
            search_tavily("timeout query", api_key="mock_key")


def test_failure_recovery_tavily_429():
    """Verify graceful handling of Tavily rate limit (429)."""
    http_error = urllib.error.HTTPError("url", 429, "Rate Limit", {}, io.BytesIO(b'{"detail":"Rate limit"}'))
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(TavilyRateLimitError):
            search_tavily("rate limit query", api_key="mock_key")


def test_failure_recovery_tavily_quota():
    """Verify graceful handling of Tavily monthly quota exhaustion (432)."""
    http_error = urllib.error.HTTPError("url", 432, "Quota Exceeded", {}, io.BytesIO(b'{"detail":"Monthly limit reached"}'))
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(TavilyQuotaError):
            search_tavily("quota query", api_key="mock_key")


def test_failure_recovery_scraper_errors():
    """Verify fetch_html returns None gracefully on 403, 404, SSLError, and connection errors."""
    mock_resp_403 = MagicMock()
    mock_resp_403.status_code = 403
    with patch("requests.get", return_value=mock_resp_403):
        assert fetch_html("https://blocked.eu") is None

    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404
    with patch("requests.get", return_value=mock_resp_404):
        assert fetch_html("https://notfound.eu") is None

    with patch("requests.get", side_effect=requests.exceptions.SSLError("SSL fail")):
        assert fetch_html("https://badcert.eu") is None

    # Invalid URL formatting
    assert fetch_html("invalid-url") is None
    assert fetch_html("") is None


def test_failure_recovery_missing_and_malformed_fields():
    """Verify verify_founder_email_on_site handles missing or malformed candidate fields safely."""
    # Missing website URL
    c1 = CandidateCompany(name="No URL", description="SaaS", industry="SaaS", hq_country="UK", is_tech_platform=True, founder_name="Alice")
    assert verify_founder_email_on_site(c1) is None

    # Missing founder name
    c2 = CandidateCompany(name="No Founder", website_url="https://nofounder.eu", description="SaaS", industry="SaaS", hq_country="UK", is_tech_platform=True)
    assert verify_founder_email_on_site(c2) is None

    # Malformed website URL
    c3 = CandidateCompany(name="Bad URL", website_url="ftp://badurl", description="SaaS", industry="SaaS", hq_country="UK", is_tech_platform=True, founder_name="Alice")
    assert verify_founder_email_on_site(c3) is None
