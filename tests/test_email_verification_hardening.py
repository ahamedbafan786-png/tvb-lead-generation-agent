"""
tests/test_email_verification_hardening.py
Deterministic unit test suite covering the 10 failure modes and attribution scenarios
for on-site founder email verification.
ZERO live network calls — all responses mocked with BeautifulSoup/requests mocks.
"""

import pytest
from unittest.mock import patch
from bs4 import BeautifulSoup

from core.models import CandidateCompany
from core.email_verifier import (
    verify_founder_email_on_site,
    extract_dom_emails_with_context,
    get_enclosing_card_context
)
from core.scraper import find_subpages


# ==============================================================================
# 1. Founder + email in adjacent DOM siblings -> ACCEPT
# ==============================================================================
def test_1_adjacent_dom_siblings_accepted():
    html = """
    <html>
      <body>
        <div class="team-grid">
          <div class="bio-block">
            <h3>Patrick Collins</h3>
            <p class="role">Chief Executive Officer</p>
            <a href="mailto:pcollins@damsecure.ai">Contact Patrick</a>
          </div>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.ai",
        description="AI code security platform",
        industry="Cybersecurity",
        hq_country="Australia",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    assert result is not None
    email, page_url = result
    assert email == "pcollins@damsecure.ai"
    assert page_url == "https://damsecure.ai"


# ==============================================================================
# 2. Founder + email in nested team card -> ACCEPT
# ==============================================================================
def test_2_nested_team_card_accepted():
    html = """
    <html>
      <body>
        <div class="team-card">
          <div class="info-wrapper">
            <span class="founder-title">Nick Fogle</span>
            <span class="founder-role">Co-Founder & CEO</span>
          </div>
          <div class="action-buttons">
            <a href="mailto:nick@churnkey.co">Email Nick</a>
          </div>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Churnkey",
        website_url="https://churnkey.co",
        description="Subscription retention platform",
        industry="B2B SaaS",
        hq_country="Canada",
        is_tech_platform=True,
        founder_name="Nick Fogle",
        founder_title="CEO & Co-Founder"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    assert result is not None
    email, page_url = result
    assert email == "nick@churnkey.co"
    assert page_url == "https://churnkey.co"


# ==============================================================================
# 3. Founder name split across multiple spans -> ACCEPT
# ==============================================================================
def test_3_founder_name_split_across_spans_accepted():
    html = """
    <html>
      <body>
        <article class="founder-profile">
          <h2><span>Eugene</span> <span>Zolotarenko</span></h2>
          <div class="subtitle">Founder & Managing Director</div>
          <a href="mailto:eugene@outrank.so">eugene@outrank.so</a>
        </article>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Outrank",
        website_url="https://outrank.so",
        description="SEO automation platform",
        industry="B2B SaaS",
        hq_country="United Kingdom",
        is_tech_platform=True,
        founder_name="Eugene Zolotarenko",
        founder_title="Founder"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    assert result is not None
    email, page_url = result
    assert email == "eugene@outrank.so"
    assert page_url == "https://outrank.so"


# ==============================================================================
# 4. Generic company inbox -> REJECT
# ==============================================================================
def test_4_generic_company_inbox_rejected():
    html = """
    <html>
      <body>
        <div class="footer">
          <p>Leadership: Patrick Collins (CEO)</p>
          <a href="mailto:info@damsecure.ai">info@damsecure.ai</a>
          <a href="mailto:support@damsecure.ai">support@damsecure.ai</a>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.ai",
        description="AI code security platform",
        industry="Cybersecurity",
        hq_country="Australia",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    assert result is None, "Generic addresses (info@, support@) must be rejected without direct personal email context"


# ==============================================================================
# 5. Unrelated employee email -> REJECT
# ==============================================================================
def test_5_unrelated_employee_email_rejected():
    html = """
    <html>
      <body>
        <div class="card member">
          <h3>Sarah Connor</h3>
          <p class="role">Lead Security Engineer</p>
          <a href="mailto:sarah@damsecure.ai">sarah@damsecure.ai</a>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.ai",
        description="AI code security platform",
        industry="Cybersecurity",
        hq_country="Australia",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    assert result is None, "Non-founder employee email must not be attributed to founder"


# ==============================================================================
# 6. Founder name absent from context -> REJECT
# ==============================================================================
def test_6_founder_name_absent_from_context_rejected():
    html = """
    <html>
      <body>
        <div class="contact-box">
          <h3>Security Operations</h3>
          <p>Managing Director</p>
          <a href="mailto:security-lead@damsecure.ai">security-lead@damsecure.ai</a>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.ai",
        description="AI code security platform",
        industry="Cybersecurity",
        hq_country="Australia",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    assert result is None, "Email in context without founder name must be rejected"


# ==============================================================================
# 7. Founder email found on /leadership/ subpage -> ACCEPT
# ==============================================================================
def test_7_founder_email_found_on_leadership_subpage_accepted():
    home_html = """
    <html>
      <body>
        <header><a href="/leadership">Leadership</a></header>
      </body>
    </html>
    """
    leadership_html = """
    <html>
      <body>
        <div class="leader-card">
          <h3>Patrick Collins</h3>
          <p>CEO & Co-Founder</p>
          <a href="mailto:patrick@damsecure.ai">patrick@damsecure.ai</a>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.ai",
        description="AI code security platform",
        industry="Cybersecurity",
        hq_country="Australia",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )

    home_soup = BeautifulSoup(home_html, "html.parser")
    leadership_soup = BeautifulSoup(leadership_html, "html.parser")

    def mock_fetch(url):
        if "leadership" in url:
            return leadership_soup
        return home_soup

    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        result = verify_founder_email_on_site(candidate)

    assert result is not None
    email, page_url = result
    assert email == "patrick@damsecure.ai"
    assert "leadership" in page_url


# ==============================================================================
# 8. Founder email found on /meet-the-team/ subpage -> ACCEPT
# ==============================================================================
def test_8_founder_email_found_on_meet_the_team_subpage_accepted():
    home_html = """
    <html>
      <body>
        <nav><a href="/people">Meet the Team</a></nav>
      </body>
    </html>
    """
    people_html = """
    <html>
      <body>
        <div class="member-box">
          <h3>Nick Fogle</h3>
          <p>Founder & CEO</p>
          <a href="mailto:nick@churnkey.co">Contact Nick</a>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Churnkey",
        website_url="https://churnkey.co",
        description="Subscription retention platform",
        industry="B2B SaaS",
        hq_country="Canada",
        is_tech_platform=True,
        founder_name="Nick Fogle",
        founder_title="CEO"
    )

    home_soup = BeautifulSoup(home_html, "html.parser")
    people_soup = BeautifulSoup(people_html, "html.parser")

    def mock_fetch(url):
        if "people" in url:
            return people_soup
        return home_soup

    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        result = verify_founder_email_on_site(candidate)

    assert result is not None
    email, page_url = result
    assert email == "nick@churnkey.co"
    assert "people" in page_url


# ==============================================================================
# 9. Email appears only on unrelated external domain -> REJECT
# ==============================================================================
def test_9_email_on_unrelated_external_domain_rejected():
    html = """
    <html>
      <body>
        <div class="card founder">
          <h3>Patrick Collins</h3>
          <p>CEO</p>
          <a href="mailto:patrick@gmail.com">patrick@gmail.com</a>
          <a href="mailto:patrick@external-investor.com">patrick@external-investor.com</a>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.ai",
        description="AI code security platform",
        industry="Cybersecurity",
        hq_country="Australia",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    assert result is None, "Emails on external domains (gmail.com, third-party) must be strictly rejected"


# ==============================================================================
# 10. Generated / synthetic email -> REJECT
# ==============================================================================
def test_10_generated_synthetic_email_rejected():
    html = """
    <html>
      <body>
        <div class="profile">
          <h3>Patrick Collins</h3>
          <p>CEO & Founder</p>
          <p>Follow me on Twitter or connect via LinkedIn.</p>
        </div>
      </body>
    </html>
    """
    candidate = CandidateCompany(
        name="Dam Secure",
        website_url="https://damsecure.ai",
        description="AI code security platform",
        industry="Cybersecurity",
        hq_country="Australia",
        is_tech_platform=True,
        founder_name="Patrick Collins",
        founder_title="CEO"
    )

    with patch("core.email_verifier.fetch_html") as mock_fetch, \
         patch("core.email_verifier.find_subpages", return_value=[]):
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")
        result = verify_founder_email_on_site(candidate)

    # Verifier MUST return None and NEVER synthesize patrick@damsecure.ai or p.collins@damsecure.ai
    assert result is None, "Zero-guessing mandate: Verifier must never synthesize unevidenced emails"
