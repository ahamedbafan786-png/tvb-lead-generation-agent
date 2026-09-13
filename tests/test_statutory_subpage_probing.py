"""
tests/test_statutory_subpage_probing.py
Deterministic mocked unit tests for bounded direct statutory/contact subpage probing:
1. SPA homepage with no anchors -> direct /impressum probe attempted
2. direct /contact probe found -> page included in email scan
3. direct probe remains same-domain
4. direct probe cannot escape to another domain
5. 404 direct probe is handled gracefully
6. direct probe respects maximum-page budget
7. founder email discovered on direct statutory page can be accepted
8. generic email on statutory page remains rejected
9. unrelated employee email on statutory page remains rejected
10. missing founder email still returns rejection
"""

from bs4 import BeautifulSoup
from unittest.mock import patch, MagicMock
from core.models import CandidateCompany
from core.email_verifier import verify_founder_email_on_site
from core.scraper import probe_statutory_subpages, MAX_DIRECT_PROBES, MAX_DISCOVERED_SUBPAGES, is_same_domain, fetch_html


def test_1_spa_homepage_no_anchors_impressum_probe_attempted():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    spa_html = "<html><body><div id=\"root\"></div></body></html>"
    home_soup = BeautifulSoup(spa_html, "html.parser")

    def mock_fetch(url, **kwargs):
        if url == "https://webertech.de":
            return home_soup
        return None

    audit = {}
    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        verify_founder_email_on_site(cand, audit_trail=audit)

    assert audit["linked_pages_discovered"] == []
    assert any("/impressum" in p for p in audit["direct_probes_attempted"])


def test_2_direct_contact_probe_found_included_in_scan():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    spa_html = "<html><body><div id=\"app\"></div></body></html>"
    contact_html = "<html><body><p>Contact WeberTech</p></body></html>"
    home_soup = BeautifulSoup(spa_html, "html.parser")
    contact_soup = BeautifulSoup(contact_html, "html.parser")

    def mock_fetch(url, **kwargs):
        if "/contact" in url:
            return contact_soup
        if url == "https://webertech.de":
            return home_soup
        return None

    audit = {}
    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        verify_founder_email_on_site(cand, audit_trail=audit)

    assert any(p.endswith("/contact") for p in audit["pages_successfully_fetched"])


def test_3_direct_probe_remains_same_domain():
    base = "https://webertech.de/en/overview"
    probes_attempted, _ = probe_statutory_subpages(base, fetcher=lambda u, **kw: None)
    assert len(probes_attempted) > 0
    for p in probes_attempted:
        assert p.startswith("https://webertech.de/")
        assert is_same_domain(p, base)


def test_4_direct_probe_cannot_escape_to_another_domain():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_resp.url = "https://external-tracker.com/login"
    mock_resp.text = "<html><body>External Page</body></html>"

    with patch("requests.get", return_value=mock_resp):
        soup = fetch_html("https://webertech.de/impressum", enforce_domain="webertech.de")
        assert soup is None


def test_5_404_direct_probe_handled_gracefully():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    home_soup = BeautifulSoup("<html><body></body></html>", "html.parser")

    def mock_fetch_404(url, **kwargs):
        if url == "https://webertech.de":
            return home_soup
        return None

    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch_404):
        res = verify_founder_email_on_site(cand)
        assert res is None


def test_6_direct_probe_respects_maximum_page_budget():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    dummy_soup = BeautifulSoup("<html><body><p>Hello</p></body></html>", "html.parser")

    def mock_fetch_all(url, **kwargs):
        return dummy_soup

    audit = {}
    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch_all):
        verify_founder_email_on_site(cand, audit_trail=audit)

    assert len(audit["direct_probes_attempted"]) <= MAX_DIRECT_PROBES
    assert len(audit["pages_successfully_fetched"]) <= MAX_DISCOVERED_SUBPAGES


def test_7_founder_email_discovered_on_direct_statutory_page_accepted():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    spa_home = "<html><body><div id=\"root\"></div></body></html>"
    impressum_html = """
    <html>
      <body>
        <div class="imprint">
          <h3>WeberTech GmbH</h3>
          <p>Geschäftsführer: Klaus Weber (CEO)</p>
          <p>Direktkontakt: <a href="mailto:klaus@webertech.de">klaus@webertech.de</a></p>
        </div>
      </body>
    </html>
    """
    home_soup = BeautifulSoup(spa_home, "html.parser")
    impressum_soup = BeautifulSoup(impressum_html, "html.parser")

    def mock_fetch(url, **kwargs):
        if "/impressum" in url:
            return impressum_soup
        if url == "https://webertech.de":
            return home_soup
        return None

    audit = {}
    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        result = verify_founder_email_on_site(cand, audit_trail=audit)

    assert result is not None
    email, page_url = result
    assert email == "klaus@webertech.de"
    assert page_url == "https://webertech.de/impressum"
    assert audit["attribution_result"].startswith("ACCEPTED")


def test_8_generic_email_on_statutory_page_rejected():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    spa_home = "<html><body><div id=\"root\"></div></body></html>"
    impressum_html = """
    <html>
      <body>
        <div class="imprint">
          <h3>WeberTech GmbH</h3>
          <p>Geschäftsführer: Klaus Weber</p>
          <a href="mailto:info@webertech.de">info@webertech.de</a>
        </div>
      </body>
    </html>
    """
    home_soup = BeautifulSoup(spa_home, "html.parser")
    impressum_soup = BeautifulSoup(impressum_html, "html.parser")

    def mock_fetch(url, **kwargs):
        if "/impressum" in url:
            return impressum_soup
        if url == "https://webertech.de":
            return home_soup
        return None

    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        result = verify_founder_email_on_site(cand)

    assert result is None


def test_9_unrelated_employee_email_on_statutory_page_rejected():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    spa_home = "<html><body><div id=\"root\"></div></body></html>"
    team_html = """
    <html>
      <body>
        <div class="team-card">
          <h3>Dave Miller</h3>
          <p class="role">Lead Engineer</p>
          <a href="mailto:dave@webertech.de">dave@webertech.de</a>
        </div>
      </body>
    </html>
    """
    home_soup = BeautifulSoup(spa_home, "html.parser")
    team_soup = BeautifulSoup(team_html, "html.parser")

    def mock_fetch(url, **kwargs):
        if "/team" in url:
            return team_soup
        if url == "https://webertech.de":
            return home_soup
        return None

    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        result = verify_founder_email_on_site(cand)

    assert result is None


def test_10_missing_founder_email_still_returns_rejection():
    cand = CandidateCompany(
        name="WeberTech",
        website_url="https://webertech.de",
        description="B2B software",
        industry="DevTools",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Klaus Weber",
        founder_title="CEO"
    )
    spa_home = "<html><body><div id=\"root\"></div></body></html>"
    impressum_html = """
    <html>
      <body>
        <div class="imprint">
          <h3>WeberTech GmbH</h3>
          <p>Geschäftsführer: Klaus Weber</p>
          <p>Handelsregister: HRB 12345, Amtsgericht München</p>
          <p>USt-IdNr.: DE 987654321</p>
        </div>
      </body>
    </html>
    """
    home_soup = BeautifulSoup(spa_home, "html.parser")
    impressum_soup = BeautifulSoup(impressum_html, "html.parser")

    def mock_fetch(url, **kwargs):
        if "/impressum" in url:
            return impressum_soup
        if url == "https://webertech.de":
            return home_soup
        return None

    with patch("core.email_verifier.fetch_html", side_effect=mock_fetch):
        result = verify_founder_email_on_site(cand)

    assert result is None