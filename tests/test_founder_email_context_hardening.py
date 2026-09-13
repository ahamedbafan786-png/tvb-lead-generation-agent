"""
tests/test_founder_email_context_hardening.py
Comprehensive unit test suite for founder email context verification hardening.
Guarantees:
- Model-provided or crawled emails are NEVER accepted based solely on name token match.
- Guest author and article bylines are rejected.
- Adversarial surname collisions are rejected.
- Unrelated employee roles are rejected.
- Role aliases and generic inboxes are rejected.
- True founder card / executive co-location emails are accepted.
- 0 live network calls, 0 Tavily calls, 0 Gemini calls.
"""

import pytest
from unittest.mock import patch
from bs4 import BeautifulSoup

from core.models import CandidateCompany
from core.email_verifier import (
    verify_founder_email_on_site,
    discover_founder_contact_path,
    evaluate_email_founder_attribution,
    decode_cloudflare_email,
)


# ==============================================================================
# CODEX REPRODUCTION TEST
# ==============================================================================

def test_codex_reproduction_guest_author_rejected():
    """
    Codex reproduced:
    Founder: Jane Smith
    Page content: Guest author: jane@example.com
    The verifier must NOT accept this merely because 'jane' matches founder's first name.
    """
    html = """
    <html>
      <body>
        <article class="blog-post">
          <h1>Modern Cloud Security in 2026</h1>
          <p>Guest author: jane@example.com</p>
          <div class="post-content">Deep dive into security topics...</div>
        </article>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Guest author email must be rejected under strict context validation"

    contact_path = discover_founder_contact_path(cand, fetcher=lambda url, **kw: soup)
    assert contact_path.exact_email_found is None, "discover_founder_contact_path must also reject guest author"


# ==============================================================================
# 16 TEST MATRIX SCENARIOS
# ==============================================================================

def test_matrix_case_1_founder_card_exact_personal_email_accepted():
    """Case 1: Founder card + exact personal email -> ACCEPT"""
    html = """
    <html>
      <body>
        <div class="founder-card">
          <h3>Jane Smith</h3>
          <p class="title">Founder & CEO</p>
          <a href="mailto:jane@example.com">jane@example.com</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    email, url = result
    assert email == "jane@example.com"


def test_matrix_case_2_founder_card_first_last_email_accepted():
    """Case 2: Founder card + personal email with first+last -> ACCEPT"""
    html = """
    <html>
      <body>
        <div class="team-card">
          <h3>Jane Smith</h3>
          <p class="role">Co-Founder & CEO</p>
          <a href="mailto:jane.smith@example.com">Email Jane</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    email, url = result
    assert email == "jane.smith@example.com"


def test_matrix_case_3_founder_name_elsewhere_email_elsewhere_rejected():
    """Case 3: Founder name elsewhere + email elsewhere -> REJECT"""
    html = """
    <html>
      <body>
        <div class="hero">
          <h1>Founded by Jane Smith in 2021</h1>
        </div>
        <div class="footer">
          <p>For inquiries: <a href="mailto:jane@example.com">jane@example.com</a></p>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Separated founder name and email with insufficient card context must be rejected"


def test_matrix_case_4_guest_author_rejected():
    """Case 4: Guest author + matching email without explicit contact attribution -> REJECT"""
    html = """
    <div class="article-meta">
      <span class="byline">Guest Contributor: Jane Smith</span>
      <a href="mailto:jane@example.com">jane@example.com</a>
    </div>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Guest contributor must be rejected"


def test_matrix_case_5_article_byline_rejected():
    """Case 5: Article byline + matching email -> REJECT unless executive standing exists"""
    html = """
    <div class="blog-header">
      <p>Written by Jane Smith</p>
      <a href="mailto:jane@example.com">jane@example.com</a>
    </div>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Article byline without executive title must be rejected"


def test_matrix_case_6_metadata_founder_footer_email_rejected():
    """Case 6: Founder name in metadata + matching email in footer -> REJECT"""
    html = """
    <html>
      <head>
        <meta name="author" content="Jane Smith, Founder & CEO">
      </head>
      <body>
        <div class="footer">
          <p>Drop us a line at <a href="mailto:jane@example.com">jane@example.com</a></p>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Metadata separation must be rejected"


def test_matrix_case_7_navigation_founder_body_email_rejected():
    """Case 7: Founder name in navigation + matching email in body -> REJECT"""
    html = """
    <html>
      <body>
        <nav>
          <a href="/team">Our Leadership: Jane Smith (CEO)</a>
        </nav>
        <div class="content-body">
          <p>Have technical questions? Reach out to <a href="mailto:jane@example.com">jane@example.com</a></p>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Nav bar separation must be rejected"


def test_matrix_case_8_same_name_unrelated_employee_rejected():
    """Case 8: Same name + unrelated employee email -> REJECT"""
    html = """
    <html>
      <body>
        <div class="member-card">
          <h3>Jane Smith</h3>
          <p class="role">Software Engineer</p>
          <a href="mailto:jane@example.com">jane@example.com</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Employee non-executive role must be rejected"


def test_matrix_case_9_ceo_page_ceo_email_rejected():
    """Case 9: CEO page + ceo@ -> REJECT"""
    html = """
    <html>
      <body>
        <div class="founder-card">
          <h3>Jane Smith</h3>
          <p>Chief Executive Officer</p>
          <a href="mailto:ceo@example.com">ceo@example.com</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "ceo@ role alias must be strictly rejected"


def test_matrix_case_10_founder_page_founder_email_rejected():
    """Case 10: Founder page + founder@ -> REJECT"""
    html = """
    <html>
      <body>
        <div class="founder-card">
          <h3>Jane Smith</h3>
          <p>Founder & CEO</p>
          <a href="mailto:founder@example.com">founder@example.com</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "founder@ role alias must be strictly rejected"


def test_matrix_case_11_cloudflare_encoded_personal_email_accepted():
    """Case 11: Cloudflare encoded personal email inside founder card -> ACCEPT"""
    email_str = "jane@example.com"
    r = 0x55
    hex_body = "".join(f"{ord(c) ^ r:02x}" for c in email_str)
    cf_hex = f"{r:02x}{hex_body}"

    html = f"""
    <html>
      <body>
        <div class="team-card">
          <h3>Jane Smith</h3>
          <span class="role">Founder & CEO</span>
          <span data-cfemail="{cf_hex}">[email&#160;protected]</span>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    email, url = result
    assert email == "jane@example.com"


def test_matrix_case_12_obfuscated_personal_email_accepted():
    """Case 12: Obfuscated personal email inside founder card -> ACCEPT"""
    html = """
    <html>
      <body>
        <div class="founder-box">
          <h3>Jane Smith</h3>
          <p>Co-Founder & Managing Director</p>
          <p>Direct: jane [at] example [dot] com</p>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    email, url = result
    assert email == "jane@example.com"


def test_matrix_case_13_personal_email_external_domain_rejected():
    """Case 13: Personal email on external domain -> REJECT"""
    html = """
    <html>
      <body>
        <div class="founder-card">
          <h3>Jane Smith</h3>
          <p>Founder & CEO</p>
          <a href="mailto:jane.smith@gmail.com">jane.smith@gmail.com</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "External personal email must be rejected"


def test_matrix_case_14_personal_local_part_without_founder_context_rejected():
    """Case 14: Personal local-part match without founder context -> REJECT"""
    html = """
    <html>
      <body>
        <div class="contact-section">
          <p>Send feedback to <a href="mailto:jane@example.com">jane@example.com</a></p>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Unconnected email without founder identity in context must be rejected"


def test_matrix_case_15_explicit_sentence_tying_founder_to_email_accepted():
    """Case 15: Explicit sentence tying founder to email -> ACCEPT"""
    html = """
    <html>
      <body>
        <section class="about-us">
          <p>Jane Smith is the Founder and CEO of Example Inc and can be reached directly at <a href="mailto:jane@example.com">jane@example.com</a>.</p>
        </section>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    email, url = result
    assert email == "jane@example.com"


def test_matrix_case_16_founder_title_email_structured_block_accepted():
    """Case 16: Founder title + email in same structured semantic block -> ACCEPT"""
    html = """
    <html>
      <body>
        <section class="leadership">
          <h2>Executive Team</h2>
          <div class="member">
            <h4>Jane Smith</h4>
            <span class="role">Chief Executive Officer</span>
            <a href="mailto:jane.smith@example.com">Contact</a>
          </div>
        </section>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Example Inc",
        website_url="https://example.com",
        description="Security SaaS",
        industry="Cybersecurity",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    email, url = result
    assert email == "jane.smith@example.com"


# ==============================================================================
# ADVERSARIAL NAME COLLISION TESTS
# ==============================================================================

def test_adversarial_name_collision_john_smith_vs_john_jones():
    """Founder is John Smith (CEO), but DOM contains John Jones (Lead Engineer) with john@"""
    html = """
    <html>
      <body>
        <div class="team-card">
          <h3>John Jones</h3>
          <p class="role">Lead Security Engineer</p>
          <a href="mailto:john@acme.com">john@acme.com</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Acme",
        website_url="https://acme.com",
        description="Security SaaS",
        industry="DevSecOps",
        hq_country="United States",
        is_tech_platform=True,
        founder_name="John Smith",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Must not attribute John Jones's email to founder John Smith"


def test_adversarial_name_collision_alex_rivera_vs_alex_chen():
    """Founder is Alex Rivera (CEO), but DOM contains Alex Chen (Developer) with alex@"""
    html = """
    <html>
      <body>
        <div class="team-card">
          <h3>Alex Chen</h3>
          <p class="role">Frontend Developer</p>
          <a href="mailto:alex@tech.io">alex@tech.io</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Tech",
        website_url="https://tech.io",
        description="Dev tools",
        industry="DevTools",
        hq_country="United Kingdom",
        is_tech_platform=True,
        founder_name="Alex Rivera",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Must not attribute Alex Chen's email to founder Alex Rivera"


def test_adversarial_name_collision_sam_adams_vs_sam_wilson():
    """Founder is Sam Adams (Founder), but DOM contains Sam Wilson (UI Designer) with sam@"""
    html = """
    <html>
      <body>
        <div class="member">
          <h3>Sam Wilson</h3>
          <p class="role">UI Designer</p>
          <a href="mailto:sam@design.co">sam@design.co</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Design",
        website_url="https://design.co",
        description="Design tools",
        industry="Design",
        hq_country="France",
        is_tech_platform=True,
        founder_name="Sam Adams",
        founder_title="Founder"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Must not attribute Sam Wilson's email to founder Sam Adams"


def test_adversarial_name_collision_chris_evans_vs_chris_hemsworth():
    """Founder is Chris Evans (CEO), but DOM contains Chris Hemsworth (Sales) with chris@"""
    html = """
    <html>
      <body>
        <div class="member">
          <h3>Chris Hemsworth</h3>
          <p class="role">Account Manager</p>
          <a href="mailto:chris@media.net">chris@media.net</a>
        </div>
      </body>
    </html>
    """
    cand = CandidateCompany(
        name="Media",
        website_url="https://media.net",
        description="Media platform",
        industry="Media",
        hq_country="Norway",
        is_tech_platform=True,
        founder_name="Chris Evans",
        founder_title="CEO"
    )
    soup = BeautifulSoup(html, "html.parser")
    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Must not attribute Chris Hemsworth's email to founder Chris Evans"
