"""
tests/test_public_founder_email_expansion.py
Unit test suite verifying legitimate public founder-email expansion across 13 offline scenarios.
Guarantees:
1. official company founder card + personal email -> ACCEPT
2. official press release explicitly gives founder email -> ACCEPT
3. official company blog explicitly gives founder email -> ACCEPT
4. founder's own public professional page explicitly gives email -> ACCEPT
5. public corporate document explicitly associates founder + email -> ACCEPT
6. third-party article merely mentions founder + unrelated email -> REJECT
7. founder name + matching email elsewhere -> REJECT
8. guest author email -> REJECT
9. ceo@ -> REJECT
10. founder@ -> REJECT
11. inferred firstname@domain -> REJECT
12. fake/missing source URL -> REJECT
13. external-domain email without explicit founder ownership -> REJECT
"""

import pytest
from bs4 import BeautifulSoup
from unittest.mock import patch

from core.models import CandidateCompany, EmailSourceType
from core.email_verifier import (
    classify_email_source,
    evaluate_email_founder_attribution,
    verify_founder_email_on_site,
    verify_founder_email_from_public_source,
)


@pytest.fixture
def sample_candidate():
    return CandidateCompany(
        name="Nexus Automation",
        website_url="https://nexusauto.io",
        description="Enterprise process automation software",
        industry="Enterprise Software",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Jane Smith",
        founder_title="CEO & Co-Founder"
    )


# 1. Official company founder card + personal email -> ACCEPT
def test_case_1_official_founder_card_accepted(sample_candidate):
    html = """
    <html>
      <body>
        <div class="team-grid">
          <div class="card">
            <h3>Jane Smith</h3>
            <p>CEO & Co-Founder</p>
            <a href="mailto:jane@nexusauto.io">jane@nexusauto.io</a>
          </div>
        </div>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    st = classify_email_source("https://nexusauto.io/about", "nexusauto.io", "Jane Smith")
    assert st == EmailSourceType.FIRST_PARTY_OFFICIAL

    with patch("core.email_verifier.fetch_html", return_value=soup), \
         patch("core.email_verifier.find_subpages", return_value=[]):
        result = verify_founder_email_on_site(sample_candidate)

    assert result is not None
    email, source = result
    assert email == "jane@nexusauto.io"
    assert "nexusauto.io" in source


# 2. Official press release explicitly gives founder email -> ACCEPT
def test_case_2_official_press_release_accepted(sample_candidate):
    html = """
    <html>
      <body>
        <div class="press-release">
          <h1>Nexus Automation Secures Growth Financing</h1>
          <p>Berlin, Germany -- Nexus Automation announced new financing.</p>
          <div class="media-contact">
            <p>Executive Contact: Jane Smith, CEO</p>
            <p>Email: jane.smith@nexusauto.io</p>
          </div>
        </div>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    pr_url = "https://press.nexusauto.io/releases/2026-series-a"
    st = classify_email_source(pr_url, "nexusauto.io", "Jane Smith")
    assert st == EmailSourceType.FIRST_PARTY_OFFICIAL

    result = verify_founder_email_from_public_source(
        candidate=sample_candidate,
        source_url=pr_url,
        soup=soup
    )
    assert result is not None
    email, url, source_type = result
    assert email == "jane.smith@nexusauto.io"
    assert source_type == EmailSourceType.FIRST_PARTY_OFFICIAL


# 3. Official company blog explicitly gives founder email -> ACCEPT
def test_case_3_official_blog_accepted(sample_candidate):
    html = """
    <html>
      <body>
        <article class="post">
          <h1>Building the Future of Enterprise AI</h1>
          <p class="author">By Jane Smith, Founder & CEO</p>
          <p>If you are exploring similar architectures, reach out directly to our founder:</p>
          <p>Contact Jane: <a href="mailto:jane@nexusauto.io">jane@nexusauto.io</a></p>
        </article>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    blog_url = "https://blog.nexusauto.io/building-the-future"
    st = classify_email_source(blog_url, "nexusauto.io", "Jane Smith")
    assert st == EmailSourceType.FIRST_PARTY_OFFICIAL

    result = verify_founder_email_from_public_source(
        candidate=sample_candidate,
        source_url=blog_url,
        soup=soup
    )
    assert result is not None
    email, url, source_type = result
    assert email == "jane@nexusauto.io"
    assert source_type == EmailSourceType.FIRST_PARTY_OFFICIAL


# 4. Founder's own public professional page explicitly gives email -> ACCEPT
def test_case_4_founder_personal_page_accepted(sample_candidate):
    html = """
    <html>
      <body>
        <div class="bio-container">
          <h1>Jane Smith</h1>
          <p>Founder and Chief Executive Officer of Nexus Automation.</p>
          <p>Direct inquiries: jane@janesmith.io</p>
        </div>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    founder_page_url = "https://janesmith.io/contact"
    st = classify_email_source(founder_page_url, "nexusauto.io", "Jane Smith")
    assert st == EmailSourceType.FIRST_PARTY_FOUNDER_PUBLISHED

    result = verify_founder_email_from_public_source(
        candidate=sample_candidate,
        source_url=founder_page_url,
        soup=soup
    )
    assert result is not None
    email, url, source_type = result
    assert email == "jane@janesmith.io"
    assert source_type == EmailSourceType.FIRST_PARTY_FOUNDER_PUBLISHED


# 5. Public corporate document explicitly associates founder + email -> ACCEPT
def test_case_5_public_corporate_document_accepted(sample_candidate):
    html = """
    <html>
      <body>
        <div class="company-officers">
          <h2>Handelsregister Bekanntmachung / Impressum</h2>
          <p>Vertreten durch die Geschäftsführung: Jane Smith, Founder & CEO</p>
          <p>Elektronische Kontaktaufnahme: jane.smith@nexusauto.io</p>
        </div>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    corp_url = "https://nexusauto.io/imprint"
    st = classify_email_source(corp_url, "nexusauto.io", "Jane Smith")
    assert st == EmailSourceType.PUBLIC_CORPORATE_RECORD

    result = verify_founder_email_from_public_source(
        candidate=sample_candidate,
        source_url=corp_url,
        soup=soup
    )
    assert result is not None
    email, url, source_type = result
    assert email == "jane.smith@nexusauto.io"
    assert source_type == EmailSourceType.PUBLIC_CORPORATE_RECORD


# 6. Third-party article merely mentions founder + unrelated email -> REJECT
def test_case_6_third_party_article_unrelated_email_rejected(sample_candidate):
    html = """
    <html>
      <body>
        <div class="news-article">
          <h1>European Tech Startups on the Rise</h1>
          <p>Nexus Automation was established by founder Jane Smith in Berlin.</p>
          <div class="sidebar">
            <p>Send news tips and feedback to our editor: editor@techstartupsnews.com</p>
          </div>
        </div>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    third_party_url = "https://techstartupsnews.com/articles/european-tech-2026"
    st = classify_email_source(third_party_url, "nexusauto.io", "Jane Smith")
    assert st == EmailSourceType.THIRD_PARTY_REPORTED

    result = verify_founder_email_from_public_source(
        candidate=sample_candidate,
        source_url=third_party_url,
        soup=soup
    )
    assert result is None


# 7. Founder name + matching email elsewhere in unrelated context -> REJECT
def test_case_7_founder_name_matching_email_unrelated_context_rejected(sample_candidate):
    # Context contains founder name and an email containing 'jane', but separated/unrelated
    context = "Jane Smith founded Nexus Automation in 2024. For event catering inquiries contact jane@cateringpartners.de"
    is_attributed, reason, _ = evaluate_email_founder_attribution(
        email="jane@cateringpartners.de",
        context=context,
        founder_name="Jane Smith",
        domain="nexusauto.io",
        source_url="https://cateringpartners.de/events",
        source_type=EmailSourceType.THIRD_PARTY_REPORTED
    )
    assert not is_attributed
    assert "REJECTED" in reason


# 8. Guest author email -> REJECT
def test_case_8_guest_author_rejected(sample_candidate):
    context = "Guest author: jane@nexusauto.io. Jane Smith writes about industry trends."
    is_attributed, reason, _ = evaluate_email_founder_attribution(
        email="jane@nexusauto.io",
        context=context,
        founder_name="Jane Smith",
        domain="nexusauto.io",
        source_type=EmailSourceType.FIRST_PARTY_OFFICIAL
    )
    assert not is_attributed
    assert reason == "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT"


# 9. ceo@ -> REJECT
def test_case_9_role_alias_ceo_rejected(sample_candidate):
    context = "Contact our Founder & CEO Jane Smith directly at ceo@nexusauto.io"
    is_attributed, reason, _ = evaluate_email_founder_attribution(
        email="ceo@nexusauto.io",
        context=context,
        founder_name="Jane Smith",
        domain="nexusauto.io",
        source_type=EmailSourceType.FIRST_PARTY_OFFICIAL
    )
    assert not is_attributed
    assert "ROLE_ALIAS" in reason


# 10. founder@ -> REJECT
def test_case_10_role_alias_founder_rejected(sample_candidate):
    context = "Reach our Founder & CEO Jane Smith at founder@nexusauto.io"
    is_attributed, reason, _ = evaluate_email_founder_attribution(
        email="founder@nexusauto.io",
        context=context,
        founder_name="Jane Smith",
        domain="nexusauto.io",
        source_type=EmailSourceType.FIRST_PARTY_OFFICIAL
    )
    assert not is_attributed
    assert "ROLE_ALIAS" in reason


# 11. Inferred firstname@domain -> REJECT
def test_case_11_inferred_email_rejected(sample_candidate):
    st = classify_email_source(None, "nexusauto.io", "Jane Smith")
    assert st == EmailSourceType.INFERRED

    st_synth = classify_email_source("https://inferred-leads.com/guess", "nexusauto.io", "Jane Smith")
    assert st_synth == EmailSourceType.INFERRED

    is_attributed, reason, _ = evaluate_email_founder_attribution(
        email="jane@nexusauto.io",
        context="Jane Smith is CEO",
        founder_name="Jane Smith",
        domain="nexusauto.io",
        source_type=EmailSourceType.INFERRED
    )
    assert not is_attributed
    assert reason == "EMAIL_SOURCE_INFERRED_REJECTED"


# 12. Fake / missing source URL -> REJECT
def test_case_12_fake_missing_source_url_rejected(sample_candidate):
    audit = {}
    result_none = verify_founder_email_from_public_source(
        candidate=sample_candidate,
        source_url="",
        audit_trail=audit
    )
    assert result_none is None
    assert "Missing" in audit.get("rejection_reason", "")

    audit_ssrf = {}
    result_ssrf = verify_founder_email_from_public_source(
        candidate=sample_candidate,
        source_url="http://169.254.169.254/metadata",
        audit_trail=audit_ssrf
    )
    assert result_ssrf is None
    assert "Blocked" in audit_ssrf.get("rejection_reason", "")


# 13. External-domain email without explicit founder ownership -> REJECT
def test_case_13_external_domain_without_founder_ownership_rejected(sample_candidate):
    # Context mentions Jane Smith as CEO, but the email is on an external agency domain
    context = "Jane Smith, Founder & CEO. Contact: jane@externalmarketingagency.com"
    is_attributed, reason, _ = evaluate_email_founder_attribution(
        email="jane@externalmarketingagency.com",
        context=context,
        founder_name="Jane Smith",
        domain="nexusauto.io",
        source_url="https://nexusauto.io/about",
        source_type=EmailSourceType.FIRST_PARTY_OFFICIAL
    )
    assert not is_attributed
    assert reason == "EMAIL_EXTERNAL_DOMAIN_REJECTED"

    # Even if source was a corporate record, an external agency domain without founder ownership is rejected
    is_attr_corp, reason_corp, _ = evaluate_email_founder_attribution(
        email="jane@externalmarketingagency.com",
        context=context,
        founder_name="Jane Smith",
        domain="nexusauto.io",
        source_url="https://nexusauto.io/imprint",
        source_type=EmailSourceType.PUBLIC_CORPORATE_RECORD
    )
    assert not is_attr_corp
    assert reason_corp == "EMAIL_EXTERNAL_DOMAIN_REJECTED"
