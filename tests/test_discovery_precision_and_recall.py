"""
tests/test_discovery_precision_and_recall.py
Deterministic unit tests for:
- Extraction prompt excluding VC firms, agencies, market reports
- Extraction prompt preserving plausible target companies
- Negative-query behavior in query templates
- Candidate deduplication normalization
- Email recall: Cloudflare de-obfuscation and text pattern de-obfuscation
- Preservation of strict zero-guessing on generic addresses
"""

import re
from bs4 import BeautifulSoup
import pytest

from core.config import REVENUE_QUERY_TEMPLATES, FUNDING_QUERY_TEMPLATES, NON_US_REGIONS, TECH_SECTORS
from core.discovery import generate_queries
from core.extractor import EXTRACTION_SYSTEM_PROMPT
from core.pipeline import normalize_company_key
from core.email_verifier import (
    decode_cloudflare_email, extract_dom_emails_with_context,
    verify_founder_email_on_site, EXECUTIVE_TITLE_REGEX
)
from core.scraper import find_subpages
from core.models import CandidateCompany


def test_extraction_prompt_excludes_vc_firms():
    """Verify system prompt explicitly excludes venture capital firms and investment funds."""
    prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
    assert "venture capital" in prompt_lower or "vc firms" in prompt_lower
    assert "investment funds" in prompt_lower or "funds" in prompt_lower


def test_extraction_prompt_excludes_agencies():
    """Verify system prompt explicitly excludes consulting, dev shops, and recruiting agencies."""
    prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
    assert "consulting" in prompt_lower
    assert "agencies" in prompt_lower or "agency" in prompt_lower
    assert "recruitment" in prompt_lower or "staffing" in prompt_lower


def test_extraction_prompt_excludes_market_reports():
    """Verify system prompt explicitly excludes market research reports and directories."""
    prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
    assert "market research" in prompt_lower or "market reports" in prompt_lower
    assert "directories" in prompt_lower or "publishers" in prompt_lower


def test_extraction_prompt_preserves_plausible_target_companies():
    """Verify system prompt explicitly instructs extracting plausible early-to-growth startups."""
    prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
    assert "operating software companies only" in prompt_lower or "operating software" in prompt_lower
    assert "plausible early-to-growth stage startups" in prompt_lower or "startups" in prompt_lower
    assert "1m" in prompt_lower and "5m" in prompt_lower


def test_negative_query_generation_templates():
    """Verify generated queries contain negative filtering terms to prevent market report and VC noise."""
    queries = generate_queries(count=20, seed=42)
    assert len(queries) == 20

    # Ensure negative exclusions exist in generated queries
    negative_terms = ["-fund", "-consulting", "-agency", "-recruiter", "-report", "-vc", "-directory", "-venture"]
    queries_with_negatives = [q for q in queries if any(neg in q for neg in negative_terms)]
    assert len(queries_with_negatives) == 20, "Every query template must include negative keywords to suppress noise"


def test_candidate_deduplication_normalization():
    """Verify candidate keys are accurately normalized across legal suffixes and punctuation."""
    variants = [
        "Acme Cloud Inc.",
        "Acme Cloud Ltd",
        "ACME CLOUD",
        "Acme Cloud, LLC",
        "Acme Cloud AI",
        "Acme Cloud Technologies"
    ]
    keys = {normalize_company_key(v) for v in variants}
    assert len(keys) == 1
    assert "acmecloud" in keys.pop()


def test_email_recall_cloudflare_decoding():
    """Verify deterministic Cloudflare email-protection XOR decoding."""
    # Build a test vector: key = 0x5a, email = "founder@platform.eu"
    target_email = "founder@platform.eu"
    key = 0x5a
    hex_body = "".join(f"{ord(c) ^ key:02x}" for c in target_email)
    cf_hex = f"{key:02x}{hex_body}"

    decoded = decode_cloudflare_email(cf_hex)
    assert decoded == target_email

    # Verify invalid hex returns None safely
    assert decode_cloudflare_email("invalid") is None
    assert decode_cloudflare_email("") is None


def test_email_recall_obfuscated_text():
    """Verify detection of human-obfuscated email syntax in static DOM elements."""
    html = """
    <div class="team-card">
        <h3>Alice Dupont</h3>
        <span class="title">Co-Founder & CEO</span>
        <p class="email">alice [at] dupontech [dot] eu</p>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    emails = extract_dom_emails_with_context(soup)
    assert len(emails) == 1
    email, ctx = emails[0]
    assert email == "alice@dupontech.eu"
    assert "Alice Dupont" in ctx
    assert "CEO" in ctx

    # Verify attribution accepts this email
    cand = CandidateCompany(
        name="DuponTech",
        website_url="https://dupontech.eu",
        description="B2B platform",
        industry="SaaS",
        hq_country="France",
        is_tech_platform=True,
        founder_name="Alice Dupont",
        founder_title="CEO"
    )
    from unittest.mock import patch
    with patch("core.email_verifier.fetch_html", return_value=soup):
        result = verify_founder_email_on_site(cand)
    assert result is not None
    verified_email, source_url = result
    assert verified_email == "alice@dupontech.eu"


def test_email_recall_statutory_subpages():
    """Verify European legal notice and imprint paths are discovered."""
    html = """
    <html>
        <body>
            <a href="/about-us">About Us</a>
            <a href="/impressum">Impressum</a>
            <a href="/mentions-legales">Mentions Légales</a>
            <a href="/leadership-team">Leadership Team</a>
            <a href="/careers">Careers</a>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    subpages = find_subpages(soup, "https://example.de")
    assert "https://example.de/impressum" in subpages
    assert "https://example.de/mentions-legales" in subpages
    assert "https://example.de/leadership-team" in subpages


def test_executive_title_regex_international():
    """Verify executive regex matches common international leadership titles."""
    assert EXECUTIVE_TITLE_REGEX.search("CEO & Founder")
    assert EXECUTIVE_TITLE_REGEX.search("Geschäftsführer")
    assert EXECUTIVE_TITLE_REGEX.search("Fondateur et CEO")
    assert EXECUTIVE_TITLE_REGEX.search("Managing Director")
    assert not EXECUTIVE_TITLE_REGEX.search("Junior Sales Associate")


def test_zero_guessing_preserved_for_generic_emails():
    """Verify generic info/contact inboxes without personal attribution are strictly rejected."""
    html = """
    <div class="footer">
        <p>Contact us: <a href="mailto:info@startup.eu">info@startup.eu</a></p>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    cand = CandidateCompany(
        name="Startup EU",
        website_url="https://startup.eu",
        description="SaaS platform",
        industry="SaaS",
        hq_country="Germany",
        is_tech_platform=True,
        founder_name="Max Müller",
        founder_title="CEO"
    )
    from unittest.mock import patch
    with patch("core.email_verifier.fetch_html", return_value=soup):
        result = verify_founder_email_on_site(cand)
    assert result is None, "Generic inbox without personal attribution must be rejected under zero-guessing"
