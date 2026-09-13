"""
tests/test_role_alias_rejection.py

Adversarial tests for the Codex-identified critical defect:
Role-alias emails (ceo@, founder@, etc.) must NOT satisfy person-level
founder email attribution. A role mailbox identifies a ROLE, not a PERSON.

Cases A-F are the mandatory reproduction cases from the Codex review.
Cases G-N are additional hardening tests.
"""

import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup

from core.email_verifier import (
    is_email_attributed_to_founder,
    verify_founder_email_on_site,
    GENERIC_PREFIXES,
)
from core.models import CandidateCompany


# ---------------------------------------------------------------------------
# Unit-level: is_email_attributed_to_founder
# ---------------------------------------------------------------------------

class TestRoleAliasAttribution:
    """Ensure role aliases are never treated as person-level identity."""

    def _tokens(self, name: str):
        return [t for t in name.lower().split() if len(t) >= 3]

    # --- CASE A: ceo@ for Jane Doe -> REJECT ---
    def test_case_a_ceo_rejected(self):
        assert is_email_attributed_to_founder("ceo", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE B: founder@ for Jane Doe -> REJECT ---
    def test_case_b_founder_rejected(self):
        assert is_email_attributed_to_founder("founder", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE C: jane@ in founder card -> ACCEPT ---
    def test_case_c_jane_accepted(self):
        assert is_email_attributed_to_founder("jane", self._tokens("Jane Doe"), "jane doe") is True

    # --- CASE D: jane.doe@ in founder card -> ACCEPT ---
    def test_case_d_jane_doe_accepted(self):
        assert is_email_attributed_to_founder("jane.doe", self._tokens("Jane Doe"), "jane doe") is True

    # --- CASE E: dave.engineer@ for Jane Doe -> REJECT ---
    def test_case_e_wrong_person_rejected(self):
        assert is_email_attributed_to_founder("dave.engineer", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE G: owner@ -> REJECT ---
    def test_case_g_owner_rejected(self):
        assert is_email_attributed_to_founder("owner", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE H: president@ -> REJECT ---
    def test_case_h_president_rejected(self):
        assert is_email_attributed_to_founder("president", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE I: director@ -> REJECT ---
    def test_case_i_director_rejected(self):
        assert is_email_attributed_to_founder("director", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE J: leadership@ -> REJECT ---
    def test_case_j_leadership_rejected(self):
        assert is_email_attributed_to_founder("leadership", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE K: co-founder@ -> REJECT ---
    def test_case_k_co_founder_rejected(self):
        assert is_email_attributed_to_founder("co-founder", self._tokens("Jane Doe"), "jane doe") is False

    # --- CASE L: cofounder@ -> REJECT ---
    def test_case_l_cofounder_rejected(self):
        assert is_email_attributed_to_founder("cofounder", self._tokens("Jane Doe"), "jane doe") is False

    # --- Positive: first-initial+lastname (jdoe) -> ACCEPT ---
    def test_positive_initial_lastname(self):
        assert is_email_attributed_to_founder("jdoe", self._tokens("Jane Doe"), "jane doe") is True

    # --- Positive: p.collins -> ACCEPT for Patrick Collins ---
    def test_positive_dot_format(self):
        assert is_email_attributed_to_founder("p.collins", self._tokens("Patrick Collins"), "patrick collins") is True


# ---------------------------------------------------------------------------
# GENERIC_PREFIXES coverage: role aliases must be present
# ---------------------------------------------------------------------------

class TestGenericPrefixesIncludeRoleAliases:
    """All role aliases must be in GENERIC_PREFIXES so the is_generic gate blocks them."""

    @pytest.mark.parametrize("prefix", [
        "ceo@", "founder@", "cofounder@", "co-founder@",
        "president@", "director@", "owner@", "leadership@",
        "managing-director@",
    ])
    def test_role_prefix_in_generic(self, prefix):
        assert prefix in GENERIC_PREFIXES, f"{prefix} missing from GENERIC_PREFIXES"


# ---------------------------------------------------------------------------
# Integration-level: verify_founder_email_on_site end-to-end
# ---------------------------------------------------------------------------

def _mock_html(body_html: str) -> BeautifulSoup:
    return BeautifulSoup(f"<html><body>{body_html}</body></html>", "html.parser")


class TestRoleAliasEndToEnd:
    """
    End-to-end tests verifying that role-alias emails are rejected even when
    they appear inside a founder card with the founder's name and title.
    """

    def _make_candidate(self, domain="example.com", founder="Jane Doe", title="CEO"):
        return CandidateCompany(
            name="ExampleCo",
            website_url=f"https://{domain}",
            description="A test company",
            industry="SaaS",
            hq_country="Germany",
            is_tech_platform=True,
            founder_name=founder,
            founder_title=title,
        )

    # --- CASE F: ceo@ even when page says "Jane Doe -- CEO" -> REJECT ---
    def test_case_f_ceo_with_name_on_page_rejected(self):
        html = _mock_html("""
            <div class="team-card">
                <h3>Jane Doe</h3>
                <p>CEO & Co-Founder</p>
                <a href="mailto:ceo@example.com">ceo@example.com</a>
            </div>
        """)
        candidate = self._make_candidate()

        with patch("core.email_verifier.fetch_html") as mock_fetch, \
             patch("core.email_verifier.find_subpages", return_value=[]), \
             patch("core.email_verifier.probe_statutory_subpages", return_value=(0, [])):
            mock_fetch.return_value = html
            result = verify_founder_email_on_site(candidate)

        assert result is None or result == (None, None), (
            f"ceo@example.com was incorrectly accepted: {result}"
        )

    # --- founder@ in founder card -> REJECT ---
    def test_founder_email_in_card_rejected(self):
        html = _mock_html("""
            <div class="founder-profile">
                <h2>Jane Doe</h2>
                <span>Founder</span>
                <a href="mailto:founder@example.com">founder@example.com</a>
            </div>
        """)
        candidate = self._make_candidate(founder="Jane Doe", title="Founder")

        with patch("core.email_verifier.fetch_html") as mock_fetch, \
             patch("core.email_verifier.find_subpages", return_value=[]), \
             patch("core.email_verifier.probe_statutory_subpages", return_value=(0, [])):
            mock_fetch.return_value = html
            result = verify_founder_email_on_site(candidate)

        assert result is None or result == (None, None), (
            f"founder@example.com was incorrectly accepted: {result}"
        )

    # --- jane@example.com in founder card -> ACCEPT ---
    def test_legitimate_name_email_accepted(self):
        html = _mock_html("""
            <div class="team-card">
                <h3>Jane Doe</h3>
                <p>CEO & Co-Founder</p>
                <a href="mailto:jane@example.com">jane@example.com</a>
            </div>
        """)
        candidate = self._make_candidate()

        with patch("core.email_verifier.fetch_html") as mock_fetch, \
             patch("core.email_verifier.find_subpages", return_value=[]), \
             patch("core.email_verifier.probe_statutory_subpages", return_value=(0, [])):
            mock_fetch.return_value = html
            result = verify_founder_email_on_site(candidate)

        assert result is not None and result != (None, None), (
            "jane@example.com was incorrectly rejected"
        )
        email, source = result
        assert email == "jane@example.com"

    # --- president@ in executive card -> REJECT ---
    def test_president_in_card_rejected(self):
        html = _mock_html("""
            <div class="executive-card">
                <h3>John Smith</h3>
                <p>President</p>
                <a href="mailto:president@example.com">president@example.com</a>
            </div>
        """)
        candidate = self._make_candidate(founder="John Smith", title="President")

        with patch("core.email_verifier.fetch_html") as mock_fetch, \
             patch("core.email_verifier.find_subpages", return_value=[]), \
             patch("core.email_verifier.probe_statutory_subpages", return_value=(0, [])):
            mock_fetch.return_value = html
            result = verify_founder_email_on_site(candidate)

        assert result is None or result == (None, None), (
            f"president@example.com was incorrectly accepted: {result}"
        )
