"""
core/email_verifier.py
Strict DOM card/sibling co-location and generic prefix blacklist. Zero guessing.
"""

import re
from typing import Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from core.models import CandidateCompany, ContactPathResult
from core.scraper import (
    fetch_html, find_subpages, probe_statutory_subpages,
    MAX_DISCOVERED_SUBPAGES, MAX_DIRECT_PROBES, validate_public_url
)

EMAIL_REGEX = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b")

GENERIC_PREFIXES = (
    "info@", "contact@", "hello@", "support@", "sales@", "press@", 
    "jobs@", "careers@", "admin@", "general@", "media@", "team@", 
    "office@", "help@", "privacy@", "billing@", "legal@",
    # Role aliases: identify a ROLE, not a PERSON — must not satisfy person-level attribution
    "ceo@", "founder@", "cofounder@", "co-founder@", "president@",
    "director@", "owner@", "leadership@", "managing-director@",
)

CARD_CONTAINER_HINTS = {
    "card", "member", "person", "profile", "founder", "team", 
    "bio", "leader", "leadership", "item", "col", "column", 
    "speaker", "executive", "officer", "staff", 
    "management", "user", "box", "tile", "row", "wrapper", "group",
    "imprint", "impressum", "legal"
}

ROLE_ALIASES = {
    "ceo", "founder", "cofounder", "co-founder", "president", 
    "director", "owner", "leadership", "managing-director", 
    "admin", "info", "contact", "support", "sales", "press", 
    "jobs", "careers", "general", "media", "team", "office", "help",
    "privacy", "billing", "legal"
}

EXECUTIVE_TITLE_REGEX = re.compile(
    r"\b(ceo|co-founder|cofounder|founder|managing director|chief executive|president|geschäftsführer|geschäftsführung|fondateur|gerente|gruender|gründer|vorstand|inhaber)\b",
    re.IGNORECASE
)

OBFUSCATED_EMAIL_REGEX = re.compile(
    r"\b([a-zA-Z0-9_.+-]+)\s*(?:\[at\]|\(at\))\s*([a-zA-Z0-9-]+)\s*(?:\[dot\]|\(dot\)|\.)\s*([a-zA-Z0-9-.]+)\b",
    re.IGNORECASE
)

def decode_cloudflare_email(cf_hex: str) -> Optional[str]:
    """Decodes Cloudflare email-protection hex ciphertext deterministically."""
    if not cf_hex or len(cf_hex) < 4:
        return None
    try:
        r = int(cf_hex[:2], 16)
        email = "".join([chr(int(cf_hex[i:i+2], 16) ^ r) for i in range(2, len(cf_hex), 2)])
        return email.strip().lower()
    except Exception:
        return None


def get_enclosing_card_context(tag) -> str:
    current = tag
    best_context = tag.get_text(separator=" ", strip=True)

    for _ in range(4):
        parent = current.parent
        if not parent or parent.name in ["body", "html", "[document]"]:
            break

        class_id_str = " ".join(parent.get("class", [])) + " " + str(parent.get("id", ""))
        class_id_lower = class_id_str.lower()
        parent_text = parent.get_text(separator=" ", strip=True)

        if parent.name in ["article", "li", "tr", "dl", "section"] or any(hint in class_id_lower for hint in CARD_CONTAINER_HINTS):
            if len(parent_text) < 1600:
                return parent_text

        if len(parent_text) < 1000:
            best_context = parent_text

        current = parent

    return best_context


def extract_dom_emails_with_context(soup: BeautifulSoup) -> list[tuple[str, str]]:
    found = []

    # 1. Cloudflare protected email tags
    for tag in soup.find_all(attrs={"data-cfemail": True}):
        cf_hex = tag.get("data-cfemail", "").strip()
        decoded = decode_cloudflare_email(cf_hex)
        if decoded and EMAIL_REGEX.match(decoded):
            card_context = get_enclosing_card_context(tag)
            found.append((decoded, card_context))

    # 2. Standard mailto links (including URL percent-encoded %40)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().startswith("mailto:"):
            clean_email = href[7:].split("?")[0].strip().replace("%40", "@").replace("%20", "").lower()
            if EMAIL_REGEX.match(clean_email):
                card_context = get_enclosing_card_context(a)
                found.append((clean_email, card_context))

    # 3. Plain text and obfuscated patterns in DOM tags
    for tag in soup.find_all(["p", "div", "li", "span", "td", "dd", "h1", "h2", "h3", "h4", "h5", "h6"]):
        tag_text = tag.get_text(separator=" ", strip=True)
        # Standard email match
        for match in EMAIL_REGEX.findall(tag_text):
            clean_email = match.lower()
            card_context = get_enclosing_card_context(tag)
            found.append((clean_email, card_context))
        # Obfuscated pattern match (e.g. name [at] domain [dot] com)
        for ob_match in OBFUSCATED_EMAIL_REGEX.findall(tag_text):
            norm_email = f"{ob_match[0]}@{ob_match[1]}.{ob_match[2]}".lower()
            if EMAIL_REGEX.match(norm_email):
                card_context = get_enclosing_card_context(tag)
                found.append((norm_email, card_context))

    seen = set()
    unique_found = []
    for em, ctx in found:
        pair_key = (em, ctx)
        if pair_key not in seen:
            seen.add(pair_key)
            unique_found.append((em, ctx))

    return unique_found




def is_email_attributed_to_founder(local_part: str, founder_tokens: list[str], founder_name: str) -> bool:
    """
    Validates whether an email's local part is legitimately attributable to the founder.
    Returns True if:
    - Any founder name token (>= 3 chars) is in local_part (e.g. 'patrick', 'collins')
    - First initial + last name is in local_part (e.g. 'pcollins', 'p.collins')
    Returns False if:
    - Local part is a generic role alias (e.g. 'ceo', 'founder', 'president')
      — a role mailbox identifies a ROLE, not a PERSON
    - Local part appears to belong to an unrelated employee or person
    """
    local_lower = local_part.lower()
    clean_local = re.sub(r"[^a-zA-Z0-9]", "", local_lower)

    if local_lower in ROLE_ALIASES or clean_local in {"ceo", "founder", "cofounder", "president", "director", "owner"}:
        return False

    valid_tokens = [tok.lower() for tok in founder_tokens if len(tok) >= 3]
    if any(tok in local_lower for tok in valid_tokens):
        return True

    if founder_tokens and len(founder_name) > 0:
        first_init = founder_name[0].lower()
        last_tok = founder_tokens[-1].lower()
        if f"{first_init}{last_tok}" in clean_local:
            return True

    return False


def evaluate_email_founder_attribution(
    email: str,
    context: str,
    founder_name: str,
    domain: str,
    candidate_founder_title: Optional[str] = None
) -> tuple[bool, str, Optional[str]]:
    """
    Strict, fail-closed email attribution to founder.
    Requires:
    1. Email belongs to candidate's domain (or subdomain).
    2. Email is not generic and not a role alias.
    3. Email local-part legitimately matches founder name tokens.
    4. Email appears in a bounded, structured context (card, block, explicit sentence).
    5. Context establishes founder identity (full name, or first+last tokens, no surname collision).
    6. Context establishes executive/founder standing (CEO, Founder, etc. or explicit contact phrasing).
    7. Context is not a guest byline, article author, or non-executive employee role.
    
    Returns: (is_attributed, rule_or_reason, context_snippet)
    """
    parts = email.strip().lower().split("@")
    if len(parts) != 2:
        return False, "EMAIL_FORMAT_INVALID", None
    local_part, email_domain = parts[0], parts[1]

    clean_domain = domain.lower().replace("www.", "").split(":")[0].strip()
    clean_email_domain = email_domain.split(":")[0].strip()
    if clean_email_domain != clean_domain and not clean_email_domain.endswith("." + clean_domain):
        return False, "EMAIL_EXTERNAL_DOMAIN_REJECTED", None

    if any(email.lower().startswith(gp) for gp in GENERIC_PREFIXES):
        return False, "EMAIL_ROLE_ALIAS_OR_GENERIC_REJECTED", None

    clean_local = re.sub(r"[^a-zA-Z0-9]", "", local_part)
    if local_part in ROLE_ALIASES or clean_local in {"ceo", "founder", "cofounder", "president", "director", "owner", "managingdirector", "leadership"}:
        return False, "EMAIL_ROLE_ALIAS_REJECTED", None

    f_name_clean = " ".join(founder_name.lower().split())
    f_tokens = [t for t in re.split(r"[\s.-]+", f_name_clean) if len(t) >= 2]
    if not f_tokens or not is_email_attributed_to_founder(local_part, f_tokens, f_name_clean):
        return False, "EMAIL_LOCAL_PART_NAME_MISMATCH", None

    raw_ctx = (context or "").strip()
    if not raw_ctx or len(raw_ctx) < 5:
        return False, "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT", None

    if len(raw_ctx) > 1600:
        return False, "EMAIL_ATTRIBUTION_CONTEXT_UNBOUNDED", None

    context_lower = " ".join(raw_ctx.lower().split())

    # Executive title check
    has_exec_title = bool(EXECUTIVE_TITLE_REGEX.search(context_lower))
    if candidate_founder_title and candidate_founder_title.lower() in context_lower:
        has_exec_title = True

    # Guest Author check (Codex Reproduction: guest authors are never the company founder)
    if re.search(r"\bguest\s+(?:author|writer|contributor|blogger|post)\b", context_lower):
        return False, "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT", None

    # Byline / Article Author check without executive title
    has_byline_indicator = bool(re.search(
        r"\b(contributing\s+(?:author|writer|editor)|"
        r"author\s*:\s*|"
        r"written\s+by\b|"
        r"posted\s+by\b|"
        r"article\s+by\b|"
        r"byline\b|"
        r"blog\s+author\b|"
        r"columnist\b|"
        r"commenter\b|"
        r"reviewer\b)\b",
        context_lower
    ))
    if has_byline_indicator and not has_exec_title:
        return False, "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT", None

    # Employee non-executive role check
    has_employee_role = bool(re.search(
        r"\b(lead\s+security\s+engineer|software\s+engineer|security\s+engineer|engineer|ui\s+designer|designer|"
        r"developer|intern|assistant|specialist|coordinator|recruiter|sales\s+rep|account\s+manager)\b",
        context_lower
    ))
    if has_employee_role and not has_exec_title:
        return False, "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT", None

    # Name collision check (Section 12)
    if len(f_tokens) >= 2:
        first_tok, last_tok = f_tokens[0], f_tokens[-1]
        ignored_words = {
            "is", "was", "our", "the", "ceo", "founder", "cofounder", "co-founder",
            "at", "and", "can", "has", "will", "from", "for", "with",
            "direct", "email", "contact", "reach", "who", "an", "a", "all", "in",
            "to", "on", "by", "as", "he", "she", "his", "her"
        }
        for collision_match in re.finditer(r"\b" + re.escape(first_tok) + r"\s+([a-z]+)\b", context_lower):
            other_surname = collision_match.group(1).lower()
            if other_surname not in f_tokens and other_surname not in ignored_words and len(other_surname) >= 3:
                if last_tok not in context_lower:
                    return False, "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT", None

    # Founder identity in context
    has_full_name = f_name_clean in context_lower
    has_first_and_last = len(f_tokens) >= 2 and (f_tokens[0] in context_lower and f_tokens[-1] in context_lower)
    has_single_name = len(f_tokens) == 1 and f_tokens[0] in context_lower
    has_first_and_email_surname = (
        len(f_tokens) >= 2 and
        f_tokens[0] in context_lower and
        f_tokens[-1] in local_part
    )

    founder_identified = (
        has_full_name or
        has_first_and_last or
        has_single_name or
        has_first_and_email_surname
    )

    if not founder_identified:
        return False, "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT", None

    # Executive phrasing check
    has_exec_phrases = bool(re.search(
        r"\b(?:our\s+(?:co-?founder|ceo|founder|leader|executive)|"
        r"(?:contact|reach|email)\s+(?:the\s+)?(?:founder|ceo|co-?founder|executive)|"
        r"(?:executive|leadership|founding)\s+(?:team|leadership|member|profile)|"
        r"(?:vertreten\s+durch|geschäftsführung|impressum|imprint|legal\s+notice))\b",
        context_lower
    ))
    has_direct_contact_link = False
    if f_tokens:
        has_direct_contact_link = bool(re.search(
            r"\b(?:contact|email|reach)\s+" + re.escape(f_tokens[0]) + r"\b",
            context_lower
        ))

    if not (has_exec_title or has_exec_phrases or has_direct_contact_link):
        return False, "EMAIL_ATTRIBUTION_CONTEXT_INSUFFICIENT", None

    concise_snippet = raw_ctx[:180]
    return True, "ACCEPTED_RULE_1", concise_snippet


def _safe_fetch(fetcher, url: str, domain: Optional[str] = None) -> Optional[BeautifulSoup]:
    try:
        return fetcher(url, enforce_domain=domain)
    except TypeError:
        return fetcher(url)


def extract_founder_from_dom(soup: BeautifulSoup) -> tuple[Optional[str], Optional[str]]:
    """
    Scans DOM for executive names and titles (CEO, Co-Founder, Founder, etc.).
    Returns (founder_name, founder_title) if found, else (None, None).
    """
    patterns = [
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s*[-–:,|/]?\s*(CEO|Co-Founder|Cofounder|Founder|Managing Director|President)",
        r"(CEO|Co-Founder|Cofounder|Founder|Managing Director|President)\s*[-–:,|/]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
    ]
    non_names = {"about", "contact", "home", "company", "team", "our", "the", "read", "view", "more", "learn", "blog", "press", "privacy", "legal"}

    for tag in soup.find_all(["div", "li", "article", "section", "tr", "p", "h1", "h2", "h3", "h4", "h5"]):
        tag_text = tag.get_text(separator=" ", strip=True)
        if not tag_text or len(tag_text) > 300:
            continue
        if not EXECUTIVE_TITLE_REGEX.search(tag_text):
            continue

        for pat in patterns:
            m = re.search(pat, tag_text, re.IGNORECASE)
            if m:
                g1, g2 = m.group(1).strip(), m.group(2).strip()
                if EXECUTIVE_TITLE_REGEX.match(g1):
                    title, name = g1, g2
                else:
                    name, title = g1, g2

                name_tokens = name.split()
                if len(name_tokens) in (2, 3) and not any(t.lower() in non_names for t in name_tokens):
                    return name, title.title()
    return None, None


def discover_founder_contact_path(
    candidate: CandidateCompany,
    fetcher=fetch_html
) -> ContactPathResult:
    """
    FREE Bounded Founder/Contact-Path Discovery (0 Tavily Calls).
    Inspects official company website, linked subpages, and bounded statutory/contact probes.
    Returns structured ContactPathResult.
    """
    if not candidate.website_url or not candidate.website_url.strip():
        return ContactPathResult(
            contact_path_found=False,
            reason_if_no_contact_path="No official website URL provided."
        )

    clean_url = candidate.website_url.strip()
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    is_safe, reason, canonical_url = validate_public_url(clean_url, resolve_dns=False)
    if not is_safe or not canonical_url:
        return ContactPathResult(
            contact_path_found=False,
            reason_if_no_contact_path=f"Blocked unsafe website URL: {clean_url} ({reason})"
        )
    clean_url = canonical_url

    parsed = urlparse(clean_url)
    domain = parsed.netloc.lower().replace("www.", "")
    if not domain:
        return ContactPathResult(
            contact_path_found=False,
            reason_if_no_contact_path="Invalid official website URL."
        )

    home_soup = _safe_fetch(fetcher, clean_url, domain)
    if not home_soup:
        return ContactPathResult(
            contact_path_found=False,
            reason_if_no_contact_path=f"Failed to fetch homepage: {clean_url}"
        )

    pages_to_check: list[tuple[str, BeautifulSoup]] = [(clean_url, home_soup)]
    pages_checked_urls: list[str] = [clean_url]

    # 1. Discover linked subpages from homepage
    linked_urls = find_subpages(home_soup, clean_url)
    for sub_url in linked_urls:
        if len(pages_to_check) >= MAX_DISCOVERED_SUBPAGES:
            break
        sub_soup = _safe_fetch(fetcher, sub_url, domain)
        if sub_soup:
            pages_to_check.append((sub_url, sub_soup))
            pages_checked_urls.append(sub_url)

    # 2. Bounded direct statutory probing (if budget remains)
    if len(pages_to_check) < MAX_DISCOVERED_SUBPAGES:
        probes_attempted, probed_pages = probe_statutory_subpages(
            base_url=clean_url,
            existing_urls=pages_checked_urls,
            max_probes=min(MAX_DIRECT_PROBES, max(0, MAX_DISCOVERED_SUBPAGES - len(pages_to_check))),
            fetcher=fetcher
        )
        for p_url, p_soup in probed_pages:
            if len(pages_to_check) >= MAX_DISCOVERED_SUBPAGES:
                break
            pages_to_check.append((p_url, p_soup))
            pages_checked_urls.append(p_url)

    # 3. Founder identification
    founder_name = candidate.founder_name
    founder_title = candidate.founder_title
    founder_source_url = candidate.founder_source_url
    if not founder_name:
        for page_url, soup in pages_to_check:
            f_name, f_title = extract_founder_from_dom(soup)
            if f_name:
                founder_name = f_name
                founder_title = f_title
                founder_source_url = page_url
                break

    # 4. Email discovery and attribution
    exact_email_found = None
    email_source_url = None
    attribution_context = None

    if founder_name:
        for page_url, soup in pages_to_check:
            extracted = extract_dom_emails_with_context(soup)
            for email, context in extracted:
                is_attributed, rule_or_reason, snippet = evaluate_email_founder_attribution(
                    email=email,
                    context=context,
                    founder_name=founder_name,
                    domain=domain,
                    candidate_founder_title=founder_title
                )
                if is_attributed:
                    exact_email_found = email
                    email_source_url = page_url
                    attribution_context = f"{rule_or_reason} ({page_url})"
                    break
            if exact_email_found:
                break

    # 5. Evaluate contact path
    contact_path_found = False
    reason_if_no = None

    if exact_email_found:
        contact_path_found = True
    elif founder_name:
        # Founder is identified and official website was reachable with pages checked
        contact_path_found = True
    else:
        # Check for contact page, team/about page, or generic email in DOM
        has_contact_page = any(
            any(cp in u.lower() for cp in ["contact", "about", "team", "impressum", "imprint"])
            for u in pages_checked_urls
        )
        has_any_email = False
        for _, soup in pages_to_check:
            if extract_dom_emails_with_context(soup):
                has_any_email = True
                break

        if has_contact_page or has_any_email:
            contact_path_found = True
        else:
            contact_path_found = False
            reason_if_no = "No founder identified and no contact/team subpages or contact email found on official site."

    return ContactPathResult(
        founder_name_found=founder_name,
        founder_title_found=founder_title,
        founder_source_url=founder_source_url,
        exact_email_found=exact_email_found,
        email_source_url=email_source_url,
        attribution_context=attribution_context,
        contact_path_found=contact_path_found,
        pages_checked=pages_checked_urls,
        reason_if_no_contact_path=reason_if_no
    )


def verify_founder_email_on_site(
    candidate: CandidateCompany,
    audit_trail: Optional[dict] = None
) -> Optional[tuple[str, str]]:
    if not candidate.website_url or not candidate.founder_name:
        if audit_trail is not None:
            audit_trail["rejection_reason"] = "Missing website_url or founder_name"
        return None

    founder_name = candidate.founder_name.lower().strip()
    clean_site = candidate.website_url.strip()
    if not clean_site.startswith(("http://", "https://")):
        clean_site = "https://" + clean_site

    is_safe, reason, canonical_site = validate_public_url(clean_site, resolve_dns=False)
    if not is_safe or not canonical_site:
        if audit_trail is not None:
            audit_trail["rejection_reason"] = f"Blocked unsafe website URL: {candidate.website_url} ({reason})"
        return None

    domain = urlparse(canonical_site).netloc.lower().replace("www.", "")
    founder_tokens = [t for t in re.split(r"[\s.-]+", founder_name) if len(t) >= 3]

    if audit_trail is not None:
        audit_trail["homepage"] = canonical_site
        audit_trail["linked_pages_discovered"] = []
        audit_trail["direct_probes_attempted"] = []
        audit_trail["pages_successfully_fetched"] = []
        audit_trail["emails_found_count"] = 0
        audit_trail["attribution_result"] = None
        audit_trail["rejection_reason"] = None

    home_soup = _safe_fetch(fetch_html, canonical_site, domain)
    if not home_soup:
        if audit_trail is not None:
            audit_trail["rejection_reason"] = f"Failed to fetch homepage: {candidate.website_url}"
        return None

    if audit_trail is not None:
        audit_trail["pages_successfully_fetched"].append(candidate.website_url)

    pages_to_check: list[tuple[str, BeautifulSoup]] = [(candidate.website_url, home_soup)]

    # 1. Discover linked subpages from homepage
    linked_urls = find_subpages(home_soup, candidate.website_url)
    if audit_trail is not None:
        audit_trail["linked_pages_discovered"] = list(linked_urls)

    for sub_url in linked_urls:
        if len(pages_to_check) >= MAX_DISCOVERED_SUBPAGES:
            break
        sub_soup = _safe_fetch(fetch_html, sub_url, domain)
        if sub_soup:
            pages_to_check.append((sub_url, sub_soup))
            if audit_trail is not None:
                audit_trail["pages_successfully_fetched"].append(sub_url)

    # 2. Bounded direct statutory probing (if budget remains)
    if len(pages_to_check) < MAX_DISCOVERED_SUBPAGES:
        discovered_urls = [p[0] for p in pages_to_check]
        probes_attempted, probed_pages = probe_statutory_subpages(
            base_url=candidate.website_url,
            existing_urls=discovered_urls,
            max_probes=min(MAX_DIRECT_PROBES, max(0, MAX_DISCOVERED_SUBPAGES - len(pages_to_check))),
            fetcher=fetch_html
        )
        if audit_trail is not None:
            audit_trail["direct_probes_attempted"] = probes_attempted

        for p_url, p_soup in probed_pages:
            if len(pages_to_check) >= MAX_DISCOVERED_SUBPAGES:
                break
            pages_to_check.append((p_url, p_soup))
            if audit_trail is not None:
                audit_trail["pages_successfully_fetched"].append(p_url)

    # 3. Check all gathered pages for verified founder email
    total_emails_found = 0
    for page_url, soup in pages_to_check:
        extracted = extract_dom_emails_with_context(soup)
        total_emails_found += len(extracted)

        for email, context in extracted:
            is_attributed, rule_or_reason, snippet = evaluate_email_founder_attribution(
                email=email,
                context=context,
                founder_name=candidate.founder_name,
                domain=domain,
                candidate_founder_title=candidate.founder_title
            )
            if is_attributed:
                if audit_trail is not None:
                    audit_trail["emails_found_count"] = total_emails_found
                    audit_trail["attribution_result"] = f"{rule_or_reason} ({page_url})"
                    if snippet:
                        audit_trail["attribution_snippet"] = snippet
                return email, page_url

    if audit_trail is not None:
        audit_trail["emails_found_count"] = total_emails_found
        audit_trail["attribution_result"] = "NO_ATTRIBUTED_FOUNDER_EMAIL"
        audit_trail["rejection_reason"] = f"No verified founder email on site for {candidate.founder_name}. Zero-guessing enforced."

    return None


