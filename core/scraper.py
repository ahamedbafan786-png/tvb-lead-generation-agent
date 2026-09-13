"""
core/scraper.py
Defensive HTTP fetching and broadened subpage link discovery.
Bounded crawl with anchor text recognition and URL normalization.
"""

import ipaddress
import logging
import re
import socket
from typing import Optional, Callable, Tuple, Union
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit
import requests
from bs4 import BeautifulSoup
from core.config import HTTP_HEADERS, SCRAPER_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
    "metadata",
    "kubernetes.default",
}

BLOCKED_HOST_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".localdomain",
)

MAX_REDIRECTS = 5

SUBPAGE_PATTERNS = [
    "about", "team", "contact", "people", "leadership", 
    "company", "founder", "founders", "management", "imprint", 
    "impressum", "legal", "investor", "investors", "press", 
    "news", "our-team", "meet-the-team", "who-we-are", "about-us", "contact-us",
    "mentions-legales", "aviso-legal", "legal-notice", "leadership-team", "our-story"
]

ANCHOR_PATTERNS = [
    "about", "team", "contact", "people", "leadership",
    "founder", "founders", "management", "imprint", "impressum",
    "legal", "our team", "meet the team", "who we are", "about us", "contact us",
    "mentions légales", "aviso legal", "legal notice", "leadership team", "our story"
]

MAX_DISCOVERED_SUBPAGES = 10
MAX_DIRECT_PROBES = 6

STATUTORY_PROBE_PATHS = [
    "/impressum",
    "/imprint",
    "/legal-notice",
    "/mentions-legales",
    "/aviso-legal",
    "/contact",
    "/contact-us",
    "/about",
    "/about-us",
    "/leadership",
    "/leadership-team",
    "/team",
    "/our-team",
    "/people",
    "/founders",
]


def parse_ip_address(host_str: str) -> Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]:
    """
    Parses an IP address in standard IPv4/IPv6, bracketed IPv6, decimal integer,
    hexadecimal integer, or octal dotted format. Returns None if host_str is a domain name.
    """
    if not host_str:
        return None

    cleaned = host_str.strip().strip("[]").lower()

    # 1. Standard ipaddress parser (IPv4, standard IPv6, IPv4-mapped IPv6)
    try:
        return ipaddress.ip_address(cleaned)
    except ValueError:
        pass

    # 2. Decimal integer IPv4 (e.g. 2130706433 -> 127.0.0.1)
    if cleaned.isdigit():
        try:
            val = int(cleaned)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.ip_address(val)
        except ValueError:
            pass

    # 3. Hexadecimal IPv4 (e.g. 0x7f000001 -> 127.0.0.1)
    if cleaned.startswith("0x") or cleaned.startswith("0X"):
        try:
            val = int(cleaned, 16)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.ip_address(val)
        except ValueError:
            pass

    # 4. Octal / alternate dotted IPv4 (e.g. 0177.0.0.1)
    try:
        packed = socket.inet_aton(cleaned)
        if cleaned.count(".") == 3:
            return ipaddress.ip_address(packed)
    except (socket.error, OSError, ValueError):
        pass

    return None


def is_safe_public_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> Tuple[bool, Optional[str]]:
    """
    Evaluates whether an IP address is a safe, publicly routable address.
    Blocks loopback, private RFC1918, link-local, cloud metadata, multicast,
    unspecified, carrier-grade NAT, and reserved networks.
    """
    # If IPv6 has mapped IPv4 address (e.g. ::ffff:127.0.0.1), inspect the mapped address
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return is_safe_public_ip(ip.ipv4_mapped)

    if ip.is_loopback:
        return False, f"Loopback IP address blocked: {ip}"

    if ip.is_private:
        return False, f"Private RFC1918/local IP address blocked: {ip}"

    if ip.is_link_local:
        return False, f"Link-local IP address blocked: {ip}"

    if ip.is_multicast:
        return False, f"Multicast IP address blocked: {ip}"

    if ip.is_reserved:
        return False, f"Reserved IP address blocked: {ip}"

    if ip.is_unspecified:
        return False, f"Unspecified IP address blocked: {ip}"

    if not ip.is_global:
        return False, f"Non-global IP address blocked: {ip}"

    # Explicit cloud metadata endpoint check
    ip_str = str(ip)
    if ip_str in {"169.254.169.254", "169.254.169.253", "fd00:ec2::254"}:
        return False, f"Cloud metadata endpoint blocked: {ip}"

    return True, None


def validate_public_url(url: str, resolve_dns: bool = True) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validates that a URL is a safe, public web URL.
    - Enforces scheme (strictly HTTP/HTTPS)
    - Rejects credentials/userinfo
    - Validates port numbers (1-65535)
    - Rejects localhost and internal hostnames
    - Identifies and rejects loopback, RFC1918, link-local, metadata, and non-global IPs
    - Resolves DNS and verifies all resolved IP addresses are safe and public
    Returns: (is_safe, error_reason, canonical_url)
    """
    if not url or not isinstance(url, str):
        return False, "Empty or invalid URL type", None

    clean_url = url.strip()
    if any(c in clean_url for c in ["\r", "\n", "\t", " "]):
        return False, "URL contains illegal whitespace or control characters", None

    try:
        parsed = urlsplit(clean_url)
    except Exception as e:
        return False, f"Failed to parse URL: {e}", None

    # 1. Scheme validation (strictly http or https)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, f"Disallowed scheme '{scheme}' (only HTTP and HTTPS allowed)", None

    # 2. Reject credentials / userinfo
    if parsed.username or parsed.password or "@" in parsed.netloc:
        return False, "URL contains embedded user credentials", None

    # 3. Hostname extraction & syntax
    hostname = parsed.hostname
    if not hostname:
        return False, "URL is missing a valid hostname", None

    hostname = hostname.lower().rstrip(".")
    if not hostname:
        return False, "Empty hostname", None

    # 4. Port validation
    port = parsed.port
    if port is not None:
        if port < 1 or port > 65535:
            return False, f"Invalid port: {port}", None

    # 5. Check hostname blacklist
    if hostname in BLOCKED_HOSTNAMES or any(hostname.endswith(sfx) for sfx in BLOCKED_HOST_SUFFIXES):
        return False, f"Blocked internal/loopback hostname: '{hostname}'", None

    # 6. Check if hostname is an IP literal
    ip = parse_ip_address(hostname)
    if ip is not None:
        is_safe, reason = is_safe_public_ip(ip)
        if not is_safe:
            return False, reason, None
    elif resolve_dns:
        # 7. DNS Resolution check
        target_port = port or (443 if scheme == "https" else 80)
        try:
            addr_info = socket.getaddrinfo(
                hostname, target_port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM
            )
        except socket.gaierror as e:
            return False, f"DNS resolution failed for '{hostname}': {e}", None
        except Exception as e:
            return False, f"DNS resolution error for '{hostname}': {e}", None

        if not addr_info:
            return False, f"No IP addresses resolved for '{hostname}'", None

        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            try:
                resolved_ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return False, f"Invalid IP returned by DNS: {ip_str}", None

            is_safe, reason = is_safe_public_ip(resolved_ip)
            if not is_safe:
                return False, f"DNS resolution for '{hostname}' returned blocked IP ({ip_str}): {reason}", None

    # 8. Reconstruct canonical URL (strip fragment, preserve exact path)
    netloc = hostname
    if port and port not in (80, 443):
        netloc = f"{netloc}:{port}"
    elif port == 80 and scheme == "https":
        netloc = f"{netloc}:80"
    elif port == 443 and scheme == "http":
        netloc = f"{netloc}:443"

    canonical = urlunsplit((scheme, netloc, parsed.path, parsed.query, ""))
    return True, None, canonical


def is_same_domain(url1: str, url2: str) -> bool:
    """Verifies that two URLs belong strictly to the same registered root domain or subdomain."""
    d1 = urlparse(url1).netloc.lower().replace("www.", "")
    d2 = urlparse(url2).netloc.lower().replace("www.", "")
    if not d1 or not d2:
        return False
    return d1 == d2 or d1.endswith("." + d2) or d2.endswith("." + d1)


def fetch_html(url: str, verify: bool = True, enforce_domain: Optional[str] = None) -> Optional[BeautifulSoup]:
    """
    Fetches an HTML document with timeout, SSRF protection, redirect validation,
    and safe error handling.
    - Strictly blocks loopback, private RFC1918, link-local, cloud metadata, and non-global IPs.
    - Validates scheme (HTTP/HTTPS only), rejects credentials, and validates ports.
    - Validates each redirect target before following (preventing public -> private SSRF redirects).
    - Strictly enforces TLS certificate verification (NEVER retries with verify=False).
    - If enforce_domain is provided, strictly rejects responses that redirected to an external domain.
    """
    is_safe, reason, current_url = validate_public_url(url, resolve_dns=True)
    if not is_safe or not current_url:
        logger.warning(f"fetch_html blocked unsafe URL '{url}': {reason}")
        return None

    for hop in range(MAX_REDIRECTS + 1):
        try:
            response = requests.get(
                current_url,
                headers=HTTP_HEADERS,
                timeout=SCRAPER_TIMEOUT_SECONDS,
                allow_redirects=False,
                verify=True
            )
        except requests.exceptions.SSLError as e:
            logger.warning(f"TLS certificate verification failed for {current_url}: {e}")
            # CRITICAL CODEX REQUIREMENT:
            # Never retry with verify=False! Return None safely.
            return None
        except requests.exceptions.ConnectionError:
            parsed = urlparse(current_url)
            if not parsed.netloc.startswith("www.") and hop == 0:
                www_host = f"www.{parsed.netloc}"
                www_url = parsed._replace(netloc=www_host).geturl()
                is_safe_www, _, canonical_www = validate_public_url(www_url, resolve_dns=True)
                if is_safe_www and canonical_www:
                    try:
                        return fetch_html(canonical_www, verify=True, enforce_domain=enforce_domain)
                    except Exception:
                        pass
            return None
        except Exception as e:
            logger.debug(f"HTTP fetch error for {current_url}: {e}")
            return None

        # Handle Redirects safely (verify response is actually a redirect response)
        status_code = getattr(response, "status_code", None)
        is_redirect = (getattr(response, "is_redirect", False) is True) or (status_code in (301, 302, 303, 307, 308))

        if is_redirect:
            if hop >= MAX_REDIRECTS:
                logger.warning(f"Exceeded max redirects ({MAX_REDIRECTS}) fetching {url}")
                return None

            raw_headers = getattr(response, "headers", None)
            location = raw_headers.get("Location") if hasattr(raw_headers, "get") else None
            if not location or not isinstance(location, str):
                return None

            next_url = urljoin(current_url, location)
            is_safe, reason, next_canonical = validate_public_url(next_url, resolve_dns=True)
            if not is_safe or not next_canonical:
                logger.warning(f"SSRF blocked unsafe redirect from {current_url} to {next_url}: {reason}")
                return None

            if enforce_domain:
                resp_domain = urlparse(next_canonical).netloc.lower().replace("www.", "")
                clean_enforce = enforce_domain.lower().replace("www.", "")
                if resp_domain != clean_enforce and not resp_domain.endswith("." + clean_enforce):
                    return None

            current_url = next_canonical
            continue

        # Final non-redirect response
        raw_headers = getattr(response, "headers", None)
        content_type = raw_headers.get("Content-Type", "") if hasattr(raw_headers, "get") else ""
        if status_code == 200 and "text/html" in content_type:
            final_url = getattr(response, "url", None) or current_url
            if isinstance(final_url, str) and enforce_domain:
                resp_domain = urlparse(final_url).netloc.lower().replace("www.", "")
                clean_enforce = enforce_domain.lower().replace("www.", "")
                if resp_domain != clean_enforce and not resp_domain.endswith("." + clean_enforce):
                    return None
            text = getattr(response, "text", "")
            return BeautifulSoup(text, "html.parser")

        return None

    return None


def probe_statutory_subpages(
    base_url: str,
    existing_urls: Optional[list[str]] = None,
    max_probes: int = MAX_DIRECT_PROBES,
    fetcher: Optional[Callable[..., Optional[BeautifulSoup]]] = None
) -> tuple[list[str], list[tuple[str, BeautifulSoup]]]:
    """
    Directly probes common same-domain statutory/contact paths for a company.
    Strictly bounded by max_probes and constrained to the official base domain.
    Returns:
      (probes_attempted, successfully_fetched_pages)
      where successfully_fetched_pages is a list of (url, BeautifulSoup) tuples.
    """
    if not base_url or not base_url.startswith(("http://", "https://")):
        return [], []

    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc.lower().replace("www.", "")
    if not base_domain:
        return [], []

    scheme = parsed_base.scheme or "https"
    origin = f"{scheme}://{parsed_base.netloc}".rstrip("/")

    normalized_existing = set()
    if existing_urls:
        for u in existing_urls:
            p = urlparse(u)
            normalized_existing.add(p.path.lower().rstrip("/"))

    probes_attempted: list[str] = []
    probed_pages: list[tuple[str, BeautifulSoup]] = []
    active_fetcher = fetcher or fetch_html

    for path in STATUTORY_PROBE_PATHS:
        if len(probes_attempted) >= max_probes or len(probed_pages) >= max_probes:
            break

        clean_path = path.rstrip("/")
        if clean_path.lower() in normalized_existing:
            continue

        probe_url = f"{origin}{path}"
        if not is_same_domain(probe_url, base_url):
            continue

        probes_attempted.append(probe_url)

        try:
            try:
                soup = active_fetcher(probe_url, enforce_domain=base_domain)
            except TypeError:
                soup = active_fetcher(probe_url)
            if soup is not None:
                probed_pages.append((probe_url, soup))
        except Exception:
            continue

    return probes_attempted, probed_pages



def find_subpages(soup: BeautifulSoup, base_url: str) -> list[str]:
    """
    Discovers relevant internal subpages (about, team, leadership, contact)
    from a company's homepage. Normalizes fragments and query parameters,
    matches by both URL path and anchor text, strips www. for domain matching,
    and returns a bounded list (max 10).
    """
    discovered = []
    base_parsed = urlparse(base_url)
    base_domain = base_parsed.netloc.lower().replace("www.", "")
    base_clean = base_parsed._replace(fragment="", query="").geturl().rstrip("/")

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        link_domain = parsed.netloc.lower().replace("www.", "")

        if link_domain == base_domain:
            clean_url = parsed._replace(fragment="", query="").geturl().rstrip("/")
            if clean_url == base_clean or clean_url in discovered:
                continue

            path_lower = parsed.path.lower()
            anchor_lower = a.get_text(separator=" ", strip=True).lower()

            path_match = any(pattern in path_lower for pattern in SUBPAGE_PATTERNS)
            anchor_match = any(pattern in anchor_lower for pattern in ANCHOR_PATTERNS)

            if path_match or anchor_match:
                discovered.append(clean_url)
                if len(discovered) >= MAX_DISCOVERED_SUBPAGES:
                    break

    return discovered

