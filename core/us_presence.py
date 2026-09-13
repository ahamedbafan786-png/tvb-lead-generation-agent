"""
core/us_presence.py
Evidence-backed classification and validation of non-US corporate footprint.
Distinguishes substantial US operational presence (disqualifying) from minimal
permissible signals (Delaware investment holdings, global/US customer reach, isolated remote staff).
"""

import re
from enum import Enum
from typing import Optional, Tuple, List


class USPresenceCategory(str, Enum):
    # Strong / Disqualifying categories
    US_HQ = "US_HQ"
    US_OPERATIONAL_OFFICE = "US_OPERATIONAL_OFFICE"
    US_SPECIALIZED_OFFICE = "US_SPECIALIZED_OFFICE"
    US_SUBSIDIARY = "US_SUBSIDIARY"
    US_OPERATIONS = "US_OPERATIONS"
    US_EXECUTIVE_TEAM = "US_EXECUTIVE_TEAM"

    # Non-disqualifying / Permitted footprint categories
    US_CUSTOMERS_ONLY = "US_CUSTOMERS_ONLY"
    DELAWARE_INCORPORATION_ONLY = "DELAWARE_INCORPORATION_ONLY"
    INCIDENTAL_REMOTE_ONLY = "INCIDENTAL_REMOTE_ONLY"

    # Clean non-US
    NO_US_PRESENCE = "NO_US_PRESENCE"

    # Unknown / Unverified HQ
    UNKNOWN_HQ = "UNKNOWN_HQ"


# All 50 US states
US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming"
}

# Major US tech cities, tech hubs, and boroughs
US_TECH_CITIES = {
    "san francisco", "new york", "austin", "seattle", "boston",
    "los angeles", "chicago", "denver", "atlanta", "miami",
    "dallas", "houston", "palo alto", "mountain view", "sunnyvale",
    "san jose", "san diego", "cambridge", "salt lake city",
    "washington dc", "arlington", "raleigh", "philadelphia",
    "manhattan", "brooklyn", "queens", "bronx", "staten island",
    "new york city", "nyc", "sf", "bay area", "silicon valley",
    "santa clara", "oakland", "berkeley", "bellevue", "redmond",
    "san antonio", "boulder", "orlando", "tampa", "dc", "durham",
    "pittsburgh", "phoenix", "scottsdale", "portland", "minneapolis",
    "nashville", "charlotte"
}

# US State 2-letter postal abbreviations
US_STATE_ABBRS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
}

US_REGIONS = {
    "california", "texas", "massachusetts", "new york state",
    "washington state", "delaware", "silicon valley", "bay area"
}

# Sorted regex string for all US locations (cities, hubs, states)
_ALL_LOCATIONS = sorted(
    list(US_STATES) + list(US_TECH_CITIES),
    key=lambda x: -len(x)
)
_LOCATIONS_PATTERN = "|".join(re.escape(loc) for loc in _ALL_LOCATIONS)
_ABBRS_PATTERN = "|".join(US_STATE_ABBRS)

# Core operational nouns and facility signals
_OFFICE_NOUNS = r"(?:office|offices|branch|branch\s+office|operating\s+center|operations\s+center|operating\s+facility|facilities|facility|tech\s+hub|innovation\s+center|development\s+center|permanent\s+establishment)"
_SPECIALIZED_OFFICE_NOUNS = r"(?:sales|engineering|r&d|tech|development|innovation|support)\s+office(?:s)?"
_MATERIALITY_PREFIX = r"(?:\d+[-\s]person|\d+\s+(?:people|employees?|staff)|permanent|regional|dedicated|major|primary)"

# Disqualifying regex patterns mapped to categories
US_DISQUALIFYING_PATTERNS = [
    # 1. US Headquarters (US_HQ)
    (USPresenceCategory.US_HQ, re.compile(
        r"\b(?:us|u\.s\.|united states|american|north american)\s+(?:headquarters|hq|corporate hq|principal office|operating hq|regional headquarters)\b", re.I)),
    (USPresenceCategory.US_HQ, re.compile(
        rf"\bheadquarter(?:ed|s)?\s+(?:in|at)\s+(?:the\s+)?(?:us|u\.s\.|united states|america|north america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_HQ, re.compile(
        r"\b(?:co|dual)[-\s]headquarter(?:ed|s)?\b", re.I)),
    (USPresenceCategory.US_HQ, re.compile(
        rf"\b(?:{_LOCATIONS_PATTERN})\s+(?:headquarters|hq|corporate hq)\b", re.I)),

    # 2. US Operational Office (US_OPERATIONAL_OFFICE)
    # e.g. "We operate a 100-person office in Manhattan, New York", "Maintains a US office"
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\b(?:maintains?|operates?|has|opened|established|running)\s+(?:an?\s+)?(?:{_MATERIALITY_PREFIX}\s+)?(?:us|u\.s\.|united states|american)\s+{_OFFICE_NOUNS}\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\b(?:maintains?|operates?|has|opened|established|running)\s+(?:an?\s+)?{_MATERIALITY_PREFIX}\s+{_OFFICE_NOUNS}\s+(?:in|at)\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\b{_MATERIALITY_PREFIX}\s+{_OFFICE_NOUNS}\s+(?:in|at)\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\b(?:us|u\.s\.|united states|american)\s+{_OFFICE_NOUNS}\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\boffice(?:s)?\s+(?:in|at|across)\s+(?:the\s+)?(?:us|u\.s\.|united states|america)\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\boffice(?:s)?\s+(?:in|at)\s+(?:the\s+)?(?:{_LOCATIONS_PATTERN}|(?:in\s+(?:{_ABBRS_PATTERN})\b))\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\b(?:maintains?|has|opened|operates?)\s+(?:an?\s+)?office\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\b(?:{_LOCATIONS_PATTERN})\s+office(?:s)?\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        r"\b(?:multiple|several|\d+)\s+us\s+offices\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        r"\bus\s+branch(?:\s+office)?\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\bbranch\s+office\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\bemploys?\s+\d+\s+(?:people|employees?|staff)\s+in\s+(?:its\s+)?(?:(?:{_LOCATIONS_PATTERN})\s+)?office\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        rf"\bemploys?\s+\d+\s+(?:people|employees?|staff)\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_OPERATIONAL_OFFICE, re.compile(
        r"\bpermanent\s+(?:us\s+)?(?:establishment|operating\s+facility)\b", re.I)),

    # 3. US Specialized Office (Sales, Engineering, R&D)
    (USPresenceCategory.US_SPECIALIZED_OFFICE, re.compile(
        rf"\b(?:us|u\.s\.|united states|american)\s+{_SPECIALIZED_OFFICE_NOUNS}\b", re.I)),
    (USPresenceCategory.US_SPECIALIZED_OFFICE, re.compile(
        rf"\b{_SPECIALIZED_OFFICE_NOUNS}\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_SPECIALIZED_OFFICE, re.compile(
        rf"\b(?:{_LOCATIONS_PATTERN})\s+{_SPECIALIZED_OFFICE_NOUNS}\b", re.I)),
    (USPresenceCategory.US_SPECIALIZED_OFFICE, re.compile(
        r"\bpermanent\s+sales\s+office\b", re.I)),
    (USPresenceCategory.US_SPECIALIZED_OFFICE, re.compile(
        r"\b(?:us|u\.s\.|united states)\s+(?:tech\s+hub|innovation\s+center|development\s+center)\b", re.I)),
    (USPresenceCategory.US_SPECIALIZED_OFFICE, re.compile(
        rf"\bengineering\s+team\s+based\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),

    # 4. US Subsidiary & Legal Entity (US_SUBSIDIARY)
    (USPresenceCategory.US_SUBSIDIARY, re.compile(
        r"\b(?:us|u\.s\.|united states|american|north american)\s+subsidiary\b", re.I)),
    (USPresenceCategory.US_SUBSIDIARY, re.compile(
        r"\bsubsidiary\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america)\b", re.I)),
    (USPresenceCategory.US_SUBSIDIARY, re.compile(
        r"\boperates\s+through\s+(?:its\s+)?(?:us|u\.s\.|american)\s+subsidiary\b", re.I)),
    (USPresenceCategory.US_SUBSIDIARY, re.compile(
        r"\bus\s+operating\s+subsidiary\b", re.I)),
    (USPresenceCategory.US_SUBSIDIARY, re.compile(
        r"\b(?:us|u\.s\.|united states|american)\s+(?:legal\s+entity|operating\s+entity|branch\s+entity)\b", re.I)),

    # 5. US Operations & Operating Centers (US_OPERATIONS)
    (USPresenceCategory.US_OPERATIONS, re.compile(
        r"\b(?:primary|core|main)\s+operations\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america)\b", re.I)),
    (USPresenceCategory.US_OPERATIONS, re.compile(
        r"\boperational\s+base\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america)\b", re.I)),
    (USPresenceCategory.US_OPERATIONS, re.compile(
        r"\b(?:us|u\.s\.|united states|american)\s+(?:operating|operations)\s+center\b", re.I)),
    (USPresenceCategory.US_OPERATIONS, re.compile(
        r"\b(?:us|u\.s\.|united states|american)\s+operations\b", re.I)),
    (USPresenceCategory.US_OPERATIONS, re.compile(
        rf"\b(?:{_LOCATIONS_PATTERN})\s+operations\b", re.I)),
    (USPresenceCategory.US_OPERATIONS, re.compile(
        rf"\boperations\s+(?:team\s+)?based\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),

    # 6. US Executive / Leadership Team (US_EXECUTIVE_TEAM)
    (USPresenceCategory.US_EXECUTIVE_TEAM, re.compile(
        r"\b(?:us|u\.s\.|united states|american)\s+(?:executive\s+team|leadership\s+team|management\s+team)\b", re.I)),
    (USPresenceCategory.US_EXECUTIVE_TEAM, re.compile(
        r"\b(?:us|u\.s\.|united states|american)\s+management\s+office\b", re.I)),
    (USPresenceCategory.US_EXECUTIVE_TEAM, re.compile(
        rf"\b(?:executive\s+team|executives?|leadership)\s+(?:based|located)\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states|america|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.US_EXECUTIVE_TEAM, re.compile(
        r"\bsubstantial\s+(?:us|u\.s\.|american)\s+(?:team|workforce|operations|presence)\b", re.I)),
    (USPresenceCategory.US_EXECUTIVE_TEAM, re.compile(
        r"\bdedicated\s+(?:us|american)\s+team\b", re.I)),
]

# Non-disqualifying / mitigating patterns
US_MITIGATING_PATTERNS = [
    # Delaware incorporation / holding only
    (USPresenceCategory.DELAWARE_INCORPORATION_ONLY, re.compile(
        r"\bincorporated\s+in\s+delaware\b", re.I)),
    (USPresenceCategory.DELAWARE_INCORPORATION_ONLY, re.compile(
        r"\bdelaware\s+(?:c[-\s]?corp|holding|incorporat(?:ed|ion)|entity)\b", re.I)),
    (USPresenceCategory.DELAWARE_INCORPORATION_ONLY, re.compile(
        r"\bdelaware\s+for\s+(?:funding|investment|tax|venture)\b", re.I)),
    (USPresenceCategory.DELAWARE_INCORPORATION_ONLY, re.compile(
        r"\bdelaware\s+holding\s+company\b", re.I)),

    # Customers / Market presence only
    (USPresenceCategory.US_CUSTOMERS_ONLY, re.compile(
        r"\b(?:serves?|serving|sells?\s+to|customers?\s+in|clients?\s+in)\s+(?:the\s+)?(?:us|u\.s\.|united states|american\s+market)\b", re.I)),
    (USPresenceCategory.US_CUSTOMERS_ONLY, re.compile(
        r"\b(?:us|u\.s\.|united states|american)\s+(?:customers|clients|market|users|revenue|sales|accounts)\b", re.I)),
    (USPresenceCategory.US_CUSTOMERS_ONLY, re.compile(
        r"\b\d+%\s+of\s+(?:revenue|sales|customers)\s+(?:from|in)\s+(?:the\s+)?(?:us|u\.s\.|united states)\b", re.I)),
    (USPresenceCategory.US_CUSTOMERS_ONLY, re.compile(
        r"\b(?:customer\s+base|client\s+base)\s+in\s+(?:the\s+)?(?:us|u\.s\.|united states)\b", re.I)),
    (USPresenceCategory.US_CUSTOMERS_ONLY, re.compile(
        r"\bcustomers\s+across\s+(?:the\s+)?(?:us|u\.s\.|united states)\b", re.I)),
    (USPresenceCategory.US_CUSTOMERS_ONLY, re.compile(
        r"\b(?:us|american)\s+customer\s+base\b", re.I)),

    # Incidental / Isolated remote staff only
    (USPresenceCategory.INCIDENTAL_REMOTE_ONLY, re.compile(
        rf"\b(?:one|single|1|isolated|a\s+couple\s+of|few)\s+(?:remote\s+)?(?:employees?|contractors?|workers?|engineers?)\s+(?:in|from|based\s+in)\s+(?:the\s+)?(?:us|u\.s\.|united states|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.INCIDENTAL_REMOTE_ONLY, re.compile(
        rf"\bremote\s+(?:contractor|worker|employee)\s+(?:in|from|based\s+in)\s+(?:the\s+)?(?:us|u\.s\.|united states|{_LOCATIONS_PATTERN})\b", re.I)),
    (USPresenceCategory.INCIDENTAL_REMOTE_ONLY, re.compile(
        rf"\b(?:employee|worker|contractor)\s+works?\s+remotely\s+from\s+(?:the\s+)?(?:us|u\.s\.|united states|{_LOCATIONS_PATTERN})\b", re.I)),
]


def extract_us_locations(text: str) -> List[str]:
    """Extracts known US cities, states, hubs, or regions mentioned in text."""
    if not text:
        return []
    text_lower = text.lower()
    found = []
    # 1. Cities, tech hubs, boroughs
    for city in sorted(US_TECH_CITIES, key=lambda x: -len(x)):
        if re.search(rf"\b{re.escape(city)}\b", text_lower):
            found.append(city.title())
    # 2. States
    for state in sorted(US_STATES, key=lambda x: -len(x)):
        if re.search(rf"\b{re.escape(state)}\b", text_lower):
            found.append(state.title())
    # 3. State abbreviations in comma or 'in' context
    for abbr in US_STATE_ABBRS:
        if re.search(rf"(?:,\s*{abbr}\b|\bin\s+{abbr}\b)", text):
            found.append(abbr)
    return list(dict.fromkeys(found))


def evaluate_us_presence(candidate) -> Tuple[bool, str, Optional[str], Optional[str], List[str]]:
    """
    Evaluates US presence for a CandidateCompany.
    Returns:
        (is_non_us: bool, result_code: str, evidence_summary: Optional[str], category: Optional[str], locations: List[str])
    """
    country = (candidate.hq_country or "").lower().strip()
    city = (candidate.hq_city or "").lower().strip()

    # Rule 0: Unknown / Missing HQ Country cannot be verified as non-US
    if not country or country in {"unknown", "n/a", "none", "null", "undefined"}:
        return False, "REJECTED_UNKNOWN_HQ", "HQ country is unknown or unevidenced; cannot verify non-US requirement.", USPresenceCategory.UNKNOWN_HQ.value, []

    # Rule 1: HQ Country explicitly US
    if country in {"us", "usa", "united states"} or "united states" in country or country.endswith(".us"):
        cat = USPresenceCategory.US_HQ.value
        locs = [candidate.hq_city.title()] if candidate.hq_city else ["United States"]
        return False, "REJECTED_US_HQ", f"Headquartered in the US: {candidate.hq_city or ''}, {candidate.hq_country}", cat, locs

    # Rule 2: HQ City in major US tech hubs
    if city in US_TECH_CITIES:
        cat = USPresenceCategory.US_HQ.value
        locs = [candidate.hq_city.title()]
        return False, "REJECTED_US_HQ", f"City identified as major US tech hub: {candidate.hq_city}", cat, locs

    # Aggregate all evidence fields
    evidence_parts = [
        candidate.us_presence_evidence or "",
        candidate.us_operations_evidence or "",
        candidate.us_team_evidence or "",
        candidate.description or ""
    ]
    combined_evidence = " ".join(part for part in evidence_parts if part).strip()
    locations = extract_us_locations(combined_evidence)
    if candidate.us_locations:
        locations = list(dict.fromkeys(locations + candidate.us_locations))

    # Pattern analysis
    disqualifying_hits = []
    for cat, pattern in US_DISQUALIFYING_PATTERNS:
        match = pattern.search(combined_evidence)
        if match:
            disqualifying_hits.append((cat, match.group(0)))

    mitigating_hits = []
    for cat, pattern in US_MITIGATING_PATTERNS:
        match = pattern.search(combined_evidence)
        if match:
            mitigating_hits.append((cat, match.group(0)))

    # Rule 3: Disqualifying operational presence detected (Strongest evidence wins)
    if disqualifying_hits:
        primary_cat, matched_text = disqualifying_hits[0]
        if hasattr(candidate, "us_operational_phrase"):
            candidate.us_operational_phrase = matched_text
        if hasattr(candidate, "us_city_or_state") and locations:
            candidate.us_city_or_state = locations[0]
        result_code = "REJECTED_US_HQ" if primary_cat == USPresenceCategory.US_HQ else "REJECTED_US_OPERATIONS"
        loc_str = f" (locations: {', '.join(locations)})" if locations else ""
        snippet = getattr(candidate, "us_presence_evidence", None) or combined_evidence[:120]
        summary = f"Substantial US operations detected [{primary_cat.value}]: {snippet}{loc_str}"
        return False, result_code, summary, primary_cat.value, locations

    # Rule 4: Explicit us_presence_detected flag asserted without mitigating evidence
    if candidate.us_presence_detected and not mitigating_hits:
        cat = USPresenceCategory.US_OPERATIONS.value
        loc_str = f" (locations: {', '.join(locations)})" if locations else ""
        summary = f"US presence explicitly asserted without mitigating non-operational evidence.{loc_str}"
        return False, "REJECTED_US_OPERATIONS", summary, cat, locations

    # Rule 5: Mitigating non-operational presence permitted (Delaware, Customers, Isolated Remote)
    if mitigating_hits:
        primary_cat, matched_text = mitigating_hits[0]
        if hasattr(candidate, "us_operational_phrase"):
            candidate.us_operational_phrase = matched_text
        if hasattr(candidate, "us_city_or_state") and locations:
            candidate.us_city_or_state = locations[0]
        snippet = getattr(candidate, "us_presence_evidence", None) or combined_evidence[:100]
        summary = f"Minimal US footprint permitted [{primary_cat.value}] (e.g. US customers / Delaware holding): {snippet}"
        return True, "PASSED_MINIMAL_US_SIGNAL", summary, primary_cat.value, locations

    # Rule 6: Verified non-US
    return True, "PASSED_NON_US_VERIFIED", f"HQ and primary operations verified in {candidate.hq_country}.", USPresenceCategory.NO_US_PRESENCE.value, []
