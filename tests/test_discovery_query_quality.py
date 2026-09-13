"""
tests/test_discovery_query_quality.py
Offline verification of discovery query matrix:
- Signal quality and density (explicit financial amounts and founder terms)
- Autonomous dynamic generation (zero hardcoded companies/leads)
- Non-US geographic diversity and tech sector coverage
- 100% uniqueness with exactly 35 queries
- Negative filtering compliance
- Well-formed syntax (balanced quotes, no double colons)
- Downstream qualification gates uncompromised
"""

import pytest
from core.config import (
    DEFAULT_MAX_SEARCH_QUERIES, TAVILY_MAX_CALLS_PER_RUN,
    MAX_CANDIDATES_PER_RUN, MIN_FINANCIAL_USD, MAX_FINANCIAL_USD,
    NON_US_REGIONS, TECH_SECTORS,
    REVENUE_QUERY_TEMPLATES, FUNDING_QUERY_TEMPLATES
)
from core.discovery import generate_queries, audit_query_quality


def test_default_query_count_is_35():
    """Verify default query count is strictly 35 and generates 35 items."""
    assert DEFAULT_MAX_SEARCH_QUERIES == 35
    queries = generate_queries()
    assert len(queries) == 35


def test_queries_are_100_percent_unique_across_seeds():
    """Verify generated queries contain zero duplicates across multiple random seeds."""
    for test_seed in (1, 42, 100, 777, 2024, 9999):
        queries = generate_queries(count=35, seed=test_seed)
        assert len(queries) == 35
        assert len(set(queries)) == 35, f"Expected 35 unique queries for seed {test_seed}, got {len(set(queries))}"


def test_diverse_non_us_geographies():
    """Verify generated 35 queries cover a broad distribution of non-US regions (>= 15)."""
    queries = generate_queries(count=35, seed=42)
    regions_hit = {r for r in NON_US_REGIONS if any(r.lower() in q.lower() for q in queries)}
    assert len(regions_hit) >= 15, f"Expected broad non-US coverage (>=15), got {len(regions_hit)}"


def test_diverse_tech_sectors():
    """Verify generated 35 queries cover diverse tech sectors (>= 8)."""
    queries = generate_queries(count=35, seed=42)
    sectors_hit = {s for s in TECH_SECTORS if any(s.lower() in q.lower() for q in queries)}
    assert len(sectors_hit) >= 8, f"Expected broad sector coverage (>=8), got {len(sectors_hit)}"


def test_financial_signal_density():
    """Verify 100% of generated queries contain explicit financial signal terms."""
    financial_terms = [
        "$1m", "$2m", "$3m", "$4m", "$5m", "$1.5m", "$2.5m",
        "€1m", "€2m", "£1m", "£2m",
        "arr", "annual recurring revenue", "seed", "pre-seed", "revenue", "funding"
    ]
    queries = generate_queries(count=35, seed=42)
    for q in queries:
        q_lower = q.lower()
        has_fin = any(term in q_lower for term in financial_terms)
        assert has_fin, f"Query missing financial signal: '{q}'"


def test_founder_signal_density():
    """Verify generated queries contain executive/founder terms."""
    founder_terms = ["founder", "co-founder", "ceo"]
    queries = generate_queries(count=35, seed=42)
    founder_queries = [q for q in queries if any(ft in q.lower() for ft in founder_terms)]
    assert len(founder_queries) >= 28, f"Expected >= 28 founder-oriented queries, got {len(founder_queries)}"


def test_negative_filtering_compliance():
    """Verify every query includes negative exclusion keywords to suppress noisy VC funds and directories."""
    negative_terms = ["-fund", "-consulting", "-agency", "-recruiter", "-report", "-vc", "-directory", "-venture"]
    queries = generate_queries(count=35, seed=42)
    for q in queries:
        has_neg = any(term in q for term in negative_terms)
        assert has_neg, f"Query missing negative filter: '{q}'"


def test_no_hardcoded_company_names():
    """Verify query templates and generated queries do not contain hardcoded company names."""
    known_companies = [
        "stripe", "openai", "datadog", "snowflake", "salesforce",
        "postman", "canva", "celonis", "klarna", "personio"
    ]
    all_templates = REVENUE_QUERY_TEMPLATES + FUNDING_QUERY_TEMPLATES
    for tpl in all_templates:
        tpl_lower = tpl.lower()
        for company in known_companies:
            assert company not in tpl_lower, f"Template contains hardcoded company name: {company}"

    queries = generate_queries(count=35, seed=42)
    for q in queries:
        q_lower = q.lower()
        for company in known_companies:
            assert company not in q_lower, f"Query contains hardcoded company name: {company}"


def test_syntax_and_quote_balance():
    """Verify all queries have balanced quotes and no malformed double colons."""
    queries = generate_queries(count=35, seed=42)
    for q in queries:
        assert q.count('"') % 2 == 0, f"Unbalanced quotes in query: '{q}'"
        assert "::" not in q, f"Malformed double colon in query: '{q}'"


def test_audit_query_quality_telemetry():
    """Verify audit_query_quality correctly evaluates query batches."""
    queries = generate_queries(count=35, seed=42)
    telemetry = audit_query_quality(queries)

    assert telemetry["total_queries"] == 35
    assert telemetry["unique_queries"] == 35
    assert telemetry["high_signal_queries"] >= 30
    assert telemetry["financial_signal_queries"] == 35
    assert telemetry["founder_signal_queries"] >= 30
    assert telemetry["negative_filtered_queries"] == 35
    assert telemetry["distinct_regions_hit"] >= 15
    assert telemetry["distinct_sectors_hit"] >= 8
    assert telemetry["high_signal_ratio"] >= 0.85

    # Test empty input handling
    empty_telemetry = audit_query_quality([])
    assert empty_telemetry["total_queries"] == 0
    assert empty_telemetry["high_signal_ratio"] == 0.0


def test_autonomous_dynamic_discovery():
    """Verify changing seed produces different query sets, confirming dynamic autonomous generation."""
    q_seed1 = generate_queries(count=35, seed=1)
    q_seed2 = generate_queries(count=35, seed=2)
    assert q_seed1 != q_seed2, "Query generator must produce different dynamic sets across runs"


def test_downstream_gates_unaffected():
    """Verify all downstream qualification and budget thresholds remain strictly unchanged."""
    assert MIN_FINANCIAL_USD == 1_000_000
    assert MAX_FINANCIAL_USD == 5_000_000
    assert TAVILY_MAX_CALLS_PER_RUN == 60
    assert MAX_CANDIDATES_PER_RUN == 150
