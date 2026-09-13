"""
core/discovery.py
Tavily Basic Search dynamic web discovery with prompt isolation,
source traceability, in-memory caching, and zero Google Search Grounding.
"""

import json
import socket
import logging
import random
import urllib.request
import urllib.error
from typing import Optional
import threading

from google import genai
from core.config import (
    GEMINI_MODEL, FALLBACK_MODEL,
    TAVILY_API_KEY, TAVILY_SEARCH_URL, TAVILY_TIMEOUT_SECONDS, TAVILY_MAX_CALLS_PER_RUN,
    NON_US_REGIONS, TECH_SECTORS,
    REVENUE_QUERY_TEMPLATES, FUNDING_QUERY_TEMPLATES
)

logger = logging.getLogger(__name__)

# Module-level search tracking and deduplication cache
_TAVILY_CACHE: dict[str, list[dict]] = {}

# Budget lock: ensures atomic reserve-before-dispatch
_BUDGET_LOCK = threading.Lock()

# Attempt counters (incremented BEFORE urlopen — every outbound attempt costs a slot)
_DISCOVERY_CALLS_COUNT: int = 0
_FOLLOWUP_CALLS_COUNT: int = 0
_OTHER_CALLS_COUNT: int = 0

# Outcome counters (incremented AFTER urlopen resolves)
_TAVILY_REQUESTS_SUCCEEDED: int = 0
_TAVILY_REQUESTS_FAILED: int = 0
_TAVILY_CACHE_HITS: int = 0


def get_total_tavily_calls() -> int:
    """Returns the total number of outbound Tavily request ATTEMPTS in this run."""
    return _DISCOVERY_CALLS_COUNT + _FOLLOWUP_CALLS_COUNT + _OTHER_CALLS_COUNT


def get_tavily_search_count() -> int:
    """Backward-compatible accessor for total Tavily outbound attempts."""
    return get_total_tavily_calls()


def get_tavily_call_breakdown() -> dict[str, int]:
    """Returns independent counters for discovery, follow-up, outcome, and total Tavily calls."""
    return {
        "discovery_search_calls": _DISCOVERY_CALLS_COUNT,
        "financial_followup_search_calls": _FOLLOWUP_CALLS_COUNT,
        "other_search_calls": _OTHER_CALLS_COUNT,
        "total_tavily_calls": get_total_tavily_calls(),
        "discovery_calls": _DISCOVERY_CALLS_COUNT,
        "financial_followup_calls": _FOLLOWUP_CALLS_COUNT,
        "other_calls": _OTHER_CALLS_COUNT,
        "total_calls": get_total_tavily_calls(),
        # Outcome counters
        "tavily_requests_attempted": get_total_tavily_calls(),
        "tavily_requests_succeeded": _TAVILY_REQUESTS_SUCCEEDED,
        "tavily_requests_failed": _TAVILY_REQUESTS_FAILED,
        "tavily_cache_hits": _TAVILY_CACHE_HITS,
    }


def reset_tavily_search_count() -> None:
    """Resets all Tavily search and outcome counters."""
    global _DISCOVERY_CALLS_COUNT, _FOLLOWUP_CALLS_COUNT, _OTHER_CALLS_COUNT
    global _TAVILY_REQUESTS_SUCCEEDED, _TAVILY_REQUESTS_FAILED, _TAVILY_CACHE_HITS
    _DISCOVERY_CALLS_COUNT = 0
    _FOLLOWUP_CALLS_COUNT = 0
    _OTHER_CALLS_COUNT = 0
    _TAVILY_REQUESTS_SUCCEEDED = 0
    _TAVILY_REQUESTS_FAILED = 0
    _TAVILY_CACHE_HITS = 0


def clear_tavily_cache() -> None:
    """Clears the in-memory Tavily search cache."""
    global _TAVILY_CACHE
    with _BUDGET_LOCK:
        _TAVILY_CACHE.clear()


def get_tavily_cache_size() -> int:
    """Returns the current number of cached search queries in memory."""
    with _BUDGET_LOCK:
        return len(_TAVILY_CACHE)


# ==============================================================================
# Error Class Hierarchy
# ==============================================================================

class TavilySearchError(Exception):
    """Base exception for all Tavily search failures."""
    pass

class TavilyAuthError(TavilySearchError):
    """Raised when Tavily authentication fails (HTTP 401 / 403)."""
    pass

class TavilyQuotaError(TavilySearchError):
    """Raised when Tavily monthly credits are exhausted (HTTP 432 / plan quota)."""
    pass

class TavilyBudgetExhaustedError(TavilyQuotaError):
    """Raised when the per-run Tavily call budget (TAVILY_MAX_CALLS_PER_RUN) is reached."""
    pass

class TavilyRateLimitError(TavilySearchError):
    """Raised when Tavily rate limit is exceeded (HTTP 429)."""
    pass

class TavilyTimeoutError(TavilySearchError):
    """Raised when a Tavily search request times out."""
    pass

class ModelGenerationError(Exception):
    """Raised when Gemini model generation encounters an unrecoverable failure."""
    pass

class InvalidExtractionError(Exception):
    """Raised when structured candidate extraction fails to produce valid data."""
    pass

class SearchGroundingQuotaError(Exception):
    """Retained for backward compatibility with existing tests."""
    pass

def is_search_quota_exhausted(exc: Exception) -> bool:
    """Retained for backward compatibility with existing tests."""
    err_str = str(exc)
    return "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()


# ==============================================================================
# Query Generation
# ==============================================================================

def generate_queries(count: int = 35, seed: Optional[int] = None) -> list[str]:
    """Generates a randomized matrix of non-US revenue and funding search queries."""
    if seed is not None:
        random.seed(seed)

    queries = []
    half = count // 2
    rem = count - half

    for _ in range(half):
        region = random.choice(NON_US_REGIONS)
        sector = random.choice(TECH_SECTORS)
        tpl = random.choice(REVENUE_QUERY_TEMPLATES)
        queries.append(tpl.format(region=region, sector=sector))

    for _ in range(rem):
        region = random.choice(NON_US_REGIONS)
        sector = random.choice(TECH_SECTORS)
        tpl = random.choice(FUNDING_QUERY_TEMPLATES)
        queries.append(tpl.format(region=region, sector=sector))

    random.shuffle(queries)
    return queries


# ==============================================================================
# Tavily Search Implementation
# ==============================================================================

def search_tavily(
    query: str,
    max_results: int = 5,
    api_key: Optional[str] = None,
    category: str = "other",
    max_budget: Optional[int] = None
) -> list[dict]:
    """
    Executes a Tavily Basic Search request via HTTP standard library.
    Returns structured results: [{'title': ..., 'url': ..., 'content': ..., 'score': ...}].
    Includes caching to prevent duplicate searches and credit waste.

    HARD BUDGET GUARANTEE:
    The attempt slot is reserved (counter incremented) BEFORE urlopen is called.
    This ensures that failed, timed-out, rate-limited, or errored requests still
    consume a budget slot. No more than TAVILY_MAX_CALLS_PER_RUN outbound request
    attempts can ever be dispatched in a single run.

    Cache hits consume zero budget slots.
    """
    global _DISCOVERY_CALLS_COUNT, _FOLLOWUP_CALLS_COUNT, _OTHER_CALLS_COUNT
    global _TAVILY_REQUESTS_SUCCEEDED, _TAVILY_REQUESTS_FAILED, _TAVILY_CACHE_HITS

    key = TAVILY_API_KEY if api_key is None else api_key
    if not key or not key.strip():
        raise TavilyAuthError("TAVILY_AUTH_ERROR: Tavily API key is missing. Set TAVILY_API_KEY in .env.")

    # --- Cache check: zero budget cost ---
    norm_query = " ".join(query.strip().lower().split())
    cache_key = f"{norm_query}:{max_results}"
    with _BUDGET_LOCK:
        if cache_key in _TAVILY_CACHE:
            logger.debug(f"Reusing cached Tavily search result for: '{query}'")
            _TAVILY_CACHE_HITS += 1
            return list(_TAVILY_CACHE[cache_key])

    # --- Atomic budget reservation: reserve BEFORE dispatch ---
    budget_limit = max_budget if max_budget is not None else TAVILY_MAX_CALLS_PER_RUN
    with _BUDGET_LOCK:
        if get_total_tavily_calls() >= budget_limit:
            raise TavilyBudgetExhaustedError(
                f"TAVILY_BUDGET_EXHAUSTED: Per-run search budget reached ({budget_limit} calls). Halting new outbound requests."
            )
        # Reserve the slot BEFORE the network call
        if category == "discovery":
            _DISCOVERY_CALLS_COUNT += 1
        elif category == "financial_followup":
            _FOLLOWUP_CALLS_COUNT += 1
        else:
            _OTHER_CALLS_COUNT += 1

    # --- Dispatch the request (slot already reserved) ---
    payload = {
        "api_key": key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False
    }

    req = urllib.request.Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "TVM-Agent-Discovery/1.0"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=TAVILY_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            _TAVILY_REQUESTS_SUCCEEDED += 1
            with _BUDGET_LOCK:
                _TAVILY_CACHE[cache_key] = results
            return list(results)

    except urllib.error.HTTPError as e:
        _TAVILY_REQUESTS_FAILED += 1
        status_code = e.code
        err_msg = e.read().decode("utf-8", errors="ignore")
        if status_code in (401, 403):
            raise TavilyAuthError(f"TAVILY_AUTH_ERROR (HTTP {status_code}): Invalid or unauthorized Tavily API key.") from e
        elif status_code == 429:
            raise TavilyRateLimitError("TAVILY_RATE_LIMIT (HTTP 429): Tavily request rate limit exceeded.") from e
        elif status_code == 432 or "quota" in err_msg.lower() or "credit" in err_msg.lower():
            raise TavilyQuotaError(f"TAVILY_QUOTA_ERROR (HTTP {status_code}): Tavily monthly credits exhausted.") from e
        else:
            raise TavilySearchError(f"Tavily HTTP error {status_code}: {err_msg[:200]}") from e

    except (socket.timeout, TimeoutError) as e:
        _TAVILY_REQUESTS_FAILED += 1
        raise TavilyTimeoutError(f"TAVILY_TIMEOUT: Request timed out after {TAVILY_TIMEOUT_SECONDS}s.") from e

    except urllib.error.URLError as e:
        _TAVILY_REQUESTS_FAILED += 1
        if isinstance(e.reason, socket.timeout):
            raise TavilyTimeoutError(f"TAVILY_TIMEOUT: Request timed out after {TAVILY_TIMEOUT_SECONDS}s.") from e
        raise TavilySearchError(f"Tavily connection error: {e.reason}") from e

    except json.JSONDecodeError as e:
        _TAVILY_REQUESTS_FAILED += 1
        raise TavilySearchError(f"Malformed JSON response from Tavily: {e}") from e


def format_tavily_sources(results: list[dict]) -> str:
    """
    Formats structured Tavily results into traceable source text blocks.
    Retains title, source URL, and extracted content.
    """
    if not results:
        return ""

    blocks = []
    for idx, r in enumerate(results, 1):
        title = r.get("title", "").strip() or "Untitled Source"
        url = r.get("url", "").strip() or "No URL"
        content = r.get("content", "").strip() or "No content available."
        blocks.append(f"SOURCE {idx}\nTitle: {title}\nURL: {url}\nContent: {content}")

    return "\n\n".join(blocks)


# ==============================================================================
# Discovery Interface (Zero Google Search Grounding)
# ==============================================================================

def search_grounded_candidates(
    client: Optional[genai.Client],
    query: str,
    tavily_api_key: Optional[str] = None
) -> tuple[str, list[str]]:
    """
    Primary discovery interface for the TVB pipeline.
    Executes Tavily Basic Search to gather fresh web evidence, formats source blocks,
    and returns (raw_source_text, source_urls).
    NOTE: Google Search Grounding is completely eliminated; web search is performed by Tavily.
    """
    results = search_tavily(query=query, max_results=5, api_key=tavily_api_key, category="discovery")
    if not results:
        return "", []

    raw_text = format_tavily_sources(results)
    sources = [
        r["url"] for r in results 
        if isinstance(r, dict) and r.get("url") and r["url"].strip()
    ]
    return raw_text, sources
