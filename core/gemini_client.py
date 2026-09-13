"""
core/gemini_client.py
Resilient Gemini generation with automatic model fallback rotation
and inter-call pacing for Free Tier rate limits (20 RPD per model).

Strategy:
  1. Pace calls with MIN_INTER_CALL_DELAY_SECONDS between consecutive requests.
  2. On 429 daily RPD quota -> mark model as exhausted, try fallback model.
  3. On 429 RPM rate limit -> sleep for retryDelay, retry once on same model.
  4. Raise AllModelsExhaustedError when no models remain for the day.
"""

import re
import time
import logging
from typing import Optional

from google import genai
from core.config import GEMINI_MODEL, FALLBACK_MODEL

logger = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────
_EXHAUSTED_MODELS: set = set()
_LAST_GEMINI_CALL_TIME: float = 0.0
MIN_INTER_CALL_DELAY_SECONDS: float = 4.0


# ── Error ─────────────────────────────────────────────────────────────
class AllModelsExhaustedError(Exception):
    """All configured Gemini models have hit their daily RPD quota."""
    pass


# ── Public helpers ────────────────────────────────────────────────────
def reset_model_state() -> None:
    """Reset exhausted-models tracking.  Call at the start of each pipeline run."""
    global _EXHAUSTED_MODELS, _LAST_GEMINI_CALL_TIME
    _EXHAUSTED_MODELS.clear()
    _LAST_GEMINI_CALL_TIME = 0.0


def all_models_exhausted() -> bool:
    """True when both primary and fallback models are exhausted for the day."""
    return GEMINI_MODEL in _EXHAUSTED_MODELS and FALLBACK_MODEL in _EXHAUSTED_MODELS


def get_exhausted_models() -> set:
    """Returns the current set of exhausted model names."""
    return set(_EXHAUSTED_MODELS)


# ── Internal helpers ──────────────────────────────────────────────────
def _is_daily_quota_error(error_str: str) -> bool:
    """Distinguish daily RPD quota from per-minute RPM rate limit."""
    return "PerDay" in error_str or "FreeTier" in error_str


def _parse_retry_delay(error_str: str) -> Optional[float]:
    """Extract retryDelay seconds from a Gemini 429 error response."""
    match = re.search(r"retryDelay.*?['\"]?(\d+(?:\.\d+)?)\s*s", error_str)
    if match:
        return float(match.group(1))
    return None


def _pace_call() -> None:
    """Enforce minimum inter-call delay to avoid per-minute rate limits."""
    global _LAST_GEMINI_CALL_TIME
    now = time.time()
    if _LAST_GEMINI_CALL_TIME > 0:
        elapsed = now - _LAST_GEMINI_CALL_TIME
        if elapsed < MIN_INTER_CALL_DELAY_SECONDS:
            wait = MIN_INTER_CALL_DELAY_SECONDS - elapsed
            logger.debug(f"Pacing: sleeping {wait:.1f}s before next Gemini call")
            time.sleep(wait)
    _LAST_GEMINI_CALL_TIME = time.time()


# ── Core generation function ─────────────────────────────────────────
def generate_with_fallback(
    client: genai.Client,
    contents: str,
    config: dict,
    primary_model: Optional[str] = None,
    fallback_model: Optional[str] = None,
) -> str:
    """
    Generate content with automatic model fallback on daily quota exhaustion.

    Behaviour
    ---------
    1. Tries *primary_model* first (with inter-call pacing).
    2. On 429 **daily RPD**: marks that model as exhausted, falls through to
       *fallback_model*.
    3. On 429 **RPM**: sleeps for the server-indicated retryDelay, retries once
       on the same model.
    4. Returns ``response.text`` on success.
    5. Raises ``AllModelsExhaustedError`` if every model is exhausted.

    Parameters
    ----------
    client : genai.Client
        Authenticated Gemini client.
    contents : str
        Prompt / user message.
    config : dict
        Generation config (system_instruction, response_mime_type, temperature, etc.).
    primary_model : str, optional
        Defaults to ``GEMINI_MODEL`` from config.
    fallback_model : str, optional
        Defaults to ``FALLBACK_MODEL`` from config.

    Returns
    -------
    str
        The raw response text from the first model that succeeds.
    """
    primary = primary_model or GEMINI_MODEL
    fallback = fallback_model or FALLBACK_MODEL
    models_to_try = [m for m in [primary, fallback] if m not in _EXHAUSTED_MODELS]

    if not models_to_try:
        raise AllModelsExhaustedError(
            f"All Gemini models exhausted for today (daily RPD limit). "
            f"Exhausted: {sorted(_EXHAUSTED_MODELS)}"
        )

    last_error: Optional[Exception] = None
    for model in models_to_try:
        _pace_call()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text

        except Exception as e:
            error_str = str(e)

            if "429" not in error_str or "RESOURCE_EXHAUSTED" not in error_str:
                raise  # Non-rate-limit error -> propagate immediately

            # ── Rate-limit branch ──
            if _is_daily_quota_error(error_str):
                # Daily RPD exhausted for this model
                _EXHAUSTED_MODELS.add(model)
                logger.warning(
                    f"Daily RPD quota exhausted for '{model}'. "
                    f"Trying fallback model..."
                )
                last_error = e
                continue  # Fall through to next model

            # Per-minute RPM limit -> sleep and retry once
            delay = _parse_retry_delay(error_str) or 15.0
            delay = min(delay, 60.0)
            logger.info(
                f"RPM rate limit on '{model}'. "
                f"Sleeping {delay:.0f}s before retry..."
            )
            time.sleep(delay)

            _pace_call()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                return response.text
            except Exception as retry_err:
                error_str2 = str(retry_err)
                if "429" in error_str2 and _is_daily_quota_error(error_str2):
                    _EXHAUSTED_MODELS.add(model)
                    logger.warning(
                        f"Daily RPD quota exhausted for '{model}' after retry."
                    )
                last_error = retry_err
                continue

    raise AllModelsExhaustedError(
        f"All Gemini models exhausted for today. Last error: {last_error}"
    )
