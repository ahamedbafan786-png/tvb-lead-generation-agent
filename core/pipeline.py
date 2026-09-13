"""
core/pipeline.py
Production pipeline coordinator with legal-suffix deduplication and budget handling.
"""

import re
import time
import logging
from typing import Callable, Optional, Any
from google import genai
from core.models import (
    CandidateCompany, LeadRecord, FunnelMetrics, PipelineSummary, CandidateAuditRecord,
    ContactPathResult, AuditEvidence, EmailEvidence, USPresenceEvidence,
    TechPlatformEvidence, FounderEvidence
)
from core.config import (
    DEFAULT_MAX_SEARCH_QUERIES, MAX_SEARCH_QUERIES_CEILING, DEFAULT_TIMEOUT_SECONDS,
    TAVILY_MAX_CALLS_PER_RUN, MAX_FINANCIAL_USD, MAX_CANDIDATES_PER_RUN
)
from core.discovery import (
    generate_queries, search_grounded_candidates,
    TavilyAuthError, TavilyQuotaError, TavilyRateLimitError, TavilyTimeoutError,
    TavilyBudgetExhaustedError, get_tavily_call_breakdown, reset_tavily_search_count, clear_tavily_cache,
    SearchGroundingQuotaError, ModelGenerationError
)
from core.extractor import (
    extract_structured_candidates, execute_targeted_funding_follow_up,
    evaluate_financial_qualification, passes_non_us_criterion, passes_tech_platform_criterion
)
from core.gemini_client import (
    AllModelsExhaustedError, reset_model_state, all_models_exhausted
)
from core.email_verifier import verify_founder_email_on_site, discover_founder_contact_path

logger = logging.getLogger(__name__)

LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b(inc|ltd|limited|llc|gmbh|s\.r\.o\.|sro|pty|technologies|software|labs|app|io|ai)\b\.?",
    re.IGNORECASE
)

def normalize_company_key(name: str) -> str:
    clean = LEGAL_SUFFIXES_REGEX.sub("", name)
    clean = re.sub(r"[^a-zA-Z0-9]", "", clean)
    return clean.lower()

def has_initial_financial_evidence(cand: CandidateCompany) -> bool:
    """
    Determines if candidate has sufficient financial signals from initial discovery:
    - Positive total_cumulative_funding_usd
    - Positive latest_round_amount_usd
    - Positive revenue_amount_usd
    - Non-empty financial_evidence_snippet containing financial keywords and numbers
    """
    if cand.total_cumulative_funding_usd is not None and cand.total_cumulative_funding_usd > 0:
        return True
    if cand.latest_round_amount_usd is not None and cand.latest_round_amount_usd > 0:
        return True
    if cand.revenue_amount_usd is not None and cand.revenue_amount_usd > 0:
        return True
    snippet = (cand.financial_evidence_snippet or "").lower()
    financial_keywords = ["seed", "series", "funding", "raised", "arr", "revenue", "valuation", "$", "€", "£"]
    if any(kw in snippet for kw in financial_keywords) and any(ch.isdigit() for ch in snippet):
        return True
    return False

def should_skip_financial_followup(
    contact_result: ContactPathResult,
    cand: CandidateCompany
) -> tuple[bool, str]:
    """
    Safest Credit-Saving Rule:
    Skip expensive Tavily financial follow-up when BOTH are true:
    A. No credible founder/contact path was found in the bounded free inspection
       (not contact_result.contact_path_found)
    AND
    B. The candidate lacks enough financial evidence in initial discovery data
       to justify further verification (not has_initial_financial_evidence(cand))

    If either side has meaningful evidence, retain the candidate for appropriate processing.
    """
    has_fin = has_initial_financial_evidence(cand)
    if not contact_result.contact_path_found and not has_fin:
        reason = (
            f"Candidate lacks credible founder/contact path ({contact_result.reason_if_no_contact_path}) "
            f"and lacks initial financial evidence."
        )
        return True, reason
    return False, ""

def run_lead_pipeline(
    api_key: str,
    target_count: int = 15,
    max_queries: int = DEFAULT_MAX_SEARCH_QUERIES,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    tavily_api_key: Optional[str] = None,
    max_candidates: Optional[int] = None
) -> PipelineSummary:
    start_time = time.time()
    max_queries = min(max_queries, MAX_SEARCH_QUERIES_CEILING)

    candidate_cap = max_candidates if max_candidates is not None else MAX_CANDIDATES_PER_RUN
    if candidate_cap is not None and candidate_cap < 0:
        raise ValueError(f"Invalid MAX_CANDIDATES_PER_RUN: {candidate_cap}. Must be non-negative or None.")

    client = genai.Client(api_key=api_key)
    reset_model_state()
    reset_tavily_search_count()
    clear_tavily_cache()

    queries = generate_queries(count=max_queries)
    metrics = FunnelMetrics()
    qualified_leads: list[LeadRecord] = []
    audit_log: list[CandidateAuditRecord] = []
    seen_companies = set()

    exit_reason: Optional[str] = None
    for idx, query in enumerate(queries):
        if exit_reason:
            break
        if len(qualified_leads) >= target_count:
            exit_reason = "Target reached"
            break
        if time.time() - start_time > DEFAULT_TIMEOUT_SECONDS:
            exit_reason = "Pipeline timeout reached"
            break

        if progress_callback:
            progress_callback(f"Query {idx+1}/{len(queries)}: {query[:45]}...", idx / len(queries))

        try:
            raw_text, sources = search_grounded_candidates(client, query, tavily_api_key=tavily_api_key)
        except TavilyBudgetExhaustedError as e:
            logger.error(f"Tavily run budget limit reached on query {idx+1}: {e}")
            exit_reason = "TAVILY_BUDGET_EXHAUSTED"
            break
        except TavilyAuthError as e:
            logger.error(f"Tavily authentication failure on query {idx+1}: {e}")
            exit_reason = "TAVILY_AUTH_ERROR"
            break
        except TavilyQuotaError as e:
            logger.error(f"Tavily credits exhausted on query {idx+1}: {e}")
            exit_reason = "TAVILY_QUOTA_EXHAUSTED"
            break
        except TavilyRateLimitError as e:
            logger.warning(f"Tavily rate limit on query {idx+1}: {e}")
            time.sleep(2)
            continue
        except TavilyTimeoutError as e:
            logger.warning(f"Tavily timeout on query {idx+1}: {e}")
            continue
        except SearchGroundingQuotaError as e:
            logger.error(f"Search Grounding Quota Exhausted on query {idx+1}: {e}")
            exit_reason = "SEARCH_GROUNDING_QUOTA_EXHAUSTED"
            break
        except ModelGenerationError as e:
            logger.warning(f"Query {idx+1} failed with model error: {e}")
            continue
        except Exception as e:
            logger.warning(f"Query {idx+1} unexpected error: {e}")
            continue

        if not raw_text:
            continue

        # Check if all Gemini models are exhausted before attempting extraction
        if all_models_exhausted():
            logger.warning(f"All Gemini models exhausted. Skipping extraction for query {idx+1}.")
            exit_reason = "ALL_GEMINI_MODELS_EXHAUSTED_RPD"
            break

        try:
            candidates = extract_structured_candidates(client, raw_text, trusted_sources=sources)
        except AllModelsExhaustedError:
            logger.error(f"All Gemini models exhausted during extraction on query {idx+1}.")
            exit_reason = "ALL_GEMINI_MODELS_EXHAUSTED_RPD"
            break

        metrics.candidates_found += len(candidates)
        metrics.raw_candidates_found += len(candidates)

        for cand in candidates:
            c_key = normalize_company_key(cand.name)
            if not c_key or c_key in seen_companies:
                continue
            seen_companies.add(c_key)
            metrics.unique_candidates_found += 1

            if candidate_cap is not None and metrics.candidates_processed >= candidate_cap:
                metrics.candidates_skipped_candidate_budget += 1
                audit_record = CandidateAuditRecord(
                    company_name=cand.name,
                    qualification_status="REJECTED",
                    rejection_stage="CANDIDATE_BUDGET",
                    rejection_reason=f"Skipped because MAX_CANDIDATES_PER_RUN ({candidate_cap}) was reached.",
                    hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                    audit_summary=f"Skipped because MAX_CANDIDATES_PER_RUN was reached ({candidate_cap})"
                )
                audit_log.append(audit_record)
                continue

            metrics.candidates_processed += 1

            # --- Cheap Deterministic Pre-Filters (Zero-Tavily-Burn Gate) ---
            # 1. Non-US check on initial data (Obvious US HQ / major US tech hubs)
            non_us_pre_ok, non_us_pre_result, non_us_pre_evidence = passes_non_us_criterion(cand)
            if not non_us_pre_ok:
                audit_record = CandidateAuditRecord(
                    company_name=cand.name,
                    qualification_status="REJECTED",
                    rejection_stage="NON_US",
                    rejection_reason=non_us_pre_evidence,
                    hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                    us_presence_result=non_us_pre_result,
                    us_presence_evidence=non_us_pre_evidence,
                    us_presence_category=cand.us_presence_category,
                    us_locations=cand.us_locations,
                    us_operational_phrase=cand.us_operational_phrase,
                    us_city_or_state=cand.us_city_or_state,
                    audit_summary=f"Disqualified before follow-up: {non_us_pre_evidence}"
                )
                audit_log.append(audit_record)
                continue

            # 2. Tech platform check on initial data
            tech_pre_ok, tech_pre_reason = passes_tech_platform_criterion(cand)
            if not tech_pre_ok:
                audit_record = CandidateAuditRecord(
                    company_name=cand.name,
                    qualification_status="REJECTED",
                    rejection_stage="TECH_PLATFORM",
                    rejection_reason=tech_pre_reason,
                    hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                    tech_platform_notes=tech_pre_reason,
                    audit_summary=f"Disqualified before follow-up: {tech_pre_reason}"
                )
                audit_log.append(audit_record)
                continue

            # 3. Known acquired or defunct status
            if cand.acquisition_or_closure_discovered:
                audit_record = CandidateAuditRecord(
                    company_name=cand.name,
                    qualification_status="REJECTED",
                    rejection_stage="FINANCIAL",
                    rejection_reason="Entity was acquired, closed, or ceased independent operations.",
                    hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                    audit_summary="Disqualified before follow-up: Acquired or closed."
                )
                audit_log.append(audit_record)
                continue

            # 4. Known cumulative funding already exceeding $5M cap
            if cand.total_cumulative_funding_usd is not None and cand.total_cumulative_funding_usd > MAX_FINANCIAL_USD:
                audit_record = CandidateAuditRecord(
                    company_name=cand.name,
                    qualification_status="REJECTED",
                    rejection_stage="FINANCIAL",
                    rejection_reason=f"Cumulative funding (${cand.total_cumulative_funding_usd:,.0f}) exceeds ${MAX_FINANCIAL_USD:,.0f}.",
                    financial_figure_used=f"${cand.total_cumulative_funding_usd:,.0f}",
                    financial_type="CUMULATIVE_FUNDING",
                    hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                    audit_summary=f"Disqualified before follow-up: Cumulative funding (${cand.total_cumulative_funding_usd:,.0f}) > ${MAX_FINANCIAL_USD:,.0f} cap."
                )
                audit_log.append(audit_record)
                continue

            # --- Contact-path Discovery & Gating (New Search Order) ---
            metrics.potential_followups_before_contact_gate += 1

            # 1. Official-domain check
            if not cand.website_url or not cand.website_url.strip():
                audit_record = CandidateAuditRecord(
                    company_name=cand.name,
                    qualification_status="REJECTED",
                    rejection_stage="EMAIL_VERIFICATION",
                    rejection_reason="No official website URL provided.",
                    hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                    audit_summary="Disqualified before follow-up: No official website URL."
                )
                audit_log.append(audit_record)
                metrics.followups_skipped_no_contact_path += 1
                continue

            # 2. FREE bounded contact-path discovery (0 Tavily calls)
            contact_result = discover_founder_contact_path(cand)
            metrics.contact_path_checks += 1
            if contact_result.contact_path_found:
                metrics.contact_paths_found += 1
            if contact_result.exact_email_found:
                metrics.exact_emails_found_early += 1

            # Preserve founder if discovered by official site
            if not cand.founder_name and contact_result.founder_name_found:
                cand.founder_name = contact_result.founder_name_found
                cand.founder_title = contact_result.founder_title_found or "Co-Founder"
            if contact_result.founder_source_url and not cand.founder_source_url:
                cand.founder_source_url = contact_result.founder_source_url

            # 3. Follow-up Gating Condition:
            # Skip expensive financial follow-up when BOTH are true:
            # A. No credible founder/contact path was found in the bounded free inspection
            # AND
            # B. The candidate lacks enough financial evidence in initial discovery data to justify further verification.
            skip_followup, skip_reason = should_skip_financial_followup(contact_result, cand)
            if skip_followup:
                metrics.followups_skipped_no_contact_path += 1
                audit_record = CandidateAuditRecord(
                    company_name=cand.name,
                    qualification_status="REJECTED",
                    rejection_stage="EMAIL_VERIFICATION",
                    rejection_reason=skip_reason,
                    hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                    audit_summary=f"Disqualified before follow-up: {contact_result.reason_if_no_contact_path}"
                )
                audit_log.append(audit_record)
                continue

            # --- Candidate passed contact gate: Execute targeted funding follow-up ---
            metrics.actual_followups_after_contact_gate += 1
            metrics.followups_executed += 1
            try:
                cand = execute_targeted_funding_follow_up(client, cand, tavily_api_key=tavily_api_key)
            except TavilyBudgetExhaustedError as e:
                logger.error(f"Tavily run budget limit reached during candidate follow-up: {e}")
                exit_reason = "TAVILY_BUDGET_EXHAUSTED"
                break

            fin_ok, audit_record = evaluate_financial_qualification(cand)
            if not fin_ok:
                audit_log.append(audit_record)
                continue
            metrics.passed_financial += 1

            non_us_ok, non_us_result, non_us_evidence = passes_non_us_criterion(cand)
            audit_record.us_presence_result = non_us_result
            audit_record.us_presence_evidence = non_us_evidence
            audit_record.us_presence_category = cand.us_presence_category
            audit_record.us_locations = cand.us_locations
            audit_record.us_operational_phrase = cand.us_operational_phrase
            audit_record.us_city_or_state = cand.us_city_or_state

            if audit_record.evidence:
                loc_str = ", ".join(cand.us_locations) if cand.us_locations else None
                audit_record.evidence.us_presence = USPresenceEvidence(
                    category=cand.us_presence_category,
                    locations=cand.us_locations,
                    location=loc_str,
                    country="United States" if (cand.us_locations or not non_us_ok or cand.us_presence_detected) else None,
                    city_or_state=cand.us_city_or_state or (cand.us_locations[0] if cand.us_locations else None),
                    operational_phrase=cand.us_operational_phrase,
                    source_url=cand.us_presence_source_url or cand.website_url,
                    source_quote=cand.us_presence_source_quote or cand.us_presence_evidence or non_us_evidence,
                    verification_status="REJECTED" if not non_us_ok else "VERIFIED"
                )

            if not non_us_ok:
                audit_record.qualification_status = "REJECTED"
                audit_record.rejection_stage = "NON_US"
                audit_record.rejection_reason = non_us_evidence
                audit_log.append(audit_record)
                continue
            metrics.passed_non_us += 1

            tech_ok, tech_reason = passes_tech_platform_criterion(cand)
            if audit_record.evidence:
                audit_record.evidence.tech_platform = TechPlatformEvidence(
                    is_tech_platform=bool(cand.is_tech_platform),
                    platform_type=cand.industry,
                    source_url=cand.tech_source_url or cand.website_url,
                    source_quote=cand.tech_source_quote or cand.description,
                    verification_status="VERIFIED" if tech_ok else "REJECTED"
                )
            if not tech_ok:
                audit_record.qualification_status = "REJECTED"
                audit_record.rejection_stage = "TECH_PLATFORM"
                audit_record.rejection_reason = tech_reason
                audit_log.append(audit_record)
                continue
            metrics.passed_tech_check += 1

            if not cand.founder_name:
                audit_record.qualification_status = "REJECTED"
                audit_record.rejection_stage = "EMAIL_VERIFICATION"
                audit_record.rejection_reason = "No CEO or Co-founder identified."
                audit_log.append(audit_record)
                continue

            if audit_record.evidence:
                audit_record.evidence.founder = FounderEvidence(
                    founder_name=cand.founder_name,
                    founder_title=cand.founder_title,
                    source_url=cand.founder_source_url or cand.website_url,
                    source_quote=f"{cand.founder_name} ({cand.founder_title})" if cand.founder_name else None,
                    verification_status="VERIFIED"
                )

            email_audit: dict[str, Any] = {}
            try:
                email_verified = verify_founder_email_on_site(cand, audit_trail=email_audit)
            except TypeError:
                email_verified = verify_founder_email_on_site(cand)
            if not email_verified:
                audit_record.qualification_status = "REJECTED"
                audit_record.rejection_stage = "EMAIL_VERIFICATION"
                audit_record.rejection_reason = (
                    f"No verified founder email on site for {cand.founder_name}. Zero-guessing enforced."
                )
                audit_log.append(audit_record)
                continue

            email, source_url = email_verified
            metrics.passed_email_verification += 1

            audit_record.qualification_status = "ACCEPTED"
            audit_record.verified_email = email
            audit_record.email_source_url = source_url
            if cand.founder_source_url:
                audit_record.founder_source_url = cand.founder_source_url

            email_ev = EmailEvidence(
                email=email,
                source_url=source_url,
                attribution_result=email_audit.get("attribution_result") or "ACCEPTED_RULE_1",
                verification_status="VERIFIED"
            )
            if audit_record.evidence:
                audit_record.evidence.email = email_ev

            audit_log.append(audit_record)

            lead = LeadRecord(
                company_name=cand.name,
                description=cand.description,
                industry=cand.industry,
                ceo_cofounder_name=f"{cand.founder_name} ({cand.founder_title or 'Co-Founder'})",
                verified_email=email,
                email_source_url=source_url,
                hq_location=f"{cand.hq_city or 'N/A'}, {cand.hq_country or 'Unknown'}",
                financial_signal=audit_record.financial_figure_used or "Verified Signal",
                revenue_period=audit_record.revenue_period,
                financial_evidence_date=cand.financial_evidence_date,
                latest_round_date=cand.latest_round_date,
                recency_basis=audit_record.recency_basis or "Recent",
                us_presence_result=non_us_result,
                us_presence_evidence=non_us_evidence,
                financial_confidence=audit_record.confidence_rating,
                recency_audit_note=audit_record.audit_summary,
                financial_source_url=cand.financial_source_url,
                financial_source_quote=cand.financial_source_quote or cand.financial_evidence_snippet,
                founder_source_url=cand.founder_source_url,
                founder_source_quote=f"{cand.founder_name} ({cand.founder_title})" if cand.founder_name else None,
                tech_source_url=cand.tech_source_url or cand.website_url,
                tech_source_quote=cand.tech_source_quote or cand.description,
                us_presence_source_url=cand.us_presence_source_url or cand.website_url,
                us_presence_source_quote=cand.us_presence_source_quote or cand.us_presence_evidence or non_us_evidence,
                evidence=audit_record.evidence
            )
            qualified_leads.append(lead)

            if len(qualified_leads) >= target_count:
                exit_reason = "Target reached"
                break

        if exit_reason or len(qualified_leads) >= target_count:
            break

        if candidate_cap is not None and metrics.candidates_processed >= candidate_cap:
            exit_reason = f"MAX_CANDIDATES_PER_RUN reached ({candidate_cap})"
            break

    duration = time.time() - start_time
    shortfall = max(0, target_count - len(qualified_leads))
    if exit_reason is None:
        exit_reason = "Target reached" if shortfall == 0 else "Query/timeout budget reached"

    if progress_callback:
        progress_callback("Run complete.", 1.0)

    call_breakdown = get_tavily_call_breakdown()
    total_calls = call_breakdown.get("total_calls", 0)
    disc_calls = call_breakdown.get("discovery_calls", 0)
    follow_calls = call_breakdown.get("financial_followup_calls", 0)
    remaining_budget = max(0, TAVILY_MAX_CALLS_PER_RUN - total_calls)

    leads_per_call = round(len(qualified_leads) / total_calls, 4) if total_calls > 0 else 0.0
    metrics.verified_leads_per_tavily_call = leads_per_call

    return PipelineSummary(
        total_queries_run=idx + 1 if 'idx' in locals() else 0,
        duration_seconds=round(duration, 2),
        funnel=metrics,
        leads=qualified_leads,
        audit_log=audit_log,
        shortfall=shortfall,
        exit_reason=exit_reason,
        total_tavily_calls=total_calls,
        tavily_discovery_calls=disc_calls,
        tavily_followup_calls=follow_calls,
        tavily_budget_remaining=remaining_budget,
        verified_leads_per_tavily_call=leads_per_call,
        candidates_processed=metrics.candidates_processed,
        candidates_skipped_candidate_budget=metrics.candidates_skipped_candidate_budget
    )
