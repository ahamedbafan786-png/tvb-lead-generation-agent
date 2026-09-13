"""
core/models.py
Pydantic data models for candidate tracking, structured audits, and qualified leads.
"""

from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field

class RevenuePeriod(str, Enum):
    ARR = "ARR"
    ANNUAL = "ANNUAL"
    TTM = "TTM"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    LIFETIME = "LIFETIME"
    UNKNOWN = "UNKNOWN"

class CandidateCompany(BaseModel):
    name: str = Field(description="Commercial name of the company")
    website_url: Optional[str] = Field(default=None, description="Official website URL")
    description: Optional[str] = Field(default=None, description="Summary of platform or software product")
    industry: Optional[str] = Field(default=None, description="Primary tech sector")
    hq_country: Optional[str] = Field(default=None, description="Country where HQ and operations reside")
    hq_city: Optional[str] = Field(default=None)
    is_tech_platform: Optional[bool] = Field(default=None, description="True if software/SaaS/platform")

    total_cumulative_funding_usd: Optional[float] = Field(default=None)
    latest_round_name: Optional[str] = Field(default=None)
    latest_round_amount_usd: Optional[float] = Field(default=None)
    latest_round_date: Optional[str] = Field(default=None)
    financial_evidence_date: Optional[str] = Field(default=None)
    revenue_amount_usd: Optional[float] = Field(default=None)
    revenue_date: Optional[str] = Field(default=None)
    revenue_period: Optional[str] = Field(default=None, description="ARR, ANNUAL, TTM, MONTHLY, QUARTERLY, LIFETIME, UNKNOWN")
    financial_conflict_detected: bool = Field(default=False)
    subsequent_rounds_discovered: bool = Field(default=False)
    acquisition_or_closure_discovered: bool = Field(default=False)
    financial_followup_complete: bool = Field(
        default=False,
        description=(
            "True ONLY when a financial follow-up search was successfully executed "
            "and the extraction completed without error. When False, the values of "
            "subsequent_rounds_discovered and financial_conflict_detected are UNKNOWN "
            "(not 'verified absent') and Policy B must not accept."
        )
    )
    financial_evidence_snippet: str = Field(default="")

    us_presence_detected: bool = Field(default=False)
    us_presence_evidence: Optional[str] = Field(default=None)
    us_operations_evidence: Optional[str] = Field(default=None)
    us_team_evidence: Optional[str] = Field(default=None)
    us_presence_category: Optional[str] = Field(default=None)
    us_locations: list[str] = Field(default_factory=list)
    us_operational_phrase: Optional[str] = Field(default=None)
    us_city_or_state: Optional[str] = Field(default=None)

    founder_name: Optional[str] = Field(default=None)
    founder_title: Optional[str] = Field(default=None)

    # Evidence traceability fields (bounded context)
    financial_source_url: Optional[str] = Field(default=None)
    financial_source_quote: Optional[str] = Field(default=None)
    founder_source_url: Optional[str] = Field(default=None)
    founder_source_quote: Optional[str] = Field(default=None)
    tech_source_url: Optional[str] = Field(default=None)
    tech_source_quote: Optional[str] = Field(default=None)
    us_presence_source_url: Optional[str] = Field(default=None)
    us_presence_source_quote: Optional[str] = Field(default=None)
    supporting_evidence: list["EvidenceItem"] = Field(default_factory=list)
    trusted_source_urls: list[str] = Field(default_factory=list)
    trusted_source_text: Optional[str] = Field(default=None)

class EvidenceItem(BaseModel):
    evidence_type: str = Field(description="FINANCIAL, RECENCY, NON_US, TECH_PLATFORM, FOUNDER, EMAIL")
    source_url: Optional[str] = Field(default=None, description="Verified origin URL")
    source_quote: Optional[str] = Field(default=None, description="Concise quote or context from source")
    extracted_value: Optional[str] = Field(default=None, description="Extracted figure or entity")
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class FinancialEvidence(BaseModel):
    amount_usd: Optional[float] = None
    financial_type: Optional[str] = None
    revenue_period: Optional[str] = None
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    evidence_date: Optional[str] = None
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class RecencyEvidence(BaseModel):
    evidence_date: Optional[str] = None
    recency_basis: Optional[str] = None
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    is_recent: bool = False
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class HQEvidence(BaseModel):
    country: Optional[str] = None
    city: Optional[str] = None
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class USPresenceEvidence(BaseModel):
    category: Optional[str] = None
    locations: list[str] = Field(default_factory=list)
    location: Optional[str] = None
    country: Optional[str] = None
    city_or_state: Optional[str] = None
    operational_phrase: Optional[str] = None
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class TechPlatformEvidence(BaseModel):
    is_tech_platform: bool = False
    platform_type: Optional[str] = None
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class FounderEvidence(BaseModel):
    founder_name: Optional[str] = None
    founder_title: Optional[str] = None
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class EmailSourceType(str, Enum):
    FIRST_PARTY_OFFICIAL = "FIRST_PARTY_OFFICIAL"
    FIRST_PARTY_FOUNDER_PUBLISHED = "FIRST_PARTY_FOUNDER_PUBLISHED"
    PUBLIC_CORPORATE_RECORD = "PUBLIC_CORPORATE_RECORD"
    THIRD_PARTY_REPORTED = "THIRD_PARTY_REPORTED"
    INFERRED = "INFERRED"

class EmailEvidence(BaseModel):
    email: Optional[str] = None
    source_url: Optional[str] = None
    source_quote: Optional[str] = None
    source_type: Optional[str] = None
    attribution_result: Optional[str] = None
    verification_status: Literal["VERIFIED", "UNVERIFIED", "REJECTED"] = "VERIFIED"

class AuditEvidence(BaseModel):
    financial: Optional[FinancialEvidence] = None
    recency: Optional[RecencyEvidence] = None
    hq: Optional[HQEvidence] = None
    us_presence: Optional[USPresenceEvidence] = None
    tech_platform: Optional[TechPlatformEvidence] = None
    founder: Optional[FounderEvidence] = None
    email: Optional[EmailEvidence] = None
    supporting_sources: list[EvidenceItem] = Field(default_factory=list)

class ContactPathResult(BaseModel):
    founder_name_found: Optional[str] = None
    founder_title_found: Optional[str] = None
    founder_source_url: Optional[str] = None
    exact_email_found: Optional[str] = None
    email_source_url: Optional[str] = None
    email_source_type: Optional[str] = None
    attribution_context: Optional[str] = None
    contact_path_found: bool = False
    pages_checked: list[str] = Field(default_factory=list)
    reason_if_no_contact_path: Optional[str] = None

class CandidateAuditRecord(BaseModel):
    company_name: str
    qualification_status: Literal["ACCEPTED", "REJECTED"]
    rejection_stage: Optional[Literal["FINANCIAL", "RECENCY", "NON_US", "TECH_PLATFORM", "EMAIL_VERIFICATION", "CANDIDATE_BUDGET"]] = None
    rejection_reason: Optional[str] = None
    financial_figure_used: Optional[str] = None
    financial_type: Optional[Literal["CUMULATIVE_FUNDING", "SINGLE_ROUND_FUNDING", "ARR", "ANNUAL_REVENUE", "TTM_REVENUE", "MONTHLY_REVENUE", "QUARTERLY_REVENUE", "LIFETIME_REVENUE", "INSUFFICIENT", "UNKNOWN"]] = None
    revenue_period: Optional[str] = None
    latest_round_date: Optional[str] = None
    financial_evidence_date: Optional[str] = None
    recency_basis: Optional[str] = None
    is_recent: bool = False
    confidence_rating: Literal["HIGH", "MEDIUM", "UNQUALIFIED"] = "UNQUALIFIED"
    hq_location: str = "N/A, Unknown"
    us_presence_result: str = "NON_US_VERIFIED"
    us_presence_evidence: Optional[str] = None
    us_presence_category: Optional[str] = None
    us_locations: list[str] = Field(default_factory=list)
    us_operational_phrase: Optional[str] = None
    us_city_or_state: Optional[str] = None
    tech_platform_notes: str = ""
    founder_identified: Optional[str] = None
    verified_email: Optional[str] = None
    email_source_url: Optional[str] = None
    email_source_type: Optional[str] = None
    audit_summary: str = ""

    # Evidence traceability fields
    financial_source_url: Optional[str] = None
    financial_source_quote: Optional[str] = None
    founder_source_url: Optional[str] = None
    evidence: Optional[AuditEvidence] = None

class LeadRecord(BaseModel):
    company_name: str
    description: Optional[str] = Field(default=None)
    industry: Optional[str] = Field(default=None)
    ceo_cofounder_name: str
    verified_email: str
    email_source_url: str
    email_source_type: Optional[str] = None
    hq_location: str
    financial_signal: str
    financial_evidence_date: Optional[str] = None
    latest_round_date: Optional[str] = None
    revenue_period: Optional[str] = Field(default=None)
    recency_basis: str
    us_presence_result: str
    us_presence_evidence: Optional[str] = None
    financial_confidence: str
    recency_audit_note: str

    # Evidence traceability fields
    financial_source_url: Optional[str] = Field(default=None)
    financial_source_quote: Optional[str] = Field(default=None)
    founder_source_url: Optional[str] = Field(default=None)
    founder_source_quote: Optional[str] = Field(default=None)
    tech_source_url: Optional[str] = Field(default=None)
    tech_source_quote: Optional[str] = Field(default=None)
    us_presence_source_url: Optional[str] = Field(default=None)
    us_presence_source_quote: Optional[str] = Field(default=None)
    evidence: Optional[AuditEvidence] = Field(default=None)

class FunnelMetrics(BaseModel):
    candidates_found: int = 0
    raw_candidates_found: int = 0
    unique_candidates_found: int = 0
    candidates_processed: int = 0
    candidates_skipped_candidate_budget: int = 0
    passed_financial: int = 0
    passed_non_us: int = 0
    passed_tech_check: int = 0
    passed_email_verification: int = 0

    # Contact-path & follow-up telemetry
    potential_followups_before_contact_gate: int = 0
    contact_path_checks: int = 0
    contact_paths_found: int = 0
    exact_emails_found_early: int = 0
    followups_skipped_no_contact_path: int = 0
    actual_followups_after_contact_gate: int = 0
    followups_executed: int = 0
    verified_leads_per_tavily_call: float = 0.0

class PipelineSummary(BaseModel):
    total_queries_run: int
    duration_seconds: float
    funnel: FunnelMetrics
    leads: list[LeadRecord]
    audit_log: list[CandidateAuditRecord]
    shortfall: int
    exit_reason: str
    total_tavily_calls: int = 0
    tavily_discovery_calls: int = 0
    tavily_followup_calls: int = 0
    tavily_budget_remaining: int = 0
    verified_leads_per_tavily_call: float = 0.0
    candidates_processed: int = 0
    candidates_skipped_candidate_budget: int = 0
