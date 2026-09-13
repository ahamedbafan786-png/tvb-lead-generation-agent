"""
app.py
Streamlit application displaying full audit records, separated evidence dates,
recency basis, and US presence validation details.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from core.pipeline import run_lead_pipeline
from core.config import (
    DEFAULT_MAX_SEARCH_QUERIES, MAX_SEARCH_QUERIES_CEILING, TAVILY_MAX_CALLS_PER_RUN
)

load_dotenv()

st.set_page_config(page_title="TVB Lead-Gen Agent", layout="wide", page_icon="🎯")
st.title("🎯 TVB Autonomous Lead-Generation Agent")
st.caption("Evidence-backed discovery and verification of non-US tech platforms ($1M–$5M ARR or funding).")

def get_gemini_api_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")

def get_tavily_api_key() -> str:
    try:
        if "TAVILY_API_KEY" in st.secrets:
            return st.secrets["TAVILY_API_KEY"]
    except Exception:
        pass
    return os.getenv("TAVILY_API_KEY", "")

def _get_record_email_source_type(record, default: str = "N/A") -> str:
    """Safely extracts email source type from record or embedded evidence."""
    source_type = getattr(record, "email_source_type", None)
    if source_type:
        return str(source_type)
    evidence = getattr(record, "evidence", None)
    if evidence and getattr(evidence, "email", None) and getattr(evidence.email, "source_type", None):
        return str(evidence.email.source_type)
    return default

api_key = get_gemini_api_key()
tavily_key = get_tavily_api_key()

with st.sidebar:
    st.header("⚙️ Operational Controls")
    target_leads = st.number_input("Target Qualified Leads", min_value=1, max_value=25, value=15)
    query_budget = st.slider(
        "Search Query Budget",
        min_value=1,
        max_value=MAX_SEARCH_QUERIES_CEILING,
        value=DEFAULT_MAX_SEARCH_QUERIES
    )
    st.markdown(f"**Tavily Call Budget Limit:** `{TAVILY_MAX_CALLS_PER_RUN}` calls/run")
    st.markdown("---")
    st.markdown("**Codified Qualification Rules:**")
    st.markdown("- **Policy A:** Cumulative funding preferred ($1M–$5M)")
    st.markdown("- **Policy B:** Single-round fallback (Medium confidence)")
    st.markdown("- **Policy C:** Verified ARR ($1M–$5M)")
    st.markdown("- **Policy D:** Conservative scale rule (Excessive funding rejects)")
    st.markdown("- **Recency:** Distinct evidence date <= 18 months")
    st.markdown("- **Non-US:** Evidence-backed minimal US footprint")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in st.secrets or .env.")
if not tavily_key:
    st.error("⚠️ TAVILY_API_KEY not found in st.secrets or .env.")

can_run = bool(api_key and tavily_key)

if st.button("🚀 Run Lead Generation Agent", type="primary", disabled=not can_run):
    prog_bar = st.progress(0.0)
    status_text = st.empty()

    def update_ui(msg: str, frac: float):
        status_text.info(msg)
        prog_bar.progress(frac)

    summary = run_lead_pipeline(
        api_key=api_key,
        target_count=target_leads,
        max_queries=query_budget,
        progress_callback=update_ui,
        tavily_api_key=tavily_key
    )

    st.success(f"Execution completed in {summary.duration_seconds}s. Reason: {summary.exit_reason}")

    if summary.exit_reason == "TAVILY_BUDGET_EXHAUSTED":
        st.warning("⚠️ Execution halted: Tavily per-run call budget limit reached.")

    st.subheader("⚡ Tavily Credit & Budget Telemetry")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Total Calls Used", summary.total_tavily_calls)
    t2.metric("Discovery Calls", summary.tavily_discovery_calls)
    t3.metric("Follow-up Calls", summary.tavily_followup_calls)
    t4.metric("Run Budget Remaining", summary.tavily_budget_remaining)

    st.subheader("📊 Qualification Funnel")
    f = summary.funnel
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Candidates Found", f.candidates_found)
    c2.metric("Passed Financial", f.passed_financial)
    c3.metric("Passed Non-US", f.passed_non_us)
    c4.metric("Passed Tech Check", f.passed_tech_check)
    c5.metric("Verified Emails Found", f.passed_email_verification)

    if summary.leads:
        st.subheader(f"📋 Qualified Leads ({len(summary.leads)})")
        rows = [
            {
                "Company Name": lead.company_name,
                "Description": lead.description,
                "Industry / Sector": lead.industry,
                "CEO / Co-Founder": lead.ceo_cofounder_name,
                "Verified Email": lead.verified_email,
                "Email Evidence URL": lead.email_source_url,
                "Email Source Type": _get_record_email_source_type(lead, default="FIRST_PARTY_OFFICIAL"),
                "Financial Evidence URL": lead.financial_source_url or "N/A",
                "Financial Signal": lead.financial_signal,
                "Revenue Period": lead.revenue_period or "N/A",
                "Latest Round Date": lead.latest_round_date or "N/A",
                "Financial Evidence Date": lead.financial_evidence_date or "N/A",
                "Recency Basis": lead.recency_basis,
                "US Presence Result": lead.us_presence_result,
                "US Presence Evidence": lead.us_presence_evidence or "None",
                "Confidence": lead.financial_confidence,
                "HQ Location": lead.hq_location
            }
            for lead in summary.leads
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Verified Leads CSV",
            data=csv,
            file_name="tvb_qualified_leads.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No leads met all 4 criteria. Shortfall: {summary.shortfall}")

    with st.expander("🔍 Structured Audit & Disqualification Log", expanded=False):
        audit_rows = [
            {
                "Company": rec.company_name,
                "Status": rec.qualification_status,
                "Stage": rec.rejection_stage or "ACCEPTED",
                "Reason / Rationale": rec.rejection_reason or rec.audit_summary,
                "Financial Metric": rec.financial_figure_used or "None",
                "Revenue Period": rec.revenue_period or "N/A",
                "Financial Evidence URL": rec.financial_source_url or "N/A",
                "Latest Round Date": rec.latest_round_date or "N/A",
                "Evidence Date": rec.financial_evidence_date or "N/A",
                "Recency Basis": rec.recency_basis or "N/A",
                "Email Source Type": _get_record_email_source_type(rec, default="N/A"),
                "US Presence Result": rec.us_presence_result,
                "US Presence Evidence": rec.us_presence_evidence or "None",
                "HQ Location": rec.hq_location
            }
            for rec in summary.audit_log
        ]
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)
