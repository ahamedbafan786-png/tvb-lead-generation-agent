# Codex Handoff — TVB Lead-Generation Agent

## Repository Root
`c:\Users\Ahmed\Downloads\TVM_agent`

## Purpose
The TVB Lead-Generation Agent is an autonomous, evidence-backed research agent that discovers and qualifies early-stage non-US technology platforms ($1M–$5M ARR or funding) and verifies CEO/Co-Founder contact emails directly from official company domains.

---

## Core Qualification Criteria
1. **Financial Scale:** Cumulative funding OR verified ARR strictly between $1M and $5M USD (Policies A–D).
2. **Tech Platform:** Proprietary software, SaaS, or developer platform (consulting agencies and hardware excluded).
3. **Non-US Presence:** Strict non-US headquarters, operational base, and executive leadership.
4. **Verified Founder Email:** Verifiable full name and published personal email of CEO or Co-founder on official domain (zero guessing, role aliases rejected).
5. **Zero Fabrication:** If unverified, leave fields blank. Never synthesize emails or entity metadata.

---

## Primary Architecture & Modules
- `app.py`: Streamlit web interface with run controls, live telemetry, CSV export, and structured Audit Inspector.
- `.streamlit/config.toml`: Production configuration for headless Streamlit execution and UI theming.
- `core/config.py`: Search matrices across 22 non-US hubs, 12 software sectors, Tavily 60-call ceiling, and `MAX_CANDIDATES_PER_RUN` (150).
- `core/models.py`: Pydantic data schemas for candidates, audit logs (`CandidateAuditRecord`), and qualified leads (`LeadRecord`).
- `core/recency.py`: 18-month (~548 days) true date-math recency engine with future and malformed date safety.
- `core/discovery.py`: Tavily Basic Search discovery engine with query normalization, thread-safe cache, and budget enforcement.
- `core/extractor.py`: Structured JSON extraction via Gemini 3.5 Flash inside `<untrusted_source>` isolation, and Policies A–D financial evaluation.
- `core/us_presence.py`: Deterministic non-US validation and structured US operational footprint detection.
- `core/scraper.py`: Defensive HTTP fetcher with SSRF protection, IP/hostname validation, redirect re-validation, and strict TLS enforcement.
- `core/email_verifier.py`: Live DOM scraping with semantic card co-location and strict role alias filtering (`ceo@`, `founder@`, `admin@`, `info@`, etc.).
- `core/pipeline.py`: Production coordinator with deduplication, candidate cap enforcement, free contact-path gating, and early termination.

---

## Current Verified State
- **Automated Tests:** **447 automated tests passing** across 23 test files (0 failures, 0 regressions).
- **Offline Safety:** Tests execute 100% deterministically offline with zero live Tavily or Gemini credits consumed.
- **Credit Safeguards:** Hard per-run ceiling of 60 Tavily calls; cache cleared at pipeline start; failed calls count towards attempt budget.
- **Security:** Strict SSRF blocking (private IPs, link-local, cloud metadata), redirect re-validation, TLS verification enabled.

---

## Environment Configuration
```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
TAVILY_API_KEY=your_tavily_api_key_here
MAX_SEARCH_QUERIES=35
MAX_CANDIDATES_PER_RUN=150
```

---

## Verification Commands
```bash
# Run test suite:
pytest tests/ -q

# Run local UI:
streamlit run app.py
```

---

## Focus Areas for Second Codex Independent Review
1. **Candidate Budget Enforcement:** Verify that `MAX_CANDIDATES_PER_RUN` caps unique candidates, logs skips with `rejection_stage="CANDIDATE_BUDGET"`, and terminates discovery queries early without burning Tavily credits.
2. **Tavily Cache Lifecycle:** Verify that `clear_tavily_cache()` at pipeline initialization guarantees run boundary isolation and prevents stale search leakage across runs.
3. **SSRF & Network Hardening:** Verify private IP filtering, DNS resolution checks, redirect validation, and strict TLS enforcement in `core/scraper.py`.
4. **ARR vs. Revenue Distinction:** Verify that only confirmed ARR qualifies under Policy C, while generic, annual, TTM, or lifetime revenue cannot qualify without explicit ARR confirmation.
5. **Email Attribution & Role Aliases:** Verify that generic and executive role aliases (`ceo@`, `founder@`, `exec@`, etc.) are rejected as personal founder emails.
6. **Financial Hierarchy & Recency:** Verify Policies A–D, guarded Policy B follow-up requirement, and dual-date 18-month threshold handling.
