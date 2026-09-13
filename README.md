# TVB Lead-Generation Agent

An autonomous, evidence-backed lead-generation pipeline designed for venture builders to discover, verify, and audit early-stage non-US technology platforms ($1M–$5M ARR or funding) with verified founder executive emails.

---

## 1. Overview & TVB Mandate

The agent qualifies prospective portfolio candidates against four mandatory criteria:
1. **Financial Scale:** Total cumulative funding raised OR verified current ARR is strictly between **$1,000,000 and $5,000,000 USD**.
2. **Tech Platform:** Operates a proprietary software product, SaaS, developer tool, or digital platform (consulting agencies, IT services, and pure hardware entities are excluded).
3. **Minimal to No US Presence:** HQ, operational center, and executive team reside strictly outside the United States.
4. **Verified Founder Email:** Full name and verified personal email of the CEO or Co-Founder published on and extracted directly from official company domains (zero guessing).

---

## 2. Financial Qualification Hierarchy (Policies A–D)

Financial validation operates under a deterministic 4-policy hierarchy:

- **Policy A (Cumulative Priority):** Cumulative funding strictly between $1M and $5M USD qualifies with High Confidence. Any cumulative funding exceeding $5M USD is strictly rejected, regardless of round timing or recent round size.
- **Policy B (Guarded Single-Round Fallback):** A single financing round between $1M and $5M qualifies with Medium Confidence **only if** an explicit follow-up search was successfully completed (`financial_followup_complete=True`) and confirmed zero subsequent growth rounds, acquisitions, or Series B/C events. If the follow-up search failed or was skipped, the candidate cannot pass Policy B.
- **Policy C (Verified ARR):** A current ARR between $1M and $5M qualifies independently.
  - *ARR vs. Generic Revenue Distinction:* The metric must be explicitly verified as ARR. Generic revenue figures (such as annual turnover, trailing twelve months / TTM, monthly run-rate, quarterly revenue, or lifetime gross volume) do **not** qualify as ARR under Policy C.
- **Policy D (Scale Conflict / Excessive Funding):** A candidate with a qualifying ARR ($1M–$5M) that has also accumulated >$5M in cumulative equity funding is strictly rejected. Scale conflicts disqualify candidates to ensure focus on capital-efficient, early-stage platforms.

---

## 3. Recency & Date Validation Engine

Recency evaluates two distinct date signals against an 18-month (~548 days) boundary:
- **`latest_round_date`**: The date of the most recent financing round.
- **`financial_evidence_date`**: The date of the source article, filing, or public report confirming the financial figure.
- **Dual-Date Logic:** If a financing round occurred >18 months ago, the candidate can qualify only if recent confirmation evidence (dated within the past 18 months) verifies the continued absence of subsequent funding.
- **Future & Malformed Date Safety:** 
  - Future dates (dates beyond the current execution date) are rejected as unverified.
  - Malformed or invalid calendar dates (e.g., `2026-02-31`, `2025-13-01`) are safely handled and rejected without crashing the pipeline.

---

## 4. Strict Email Verification Policy

- **Zero Guessing & Zero Algorithmic Synthesis:** The pipeline never generates emails using heuristics (e.g., `first@domain.com`, `first.last@domain.com`).
- **Official Domain Sourcing:** Sourced exclusively from live DOM HTML and `mailto:` links on verified official company subpages (`/about`, `/team`, `/contact`, `/imprint`, `/impressum`).
- **Co-location & Founder Attribution:** The email must be co-located with the founder's full name and executive title within semantic card containers (`<article>`, `<li>`, `div.card`).
- **Role Alias Blacklist:** Shared and role inboxes are strictly forbidden. Emails beginning with generic or executive aliases—including `ceo@`, `founder@`, `founders@`, `exec@`, `leadership@`, `admin@`, `info@`, `contact@`, `hello@`, `support@`, `sales@`, `team@`, `office@`, and `mail@`—are rejected as unverified personal founder emails.
- **Benchmark Reality:** Because most B2B technology platforms deliberately do not publish personal founder email addresses on their public websites, strict verification causes candidates without published personal emails to be rejected. When public sources lack verified personal founder emails, the pipeline produces zero false positives rather than fabricating contact details, resulting in fewer than 15 final leads.

---

## 5. Non-US Presence Validation

- **HQ & Leadership:** Requires corporate headquarters and leadership to be physically located outside the United States.
- **Operational Footprint:** Structured detection identifies US offices, co-working spaces, and material domestic teams (e.g., "Maintains an office in San Francisco" triggers disqualification).
- **Enterprise Reach vs. Footprint:** Permissive of Delaware holding structures and US customer reach where operations, engineering, and executive leadership remain abroad.

---

## 6. Architecture & Execution Flow

```
[Tavily Basic Search] (Discovery across 22 non-US hubs & 12 sectors)
       │
       ▼
[Gemini 3.5 Flash] (Structured entity extraction in <untrusted_source> isolation)
       │
       ▼
[Deduplication & Candidate Budget Cap] (seen_companies filter; MAX_CANDIDATES_PER_RUN)
       │
       ▼
[Cheap Deterministic Pre-Filters] (HQ country, tech platform, closure status)
       │
       ▼
[Free Bounded Contact-Path Discovery] (Official domain DOM check; 0 Tavily burn)
       │
       ▼
[Targeted Financial Follow-Up] (Policy B & conflict search; Tavily 60-call budget)
       │
       ▼
[Deterministic Qualification] (Policies A–D, 18m recency, non-US presence)
       │
       ▼
[Strict Founder Email Verification] (DOM scraping, role alias filter, founder attribution)
       │
       ▼
[Audited Lead Output & Telemetry] (Full CandidateAuditRecord & LeadRecord export)
```

1. **Discovery Layer (Tavily Basic Search):**
   - Dispatches targeted search queries across non-US tech hubs and sectors.
   - Operates under a hard per-run budget ceiling of **60 Tavily calls**.
   - Thread-safe in-memory cache normalized by query content avoids duplicate outbound requests within a run.
   - Cache is explicitly purged at the beginning of each run (`clear_tavily_cache()`) to ensure no stale cross-run data leakage.
   - Every outbound request attempt consumes a budget slot, preventing runaway retries.
   - *Note on Discovery:* Production discovery relies on Tavily Basic Search. Google Search Grounding is not part of the active production architecture.
2. **Reasoning & Extraction Layer (Gemini 3.5 Flash):**
   - Search snippets are isolated inside `<untrusted_source>` tags to resist prompt injection.
   - Converts raw web text into `CandidateCompany` schemas.
   - Missing fields remain `None` rather than fabricated or default-populated.
3. **Candidate Budget Cap (`MAX_CANDIDATES_PER_RUN`):**
   - Configured via environment variable (default: 150 unique candidates).
   - Deduplication precedes cap enforcement. Candidates beyond the cap are logged with `rejection_stage="CANDIDATE_BUDGET"`.
   - Skipped candidates consume zero downstream Tavily credits.
4. **Verification & Audit:**
   - Evaluates financial figures, evidence dates, non-US presence, and founder email attribution.
   - Every evaluated candidate generates a structured `CandidateAuditRecord` detailing status, evidence URLs, and rejection rationale.

---

## 7. Security Safeguards

- **No Secrets in Code:** All API credentials are read from `.env` or `st.secrets`. `.env` is excluded via `.gitignore`.
- **SSRF & DNS-Rebinding Protection:** All external URLs fetched by the scraper pass through `validate_public_url()`. Loopback addresses (`127.0.0.0/8`, `::1`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), multicast, and cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`) are blocked.
- **Redirect Validation:** Scraper follows redirects with per-hop IP and hostname re-validation to prevent redirect-based SSRF.
- **Strict TLS Verification:** Requests maintain standard TLS certificate validation (`verify=True`). Unsafe fallback to unverified TLS (`verify=False`) is disallowed.
- **Source Authentication:** Financial and presence evidence requires authenticated public source URLs.

---

## 8. Local Setup & Reproducibility

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/your-username/tvb-lead-generation-agent.git
cd tvb-lead-generation-agent

# 2. Create and activate Python 3.11 virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and insert your personal API keys:
# GEMINI_API_KEY=your_gemini_api_key_here
# TAVILY_API_KEY=your_tavily_api_key_here
```

### Running Tests

```bash
# Run the complete automated test suite (447 tests):
pytest tests/ -q
```

### Launching Streamlit Web Application

```bash
# Run local Streamlit interface
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Configure operational parameters in the sidebar and trigger lead generation.

---

## 9. Test Verification Status

- **Automated Regression Suite:** **447 automated tests pass** across 23 test modules covering financial qualification, recency bounds, US-presence validation, candidate budget limits, SSRF protection, email role-alias rejection, cache lifecycle, financial provenance, founder context, and US operational presence.
- **Verification Guarantee:** All tests run deterministically offline with zero live Tavily or Gemini API consumption.
