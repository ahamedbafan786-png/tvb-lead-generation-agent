# TVB Lead-Generation Agent

An autonomous, evidence-backed lead-generation agent designed for venture builders to discover, verify, and audit early-stage non-US technology platforms ($1M–$5M ARR or funding) with verified founder executive emails.

Repository: [https://github.com/ahamedbafan786-png/tvb-lead-generation-agent](https://github.com/ahamedbafan786-png/tvb-lead-generation-agent)

---

## 1. Overview & TVB Mandate

The agent qualifies prospective venture portfolio candidates against four mandatory criteria:
1. **Financial Scale:** Total cumulative funding raised OR verified current ARR is strictly between **$1,000,000 and $5,000,000 USD** (converted at standard exchange rates).
2. **Tech Platform:** Operates a proprietary software product, SaaS, developer tool, or digital platform (consulting agencies, dev shops, IT services, and pure hardware entities are excluded).
3. **Minimal to No US Presence:** Corporate headquarters, operational footprint, and executive team reside strictly outside the United States.
4. **Verified Founder Email:** Full name and verified email of the CEO or Co-Founder published on legitimate public sources with verified contextual attribution (zero guessing, zero algorithmic synthesis).

---

## 2. Financial Qualification Hierarchy (Policies A–D) & Strict Provenance

Financial validation operates under a deterministic 4-policy hierarchy:

- **Policy A (Cumulative Priority):** Cumulative funding strictly between $1M and $5M USD qualifies with High Confidence. Any cumulative funding exceeding $5M USD is strictly rejected (e.g., standard European seed rounds of €5.0M convert to ~$5.4M USD and are correctly disqualified).
- **Policy B (Guarded Single-Round Fallback):** A single financing round between $1M and $5M qualifies with Medium Confidence **only if** an explicit follow-up search was successfully completed (`financial_followup_complete=True`) and confirmed zero subsequent growth rounds, acquisitions, or Series B/C events. If follow-up was skipped or failed, the candidate cannot pass Policy B.
- **Policy C (Verified ARR):** A current ARR between $1M and $5M qualifies independently.
  - *ARR vs. Generic Revenue Distinction:* The metric must be explicitly verified as recurring ARR. Generic revenue figures (such as annual turnover, trailing twelve months / TTM, monthly run-rate, quarterly revenue, or lifetime gross volume) do **not** qualify as ARR under Policy C.
- **Policy D (Scale Conflict / Excessive Funding):** A candidate with a qualifying ARR ($1M–$5M) that has also accumulated >$5M in cumulative equity funding is strictly rejected under the conservative scale rule to preserve focus on capital-efficient, early-stage platforms.
- **Strict Financial Provenance:** Every financial claim requires an authenticated source URL present in trusted search results, an exact non-empty source quote from trusted source material, and numeric corroboration. Source-free or hallucinated claims are rejected fail-closed.

---

## 3. Recency & Date Validation Engine

Recency evaluates two distinct date signals against an 18-month (~548 days) boundary:
- **`latest_round_date`**: The calendar date of the most recent financing round.
- **`financial_evidence_date`**: The publication date of the source article, filing, or public report confirming the financial figure.
- **Dual-Date Logic:** If a financing round occurred >18 months ago, the candidate can qualify only if recent confirmation evidence (dated within the past 18 months) verifies the continued absence of subsequent funding.
- **Future & Malformed Date Safety:** 
  - Future dates (dates beyond the current execution date) are rejected as unverified.
  - Malformed or invalid calendar dates (e.g., `2026-02-31`, `2025-13-01`) are safely handled and rejected without crashing the pipeline.

---

## 4. Strict Email Verification & Attribution Policy

- **Zero Guessing & Zero Algorithmic Synthesis:** The pipeline never generates emails using heuristics (e.g., `first@domain.com`, `first.last@domain.com`, `f.last@domain.com`).
- **Permitted Public Evidence Sources:** Sourced exclusively from legitimate public sources with verified founder attribution:
  1. `FIRST_PARTY_OFFICIAL`: Official company website founder cards and team subpages (`/team`, `/about`, `/contact`, `/imprint`, `/impressum`).
  2. `FIRST_PARTY_BLOG_PRESS`: Official company press releases and corporate announcements explicitly publishing executive contact.
  3. `FOUNDER_PUBLIC_PROFILE`: Founder's verified public professional page.
  4. `PUBLIC_CORPORATE_FILING`: Public corporate registry filings.
- **Semantic Card & Proximity Attribution:** The email must be co-located with the founder's full name and executive title within semantic card containers (`<article>`, `<li>`, `div.card`) or proximity paragraphs.
- **Role Alias Blacklist:** Shared and role inboxes are strictly forbidden. Emails beginning with generic or executive aliases—including `ceo@`, `founder@`, `founders@`, `exec@`, `leadership@`, `admin@`, `info@`, `contact@`, `hello@`, `support@`, `sales@`, `team@`, `office@`, and `mail@`—are rejected as unverified personal founder emails.
- **Obfuscation & Cloudflare Decoding:** Deterministically decodes hex-encoded Cloudflare email protection (`__cf_email__`) and handles bracketed text obfuscations (`name [at] domain [dot] com`) when co-located with founder credentials.
- **Production Reality:** Because early-stage B2B technology platforms rarely publish personal founder email addresses on public web pages, strict verification causes candidates without published personal founder emails to be rejected. When public sources lack verified personal founder emails, the pipeline produces zero false positives rather than fabricating contact details, resulting in 0 verified leads in live production benchmark runs rather than 15 fabricated leads.

---

## 5. Non-US Presence Validation

- **HQ & Leadership:** Requires corporate headquarters and leadership to be physically located outside the United States.
- **Operational Footprint:** Structured detection identifies US offices, co-working spaces, and material domestic teams (e.g., "Maintains an office in San Francisco" triggers disqualification).
- **Enterprise Reach vs. Footprint:** Permissive of Delaware holding structures and US customer reach where operations, engineering, and executive leadership remain abroad.

---

## 6. Architecture & Pipeline Execution Flow

```
[Tavily Basic Search] (Discovery across 32 non-US hubs & 12 sectors; 35 queries default)
       │
       ▼
[Gemini 3.5 Flash] (Structured entity extraction in <untrusted_source> isolation)
       │
       ▼
[Deduplication & Candidate Budget Cap] (seen_companies filter; MAX_CANDIDATES_PER_RUN = 150)
       │
       ▼
[Cheap Deterministic Pre-Filters] (Non-US HQ, tech platform, closure status, >$5M funding)
       │
       ▼
[Free Bounded Contact-Path Discovery] (Official domain DOM check; 0 Tavily burn)
       │
       ▼
[Follow-Up Gating Condition] (Skip Tavily follow-up if no contact path and no initial financial signal)
       │
       ▼
[Targeted Financial Follow-Up] (Policy B & conflict search; Tavily 60-call budget ceiling)
       │
       ▼
[Deterministic Qualification] (Policies A–D, strict provenance, 18m recency, non-US presence)
       │
       ▼
[Strict Founder Email Verification] (DOM scraping, role alias filter, founder attribution)
       │
       ▼
[Audited Lead Output & Telemetry] (Full CandidateAuditRecord & LeadRecord export in Streamlit UI)
```

1. **Discovery Layer (Tavily Basic Search):**
   - Dispatches targeted search queries across 32 non-US tech hubs and 12 technology sectors.
   - Configured with a default of **35 discovery queries** per run (`DEFAULT_MAX_SEARCH_QUERIES = 35`), adjustable in the UI up to a ceiling of 50.
   - Operates under a hard per-run budget ceiling of **60 Tavily calls** (`TAVILY_MAX_CALLS_PER_RUN = 60`).
   - Every outbound attempt atomics-reserves a budget slot before network dispatch, guaranteeing the hard limit is never exceeded.
   - Thread-safe in-memory cache normalized by query content avoids duplicate outbound requests within a run.
   - Cache is explicitly purged at the beginning of each pipeline run (`clear_tavily_cache()`) to prevent stale cross-run data leakage.
   - Cache hits consume zero Tavily outbound budget slots.
   - *Architecture Note:* Production web discovery relies exclusively on Tavily Basic Search. Google Search Grounding is completely eliminated from the discovery architecture.
2. **Reasoning & Extraction Layer (Gemini 3.5 Flash):**
   - Search snippets are isolated inside `<untrusted_source>` tags to resist prompt injection.
   - Model: `gemini-3.5-flash` with graceful fallback to `gemini-3.5-flash-lite`.
   - Converts raw web text into typed `CandidateCompany` schemas.
   - Missing fields remain `None` rather than fabricated or default-populated.
3. **Candidate Budget Cap (`MAX_CANDIDATES_PER_RUN`):**
   - Configured via environment variable (default: 150 unique candidates).
   - Deduplication precedes cap enforcement. Candidates beyond the cap are logged with `rejection_stage="CANDIDATE_BUDGET"`.
   - Skipped candidates consume zero downstream Tavily credits.
4. **Verification & Audit:**
   - Evaluates financial figures, evidence dates, non-US presence, and founder email attribution.
   - Every evaluated candidate generates a structured `CandidateAuditRecord` detailing qualification status, evidence URLs, source quotes, and rejection rationale.

---

## 7. Security Safeguards & Threat Defenses

- **No Secrets in Code:** All API credentials are read from `.env` or `st.secrets`. `.env` is strictly excluded via `.gitignore`.
- **SSRF & DNS-Rebinding Protection:** All external URLs fetched by the scraper pass through `validate_public_url()`. Loopback addresses (`127.0.0.0/8`, `::1`), private networks (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local (`169.254.0.0/16`), multicast, and cloud metadata endpoints (`169.254.169.254`, `metadata.google.internal`) are blocked.
- **Redirect Validation:** Scraper follows redirects with per-hop IP and hostname re-validation to prevent redirect-based SSRF.
- **Strict TLS Verification:** Requests maintain standard TLS certificate validation (`verify=True`). Unsafe fallback to unverified TLS (`verify=False`) is disallowed.
- **Prompt Injection Defense:** Third-party web data is enclosed in `<untrusted_source>` tags with strict system instructions forbidding the execution of embedded instructions.
- **Source Authentication:** Financial and presence evidence requires authenticated public source URLs verified against trusted search results.

---

## 8. Local Setup & Reproducibility

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/ahamedbafan786-png/tvb-lead-generation-agent.git
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

### Running the Automated Test Suite

```bash
# Run the complete automated test suite (481 tests):
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

- **Automated Regression Suite:** **481 automated tests pass** (100% pass rate) across 26 test modules:
  - `test_candidate_budget_cap.py`: Candidate ceiling and budget tracking
  - `test_tavily_budget_control.py`: Per-run 60-call hard budget enforcement
  - `test_tavily_cache_lifecycle.py`: Run-boundary cache purges and hit deduplication
  - `test_financial_provenance_gate.py`: Strict URL, quote, and amount provenance
  - `test_recency_hardening.py`: Dual-date 18-month boundaries and invalid date safety
  - `test_us_presence_operational_hardening.py`: US office, subsidiary, and team detection
  - `test_us_presence_hardening.py`: Non-US HQ and Delaware holding validation
  - `test_ssrf_and_tls_hardening.py`: SSRF, loopback, private IP, metadata, and TLS checks
  - `test_discovery_precision_and_recall.py`: Negative query filtering and extraction prompt
  - `test_discovery_query_quality.py`: 35-query uniqueness, financial density, and diversity
  - `test_public_founder_email_expansion.py`: 13 offline founder email attribution scenarios
  - `test_streamlit_lead_contract.py`: UI table contract and AppTest headless execution
  - And 14 additional modular test suites covering extraction fidelity and pipeline recovery.
- **Verification Guarantee:** All 481 tests execute deterministically offline with zero live Tavily or Gemini API consumption.

---

## 10. Live Production Benchmark & Audit Reality

In live production testing across 35 discovery queries and 1 follow-up call (36 total Tavily calls):
- **Candidates Discovered:** 138 candidates evaluated across European, UK, Canadian, and global tech hubs.
- **Funnel Progression:**
  - ~26% had public funding figures; all exceeded the $5,000,000 USD ceiling (predominantly standard European €5M seed rounds that convert to $5.4M USD, or Series A/B rounds).
  - ~73% lacked direct company homepage URLs in third-party news snippets and were disqualified before follow-up.
  - 0 candidates published verified personal founder emails on open web subpages.
- **Final Qualified Leads:** **0 leads**.
- **The Zero-Hallucination Mandate:** The TVB agent is engineered to prioritize absolute data integrity over artificial lead counts. In commercial venture building, delivering zero leads with a complete, transparent disqualification audit log is vastly superior to fabricating email addresses, inventing ARR metrics, or admitting out-of-scope enterprises.
