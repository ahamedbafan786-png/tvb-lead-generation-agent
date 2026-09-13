# TVB Lead-Generation Agent: Final Review Notes

## 1. Project Purpose
An autonomous data pipeline built for venture builders to discover, verify, and audit early-stage non-US technology platforms ($1M–$5M ARR or funding) with verified founder contact emails.

---

## 2. TVB Qualification Mandate
1. **Financial:** Cumulative funding OR verified current ARR strictly in [$1,000,000, $5,000,000] USD.
2. **Platform:** Proprietary software/SaaS/platform (no IT consulting, agencies, or hardware).
3. **Footprint:** Minimal to no US presence (HQ, operations, team strictly outside US).
4. **Founder Email:** Full name and verified personal email of CEO/Co-Founder published on official domain (no guessed or synthesized emails; role aliases rejected).
5. **Zero Fabrication:** Leave unverified fields blank; never guess contact info or entity metadata.

---

## 3. Architecture Overview
- **Discovery:** Tavily Basic Search across 22 non-US tech hubs and 12 B2B software sectors.
- **Budget Control:** Hard per-run cap of 60 Tavily calls, thread-safe cache isolation, and global `MAX_CANDIDATES_PER_RUN` cap (150).
- **Prompt Isolation:** Untrusted external web content isolated inside `<untrusted_source>` tags to prevent injection.
- **Deduplication:** Company name normalization strictly precedes candidate cap evaluation.
- **Deterministic Qualification:** Four-tier financial hierarchy (Policies A–D), 18-month recency boundary with calendar date safety, and non-US footprint verification.
- **Bounded Contact-Path Discovery:** Free DOM inspection before any paid financial follow-up.
- **Strict Email Verification:** DOM card co-location with founder attribution and blacklist of role/generic aliases.
- **Streamlit Interface:** Full operational controls, real-time funnel telemetry, CSV export, and structured Audit Inspector.

---

## 4. Key Design Decisions & Tradeoffs
- **Precision Over Recall:** Strict email verification rejects candidates without published personal founder emails on official domains. While this may result in fewer than 15 final leads in environments with sparse public email listings, zero false positives and zero fabricated leads are guaranteed.
- **Conservative Scale Rule (Policy D):** Entities with qualifying ARR but excessive historical funding (>$5M) are rejected.
- **Policy B Completeness:** Single-round funding fallback requires a successfully completed follow-up verification (`financial_followup_complete=True`) confirming absence of subsequent growth rounds.
- **True Date-Math:** Recency uses exact day-level delta calculations (548-day boundary), safely rejecting future and malformed calendar dates.
- **Network Security:** Public URL validation blocks SSRF, private IPs, loopback, link-local, and cloud metadata targets, with redirect re-validation and strict TLS enforcement.

---

## 5. Areas for Independent Scrutiny
1. `core/pipeline.py`: Candidate budget cap (`MAX_CANDIDATES_PER_RUN`) correctly audits skipped candidates and short-circuits discovery queries.
2. `core/discovery.py`: In-memory cache clearing (`clear_tavily_cache`) at run start and per-run 60-call budget enforcement.
3. `core/scraper.py`: SSRF defenses (`validate_public_url`), redirect validation, and strict TLS certificate enforcement.
4. `core/extractor.py`: Policies A–D evaluation, ARR versus generic revenue distinction, and source URL authentication.
5. `core/email_verifier.py`: Executive and generic role alias exclusion (`ceo@`, `founder@`, etc.) and founder co-location attribution.
6. `core/recency.py`: Dual-date 18-month calculation, future-date rejection, and malformed calendar date safety.

---

## 6. Execution Instructions
```bash
# Run test suite:
pytest tests/ -q

# Run Streamlit interface:
streamlit run app.py
```
