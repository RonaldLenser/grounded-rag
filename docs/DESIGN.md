# Grounded: Design Document

Status: Approved for build | Owner: Ron Lenser | Last updated: 2026-10-12

---

## 0. Technology Decisions (final)

| Layer | Choice | Why this one |
|---|---|---|
| LLM (generation) | **Anthropic Claude Sonnet 5.5** (`claude-sonnet-5-5`, set via `LLM_MODEL`) | Strong instruction-following for "answer only from sources." Native structured outputs (JSON schema) give reliable structured responses. You already know the API. |
| Embeddings | **OpenAI `text-embedding-3-small`** (1536 dims) | Cheapest credible embedding model (fractions of a cent per document). Anthropic has no embedding model. Most common in job posts. |
| Provider design | Both behind small Python `Protocol` interfaces | Swappable providers, fake providers in tests, zero real API calls in CI. This is the interview talking point, not the vendor. |
| Vector DB | **PostgreSQL + pgvector** | One database for metadata AND vectors. Deleting a document deletes its chunks in one transaction. SQL is a skill every backend team values. Dedicated vector DBs (Pinecone, Qdrant) add a second system for no benefit at this scale. |
| Keyword search | **Postgres full-text search** (`tsvector`) | Enables hybrid retrieval with no new infrastructure. |
| Backend | **FastAPI + Pydantic v2** | Industry standard for Python AI APIs. Typed request/response models, auto OpenAPI docs. |
| ORM / migrations | **SQLAlchemy 2.0 (async) + Alembic** | What real teams use. Migrations show schema discipline. |
| Document parsing | **pypdf** (PDF), stdlib (TXT, MD) | BSD license (PyMuPDF is AGPL, wrong for an MIT repo). Gives per-page text, which citations need. |
| Token counting | **tiktoken** | Chunk by tokens, not characters. |
| RAG framework | **None. No LangChain, no LlamaIndex.** | Every RAG step is your own readable code. You can explain every line. Frameworks hide exactly what interviewers ask about. |
| Frontend | **Single static page (HTML + vanilla JS)** served by FastAPI | One deploy, no build step. Recruiters get a clickable demo. Backend is the product here. TypeScript gets shown in a later project. |
| Logging | **structlog** (JSON logs, request IDs) | Machine-readable logs are what production teams expect. |
| Testing | **pytest + pytest-asyncio + httpx2** | Standard. Fakes for LLM/embedder, real Postgres in CI. |
| Lint / types | **ruff + mypy** + pre-commit | Clean code signal on every commit. |
| Package mgmt | **uv** + `pyproject.toml` + `uv.lock` | Modern standard. Faster and reproducible vs plain `requirements.txt`. |
| Container | **Multi-stage Dockerfile** (non-root) + **docker-compose** (app + pgvector) | One command local setup: `docker compose up`. |
| Deployment | **Google Cloud Run** (app) + **Neon** (managed Postgres w/ pgvector) | Cloud Run: Docker-native, scales to zero, free tier, matches your Google Cloud GenAI Leader cert. Neon: free tier, pgvector built in, wakes automatically (Supabase free pauses projects). |
| CI/CD | **GitHub Actions** | Lint, type check, tests on every push. Deploy on main. |

**Cost estimate:** Under $5 total to build and eval if you use small eval sets. Hosting $0 on free tiers.

---

## 1. Project Name
**Grounded** (repo: `grounded-rag`)

## 2. GitHub Description
> Production-minded RAG knowledge assistant: upload documents, ask questions, get answers with verified citations or an explicit "not in your documents." FastAPI, pgvector, Claude, evaluated.

## 3. Problem Statement
Teams store knowledge in PDFs and docs nobody reads. Generic chatbots answer confidently even when the answer is not in those documents, and users cannot verify where an answer came from. Grounded answers only from uploaded documents, cites the exact passage and page for every claim, and refuses when the documents do not support an answer.

## 4. Target User
Small teams and professionals who need trustworthy answers from their own documents: policy manuals, compliance frameworks, technical docs, SOPs. Demo corpus: NIST AI Risk Management Framework documents (US government, public domain, and relevant to AI hiring managers).

## 5. MVP Features
1. Upload PDF, TXT, MD (size-limited)
2. Parse, chunk, embed, store with page metadata
3. Deduplicate uploads by SHA-256 hash
4. List and delete documents (delete cascades to chunks)
5. Ask a question, get a structured answer with status: `answered`, `partial`, or `insufficient_context`
6. Every claim cites a source with filename, page, and verified quote
7. Refusal without calling the LLM when retrieval confidence is too low
8. Retrieval-only `/search` endpoint for debugging and evals
9. Query logging (scores, latency, tokens) to database
10. Evaluation harness with committed results
11. Minimal web UI
12. Dockerized, deployed, CI green

## 6. Intentionally Postponed
| Feature | Why postponed |
|---|---|
| Multi-user auth / accounts | Huge scope, not the AI skill being shown. Admin API key covers writes. |
| Background job queue (Celery, Pub/Sub) | Sync ingestion with size limits is fine for MVP. Great interview answer for "how would you scale?" |
| Reranker model (Cohere, cross-encoder) | Add only if eval shows retrieval is the bottleneck. |
| DOCX, HTML, OCR for scanned PDFs | Parser complexity, low hiring value. |
| Conversation memory / chat history | Single-turn Q&A first. Multi-turn belongs in the study app project. |
| Streaming | Milestone 9, after core works. |
| React/Next.js frontend | Second deploy target, no AI value. |
| Agents / tool calling | That is Project 1's job. Keep this project focused. |

## 7. Architecture Overview

```
                ┌──────────────── FastAPI app (Cloud Run) ────────────────┐
 Browser/curl → │ api/ routes → services                                   │
                │   IngestionService: parse → chunk → embed → store         │
                │   Retriever: vector search (+ keyword, RRF fusion)        │
                │   Answerer: context build → LLM (structured) → validate  │
                │ providers/: Embedder (OpenAI)  LLM (Claude)  [Protocols] │
                └───────────────┬───────────────────────────┬──────────────┘
                                │                           │
                     PostgreSQL + pgvector (Neon)     External APIs
                     documents, chunks, query_logs    OpenAI, Anthropic
```

Layering rule: **routes** handle HTTP only. **services** hold logic. **providers** wrap external APIs. **db** owns persistence. Routes never call OpenAI or SQL directly.

## 8. Data Flow

**Ingestion** (atomic: nothing is written until every step succeeds)
1. `POST /documents` receives file (multipart). Check admin key, collection exists, size limit (read at most limit+1 bytes).
2. Detect type by extension, then verify real contents (PDF magic bytes, UTF-8, not binary).
3. Hash bytes (SHA-256). If it already exists in the collection, return 409 with the existing document ID.
4. Parse into `(page_number, text)` and normalize whitespace (worker thread: CPU-bound).
   - PDFs only (M10): repair words hyphenated across line breaks and stray spaces before punctuation. Citation quotes showed "illus- trated" and "GOVERN , MAP", and split words were invisible to keyword search. A trailing hyphen can be soft ("illus-trated" -> "illustrated") or real ("red-teaming"), so the rule uses the whole document as evidence: keep the hyphen if the compound appears inline elsewhere, join if the joined word appears elsewhere, otherwise follow the document's majority (typesetters are consistent within one document; ties keep the hyphen). Stopword, number, and capitalized fragments are never guessed without evidence. On the demo PDFs, leftover line-break hyphens went from 20 to 5 (RMF) and 13 to 9 (GenAI profile); the rest are URLs, page footers, and splits with no evidence. Known miss: "cost-effective" gets joined in the RMF (soft-hyphen majority, no inline evidence). Existing documents keep their old text until re-ingested (`seed_demo.py --replace`).
5. Chunk (Section 14), keeping page range + heading per chunk (worker thread).
6. Embed all chunks in batches of 100 (SDK handles retries/backoff for 429/5xx).
7. One transaction: insert document (`status=ready`) + all chunks with vectors. Any earlier failure leaves nothing behind.
8. Return document metadata + chunk count.

Tradeoff: the request waits for the full pipeline. At scale this moves to a background queue; `documents.status` (`processing`/`ready`/`failed`) already exists for that.

**Query**
1. `POST /query` with question. Validate length.
2. Embed question.
3. Retrieve top candidates (vector, later hybrid).
4. If best score < `MIN_RELEVANCE_SCORE`: return `insufficient_context`. **No LLM call.** (Saves money, blocks hallucination.)
5. Build context: label chunks `S1..Sn`, wrap in `<source>` tags, cap by token budget.
6. Reserve one slot of the daily LLM budget (atomic, committed before the call).
7. Call Claude with structured outputs (JSON schema): status + a list of claims, each with evidence (source ID + verbatim quote).
8. Verify every claim's evidence in code. Build the visible answer ONLY from supported claims, with code-generated `[S#]` markers. Drop unsupported claims entirely; downgrade status.
9. Map `S#` to filename, page, document ID.
10. Log query (status, chunk IDs, scores, tokens, latency).
11. Return structured response.

## 9. Exact Stack
Python 3.12, FastAPI, Uvicorn, Pydantic v2, pydantic-settings, SQLAlchemy 2.0 async, asyncpg, Alembic, pgvector (Python + Postgres ext), pypdf, tiktoken, anthropic SDK, openai SDK (built-in retries), python-multipart, structlog, slowapi (rate limiting), pytest, pytest-asyncio, httpx2, ruff, mypy, pre-commit, uv, Docker, docker-compose, GitHub Actions, Google Cloud Run, Artifact Registry, Neon Postgres.

## 10. Repository Structure

```
grounded-rag/
├── src/grounded/
│   ├── main.py                 # app factory, middleware, router registration
│   ├── config.py               # pydantic-settings, all env vars
│   ├── logging_config.py       # structlog setup (not logging.py: avoids shadowing stdlib)
│   ├── api/
│   │   ├── deps.py             # DI: db session, services, auth
│   │   ├── middleware.py       # request ID + access log (pure ASGI)
│   │   ├── errors.py           # exception → HTTP mapping
│   │   └── routes/ health.py collections.py documents.py query.py
│   ├── errors.py               # domain errors (no HTTP knowledge)
│   ├── schemas/                # Pydantic request/response models
│   │   └── collections.py query.py
│   ├── db/
│   │   ├── session.py          # async engine/session
│   │   ├── models.py           # SQLAlchemy tables
│   │   └── repositories.py     # all SQL lives here
│   ├── providers/
│   │   ├── base.py             # Embedder, LLMClient Protocols
│   │   ├── openai_embedder.py
│   │   └── anthropic_llm.py
│   ├── ingestion/ parsers.py chunker.py service.py
│   ├── retrieval/ retriever.py fusion.py
│   ├── generation/ prompts.py answerer.py citations.py
│   └── web/ index.html app.js styles.css
├── migrations/                 # Alembic
├── evals/
│   ├── corpus/README.md        # how to download demo docs (not committed if large)
│   ├── golden.jsonl            # questions + expected sources
│   ├── metrics.py
│   └── run.py                  # python -m evals.run → docs/evals/*.md
├── tests/
│   ├── conftest.py  fakes.py   # FakeEmbedder, FakeLLM
│   ├── unit/                   # chunker, citations, fusion, prompts
│   └── integration/            # API + real Postgres (own conftest: DB check)
├── docs/
│   ├── DESIGN.md (this file)  architecture.md  api.md
│   ├── decisions/              # ADRs: 001-pgvector.md, 002-no-framework.md ...
│   ├── evals/                  # committed eval reports
│   └── images/                 # diagram, screenshots, demo GIF
├── scripts/seed_demo.py
├── .github/workflows/ ci.yml deploy.yml
├── Dockerfile  docker-compose.yml  .dockerignore
├── pyproject.toml  uv.lock  alembic.ini
├── .env.example  .gitignore  .pre-commit-config.yaml
├── LICENSE (MIT)  README.md
```

## 11. API Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | Liveness |
| GET | `/ready` | none | DB reachable |
| POST | `/v1/collections` | admin | Create collection |
| GET | `/v1/collections` | none | List collections |
| POST | `/v1/collections/{id}/documents` | admin | Upload + ingest |
| GET | `/v1/collections/{id}/documents` | none | List docs |
| GET | `/v1/documents/{id}` | none | Doc detail + status |
| DELETE | `/v1/collections/{id}` | admin | Delete collection + its docs, chunks, query logs |
| DELETE | `/v1/documents/{id}` | admin | Delete doc + chunks |
| POST | `/v1/collections/{id}/search` | none | Retrieval only (debug/eval) |
| POST | `/v1/collections/{id}/query` | none, rate limited | Cited answer |
| POST | `/v1/collections/{id}/query/stream` | none, rate limited | SSE (Milestone 9) |

**Query response shape**
```json
{
  "status": "answered | partial | insufficient_context",
  "answer": "Leaders are accountable [S2].",
  "notice": "string or null (e.g. some statements could not be verified)",
  "claims": [{"text": "Leaders are accountable.", "source_ids": ["S2"]}],
  "citations": [
    {"source_id": "S2", "document_id": "uuid", "filename": "nist-ai-rmf.pdf",
     "page_start": 12, "page_end": 12, "quote": "exact supporting text"}
  ],
  "retrieval": {"chunks_retrieved": 6, "top_score": 0.71, "strategy": "hybrid"},
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "latency_ms": 0,
  "query_id": "uuid"
}
```
Errors use one consistent shape: `{"error": {"code": "...", "message": "...", "request_id": "..."}}`.

## 12. Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

collections (
  id uuid PK, name text UNIQUE NOT NULL, created_at timestamptz
)

documents (
  id uuid PK,
  collection_id uuid FK → collections ON DELETE CASCADE,
  filename text, content_type text, size_bytes int,
  sha256 char(64), page_count int,
  status text CHECK (status IN ('processing','ready','failed')),
  error text NULL, created_at timestamptz,
  UNIQUE (collection_id, sha256)
)

chunks (
  id uuid PK,
  document_id uuid FK → documents ON DELETE CASCADE,
  chunk_index int, content text, token_count int,
  page_start int, page_end int, heading text NULL,
  embedding vector(1536) NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  UNIQUE (document_id, chunk_index)
)
INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
INDEX chunks_tsv_gin ON chunks USING gin (tsv)

query_logs (
  id uuid PK, collection_id uuid, question text, status text,
  retrieved_chunk_ids uuid[], scores float[], model text,
  input_tokens int, output_tokens int, latency_ms int, created_at timestamptz
)

llm_daily_usage (            -- global LLM budget, one row per UTC day (migration 0002)
  day date PK, calls int NOT NULL CHECK (calls >= 0), updated_at timestamptz
)
```
Raw files are **not stored**. Extracted text lives in chunks. No object storage needed. (Tradeoff: cannot re-chunk without re-upload. Document it.)

## 13. RAG Pipeline Design
`parse → normalize → chunk → embed → store` and `embed query → retrieve → threshold gate → build context → generate (structured) → validate citations → respond + log`.
Each step is a pure-ish function or small class with its own unit tests. Providers injected via FastAPI dependencies so tests swap fakes.

## 14. Chunking Strategy
- **Size:** ~500 tokens, **overlap** ~75 tokens (env-configurable, tuned with evals).
- **Recursive split** on boundaries in priority order: markdown headings → blank lines (paragraphs) → sentences → hard token cut (last resort).
- **Metadata per chunk:** `page_start`, `page_end`, nearest preceding heading, `chunk_index`.
- **Why tokens not characters:** embedding and LLM limits are in tokens.
- **Why overlap:** a fact split across a boundary still appears whole in one chunk.
- **Why not semantic chunking:** extra embedding cost and complexity for unproven gain. Mention as future work backed by eval.

## 15. Retrieval Strategy
- **MVP (M3):** pgvector cosine similarity, top_k = 6 (max 20), `hnsw.ef_search = 100`.
- **Filtered HNSW:** the index finds neighbors across all collections, then the collection filter applies. `hnsw.iterative_scan = strict_order` (pgvector 0.8) keeps scanning so small collections still get `top_k` results. Both settings are transaction-local (`set_config(..., true)`).
- Small collections: the planner may skip HNSW and do an exact scan + sort (cheaper for few rows). A test proves the query shape stays index-compatible for large collections.
- **M8: hybrid retrieval** (`RETRIEVAL_STRATEGY=hybrid`): vector top 20 + keyword top 20, merged with **Reciprocal Rank Fusion** (`score = sum of 1/(60 + rank)`), keep top 6.
  - Why hybrid: the M7 eval showed both false refusals were retrieval misses on exact terms ("residual risk", "eliminate negative risk entirely") that embeddings ranked below 6.
  - **Keyword search is IDF-weighted term matching, not Postgres's default ranking.** A chunk scores the sum of `ln(N / df)` over the question's words it contains, so rare words count and words on every page ("AI", "risk") count ~0. It is BM25's core idea without the term-frequency and length terms (chunks are uniform ~500 tokens). Measured on the golden set before building (top-6 hit rate, keyword search alone): all-terms `websearch_to_tsquery` 11/24, any-term `ts_rank_cd` 19/24, IDF-weighted 22/24, finding both vector misses at rank 1-2. Same `english` stemming as the stored `tsv` column; an OR-tsquery prefilter keeps the GIN index usable.
  - Why RRF: cosine (0..1) and keyword weights (0..~20) are on different scales; RRF uses only ranks. k=60 rewards agreement between the two lists over a single strong hit.
  - Keyword-only results still carry their cosine `score`, and **the gate uses the best cosine among the results**, not the first result, so the calibrated 0.20 threshold keeps its meaning.
  - Shipped behind a flag, then made the default by the eval ([v1](evals/2026-10-12-v1.md) vector vs [v2](evals/2026-10-12-v2-hybrid.md) hybrid, live, same 33 questions): right page in top 3 / top 6 went 88% / 92% -> 100% / 100%, MRR 0.81 -> 0.88, answerable questions answered 22/24 -> 24/24, judged correct 24/24, 0 unsupported claims in both. Cost: p50 latency 2.9s -> 3.4s (mostly LLM variance; the keyword query takes ~5 ms) and ~60 more input tokens per answer.
  - Honest caveat: the keyword method was chosen after looking at this golden set's two failures. The method is generic (IDF weighting, not tuned per question), but the numbers are not a held-out estimate. Next step for an unbiased number: 10-15 new questions written without looking at results.
- Collapse adjacent chunks from the same doc to save context tokens.
- **Threshold gate** uses the vector similarity score. Value is calibrated from eval data, not guessed. (Cosine scores are model-specific.)

## 16. Citation Strategy
1. Context sources labeled `S1..Sn` (relevance order, capped by `MAX_CONTEXT_TOKENS`), each in `<source id="S1" file="..." pages="3-4">` tags. Source text, filenames and the question are HTML-escaped so document text can never close or forge tags.
2. Claude replies via **structured outputs** (`output_config.format` = JSON Schema): `{status, claims: [{text, evidence: [{source_id, quote}]}]}`. Each claim is one factual statement with its own evidence. (Originally forced tool use; Sonnet 5.5 rejects forced `tool_choice` with a 400, so we moved to the provider's native structured-output feature.) The output is validated with Pydantic before use (malformed = 502). `source_id` and `quote` are deliberately unconstrained in the schema, so one bad item fails its claim, not the whole answer.
3. **Server-side verification** (`generation/citations.py`), per evidence item:
   - `source_id` must be one we sent, else `unknown_source`.
   - `quote` must be at least 12 characters, else `quote_too_short`.
   - `quote` must appear in that source (case/whitespace/curly-quote/HTML-entity insensitive), else `quote_not_found`.
4. A claim is **supported** only if it has evidence and ALL of its evidence verifies. One fabricated quote rejects the whole claim.
5. The answer is **built** from supported claims, with `[S#]` markers generated from verified evidence (markers the model wrote are discarded). Rejected claims are dropped, never edited, and their text never leaves the verifier (results carry rejection reasons only).
6. Response maps verified evidence to filename, pages, heading, document and chunk IDs.

**Why claims instead of an answer string (changed in M6.1):** the first version verified citations, then stripped markers whose citation failed. "Budgets are unlimited [S9]." became "Budgets are unlimited." and stayed visible with no support. Sentence-splitting the answer to find the unsupported sentence is brittle (abbreviations, lists, multi-sentence claims). Making the claim the unit the model returns removes the problem: there is no free-form text left to leak.

Interview line: "The model never writes the answer. It proposes claims with evidence; code verifies each one and assembles the answer from what passed."

## 17. Hallucination Handling (layered)
1. **Relevance gate:** best similarity below `MIN_RELEVANCE_SCORE` (0.20, provisional until M7) → `insufficient_context` with **no LLM call**.
2. **Prompt contract:** sources only; one claim per statement, each with exact quotes; sources are untrusted data, not instructions.
3. **Structured output:** API-enforced JSON schema of claims with evidence; the model must declare a status.
4. **Claim verification** (Section 16): only supported claims are shown.
5. **Status rules:** model refusal → fixed refusal message (model-written refusal text is not shown); no supported claims → `insufficient_context`; `answered` with any rejected claim → `partial` with a notice that some statements could not be verified; model `partial` → notice that the documents only partly answer. `downgraded_from` is returned.
6. **Logged + measured:** every query writes a `query_logs` row (status, chunk IDs, scores, model, tokens, latency). Question/answer text never goes to application logs.

Honest limit: quote matching proves the quote exists, not that the claim follows from it (entailment). A model could pair a true quote with an overreaching claim. The groundedness eval (M7) measures how often that happens. State this in README Limitations.

## 18. Evaluation Strategy (built in M7)
- **Corpus:** the two public-domain NIST documents the demo serves (AI 100-1 AI RMF, 48 pages; AI 600-1 Generative AI Profile, 64 pages).
- **Golden set:** `evals/golden.jsonl`, 33 hand-written questions:
  - 21 answerable, each with expected document + PDF page(s) and a reference answer
  - 3 multi-source (need both documents)
  - 6 unanswerable but on-topic (EU AI Act fines, GPU recommendations, certification cost...). Absence verified by full-text search of both documents. 3 of them accept `partial` (e.g. "three workshops" is in the text, attendance numbers are not).
  - 3 out-of-domain (cookies, World Cup, routers)
  - Validated by tests: schema, consistency rules, filenames match `seed_demo.py`.
- **Black-box runner:** `python -m evals.run <url>` sends each question to `/search` (top 6) and `/query` over HTTP, paced under the rate limits, retrying 429s via `Retry-After`. It measures the deployed system, not a re-implementation. One failing question is recorded, not fatal; errors count as failures in every metric.
- **Retrieval metrics:** Hit@1/3/6 and MRR against expected pages; multi-source coverage (both documents retrieved).
- **Answer metrics:** status accuracy, false refusals (and how many were retrieval misses), unanswerable handled without an unsupported answer, out-of-domain refused by the gate with no LLM call, citation page hit, claims proposed/shown/dropped by the verifier, latency p50/p95, tokens.
- **LLM judge (`--judge`, Haiku 5.5):** (1) claim support: is each SHOWN claim actually supported by its verified quotes (the entailment gap the verifier can't close); (2) answer correctness vs the reference answer. Inputs are HTML-escaped; every verdict has a one-sentence reason in the report for human spot-checks.
- **Gate calibration:** top-1 similarity per category plus a threshold sweep. Rule: middle of the gap between the highest out-of-domain and lowest answerable score, rounded down to 0.05.
- **Output:** `docs/evals/YYYY-MM-DD-<label>.md` (human report) + `.json` (raw results, re-scorable with `--from-json` without API calls). Committed.
- Real numbers only. Never fabricate results.

**What the first run found (2026-10-12 baseline):**
1. A real bug: claim-format answers are wordier than prose; one list answer hit the 1024-token output cap and returned 502 (another used 943 tokens). Fix: `LLM_MAX_TOKENS` 2048 + "at most 8 claims, group long lists" in the prompt.
2. Both false refusals were retrieval misses (expected page not in the top 6), not model failures: the model correctly refused from what it was given. This is the measured case for hybrid retrieval (M8).
3. The 0.20 gate threshold is confirmed by data: out-of-domain max 0.14, answerable min 0.31. On-topic unanswerable questions score 0.40-0.63, so no threshold can catch them; the model's refusal handled all 6.

**After the fix (2026-10-12 v1, judged, [report](evals/2026-10-12-v1.md)):** 0 errors; 22/24 answerable answered, 22/22 judged correct vs reference; 6/6 unanswerable and 3/3 off-topic refused; 78 claims shown, 0 dropped by the verifier, 72 fully supported by their quotes, 6 partially, 0 unsupported; Hit@6 92%, MRR 0.81; p50 2.9s, p95 6.5s.
- Manual spot-check of the judge's "partial" verdicts: the extra detail was in the source document but outside the quoted span (quote chosen too narrow), or attribution wording. This is the entailment gap the verifier can't close, measured: it exists, it is small, and in this sample it did not introduce false facts.
- The eval harness itself had two bugs found in use: a 300-token judge cap (judge reasons got cut off) and a judge failure aborting the whole run before results were saved. Fixed: 1024 cap, answers saved before judging, per-question judge errors.

## 19. Testing Strategy
| Level | What | Notes |
|---|---|---|
| Unit | chunker, citation verifier, RRF fusion, prompt builder, parsers, config | Pure functions, fast, majority of tests |
| Integration | API routes + real Postgres/pgvector | Fake embedder (deterministic hash vectors), fake LLM (scripted responses) |
| Contract | Provider adapters | Mock SDK responses, verify error/retry handling |
| Smoke (manual/optional) | Real API keys, marked `@pytest.mark.live`, skipped in CI | Run before releases |
| Evals | Quality, not correctness | Separate from pytest |

CI: GitHub Actions with a `pgvector/pgvector:pg16` service container. Gates: ruff, mypy, pytest. Coverage target: ~80% on `src/grounded` services (meaningful, not chased).

## 20. Security Considerations
- Secrets only via env vars / Cloud Run Secret Manager. `.env` gitignored. `.env.example` has placeholders only.
- Admin API key (header `X-API-Key`, constant-time compare) on all write endpoints.
- Upload validation: extension + magic bytes, max size (10 MB), max pages, max docs per collection.
- **Prompt injection:** documents are untrusted. Delimited in tags, system prompt says ignore instructions inside sources, model can only return the answer schema (no tools with side effects).
- Rate limiting on query (protects your API bill on a public demo).
- Question length cap. Context token budget cap.
- Parameterized SQL only (SQLAlchemy). No raw string SQL.
- CORS locked to the app's own origin.
- Web UI: all API data rendered via `textContent` (never HTML-parsed), enforced by a test; strict Content-Security-Policy (`script-src 'self'`, no inline code, `frame-ancestors 'none'`), `nosniff`, `no-referrer`. Verified with a browser XSS test (malicious filename + document text).
- Container runs as non-root.
- Error responses never leak stack traces; full details in logs with request ID.
- **Data retention (current behavior, stated plainly):** `query_logs.question` stores the raw question text, which can contain personal or confidential information. There is no automatic expiry: rows are deleted only when their collection is deleted (ON DELETE CASCADE). Answers are not stored. Application logs never contain question, answer or claim text (counts, scores and reasons only), because logs are shipped to Cloud Logging and read more widely than the database. The web UI tells users that questions are stored. Before handling real user data this needs a retention period (scheduled delete, e.g. 30 days) or hashing/redaction, plus a privacy notice.

## 21. Environment Variables
```
APP_ENV=local                     # local | test | prod
LOG_LEVEL=INFO
DATABASE_URL=postgresql+asyncpg://grounded:grounded@localhost:5433/grounded
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
LLM_MODEL=claude-sonnet-5-5
JUDGE_MODEL=claude-haiku-5-5
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
CHUNK_SIZE_TOKENS=500
CHUNK_OVERLAP_TOKENS=75
RETRIEVAL_TOP_K=6
RETRIEVAL_STRATEGY=vector         # vector | hybrid
MIN_RELEVANCE_SCORE=0.20          # calibrate via evals (M7)
MAX_CONTEXT_TOKENS=6000
MAX_UPLOAD_MB=10
ADMIN_API_KEY=
RATE_LIMIT_QUERY=10/minute
CORS_ORIGINS=http://localhost:8000
```

## 22. Deployment
Full runbook: [DEPLOY.md](DEPLOY.md). Summary:
1. Neon project (Postgres 17, AWS us-east-1), **direct** (non-pooled) connection string. The app normalizes `postgresql://...?sslmode=require` to asyncpg format and rejects `-pooler` hosts (PgBouncer transaction mode breaks asyncpg prepared statements).
2. `alembic upgrade head` against Neon from a dev machine.
3. Secrets in Google Secret Manager, typed at a hidden prompt; app strips whitespace so piped newlines can't corrupt keys.
4. `gcloud run deploy --source .` (Cloud Build builds the plain-syntax Dockerfile): us-east4, min 0 / max 2 instances, 1 GiB, concurrency 20, `APP_ENV=prod`, `TRUSTED_PROXY_HOPS=1`.
5. `scripts/seed_demo.py` loads the NIST corpus; `scripts/smoke_test.py --check-rate-limit` verifies end to end, including X-Forwarded-For spoofing resistance.

Abuse/cost controls: per-IP sliding-window limits (10 queries/min, 30 searches/min, in-memory per instance), a global cap of 300 LLM calls per UTC day (each call reserves a slot atomically in Postgres before it runs), admin-key-only writes. uvicorn runs with `--no-proxy-headers` so exactly one component (the app) derives the client IP.

Tradeoffs: scale-to-zero cold starts (a few seconds); rate limits are per instance (max 2x with 2 instances), while the daily budget is global and atomic. It is a fixed UTC-day window, so up to 2x the limit can be used across midnight (300 at 23:59 + 300 at 00:00); failed calls count against it; one "call" can be up to 1 + `LLM_MAX_RETRIES` HTTP attempts. CI/CD via GitHub Actions + Workload Identity Federation is a later improvement; deploys are a single scripted command for now.

## 23. README Outline
1. Title + one-line pitch + badges (CI, license, Python)
2. **Live demo link + demo GIF** (top, recruiters see it in 5 seconds)
3. What it does (3 bullets)
4. Why I built it / the problem
5. Architecture diagram
6. How a question becomes a cited answer (numbered flow)
7. Key engineering decisions (link ADRs)
8. Tech stack table with one-line reasons
9. Evaluation results table + link to reports
10. Hallucination handling (layers)
11. Quickstart: `docker compose up` in 3 commands
12. Environment variables
13. API usage examples (curl)
14. Testing
15. Project structure
16. Limitations (honest)
17. Future improvements
18. License

## 24. Development Milestones (build in this order)

| # | Milestone | Done when | Est. |
|---|---|---|---|
| M0 | Scaffold + tooling + CI | `/health` test passes in GitHub Actions | 0.5 day |
| M1 | Config, logging, DB, Alembic, docker-compose | `docker compose up` → `/ready` green, tables exist | 1 day |
| M2 | Ingestion: parsers, chunker, embedder, storage | Upload a PDF, chunks + vectors in DB, unit tests for chunker | 2 days |
| M3 | Retrieval: `/search` (vector) | Relevant chunks returned with scores | 1 day |
| M4 | Generation: `/query` structured + citation verifier + threshold refusal | Cited answers, refusals on off-topic Qs, verifier tests | 2 days |
| M5 | Minimal web UI | Upload + ask + citations visible in browser | 1 day |
| M6 | **Deploy** (Cloud Run + Neon) + seed demo | Public URL works | 1 day |
| M7 | Eval harness + baseline report | Report committed, threshold calibrated | 1.5 days |
| M8 | Hybrid retrieval + RRF | Before/after eval table committed | 1 day |
| M9 | SSE streaming endpoint (**deferred**: polish, not core value; the agentic project comes first) | Tokens stream in UI, citations arrive at end | 1 day |
| M10 | Polish: README, diagram, PDF text repair, demo media, resume bullets | Recruiter-ready, pinned | 1 day |

~12 to 14 focused days. Deployed and usable by M6 (about day 8). After M6, the project is shippable; everything after strengthens it.

## 25. First Git Commit
Message: `chore: scaffold project with FastAPI, tooling, and CI`

Contents:
- `README.md` (name, pitch, "under active development", planned architecture)
- `LICENSE` (MIT)
- `.gitignore` (Python, .env, .venv, IDE)
- `.env.example`
- `pyproject.toml` + `uv.lock` (deps, ruff, mypy, pytest config)
- `.pre-commit-config.yaml`
- `src/grounded/__init__.py`, `src/grounded/main.py` (app factory + `/health`)
- `tests/test_health.py`
- `.github/workflows/ci.yml`
- `docs/DESIGN.md` (this file)

Commit habits from here: small commits, conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`), one feature branch + PR per milestone (shows team workflow even solo).

---

## What This Project Proves to Employers
| Skill | Evidence in repo |
|---|---|
| RAG end to end | Own pipeline, no framework hiding it |
| Embeddings + vector search | pgvector, HNSW index, cosine similarity, threshold calibration |
| Hybrid retrieval | Full-text + vector + RRF, justified by eval delta |
| Structured outputs | API-enforced JSON schema (structured outputs) + Pydantic validation |
| Hallucination mitigation | 5-layer strategy, verified citations, measured refusal accuracy |
| AI evaluation | Golden set, retrieval + answer metrics, LLM-as-judge, committed reports |
| LLM API integration | Two providers behind interfaces, retries, token tracking |
| Backend engineering | FastAPI, layered architecture, async SQLAlchemy, migrations, error model |
| Security awareness | Prompt injection defense, upload validation, secrets handling, rate limits |
| Testing | Unit + integration with real Postgres, fakes for LLMs, CI gates |
| Observability | Structured logs, request IDs, query logs with latency/tokens |
| DevOps | Docker, compose, GitHub Actions, Cloud Run, keyless deploy auth |
| Engineering judgment | ADRs, explicit postponed list, honest limitations |

**Recruiter reality check:** RAG apps are common in portfolios. What separates this one: verified citations, a measured refusal path, committed eval numbers, and a live URL. Skip those and it looks like every other RAG demo. Do not call it "production" in your resume. Call it "production-minded" or describe the practices.

## Interview Preparation
**RAG fundamentals**
1. Walk me through what happens when a user asks a question.
2. What is an embedding? Why does cosine similarity work for semantic search?
3. Why did you chunk at 500 tokens with 75 overlap? How would you tune it?
4. What happens if a relevant answer spans two chunks?
5. Why hybrid search? Give an example where pure vector search fails.
6. Explain Reciprocal Rank Fusion. Why not just average scores?
7. How did you choose the relevance threshold? What if you swap embedding models?

**Architecture / data**
8. Why pgvector instead of Pinecone/Qdrant? When would you switch?
9. What is an HNSW index? What's the tradeoff vs exact search?
10. Why don't you store raw files? What does that cost you?
11. What happens if embedding fails halfway through ingestion?
12. How would you handle 10,000 documents or 100 concurrent uploads?
13. Why no LangChain?

**LLM / hallucination**
14. How do you make the model return valid JSON reliably? (And what broke when forced tool use was removed?)
15. How do you know a citation is real?
16. Can your system still hallucinate? How?
17. What is prompt injection, and how could a malicious PDF attack this system?
18. How would you handle documents larger than the context window?

**Evaluation**
19. How do you know your system is good? What metrics?
20. What is LLM-as-judge and what are its weaknesses?
21. How did your eval change a design decision?
22. Hit Rate vs MRR: what's the difference?

**Engineering**
23. How do you test code that calls an LLM?
24. How do you keep API keys safe in CI and on Cloud Run?
25. What would you monitor in production? What alerts would you set?
26. How would you add multi-tenant auth?
27. What was the hardest bug you hit building this?
28. What parts did AI write, and how did you verify them? (Answer honestly. This question is coming.)
