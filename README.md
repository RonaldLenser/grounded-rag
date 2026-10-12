# Grounded

[![CI](https://github.com/RonaldLenser/grounded-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/RonaldLenser/grounded-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A RAG knowledge assistant that answers only from your documents, cites the exact page and passage for every claim, and says so when the answer isn't there.**

## Live demo

**https://grounded-1004795551444.us-east4.run.app**

Preloaded with two public NIST documents (AI Risk Management Framework 1.0 and the Generative AI
Profile, 112 pages). Ask something they cover ("What are the four functions of the AI RMF
Core?") and something they don't ("What is the capital of France?") to see a cited answer and
a refusal. First load after idle takes a few seconds (scales to zero).

Rate limited per IP with a global daily cap on LLM calls. Uploads require an admin key.

> Status: deployed and usable. Evaluation harness, hybrid retrieval and streaming are next.
> See the [design doc](docs/DESIGN.md) for the full architecture and roadmap.

## What it does

- Upload PDFs, Markdown, and text files
- Ask questions in plain English
- Get answers built from claims whose quotes are **verified in code**; a claim that fails verification is dropped, never shown
- Get an explicit "not in your documents" instead of a confident guess

## Stack

FastAPI · PostgreSQL + pgvector + full-text search · SQLAlchemy 2.0 (async) + Alembic · Claude (generation) · OpenAI embeddings · structlog · Docker · Google Cloud Run

## Roadmap

- [x] M0: Project scaffold, tooling, CI
- [x] M1: Config, structured logging, database schema, migrations, Docker Compose
- [x] M2: Document ingestion (parse, chunk, embed, store)
- [x] M3: Vector search
- [x] M4: Cited answers with citation verification and refusals
- [x] M5: Web UI
- [x] M6: Deploy (Cloud Run + Neon, rate limits, daily LLM budget)
- [x] M7: Evaluation harness (33-question golden set, LLM judge, results below)
- [x] M8: Hybrid retrieval (IDF-weighted keyword search + RRF; won the eval, now the default)
- [ ] M9: Streaming
- [ ] M10: Docs and demo

## Quickstart (Docker)

Requires [Docker](https://www.docker.com/products/docker-desktop/).

```bash
docker compose up --build
```

Starts Postgres + pgvector, applies database migrations, then serves the API.

- Web app: http://localhost:8000 (upload documents, ask questions, click citations)
- API docs: http://localhost:8000/docs
- Liveness: http://localhost:8000/health
- Readiness (checks the database): http://localhost:8000/ready

## Local development

Requires [uv](https://docs.astral.sh/uv/) and Docker (for the database).

```bash
cp .env.example .env                    # then edit values as needed
uv sync                                 # install Python 3.12 + dependencies
uv run pre-commit install               # enable commit hooks
docker compose up -d db                 # start only the database
uv run alembic upgrade head             # create/upgrade the schema
uv run uvicorn grounded.main:create_app --factory --reload
```

### Try the API

Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `ADMIN_API_KEY` in `.env`, start the app, then:

```bash
# create a collection
curl -X POST localhost:8000/v1/collections \
  -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "my-docs"}'

# upload a document (PDF, .md, or .txt)
curl -X POST localhost:8000/v1/collections/<collection-id>/documents \
  -H "X-API-Key: $ADMIN_API_KEY" -F "file=@policy.pdf"

# search: the most relevant chunks for a question (no admin key needed)
curl -X POST localhost:8000/v1/collections/<collection-id>/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the policy say about accountability?", "top_k": 3}'

# ask: a cited answer, or an explicit "not in your documents"
curl -X POST localhost:8000/v1/collections/<collection-id>/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is accountable for AI risk?"}'
```

Or use the interactive docs at http://localhost:8000/docs (click **Authorize** and paste the admin key).

### Smoke test

With the app running (`docker compose up --build -d`), run the end-to-end check against
real OpenAI embeddings and Claude:

```bash
uv run python scripts/smoke_test.py                  # local
uv run python scripts/smoke_test.py https://<url>    # a deployed instance
```


### Deploy

Google Cloud Run + Neon Postgres, with per-IP rate limits and a global daily LLM budget.
Step-by-step guide: [docs/DEPLOY.md](docs/DEPLOY.md).

### Evaluation

A 33-question golden set (answerable, multi-source, unanswerable, off-topic) is run against a
live instance over HTTP and scored for retrieval, refusals, citations and, with `--judge`, claim
support and answer correctness (LLM judge with stated reasons).

```bash
uv run python -m evals.run https://<url> --judge --label my-run     # writes docs/evals/
uv run python -m evals.run --from-json docs/evals/<run>.json --judge  # re-judge a saved run
```

**Results** (live demo, 2026-10-12, Claude Sonnet 5.5, judged by Claude Haiku 5.5). Same 33
questions, before and after hybrid retrieval:

| What | Vector only ([v1](docs/evals/2026-10-12-v1.md)) | Hybrid ([v2](docs/evals/2026-10-12-v2-hybrid.md)) |
|---|---|---|
| Answerable questions answered (not refused) | 22 / 24 | **24 / 24** |
| Answers correct vs hand-written reference (LLM judge) | 22 / 22 | **24 / 24** |
| Right page retrieved: top 3 / top 6 | 88% / 92% | **100% / 100%** |
| Retrieval MRR | 0.81 | **0.88** |
| Unanswerable on-topic questions: no unsupported answer | 6 / 6 | 6 / 6 |
| Off-topic questions refused before any LLM call | 3 / 3 | 3 / 3 |
| Shown claims fully supported by their quotes (LLM judge) | 72 / 78 | 81 / 86 |
| Shown claims NOT supported by their quotes | 0 | 0 |
| Answered questions citing an expected page | 22 / 22 | 24 / 24 |
| Latency p50 / p95 (LLM answers) | 2.9s / 6.5s | 3.4s / 7.1s |

What the numbers say:
- Vector search missed two pages that keyword search finds by exact terms ("residual risk",
  "eliminate negative risk entirely"). Hybrid retrieval (vector + IDF-weighted keyword search,
  merged with Reciprocal Rank Fusion) fixed both with no regressions, so it is now the default.
- Every "partially supported" claim was checked by hand: the extra detail came from the source
  document but outside the quoted span, or was attribution wording ("MANAGE 1.3 states...").
  No fabricated fact was found.
- With hybrid, one unanswerable question ("What penalties does NIST impose?") got a `partial`
  answer: the one supported fact that the framework is voluntary, plus a notice that the
  documents only partly answer. No penalty was invented.
- The baseline run ([report](docs/evals/2026-10-12-baseline.md)) caught a real bug: claim-format
  answers hit the 1024-token output cap and returned 502. Fixed (2048 cap, at most 8 claims).
- Small sample (33 questions): one question is about 4 points, and the keyword method was
  chosen after seeing these questions' failures. Treat this as a regression check, not a
  benchmark; a held-out question set is the next step for an unbiased number.

### Privacy and data retention

The demo stores each question's raw text in the `query_logs` table (for debugging and evals),
with no automatic expiry; deleting a collection deletes its logs. Answers are not stored, and
application logs never contain question or answer text. Don't ask the public demo anything
personal. Details: "Data retention" in [docs/DESIGN.md](docs/DESIGN.md).

### Tests

```bash
uv run pytest                           # all tests (integration tests skip if the DB is down)
uv run pytest -m "not integration"      # unit tests only, no database needed
```

Integration tests run against a real PostgreSQL + pgvector database. They verify that migrations
apply and roll back cleanly, that the ORM models match the migrated schema, and that Postgres
enforces the constraints (deduplication, cascading deletes, valid statuses).

### Database migrations

```bash
uv run alembic revision --autogenerate -m "describe change"   # then REVIEW the generated file
uv run alembic upgrade head
```

## License

MIT
