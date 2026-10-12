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
- Get answers with citations that are **verified in code**, not trusted from the model
- Get an explicit "not in your documents" instead of a confident guess

## Stack

FastAPI · PostgreSQL + pgvector · SQLAlchemy 2.0 (async) + Alembic · Claude (generation) · OpenAI embeddings · structlog · Docker · Google Cloud Run

## Roadmap

- [x] M0: Project scaffold, tooling, CI
- [x] M1: Config, structured logging, database schema, migrations, Docker Compose
- [x] M2: Document ingestion (parse, chunk, embed, store)
- [x] M3: Vector search
- [x] M4: Cited answers with citation verification and refusals
- [x] M5: Web UI
- [x] M6: Deploy (Cloud Run + Neon, rate limits, daily LLM budget)
- [ ] M7: Evaluation harness
- [ ] M8: Hybrid retrieval
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
