# Grounded

[![CI](https://github.com/RonaldLenser/grounded-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/RonaldLenser/grounded-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A RAG knowledge assistant that answers only from your documents, cites the exact page and passage for every claim, and says so when the answer isn't there.**

> Status: under active development. See the [design doc](docs/DESIGN.md) for the full architecture and roadmap.

## What it will do

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
- [ ] M3: Vector search
- [ ] M4: Cited answers with citation verification and refusals
- [ ] M5: Web UI
- [ ] M6: Deploy
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

### Try ingestion

Set `OPENAI_API_KEY` and `ADMIN_API_KEY` in `.env`, start the app, then:

```bash
# create a collection
curl -X POST localhost:8000/v1/collections \
  -H "X-API-Key: $ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"name": "my-docs"}'

# upload a document (PDF, .md, or .txt)
curl -X POST localhost:8000/v1/collections/<collection-id>/documents \
  -H "X-API-Key: $ADMIN_API_KEY" -F "file=@policy.pdf"
```

Or use the interactive docs at http://localhost:8000/docs (click **Authorize** and paste the admin key).

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
