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

## Planned stack

FastAPI · PostgreSQL + pgvector · Claude (generation) · OpenAI embeddings · Docker · Google Cloud Run

## Roadmap

- [x] M0: Project scaffold, tooling, CI
- [ ] M1: Config, logging, database, migrations, Docker Compose
- [ ] M2: Document ingestion (parse, chunk, embed, store)
- [ ] M3: Vector search
- [ ] M4: Cited answers with citation verification and refusals
- [ ] M5: Web UI
- [ ] M6: Deploy
- [ ] M7: Evaluation harness
- [ ] M8: Hybrid retrieval
- [ ] M9: Streaming
- [ ] M10: Docs and demo

## Development

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # install Python 3.12 + dependencies
uv run pre-commit install                 # enable commit hooks
uv run uvicorn grounded.main:app --reload # http://localhost:8000/docs
uv run pytest                             # run tests
```

## License

MIT
