# syntax=docker/dockerfile:1

# ---------- Stage 1: builder ----------
# Has uv and build tooling. Installs dependencies into /app/.venv.
FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: this layer is reused from cache
# until pyproject.toml or uv.lock change, so code edits rebuild in seconds.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then our own code.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---------- Stage 2: runtime ----------
# Only Python + the finished venv. No uv, no build tools, no source tree:
# smaller image, smaller attack surface.
FROM python:3.12-slim-trixie AS runtime

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken

# tiktoken downloads its tokenizer file on first use. Fetch it at build time so
# the running container never needs that download (faster cold starts, and
# no runtime dependency on an external file host).
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chown -R app:app /app/.tiktoken

# Never run as root: if the app is compromised, the attacker isn't root.
USER app
EXPOSE 8000

# Shell form so $PORT expands (Cloud Run sets PORT=8080). `exec` makes uvicorn
# PID 1, so it receives shutdown signals and stops gracefully.
CMD ["sh", "-c", "exec uvicorn grounded.main:create_app --factory --host 0.0.0.0 --port ${PORT} --no-access-log --proxy-headers"]
