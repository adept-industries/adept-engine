# Adept Engine

The engine is Adept's internal Python process and background-worker foundation.

## Current status

Phase 1 provides Python 3.14, FastAPI health/readiness, SQLAlchemy access to the shared database, safe job-claim/retry primitives, tests, and a container image. The running worker intentionally remains idle until real handlers arrive in Phase 5.

The API's Flyway V1–V7 migrations exclusively own the schema. This repository must not add Alembic or create tables.

## Install

```bash
uv sync --locked
```

CI and the image use uv 0.11.16. The lockfile is the dependency source of truth.

## Run natively

Start PostgreSQL and the API first, then:

```bash
set -a
source ../.env
set +a
uv run uvicorn app.main:app --reload --port 8000
```

- `GET /health` checks only process liveness.
- `GET /ready` requires PostgreSQL and successful Flyway V1–V7.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy app tests
uv run pytest -m "not integration"
```

Database integration tests require a disposable API-migrated database, `TEST_DATABASE_URL`, and `ENGINE_TEST_DATABASE_ALLOWED=true`.

## Image

```bash
docker build -t adept-engine:phase1 .
```
