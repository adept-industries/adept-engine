# Adept Engine

The engine is Adept's internal Python process and background-worker foundation.

## Current status

Phase 1 provides Python 3.14, FastAPI health/readiness, SQLAlchemy access to the shared database, safe job-claim/retry primitives, tests, and a container image. The running worker intentionally remains idle until real handlers arrive in Phase 5.

The API's Flyway migrations exclusively own the schema. The engine supports schema versions 7, 8, 9, 10, and 11 during the forward-compatible rollout and must not add Alembic or create tables.

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
- `GET /ready` requires PostgreSQL and a supported Flyway V7, V8, V9, V10, or V11 schema.

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

After the complete `CI` workflow succeeds for a push to `main`, the publish
workflow builds Linux AMD64 and pushes exactly one immutable image tag:

```text
ghcr.io/adept-industries/adept-engine:sha-<full-commit>
```

Pull-request runs, failed CI runs, and non-main branches never publish. The
workflow uses GitHub's short-lived `GITHUB_TOKEN`; it does not require a PAT or
any application/AWS secret.
