# Adept Engine

The engine is Adept's internal Python process and background-worker foundation.

## Current status

The implementation through Phase 6 polls and claims durable jobs safely, recovers stale claims, applies bounded retries, dispatches provider synchronization/backfill/webhook work, performs idempotent normalization, completes guarded workspace deletion, and calculates versioned DORA snapshots. Production classification uses configured branch, workflow, and environment patterns; pull-request linkage uses exact SHAs, normalized commit membership, and a merge-window fallback. GitHub production failures and recoveries drive normalized incidents, while recalculation work is repository-deduplicated and limited to affected calendar periods plus their preceding periods.

The API's Flyway migrations exclusively own the schema. The engine supports schema versions 7 through 13 during the forward-compatible rollout and must not add Alembic or create tables.

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

Run the worker in a separate terminal with the same environment:

```bash
uv run python -m app.worker
```

- `GET /health` checks only process liveness.
- `GET /ready` requires PostgreSQL and a supported Flyway V7–V13 schema.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy app tests
uv run pytest -m "not integration"
```

Database integration tests require a disposable API-migrated database. Never point this command at a development, staging, or production database:

```bash
ENGINE_TEST_DATABASE_ALLOWED=true \
TEST_DATABASE_URL=postgresql+psycopg://adept:password@localhost:5432/adept_engine_test \
uv run pytest -m integration
```

The engine CI database job provisions PostgreSQL, runs the real API Flyway migrations, and then executes this integration suite.

## Image

```bash
docker build -t adept-engine:phase6 .
```

After the complete `CI` workflow succeeds for a push to `main`, the publish
workflow builds Linux AMD64 and pushes exactly one immutable image tag:

```text
ghcr.io/adept-industries/adept-engine:sha-<full-commit>
```

Pull-request runs, failed CI runs, and non-main branches never publish. A
serialized production job deploys that exact image to AWS Lightsail, waits for
the engine API and worker checks, and only then reports a terminal GitHub
Deployment status for the tested SHA and the `production` environment. The
workflow uses GitHub's short-lived `GITHUB_TOKEN` for GHCR and Deployment API
access plus the existing `LIGHTSAIL_HOST`, `LIGHTSAIL_USER`, and
`LIGHTSAIL_SSH_KEY` secrets; no PAT is required.

<!-- mock PR to verify real-time PR risk evaluation in Adept dashboard -->
