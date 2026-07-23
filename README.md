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
````

### Verify the engine locally

First run the API so it migrates the shared PostgreSQL database. In terminal 1:

```bash
cd /Users/rnp/Developer/adept-local

docker compose --env-file .env \
  -f adept-api/infra/local/compose.yaml \
  up -d postgres mailpit

cd adept-api
set -a
source ../.env
set +a
./mvnw spring-boot:run -Dspring-boot.run.profiles=local
```

In terminal 2:

```bash
cd /Users/rnp/Developer/adept-local/adept-engine

uv sync --locked
set -a
source ../.env
set +a
uv run uvicorn app.main:app --reload --port 8000
```

In terminal 3:

```bash
curl -fsS http://localhost:8000/health
curl -fsS http://localhost:8000/ready
```

For destructive-safety reasons, database integration tests should use a disposable test database prepared through the engine CI job or the full integration workflow, not the ordinary `adept` development database. Run local non-database checks with:

```bash
cd /Users/rnp/Developer/adept-local/adept-engine

uv run ruff format --check .
uv run ruff check .
uv run mypy app tests
uv run pytest -m "not integration"
docker build -t adept-engine:phase1 .
docker run --rm adept-engine:phase1 \
  uv run --frozen --no-sync python --version
```

### Commit, push, and merge the engine PR

```bash
cd /Users/rnp/Developer/adept-local/adept-engine

git status --short
git diff --check
git add .
git commit -m "feat: add phase 1 engine foundation"
git push -u origin feat/phase-1-engine-foundation
```

Wait for both engine CI jobs and the Docker image build before merging.

---