# Adept Engine

The engine is Adept's internal Python process and background-worker foundation.

## Current status

The engine polls and claims durable jobs safely, recovers stale claims, applies bounded retries, dispatches provider synchronization/backfill/webhook work, performs idempotent normalization, completes guarded workspace deletion, and calculates versioned DORA snapshots. Production classification uses configured branch, workflow, and environment patterns; pull-request linkage uses exact SHAs, normalized commit membership, and a merge-window fallback. GitHub production failures and recoveries drive normalized incidents, while recalculation work is repository-deduplicated and limited to affected calendar periods plus their preceding periods.

PR-risk inference uses only the approved `jitfine-expert-pr-risk-mvp-v1` artifact and the frozen `ns, nd, nf, entropy, la, ld, fix` order. Open-PR webhook updates and repository backfills collect complete GitHub file and commit membership, persist `jitfine-pr-features-v1`, and atomically upsert one model-version prediction. Runtime file scope is all files returned by GitHub; the persisted feature payload and model metadata document the known training/runtime limitation. The score is review-prioritization support, not proof of a defect.

Project issue ingestion is isolated from DORA and PR-risk recalculation. GitHub `issues` webhooks maintain live state, while issue-only repository jobs backfill every open issue and exclude pull requests returned by GitHub's Issues API. Jira issue-only jobs page through unresolved issues for the explicitly mapped, tracked Jira projects. Completed syncs close or resolve stale local rows that are no longer returned by the providers.

The API's Flyway migrations exclusively own the schema. The engine supports schema versions 7 through 15 during the forward-compatible rollout and must not add Alembic or create tables.

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

- `GET /health` reports HTTP-process liveness and that process's `modelReady` flag.
- `GET /ready` requires PostgreSQL and a supported Flyway V7–V15 schema.

## Worker concurrency

One worker process/container runs **two job-processing threads** by default.
`ENGINE_WORKER_THREADS` accepts `1` (sequential rollback) or `2`; higher values
are rejected to keep concurrency bounded on the shared VM. For example:

```bash
ENGINE_WORKER_THREADS=1 uv run python -m app.worker
```

For Docker, set this in the worker container's environment; adding a variable
only to Compose's interpolation `.env` file does not automatically pass it in.
No second worker container, database migration, or autoscaler is required.

- Each thread claims one job with PostgreSQL `FOR UPDATE SKIP LOCKED` and has
  a unique owner ID, including a process-start UUID. SQLAlchemy's engine/pool
  is shared, but individual database connections and transactions are not.
  The row lock ends with the claim transaction; advisory locks protect execution.
- Different repositories can run concurrently. A repository advisory lock
  serializes work for the same repository. Workspace-wide jobs (catalog/Jira
  sync and deletion) exclude repository work in that workspace. Busy jobs are
  requeued for five seconds without spending an attempt, allowing other work.
- Each active job has a lightweight heartbeat thread with its own database
  session. It refreshes the claim at most every 30 seconds and holds a job
  advisory lock. Stale recovery skips that lock, even if the heartbeat is late.
  Closing the session releases its locks; sessions holding locks are never
  returned to the connection pool.
- Losing that guard session terminates the worker process rather than letting
  an unprotected handler continue. The existing Docker restart policy starts
  it again; unfinished jobs are retried after the stale timeout (default 15
  minutes since the last heartbeat), or marked `DEAD` when attempts are exhausted.
  Retries remain **at least once**, not a guarantee of exactly-once external side effects.
- The PR-risk model is shared and loading/prediction are protected by a lock.
  Provider requests and database work can still overlap; this does not promise
  doubled CPU throughput or bypass provider rate limits. The worker loads it on
  first prediction. The separate HTTP process loads its own copy at startup;
  its health endpoints do not prove the background worker is making progress.
- SIGTERM/SIGINT stop new claims and allow in-flight handlers to finish. The
  deployment workflow allows 120 seconds before Docker force-stops the old
  container. Work interrupted after that uses the normal recovery/retry path.

Worker logs include `consumer_count` on startup and the unique `worker_id` on
claims/errors. With two active jobs, up to two additional heartbeat sessions
are open. Deploy/restart the existing worker to activate this change; no
production resources are changed by local development.

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
docker build -t adept-engine:latest .
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
