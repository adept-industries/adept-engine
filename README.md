# Adept Engine

The engine is Adept's internal Python process and background-worker foundation for polling, webhooks, and data processing.

## Tech Stack
- **Language**: Python (uv for dependency management)
- **Framework**: FastAPI (uvicorn)

## Getting Started

1. **Install Dependencies:**
   ```bash
   uv sync --locked
   ```
2. **Run the API locally:**
   ```bash
   set -a && source ../.env && set +a
   uv run uvicorn app.main:app --reload --port 8000
   ```
3. **Run the Background Worker:**
   ```bash
   set -a && source ../.env && set +a
   uv run python -m app.worker
   ```

## Testing & Quality Checks
```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy app tests
uv run pytest -m "not integration"
```

Integration tests require a test database setup:
```bash
ENGINE_TEST_DATABASE_ALLOWED=true TEST_DATABASE_URL=postgresql+psycopg://adept:password@localhost:5432/adept_engine_test uv run pytest -m integration
```
