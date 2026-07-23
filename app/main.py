import structlog
from fastapi import FastAPI, HTTPException

from app.core.logging import configure_logging
from app.db.session import current_schema_version, get_database_engine

configure_logging()
logger = structlog.get_logger()

app = FastAPI(title="Adept Engine", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        version = current_schema_version(get_database_engine())
    except Exception:
        logger.warning("engine_readiness_failed")
        raise HTTPException(
            status_code=503,
            detail="database or Flyway schema is not ready",
        ) from None

    return {"status": "READY", "schemaVersion": version}
