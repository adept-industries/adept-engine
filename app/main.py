"""Internal HTTP process for Adept Engine health and readiness."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException

from app.core.logging import configure_logging
from app.db.session import current_schema_version, get_database_engine
from app.risk.model import risk_model

configure_logging()
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    try:
        risk_model.load()
    except Exception as exc:
        logger.error("pr_risk_model_startup_failed", error=str(exc))
    yield


app = FastAPI(title="Adept Engine", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "UP", "modelReady": risk_model.ready}


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
