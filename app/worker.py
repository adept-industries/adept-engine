import time

import structlog

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import current_schema_version, get_database_engine


def run() -> None:
    configure_logging()
    logger = structlog.get_logger()
    settings = get_settings()
    database_engine = get_database_engine()

    logger.info(
        "engine_worker_starting",
        worker_id=settings.engine_worker_id,
        dispatch_enabled=False,
    )

    while True:
        try:
            version = current_schema_version(database_engine)
            logger.info(
                "engine_worker_idle",
                worker_id=settings.engine_worker_id,
                schema_version=version,
                reason="Phase 1 has no job handlers",
            )
        except Exception:
            logger.warning(
                "engine_worker_database_unavailable",
                worker_id=settings.engine_worker_id,
            )

        time.sleep(settings.engine_poll_interval_ms / 1000)


if __name__ == "__main__":
    run()
