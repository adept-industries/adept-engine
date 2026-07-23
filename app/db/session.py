from functools import lru_cache

from sqlalchemy import Engine, create_engine, text

from app.core.config import get_settings


@lru_cache
def get_database_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True)


def current_schema_version(database_engine: Engine) -> str:
    with database_engine.connect() as connection:
        version = connection.execute(
            text(
                """
                SELECT version
                FROM flyway_schema_history
                WHERE success = true
                ORDER BY installed_rank DESC
                LIMIT 1
                """
            )
        ).scalar_one_or_none()

        processing_jobs = connection.execute(
            text("SELECT to_regclass('public.processing_jobs')::text")
        ).scalar_one_or_none()

    if str(version) != "7" or processing_jobs != "processing_jobs":
        raise RuntimeError("Flyway V1-V7 schema is not ready")
    return str(version)
