from functools import lru_cache

from sqlalchemy import Engine, create_engine, text

from app.core.config import get_settings

SUPPORTED_SCHEMA_VERSIONS = frozenset({"7", "8", "9", "10", "11", "12", "13", "14"})


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

    schema_version = str(version)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS or processing_jobs != "processing_jobs":
        raise RuntimeError("supported Flyway schema V7 through V14 is not ready")
    return schema_version
