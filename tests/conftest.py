import os
from collections.abc import Iterator
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import Engine, create_engine, text


class JobFactory:
    def __init__(self, database_engine: Engine) -> None:
        self.database_engine = database_engine
        self.ids: list[UUID] = []

    def insert(
        self,
        *,
        status: str = "PENDING",
        priority: int = 100,
        attempts: int = 0,
        max_attempts: int = 8,
        available_offset_seconds: int = 0,
        created_offset_seconds: int = 0,
    ) -> UUID:
        job_id = uuid4()
        with self.database_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO processing_jobs (
                        id, job_type, status, priority, attempts, max_attempts,
                        available_at, created_at, updated_at
                    ) VALUES (
                        :id, 'RECALCULATE_METRICS', :status, :priority,
                        :attempts, :max_attempts,
                        now() + make_interval(secs => :available_offset),
                        now() + make_interval(secs => :created_offset),
                        now()
                    )
                    """
                ),
                {
                    "id": job_id,
                    "status": status,
                    "priority": priority,
                    "attempts": attempts,
                    "max_attempts": max_attempts,
                    "available_offset": available_offset_seconds,
                    "created_offset": created_offset_seconds,
                },
            )
        self.ids.append(job_id)
        return job_id

    def row(self, job_id: UUID) -> dict[str, Any]:
        with self.database_engine.connect() as connection:
            row = (
                connection.execute(
                    text("SELECT * FROM processing_jobs WHERE id = :id"),
                    {"id": job_id},
                )
                .mappings()
                .one()
            )
        return dict(row)

    def cleanup(self) -> None:
        with self.database_engine.begin() as connection:
            for job_id in self.ids:
                connection.execute(
                    text("DELETE FROM processing_jobs WHERE id = :id"),
                    {"id": job_id},
                )


@pytest.fixture(scope="session")
def database_engine() -> Iterator[Engine]:
    if os.getenv("ENGINE_TEST_DATABASE_ALLOWED") != "true":
        pytest.skip("set ENGINE_TEST_DATABASE_ALLOWED=true for database integration tests")

    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.fail("TEST_DATABASE_URL must point to a disposable Flyway-migrated database")

    engine = create_engine(database_url, pool_pre_ping=True)
    with engine.connect() as connection:
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
    if str(version) != "7":
        pytest.fail("integration database must contain API Flyway V1-V7")

    yield engine
    engine.dispose()


@pytest.fixture
def job_factory(database_engine: Engine) -> Iterator[JobFactory]:
    factory = JobFactory(database_engine)
    yield factory
    factory.cleanup()
