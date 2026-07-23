from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy.engine import RowMapping


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    id: UUID
    job_type: str
    payload: dict[str, Any]
    priority: int
    attempts: int
    max_attempts: int
    locked_by: str
    created_at: datetime
    updated_at: datetime
    version: int

    @classmethod
    def from_row(cls, row: RowMapping) -> ClaimedJob:
        return cls(
            id=cast(UUID, row["id"]),
            job_type=cast(str, row["job_type"]),
            payload=cast(dict[str, Any], row["payload"]),
            priority=cast(int, row["priority"]),
            attempts=cast(int, row["attempts"]),
            max_attempts=cast(int, row["max_attempts"]),
            locked_by=cast(str, row["locked_by"]),
            created_at=cast(datetime, row["created_at"]),
            updated_at=cast(datetime, row["updated_at"]),
            version=cast(int, row["version"]),
        )
