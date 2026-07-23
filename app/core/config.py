from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "adept"
    postgres_user: str = "adept"
    postgres_password: SecretStr = Field(
    default=SecretStr(""),
    validate_default=False,
)

    engine_poll_interval_ms: int = Field(default=1000, ge=100, le=60_000)
    engine_worker_id: str = Field(default="local-worker-1", min_length=1, max_length=128)
    engine_max_job_attempts: int = Field(default=8, ge=1, le=100)

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
