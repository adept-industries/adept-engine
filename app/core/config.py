from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
    )

    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "adept"
    postgres_user: str = "adept"
    postgres_password: SecretStr = SecretStr("change_me_local_only")

    engine_poll_interval_ms: int = Field(default=1000, ge=100, le=60_000)
    engine_worker_id: str = Field(default="local-worker-1", min_length=1, max_length=128)
    engine_max_job_attempts: int = Field(default=8, ge=1, le=100)
    engine_job_lock_timeout_seconds: int = Field(default=900, ge=30, le=86_400)

    risk_model_dir: str = Field(default="model_artifacts")
    stale_pr_hours_threshold: int = Field(default=120, ge=1)
    outcome_observation_window_days: int = Field(default=14, ge=1)
    app_internal_engine_token: str = Field(default="")

    github_app_id: str = ""
    github_app_private_key_base64: SecretStr = SecretStr("")

    jira_client_id: str = ""
    jira_client_secret: SecretStr = SecretStr("")

    app_integration_encryption_active_key_version: int = Field(default=1, ge=1)
    app_integration_encryption_key_v1_base64: SecretStr = SecretStr("")

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
