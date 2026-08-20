"""Provider API clients used by durable background jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


class ProviderError(RuntimeError):
    """A provider call failed and can be retried."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = (
            min(900.0, max(0.0, retry_after_seconds)) if retry_after_seconds is not None else None
        )


class ProviderPermanentError(ProviderError):
    """A provider rejected a request that cannot succeed unchanged."""


class ProviderConfigurationError(ProviderPermanentError):
    """Required provider configuration is absent or malformed."""


def response_retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse and bound an HTTP Retry-After delta or date."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return min(900.0, max(0.0, float(value)))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except TypeError, ValueError, OverflowError:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return min(900.0, max(0.0, (retry_at - datetime.now(UTC)).total_seconds()))
