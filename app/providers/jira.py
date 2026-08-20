from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import httpx

from app.core.config import Settings
from app.providers import (
    ProviderConfigurationError,
    ProviderError,
    ProviderPermanentError,
    response_retry_after_seconds,
)
from app.providers.github import ProviderPage


@dataclass(frozen=True, slots=True)
class JiraOAuthTokens:
    access_token: str
    refresh_token: str
    expires_in_seconds: int
    scopes: list[str]


class JiraClient:
    def __init__(
        self,
        settings: Settings,
        cloud_id: str,
        access_token: str,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._cloud_id = cloud_id
        self._access_token = access_token
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url="https://api.atlassian.com",
            timeout=20.0,
            headers={"Accept": "application/json", "User-Agent": "adept-engine"},
        )

    def __enter__(self) -> JiraClient:
        return self

    def __exit__(self, *_args: object) -> None:
        if self._owns_client:
            self._client.close()

    def list_projects(
        self, start_at: int, *, max_results: int = 50
    ) -> ProviderPage[dict[str, Any]]:
        body = self._request_json(
            "GET",
            "/project/search",
            params={"startAt": start_at, "maxResults": max_results},
        )
        values = body.get("values", [])
        if not isinstance(values, list):
            raise ProviderError("Jira projects response field values is not a list")
        projects = [cast(dict[str, Any], item) for item in values if isinstance(item, dict)]
        is_last = bool(body.get("isLast", False))
        total = int(body.get("total", start_at + len(projects)))
        next_start = start_at + len(projects)
        if is_last or not projects or next_start >= total:
            return ProviderPage(projects, None)
        return ProviderPage(projects, next_start)

    def refresh_webhook(self, webhook_id: int) -> str:
        body = self._request_json(
            "PUT",
            "/webhook/refresh",
            json_body={"webhookIds": [webhook_id]},
        )
        expiration = body.get("expirationDate")
        if not isinstance(expiration, str) or not expiration:
            raise ProviderError("Jira did not return the refreshed webhook expiration")
        return expiration

    def webhook_exists(self, webhook_id: int) -> bool:
        """Return whether Atlassian still lists the stored dynamic webhook."""
        start_at = 0
        for _page_number in range(1_000):
            body = self._request_json(
                "GET",
                "/webhook",
                params={"startAt": start_at, "maxResults": 100},
            )
            values = body.get("values")
            if not isinstance(values, list):
                raise ProviderError("Jira webhooks response field values is not a list")
            if any(
                isinstance(item, dict)
                and isinstance(item.get("id"), int)
                and item["id"] == webhook_id
                for item in values
            ):
                return True
            if body.get("isLast") is True:
                return False
            page_size_raw = body.get("maxResults", len(values))
            if isinstance(page_size_raw, bool) or not isinstance(page_size_raw, int):
                raise ProviderError("Jira returned an invalid webhooks page")
            if page_size_raw <= 0:
                raise ProviderError("Jira returned an invalid webhooks page")
            start_at += page_size_raw
        raise ProviderError("Jira returned too many webhook pages")

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, int] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                f"/ex/jira/{self._cloud_id}/rest/api/3{path}",
                params=params,
                json=json_body,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Jira request failed: {exc}") from exc
        _raise_for_provider_status(response)
        body = response.json()
        if not isinstance(body, dict):
            raise ProviderError("Jira returned an unexpected response")
        return cast(dict[str, Any], body)


def refresh_oauth_token(
    settings: Settings,
    current_refresh_token: str,
    *,
    http_client: httpx.Client | None = None,
) -> JiraOAuthTokens:
    client_id = settings.jira_client_id.strip()
    client_secret = settings.jira_client_secret.get_secret_value().strip()
    if not client_id or not client_secret:
        raise ProviderConfigurationError(
            "JIRA_CLIENT_ID and JIRA_CLIENT_SECRET are required to refresh Jira tokens"
        )
    owns_client = http_client is None
    client = http_client or httpx.Client(base_url="https://auth.atlassian.com", timeout=20.0)
    try:
        try:
            response = client.post(
                "/oauth/token",
                json={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": current_refresh_token,
                },
                headers={"Accept": "application/json", "Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"Jira token refresh failed: {exc}") from exc
        _raise_for_provider_status(response)
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("access_token"), str):
            raise ProviderError("Jira did not return a refreshed access token")
        refresh_token = body.get("refresh_token", current_refresh_token)
        if not isinstance(refresh_token, str) or not refresh_token:
            raise ProviderError("Jira did not return a usable refresh token")
        scopes_raw = body.get("scope", "")
        scopes = scopes_raw.split() if isinstance(scopes_raw, str) else []
        return JiraOAuthTokens(
            access_token=body["access_token"],
            refresh_token=refresh_token,
            expires_in_seconds=int(body.get("expires_in", 3600)),
            scopes=scopes,
        )
    finally:
        if owns_client:
            client.close()


def _raise_for_provider_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    message = f"Jira returned HTTP {response.status_code}"
    retry_after = response_retry_after_seconds(response)
    if response.status_code == 429:
        raise ProviderError(message, retry_after_seconds=retry_after)
    if response.status_code in {400, 401, 403, 404, 410, 422}:
        raise ProviderPermanentError(message)
    raise ProviderError(message, retry_after_seconds=retry_after)
