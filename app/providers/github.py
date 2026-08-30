from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from typing import Any, cast

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from app.core.config import Settings
from app.providers import (
    ProviderConfigurationError,
    ProviderError,
    ProviderPermanentError,
    response_retry_after_seconds,
)


@dataclass(frozen=True, slots=True)
class ProviderPage[T]:
    items: list[T]
    next_page: int | None
    total_count: int | None = None


class GithubClient:
    def __init__(
        self,
        settings: Settings,
        installation_id: int,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._installation_id = installation_id
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url="https://api.github.com",
            timeout=20.0,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "adept-engine",
            },
        )
        self._installation_token: str | None = None

    def __enter__(self) -> GithubClient:
        return self

    def __exit__(self, *_args: object) -> None:
        if self._owns_client:
            self._client.close()

    def list_installation_repositories(
        self, page: int, *, per_page: int = 100
    ) -> ProviderPage[dict[str, Any]]:
        body = self._request_json(
            "GET",
            "/installation/repositories",
            params={"page": page, "per_page": per_page},
        )
        repositories = _mapping_list(body, "repositories")
        total = int(body.get("total_count", len(repositories)))
        next_page = page + 1 if page * per_page < total and len(repositories) == per_page else None
        return ProviderPage(repositories, next_page)

    def list_closed_pull_requests(
        self,
        owner: str,
        repository: str,
        page: int,
        *,
        per_page: int = 50,
    ) -> ProviderPage[dict[str, Any]]:
        body = self._request_json(
            "GET",
            f"/repos/{owner}/{repository}/pulls",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "page": page,
                "per_page": per_page,
            },
        )
        items = _list_body(body)
        return ProviderPage(items, page + 1 if len(items) == per_page else None)

    def list_open_pull_requests(
        self,
        owner: str,
        repository: str,
        page: int,
        *,
        per_page: int = 50,
    ) -> ProviderPage[dict[str, Any]]:
        body = self._request_json(
            "GET",
            f"/repos/{owner}/{repository}/pulls",
            params={
                "state": "open",
                "sort": "updated",
                "direction": "desc",
                "page": page,
                "per_page": per_page,
            },
        )
        items = _list_body(body)
        return ProviderPage(items, page + 1 if len(items) == per_page else None)

    def list_open_issues(
        self,
        owner: str,
        repository: str,
        page: int,
        *,
        per_page: int = 100,
    ) -> ProviderPage[dict[str, Any]]:
        body = self._request_json(
            "GET",
            f"/repos/{owner}/{repository}/issues",
            params={
                "state": "open",
                "sort": "updated",
                "direction": "desc",
                "page": page,
                "per_page": per_page,
            },
        )
        items = _list_body(body)
        return ProviderPage(items, page + 1 if len(items) == per_page else None)

    def get_pull_request(self, owner: str, repository: str, number: int) -> dict[str, Any]:
        return self._request_json("GET", f"/repos/{owner}/{repository}/pulls/{number}")

    def list_pull_request_commits(
        self, owner: str, repository: str, number: int
    ) -> list[dict[str, Any]]:
        """Return the complete, chronological commit membership for one pull request."""
        commits: list[dict[str, Any]] = []
        page = 1
        while True:
            body = self._request_json(
                "GET",
                f"/repos/{owner}/{repository}/pulls/{number}/commits",
                params={"page": page, "per_page": 100},
            )
            batch = _list_body(body)
            commits.extend(batch)
            if len(batch) < 100:
                return commits
            page += 1

    def list_pull_request_files(
        self, owner: str, repository: str, number: int
    ) -> list[dict[str, Any]]:
        """Return every file GitHub exposes for a pull request.

        GitHub caps this endpoint at 3,000 files. Callers compare the result
        count with ``changed_files`` and decline to score incomplete changes.
        """
        files: list[dict[str, Any]] = []
        for page in range(1, 31):
            body = self._request_json(
                "GET",
                f"/repos/{owner}/{repository}/pulls/{number}/files",
                params={"page": page, "per_page": 100},
            )
            batch = _list_body(body)
            files.extend(batch)
            if len(batch) < 100:
                break
        return files

    def list_workflow_runs(
        self,
        owner: str,
        repository: str,
        page: int,
        *,
        branch: str | None,
        created_from: str,
        created_to: str,
        per_page: int = 50,
    ) -> ProviderPage[dict[str, Any]]:
        params: dict[str, str | int] = {
            "status": "completed",
            "created": f"{created_from}..{created_to}",
            "page": page,
            "per_page": per_page,
        }
        if branch:
            params["branch"] = branch
        body = self._request_json(
            "GET",
            f"/repos/{owner}/{repository}/actions/runs",
            params=params,
        )
        items = _mapping_list(body, "workflow_runs")
        total = int(body.get("total_count", len(items)))
        next_page = page + 1 if page * per_page < total and len(items) == per_page else None
        return ProviderPage(items, next_page, total)

    def list_deployments(
        self,
        owner: str,
        repository: str,
        page: int,
        *,
        per_page: int = 50,
    ) -> ProviderPage[dict[str, Any]]:
        body = self._request_json(
            "GET",
            f"/repos/{owner}/{repository}/deployments",
            params={"page": page, "per_page": per_page},
        )
        items = _list_body(body)
        return ProviderPage(items, page + 1 if len(items) == per_page else None)

    def terminal_deployment_status(
        self,
        owner: str,
        repository: str,
        deployment_id: int,
        *,
        per_page: int = 100,
    ) -> dict[str, Any] | None:
        """Return the newest success/failure even when GitHub later marks it inactive."""
        page = 1
        while True:
            body = self._request_json(
                "GET",
                f"/repos/{owner}/{repository}/deployments/{deployment_id}/statuses",
                params={"page": page, "per_page": per_page},
            )
            statuses = _list_body(body)
            for status in statuses:
                state = str(status.get("state") or "").lower()
                if state in {"success", "failure"}:
                    return status
            if len(statuses) < per_page:
                return None
            page += 1

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        token = self._installation_access_token()
        try:
            response = self._client.request(
                method,
                path,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"GitHub request failed: {exc}") from exc
        _raise_for_provider_status(response, "GitHub")
        body = response.json()
        if isinstance(body, list):
            return {"_items": body}
        if not isinstance(body, dict):
            raise ProviderError("GitHub returned an unexpected response")
        return cast(dict[str, Any], body)

    def _installation_access_token(self) -> str:
        if self._installation_token:
            return self._installation_token
        app_jwt = _generate_app_jwt(self._settings)
        try:
            response = self._client.post(
                f"/app/installations/{self._installation_id}/access_tokens",
                headers={"Authorization": f"Bearer {app_jwt}"},
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"GitHub installation authentication failed: {exc}") from exc
        _raise_for_provider_status(response, "GitHub")
        body = response.json()
        token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise ProviderError("GitHub did not return an installation token")
        self._installation_token = token
        return token


def _generate_app_jwt(settings: Settings) -> str:
    app_id = settings.github_app_id.strip()
    encoded_key = settings.github_app_private_key_base64.get_secret_value().strip()
    if not app_id or not encoded_key:
        raise ProviderConfigurationError(
            "GITHUB_APP_ID and GITHUB_APP_PRIVATE_KEY_BASE64 are required for GitHub jobs"
        )
    try:
        private_key_bytes = base64.b64decode(encoded_key, validate=True)
        private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
    except (ValueError, TypeError) as exc:
        raise ProviderConfigurationError("GITHUB_APP_PRIVATE_KEY_BASE64 is invalid") from exc

    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise ProviderConfigurationError("GitHub App private key must be an RSA key")

    now = int(time.time())
    header = _base64url_json({"alg": "RS256", "typ": "JWT"})
    payload = _base64url_json({"iat": now - 60, "exp": now + 540, "iss": app_id})
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header}.{payload}.{_base64url(signature)}"


def _base64url_json(value: dict[str, Any]) -> str:
    return _base64url(json.dumps(value, separators=(",", ":")).encode("utf-8"))


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _raise_for_provider_status(response: httpx.Response, provider: str) -> None:
    if response.is_success:
        return
    message = f"{provider} returned HTTP {response.status_code}"
    retry_after = response_retry_after_seconds(response)
    exhausted_primary_limit = (
        response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0"
    )
    if exhausted_primary_limit and retry_after is None:
        reset_at = response.headers.get("X-RateLimit-Reset")
        if reset_at is not None:
            try:
                retry_after = min(900.0, max(0.0, float(reset_at) - time.time()))
            except ValueError:
                retry_after = None
    if (
        response.status_code == 429
        or exhausted_primary_limit
        or (response.status_code == 403 and retry_after is not None)
    ):
        raise ProviderError(message, retry_after_seconds=retry_after)
    if response.status_code in {400, 401, 403, 404, 410, 422}:
        raise ProviderPermanentError(message)
    raise ProviderError(message, retry_after_seconds=retry_after)


def _mapping_list(body: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = body.get(key, [])
    if not isinstance(value, list):
        raise ProviderError(f"GitHub response field {key} is not a list")
    return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]


def _list_body(body: dict[str, Any]) -> list[dict[str, Any]]:
    return _mapping_list(body, "_items")
