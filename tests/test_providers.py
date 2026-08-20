import base64

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr

from app.core.config import Settings
from app.providers import ProviderError, ProviderPermanentError
from app.providers.github import GithubClient
from app.providers.jira import JiraClient, refresh_oauth_token


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"postgres_password": SecretStr("test")}
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _github_settings() -> Settings:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return _settings(
        github_app_id="1234",
        github_app_private_key_base64=SecretStr(base64.b64encode(pem).decode()),
    )


def test_github_client_authenticates_and_pages_installation_repositories() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        assert request.headers["Authorization"] == "Bearer installation-token"
        return httpx.Response(
            200,
            json={
                "total_count": 101,
                "repositories": [{"id": number} for number in range(100)],
            },
        )

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with GithubClient(_github_settings(), 99, http_client=http_client) as client:
        result = client.list_installation_repositories(1)

    assert len(result.items) == 100
    assert result.next_page == 2
    assert requests[0].url.path == "/app/installations/99/access_tokens"
    assert requests[1].url.params["page"] == "1"


def test_github_rate_limit_preserves_bounded_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(429, headers={"Retry-After": "1200"})

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with (
        GithubClient(_github_settings(), 99, http_client=http_client) as client,
        pytest.raises(ProviderError) as raised,
    ):
        client.list_installation_repositories(1)

    assert raised.value.retry_after_seconds == 900


def test_github_primary_rate_limit_uses_reset_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.providers.github.time.time", lambda: 1_000.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(
            403,
            headers={
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "2200",
            },
        )

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with (
        GithubClient(_github_settings(), 99, http_client=http_client) as client,
        pytest.raises(ProviderError) as raised,
    ):
        client.list_installation_repositories(1)

    assert raised.value.retry_after_seconds == 900


def test_github_permission_forbidden_remains_permanent() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json={"token": "installation-token"})
        return httpx.Response(403)

    http_client = httpx.Client(
        base_url="https://api.github.com", transport=httpx.MockTransport(handler)
    )
    with (
        GithubClient(_github_settings(), 99, http_client=http_client) as client,
        pytest.raises(ProviderPermanentError),
    ):
        client.list_installation_repositories(1)


def test_jira_client_uses_cloud_api_path_and_paginates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer access-token"
        assert request.url.path == "/ex/jira/cloud-1/rest/api/3/project/search"
        return httpx.Response(
            200,
            json={
                "values": [{"id": "100", "key": "ADEPT", "name": "Adept"}],
                "isLast": False,
                "total": 2,
            },
        )

    http_client = httpx.Client(
        base_url="https://api.atlassian.com", transport=httpx.MockTransport(handler)
    )
    with JiraClient(_settings(), "cloud-1", "access-token", http_client=http_client) as client:
        result = client.list_projects(0, max_results=1)

    assert result.items[0]["key"] == "ADEPT"
    assert result.next_page == 1


def test_jira_refresh_rotates_refresh_token_and_scopes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/oauth/token"
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "scope": "read:jira-work offline_access",
            },
        )

    client = httpx.Client(
        base_url="https://auth.atlassian.com", transport=httpx.MockTransport(handler)
    )
    result = refresh_oauth_token(
        _settings(jira_client_id="client", jira_client_secret=SecretStr("secret")),
        "old-refresh",
        http_client=client,
    )

    assert result.access_token == "new-access"
    assert result.refresh_token == "new-refresh"
    assert result.scopes == ["read:jira-work", "offline_access"]


def test_jira_webhook_exists_paginates_until_stored_id_is_found() -> None:
    starts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        starts.append(request.url.params["startAt"])
        if request.url.params["startAt"] == "0":
            return httpx.Response(
                200,
                json={"values": [{"id": 11}], "isLast": False, "maxResults": 1},
            )
        return httpx.Response(
            200,
            json={"values": [{"id": 991}], "isLast": True, "maxResults": 1},
        )

    http_client = httpx.Client(
        base_url="https://api.atlassian.com", transport=httpx.MockTransport(handler)
    )
    with JiraClient(_settings(), "cloud-1", "access-token", http_client=http_client) as client:
        assert client.webhook_exists(991) is True

    assert starts == ["0", "1"]


def test_jira_rejects_non_retryable_provider_response() -> None:
    client = httpx.Client(
        base_url="https://api.atlassian.com",
        transport=httpx.MockTransport(lambda _request: httpx.Response(401)),
    )
    with (
        JiraClient(_settings(), "cloud-1", "bad-token", http_client=client) as jira,
        pytest.raises(ProviderPermanentError),
    ):
        jira.list_projects(0)
